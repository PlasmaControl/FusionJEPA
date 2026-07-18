"""Latent-prediction training objective for Fusion-JEPA (M3, Task 3.3).

The JEPA world model (:class:`~fusion_jepa.models.jepa.JEPAModel`) predicts a
future-state latent ``z_hat`` and encodes the observed future window into a
target latent ``z_target``; this objective scores the prediction against the
target in *latent* space and returns the shared
:class:`~fusion_jepa.objectives.base.LossOutput`. It is the M3 counterpart of the
raw-prediction objective (Task 2.7) -- same LossOutput contract, same mask-aware,
NaN-safe, float32 discipline -- but the discrepancy lives between latents rather
than raw signal values.

Inputs
------
It consumes exactly the landed :class:`~fusion_jepa.models.jepa.JEPAOutput` --
``z_hat`` / ``z_target`` of shape ``[B, K, S, D]`` (``K == 1`` for the landed
single-window batch) and ``target_valid`` ``[B, K]`` -- plus ``horizon_seconds``
(``[B, K]`` float64, or ``[B]`` reshaped to ``[B, 1]``). ``horizon_seconds`` is
used ONLY to *label* the per-horizon terms and diagnostics; it never enters the
loss value. Every reduction is written K-generic, so a future multi-horizon batch
needs no change here.

Distance and normalization (a disclosed convention)
---------------------------------------------------
Two selectable per-state-token distances over the latent dim ``D``:

* ``"cosine"`` -- cosine distance ``1 - cos(z_hat, z_target)`` averaged over the
  state tokens ``S``. Cosine normalizes *both* operands by definition, so
  ``normalize_targets`` has no additional effect for it.
* ``"smooth_l1"`` -- Huber distance averaged over ``(S, D)``. When
  ``normalize_targets=True`` (default) *both* ``z_hat`` and ``z_target`` are
  L2-normalized over ``D`` first.

The ``normalize_targets`` convention is a deliberate choice: for ``"smooth_l1"``
we normalize BOTH operands, not the target alone. Normalizing only the target
would silently require the predictor to *also* learn to emit unit-norm vectors --
an implicit, ill-conditioned scale constraint whose failure shows up as a loss
dominated by norm mismatch rather than direction. Mapping both operands onto the
unit sphere keeps the Huber metric measuring direction/position on a shared scale
and makes it symmetric with cosine. With ``normalize_targets=False`` the Huber
distance runs on the raw latents.

Target-gradient policy (a constructor flag, by design)
------------------------------------------------------
:class:`JEPAOutput` carries no policy field, yet the M3 plan requires this
objective to *assert* the stop-gradient discipline for the EMA / SHARED_STOPGRAD
policies (where ``z_target`` must be detached/frozen) while still supporting the
END_TO_END_REGULARIZED policy (where gradient deliberately flows through the
target branch). The reconciliation is a single constructor flag
``allow_target_gradients`` (default ``False``): by default a ``z_target`` that
requires grad is a contract violation and raises an actionable error; the
END_TO_END caller opts in with ``allow_target_gradients=True``.

Mask-awareness and the all-invalid batch
-----------------------------------------
Only ``(example, horizon)`` cells flagged by ``target_valid`` contribute. Invalid
cells are neutralized to a finite constant *before* the distance via
``torch.where`` -- whose backward is pure masking -- so a hostile ``NaN``/``inf``
placeholder in an invalid cell can reach neither the loss value nor any gradient
(no ``0 * NaN``). A fully-invalid batch yields a finite ``0.0`` total: the
denominator floor ``_EPS`` engages only when the masked sum is already exactly
zero, so it never perturbs a real cell.

Terms vs diagnostics (the LossOutput contract)
----------------------------------------------
``terms`` are the per-horizon *additive contributions* keyed ``latent/h={seconds}``
that sum EXACTLY to ``total`` (they share one denominator), horizon-weighted.
Inspection-only readouts live in ``diagnostics`` as plain floats:
``latent_mean/h={seconds}`` (the UNWEIGHTED mean cell distance at each horizon,
which the training weights must not distort), ``horizon_weight/h={seconds}`` and
``horizon_weights_uniform`` (so the uniform default is logged as such), plus
``total`` and valid-cell counts. All reduction math runs in float32 regardless of
input/autocast dtype.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from fusion_jepa.models.jepa import JEPAOutput
from fusion_jepa.objectives.base import LossOutput

_DISTANCES = ("cosine", "smooth_l1")
# Denominator floor. Only ever engaged when a masked-out sum is exactly zero (the
# numerator is then also exactly zero, so the ratio is a clean 0); it never
# perturbs a present cell, whose denominator is >= the smallest positive weight.
_EPS = 1.0e-12
# Smallest allowed positive horizon weight (matches raw_prediction). A positive
# weight below this would sit inside the _EPS clamp's territory and silently
# distort the horizon terms it claims to weight; a horizon is either excluded
# (weight exactly 0) or weighted by a well-conditioned value.
_MIN_WEIGHT = 1.0e-6


class LatentPredictionObjective:
    """Mask-aware latent-prediction loss returning a :class:`LossOutput`.

    Args:
        distance: per-state-token discrepancy over the latent dim ``D`` --
            ``"cosine"`` (``1 - cos``) or ``"smooth_l1"`` (Huber,
            ``beta = smooth_l1_beta``).
        horizon_weights: optional 1-D weights over the horizon (``K``) axis,
            applied inside the batch-wide weighted mean. Length must equal ``K``
            at call time. ``None`` means uniform (and is logged as uniform). Each
            weight must be finite and either exactly ``0`` (horizon excluded) or
            ``>= 1e-6`` (well-conditioned against the denominator floor). Only the
            relative magnitudes matter (each mean is scale-invariant).
        normalize_targets: L2-normalize over ``D`` before the distance. For
            ``"smooth_l1"`` this normalizes BOTH operands (see module docstring);
            for ``"cosine"`` it is redundant (cosine normalizes both by
            definition).
        allow_target_gradients: when ``False`` (default) a ``z_target`` that
            requires grad raises -- the EMA / SHARED_STOPGRAD policies must detach
            or freeze the target. Set ``True`` for the END_TO_END_REGULARIZED
            policy, which deliberately backpropagates through the target branch.
        smooth_l1_beta: transition point of the smooth-L1 distance (ignored for
            ``"cosine"``).
    """

    def __init__(
        self,
        distance: str = "cosine",
        horizon_weights: Sequence[float] | Tensor | None = None,
        *,
        normalize_targets: bool = True,
        allow_target_gradients: bool = False,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {_DISTANCES}, got {distance!r}")
        if smooth_l1_beta <= 0:
            raise ValueError(f"smooth_l1_beta must be positive, got {smooth_l1_beta}")
        self.distance = distance
        self.normalize_targets = bool(normalize_targets)
        self.allow_target_gradients = bool(allow_target_gradients)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self._horizon_weights = self._validate_weights(horizon_weights)

    @staticmethod
    def _validate_weights(
        horizon_weights: Sequence[float] | Tensor | None,
    ) -> Tensor | None:
        """Validate + store horizon weights (finiteness / non-negativity)."""
        if horizon_weights is None:
            return None
        weights = torch.as_tensor(horizon_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError(
                "horizon_weights must be a 1-D sequence over horizons, got "
                f"ndim {weights.ndim}"
            )
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("horizon_weights must be finite")
        if bool((weights < 0).any()):
            raise ValueError("horizon_weights must be non-negative")
        if bool(((weights > 0) & (weights < _MIN_WEIGHT)).any()):
            raise ValueError(
                "horizon_weights must be exactly 0 (horizon excluded) or "
                f">= {_MIN_WEIGHT}; a smaller positive weight is ill-conditioned "
                "against the denominator floor"
            )
        return weights

    def __call__(self, out: JEPAOutput, horizon_seconds: Tensor) -> LossOutput:
        """Score ``out.z_hat`` against ``out.z_target`` under ``target_valid``."""
        z_hat = out.z_hat
        z_target = out.z_target
        target_valid = out.target_valid

        if z_hat.ndim != 4:
            raise ValueError(
                f"z_hat must be [B, K, S, D], got shape {tuple(z_hat.shape)}"
            )
        if z_target.shape != z_hat.shape:
            raise ValueError(
                "z_hat and z_target shapes disagree: "
                f"{tuple(z_hat.shape)} vs {tuple(z_target.shape)}"
            )
        batch, n_horizons = z_hat.shape[0], z_hat.shape[1]
        if tuple(target_valid.shape) != (batch, n_horizons):
            raise ValueError(
                "target_valid must be [B, K] matching z_hat; got "
                f"{tuple(target_valid.shape)} vs [{batch}, {n_horizons}]"
            )
        if not self.allow_target_gradients and z_target.requires_grad:
            raise ValueError(
                "z_target requires grad but allow_target_gradients=False. The "
                "EMA and SHARED_STOPGRAD target-update policies must detach or "
                "freeze the target (z_target must not require grad); only the "
                "END_TO_END_REGULARIZED policy backpropagates through the target "
                "branch -- construct "
                "LatentPredictionObjective(allow_target_gradients=True) for it."
            )

        horizons = self._resolve_horizon_seconds(
            horizon_seconds, batch, n_horizons
        )  # [B, K] float64
        weights = self._resolve_weights(n_horizons, z_hat.device)  # [K] float32

        valid = target_valid.to(torch.bool)  # [B, K]
        valid_f = valid.to(torch.float32)  # [B, K]

        cell = self._cell_distances(z_hat, z_target, valid)  # [B, K] float32
        # Zero any invalid cell's (already finite) distance so it contributes
        # nothing; the input-level neutralization already removed its gradient.
        cell = torch.where(valid, cell, torch.zeros_like(cell))

        effective = valid_f * weights.view(1, n_horizons)  # [B, K]
        num_horizon = (effective * cell).sum(dim=0)  # [K]
        # One shared denominator so per-horizon terms sum EXACTLY to total.
        total_den = effective.sum().clamp_min(_EPS)
        total = num_horizon.sum() / total_den

        # Unweighted per-horizon inspection means (detached): a faithful readout
        # of error vs. horizon that the training weights must not distort.
        cell_detached = cell.detach()
        horizon_mean = (valid_f * cell_detached).sum(dim=0) / valid_f.sum(
            dim=0
        ).clamp_min(_EPS)  # [K]

        n_valid = float(valid_f.sum().item())
        diagnostics: dict[str, float] = {
            "total": float(total.detach().item()),
            "n_valid_cells": n_valid,
            "valid_cell_fraction": n_valid / float(valid.numel()),
            "horizon_weights_uniform": (1.0 if self._horizon_weights is None else 0.0),
        }

        terms: dict[str, Tensor] = {}
        labels = self._horizon_labels(horizons, valid)  # length K, distinct
        for index, label in enumerate(labels):
            terms[f"latent/{label}"] = num_horizon[index] / total_den
            diagnostics[f"latent_mean/{label}"] = float(horizon_mean[index].item())
            diagnostics[f"horizon_weight/{label}"] = float(weights[index].item())

        return LossOutput(total=total, terms=terms, diagnostics=diagnostics)

    def _cell_distances(self, z_hat: Tensor, z_target: Tensor, valid: Tensor) -> Tensor:
        """Per-``(example, horizon)`` distance in float32 -> ``[B, K]``.

        Runs entirely in float32 (plan risk R12) regardless of the input/autocast
        dtype. Invalid cells are neutralized to a finite constant BEFORE the
        distance so a hostile placeholder there reaches neither the value nor the
        gradient: ``torch.where`` backward is pure masking, so the invalid cell's
        latent contributes an exactly-zero gradient with no ``0 * NaN``.
        """
        z_hat_f = z_hat.to(torch.float32)
        z_target_f = z_target.to(torch.float32)
        valid_expand = valid.view(valid.shape[0], valid.shape[1], 1, 1)
        neutral = torch.zeros_like(z_hat_f)
        z_hat_f = torch.where(valid_expand, z_hat_f, neutral)
        z_target_f = torch.where(valid_expand, z_target_f, neutral)

        if self.distance == "cosine":
            sim = F.cosine_similarity(z_hat_f, z_target_f, dim=-1, eps=_EPS)
            return (1.0 - sim).mean(dim=-1)  # [B, K]

        if self.normalize_targets:
            z_hat_f = F.normalize(z_hat_f, p=2.0, dim=-1, eps=_EPS)
            z_target_f = F.normalize(z_target_f, p=2.0, dim=-1, eps=_EPS)
        elementwise = F.smooth_l1_loss(
            z_hat_f, z_target_f, reduction="none", beta=self.smooth_l1_beta
        )  # [B, K, S, D]
        return elementwise.mean(dim=(-2, -1))  # [B, K]

    @staticmethod
    def _resolve_horizon_seconds(
        horizon_seconds: Tensor, batch: int, n_horizons: int
    ) -> Tensor:
        """Coerce ``horizon_seconds`` to ``[B, K]`` float64 (``[B]`` -> ``[B, 1]``)."""
        horizons = horizon_seconds.to(torch.float64)
        if horizons.ndim == 1:
            horizons = horizons.reshape(-1, 1)
        if tuple(horizons.shape) != (batch, n_horizons):
            raise ValueError(
                "horizon_seconds must be [B, K] (or [B] for K == 1) matching "
                f"z_hat; got {tuple(horizon_seconds.shape)} vs "
                f"[{batch}, {n_horizons}]"
            )
        return horizons

    def _resolve_weights(self, n_horizons: int, device: torch.device) -> Tensor:
        """Return the float32 per-horizon weights on ``device`` (uniform default)."""
        if self._horizon_weights is None:
            return torch.ones(n_horizons, dtype=torch.float32, device=device)
        if self._horizon_weights.shape[0] != n_horizons:
            raise ValueError(
                "horizon_weights length "
                f"{self._horizon_weights.shape[0]} != number of horizons "
                f"{n_horizons}"
            )
        return self._horizon_weights.to(device=device, dtype=torch.float32)

    @staticmethod
    def _horizon_labels(horizons: Tensor, valid: Tensor) -> list[str]:
        """Stable per-horizon labels ``h={seconds}`` for terms/diagnostics.

        ``horizon_seconds`` is labeling-only. A horizon *index* is a fixed lead
        time shared across the batch in a real schedule, so where an example is
        valid its seconds define the label: we average the valid examples'
        seconds for the representative value (identical to the shared value in the
        common uniform case), falling back to all examples for a horizon with none
        valid. Labels are assumed distinct per index (true for a real horizon
        schedule); a collision would silently merge term keys, so it raises.
        """
        horizons_f = horizons.to(torch.float64)  # [B, K]
        valid_f = valid.to(torch.float64)  # [B, K]
        counts = valid_f.sum(dim=0)  # [K]
        representative = torch.where(
            counts > 0,
            (valid_f * horizons_f).sum(dim=0) / counts.clamp_min(_EPS),
            horizons_f.mean(dim=0),
        )  # [K]
        labels = [f"h={value:g}" for value in representative.tolist()]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "horizon_seconds produced duplicate per-horizon labels "
                f"{labels}; horizons must be distinct per index for stable "
                "per-horizon term keys"
            )
        return labels
