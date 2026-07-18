"""Raw-prediction (reconstruction) training objective for Fusion-JEPA (M2).

The raw baseline predicts future *raw* signal values; this objective scores
those predictions against the batch's target values under the observation mask
and returns the shared :class:`~fusion_jepa.objectives.base.LossOutput`.

Inputs are three plain ``{modality: Tensor}`` dicts -- ``predictions`` (from
:meth:`RawWorldModel.forward`), ``targets`` (``batch.target``), and
``target_masks`` (``batch.target_mask``, ``True`` == observed). Keeping the
objective a function of dicts (rather than of the whole
:class:`~fusion_jepa.data.batch.FusionBatch`) makes it trivially unit-testable
with tiny hand-built cases and decouples it from the batch schema; the caller
simply forwards ``batch.target`` / ``batch.target_mask``.

Reduction order (fixed by the M2 spec)
--------------------------------------
For every ``(example, modality)`` cell we take a **masked, horizon-weighted mean
over the (channel, frame) axes** (the target-frame axis *is* the horizon axis,
because the predictor runs ``K == 1``; see the Task 2.6 report). Those cell
losses are then **averaged over the modalities present in each example**, then
**over the examples**, to a single scalar. Only cells/examples that contain at
least one observed, positively-weighted target contribute; empty ones are
excluded from their denominator, never counted as a real zero.

Mask-awareness and 100%-masked targets (a real TokaMark reality)
----------------------------------------------------------------
Real windows can have an entire target block missing for a modality. Such a cell
contributes nothing and is dropped from every denominator, so the loss stays
finite. Unobserved target entries are neutralised *before* the distance is
computed (their placeholder may be ``NaN``/``inf``), so no ``NaN`` can reach the
backward pass: masked positions produce exactly-zero loss and exactly-zero
gradient.

Terms vs diagnostics (the LossOutput contract)
----------------------------------------------
``terms`` honours the :class:`LossOutput` contract exactly: one entry per
modality, defined as that modality's *additive contribution*
``mean_b[present(b, m) * cell(b, m) / n_modalities_present(b)]`` over counted
examples, so ``sum(terms.values()) == total``. Non-additive *inspection*
readouts live in ``diagnostics`` as floats: ``modality_mean/<m>`` (mean cell
loss over the examples where the modality is present) and
``horizon_mean/<t>`` (the unweighted masked mean error at each target frame,
which the training horizon weights must not distort).

All reduction math runs in float32 regardless of autocast, and the diagnostics
are plain Python floats, per the global spec.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from fusion_jepa.objectives.base import LossOutput

_DISTANCES = ("mse", "smooth_l1")
# Denominator floor. Only ever engaged when a masked-out sum is exactly zero
# (numerator is then also exactly zero, so the ratio is a clean 0); it never
# perturbs a present cell, whose denominator is >= the smallest positive weight.
_EPS = 1.0e-12
# Smallest allowed positive horizon weight. A positive weight below this would
# sit inside the _EPS clamp's territory and silently distort the cell means it
# claims to weight; a frame is either excluded (weight exactly 0) or weighted
# by a well-conditioned value.
_MIN_WEIGHT = 1.0e-6


class RawPredictionObjective:
    """Mask-aware raw reconstruction loss returning a :class:`LossOutput`.

    Args:
        distance: per-element discrepancy, ``"mse"`` (squared error) or
            ``"smooth_l1"`` (Huber, ``beta`` = ``smooth_l1_beta``).
        horizon_weights: optional 1-D weights applied on the target-frame
            (horizon) axis inside every cell's masked mean. Length must equal the
            shared number of target frames ``T_tgt``. ``None`` means uniform.
            Only the *relative* magnitudes matter (each cell mean is
            scale-invariant); every weight must be finite and either exactly
            ``0`` (frame excluded) or ``>= 1e-6`` (well-conditioned against
            the internal denominator floor).
        smooth_l1_beta: transition point of the smooth-L1 distance (ignored for
            ``"mse"``).
    """

    def __init__(
        self,
        distance: str = "mse",
        horizon_weights: Sequence[float] | Tensor | None = None,
        *,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        if distance not in _DISTANCES:
            raise ValueError(
                f"distance must be one of {_DISTANCES}, got {distance!r}"
            )
        if smooth_l1_beta <= 0:
            raise ValueError(
                f"smooth_l1_beta must be positive, got {smooth_l1_beta}"
            )
        self.distance = distance
        self.smooth_l1_beta = float(smooth_l1_beta)
        if horizon_weights is None:
            self._horizon_weights: Tensor | None = None
        else:
            weights = torch.as_tensor(horizon_weights, dtype=torch.float32)
            if weights.ndim != 1:
                raise ValueError(
                    "horizon_weights must be a 1-D sequence over target frames, "
                    f"got ndim {weights.ndim}"
                )
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("horizon_weights must be finite")
            if bool((weights < 0).any()):
                raise ValueError("horizon_weights must be non-negative")
            if bool(((weights > 0) & (weights < _MIN_WEIGHT)).any()):
                raise ValueError(
                    "horizon_weights must be exactly 0 (frame excluded) or "
                    f">= {_MIN_WEIGHT}; a smaller positive weight is "
                    "ill-conditioned against the denominator floor"
                )
            self._horizon_weights = weights

    def _elementwise(self, pred: Tensor, target: Tensor) -> Tensor:
        """Per-element discrepancy, same shape as the inputs."""
        if self.distance == "mse":
            return (pred - target) ** 2
        return F.smooth_l1_loss(
            pred, target, reduction="none", beta=self.smooth_l1_beta
        )

    def __call__(
        self,
        predictions: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        target_masks: Mapping[str, Tensor],
    ) -> LossOutput:
        """Score ``predictions`` against ``targets`` under ``target_masks``."""
        modalities = list(targets)
        if not modalities:
            raise ValueError(
                "RawPredictionObjective requires at least one target modality"
            )

        reference = targets[modalities[0]]
        if reference.ndim != 3:
            raise ValueError(
                "targets must be [B, C, T_tgt]; "
                f"{modalities[0]!r} has ndim {reference.ndim}"
            )
        batch_size = reference.shape[0]
        n_frames = reference.shape[2]
        device = predictions[modalities[0]].device

        weights = self._resolve_weights(n_frames, device)  # [T]

        cell_losses: list[Tensor] = []  # each [B]
        cell_present: list[Tensor] = []  # each [B] bool
        per_modality_valid_fraction: dict[str, float] = {}

        # Per-horizon (frame) accumulators over (example, modality, channel).
        # Detached: horizon means are diagnostics (inspection-only), never a
        # gradient path.
        horizon_num = torch.zeros(n_frames, dtype=torch.float32, device=device)
        horizon_den = torch.zeros(n_frames, dtype=torch.float32, device=device)
        total_valid = torch.zeros((), dtype=torch.float32, device=device)
        total_elements = 0.0

        for modality in modalities:
            pred, target, mask = self._prepare(
                modality, predictions, targets, target_masks, batch_size, n_frames
            )
            # Neutralise unobserved entries BEFORE the distance so a NaN/inf
            # placeholder can never enter the graph; masked positions then yield
            # exactly-zero loss and exactly-zero gradient.
            safe_target = torch.where(mask, target, pred.detach())
            elementwise = self._elementwise(pred, safe_target)  # [B, C, T]

            mask_f = mask.to(torch.float32)
            effective = mask_f * weights.view(1, 1, n_frames)  # [B, C, T]
            cell_num = (effective * elementwise).sum(dim=(1, 2))  # [B]
            cell_den = effective.sum(dim=(1, 2))  # [B]
            present = cell_den > 0
            # cell_den is exactly 0 only when the cell is empty, and cell_num is
            # then exactly 0 too, so the floored ratio is a clean 0.
            cell_loss = cell_num / cell_den.clamp_min(_EPS)  # [B]
            cell_losses.append(cell_loss)
            cell_present.append(present)

            # Per-horizon inspection means are *unweighted* masked means at
            # each frame: a faithful readout of error vs. horizon that the
            # training weights must not distort.
            detached = elementwise.detach()
            horizon_num = horizon_num + (mask_f * detached).sum(dim=(0, 1))
            horizon_den = horizon_den + mask_f.sum(dim=(0, 1))

            observed = mask_f.sum()
            total_valid = total_valid + observed
            total_elements += float(mask.numel())
            per_modality_valid_fraction[modality] = (
                float(observed.item()) / float(mask.numel())
            )

        total, terms, modality_means = self._reduce(
            modalities, cell_losses, cell_present
        )

        diagnostics: dict[str, float] = {
            "total": float(total.detach().item()),
            "n_valid_targets": float(total_valid.item()),
            "valid_target_fraction": (
                float(total_valid.item()) / total_elements
                if total_elements > 0
                else 0.0
            ),
        }
        diagnostics.update(modality_means)
        for frame in range(n_frames):
            diagnostics[f"horizon_mean/{frame}"] = float(
                (horizon_num[frame] / horizon_den[frame].clamp_min(_EPS)).item()
            )
        for modality, fraction in per_modality_valid_fraction.items():
            diagnostics[f"valid_fraction/{modality}"] = fraction

        return LossOutput(total=total, terms=terms, diagnostics=diagnostics)

    def _resolve_weights(self, n_frames: int, device: torch.device) -> Tensor:
        """Return the fp32 per-frame weights on ``device`` (uniform by default)."""
        if self._horizon_weights is None:
            return torch.ones(n_frames, dtype=torch.float32, device=device)
        if self._horizon_weights.shape[0] != n_frames:
            raise ValueError(
                "horizon_weights length "
                f"{self._horizon_weights.shape[0]} != number of target frames "
                f"{n_frames}"
            )
        return self._horizon_weights.to(device=device, dtype=torch.float32)

    def _prepare(
        self,
        modality: str,
        predictions: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        target_masks: Mapping[str, Tensor],
        batch_size: int,
        n_frames: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Validate and cast one modality's (pred, target, mask) to fp32/bool."""
        if modality not in predictions:
            raise ValueError(
                f"predictions is missing target modality {modality!r}"
            )
        if modality not in target_masks:
            raise ValueError(
                f"target_masks is missing target modality {modality!r}"
            )
        target = targets[modality]
        pred = predictions[modality]
        mask = target_masks[modality]
        if target.ndim != 3:
            raise ValueError(
                f"target {modality!r} must be [B, C, T_tgt], got shape "
                f"{tuple(target.shape)}"
            )
        if target.shape[0] != batch_size or target.shape[2] != n_frames:
            raise ValueError(
                f"target {modality!r} shares B and T_tgt with the other "
                f"modalities; got {tuple(target.shape)} vs B={batch_size}, "
                f"T_tgt={n_frames}"
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes disagree for {modality!r}: "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if mask.shape != target.shape:
            raise ValueError(
                f"mask and target shapes disagree for {modality!r}: "
                f"{tuple(mask.shape)} vs {tuple(target.shape)}"
            )
        return pred.to(torch.float32), target.to(torch.float32), mask.bool()

    @staticmethod
    def _reduce(
        modalities: Sequence[str],
        cell_losses: list[Tensor],
        cell_present: list[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
        """Reduce cell losses to ``(total, terms, modality inspection means)``.

        ``total`` is the spec's mean over present modalities per example, then
        over present examples. ``terms`` are the per-modality *additive
        contributions* ``mean_b[present(b, m) * cell(b, m) /
        n_modalities_present(b)]`` over counted examples, so
        ``sum(terms.values()) == total`` exactly (the LossOutput contract). A
        modality absent everywhere contributes a zero term; its non-additive
        inspection mean (mean cell loss over examples where it IS present) is
        returned separately for ``diagnostics``.
        """
        loss_matrix = torch.stack(cell_losses, dim=1)  # [B, M]
        present_matrix = torch.stack(cell_present, dim=1).to(torch.float32)  # [B, M]

        example_num = (present_matrix * loss_matrix).sum(dim=1)  # [B]
        example_den = present_matrix.sum(dim=1)  # [B] == modalities present
        example_loss = example_num / example_den.clamp_min(_EPS)  # [B]

        present_f = (example_den > 0).to(torch.float32)
        n_examples = present_f.sum().clamp_min(_EPS)
        total = (present_f * example_loss).sum() / n_examples

        # Each example's cell shares: cell(b, m) / |M_b| where present, else 0.
        shares = (
            present_matrix
            * loss_matrix
            / example_den.clamp_min(_EPS).unsqueeze(1)
        )  # [B, M]

        terms: dict[str, Tensor] = {}
        modality_means: dict[str, float] = {}
        for index, modality in enumerate(modalities):
            terms[f"modality/{modality}"] = shares[:, index].sum() / n_examples
            m_present = present_matrix[:, index]
            mean = (m_present * loss_matrix[:, index]).sum() / m_present.sum(
            ).clamp_min(_EPS)
            modality_means[f"modality_mean/{modality}"] = float(
                mean.detach().item()
            )
        return total, terms, modality_means
