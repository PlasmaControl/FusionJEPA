"""Collapse diagnostics, thresholds, and the variance-covariance regularizer (M3).

Representation collapse is the failure mode of a JEPA-style objective: the target
encoder (and, downstream, the predictor) can trivially minimise the latent
prediction loss by shrinking every embedding onto a single point or a
low-dimensional subspace. This module provides the two halves of the collapse
defence used by the ``END_TO_END_REGULARIZED`` policy:

* :func:`collapse_diagnostics` -- a *pure, gradient-free* readout of a latent
  batch's health (always logged, never optimised against), plus
  :class:`CollapseThresholds` / :func:`collapse_warnings` which turn those floats
  into human-readable warnings that are *surfaced, never auto-tuned away*.
* :class:`VarianceCovarianceRegularizer` -- the *differentiable* VICReg-style
  variance + covariance penalty that actually flows gradients to keep the batch
  from collapsing.

The Trainer/validation wiring of these pieces lives in Task 3.8, not here.

Latent shape and folding
------------------------
Every entry point accepts a latent tensor whose **last dim is the latent
dimension ``D``**; all leading dims are flattened into the sample axis ``N``.
So ``[N, D]`` is used as-is, and ``[B, K, S, D]`` (batch x horizon x state x
feature) folds to ``[B*K*S, D]`` -- each ``(b, k, s)`` position is one sample.

Numerical policy (plan risk R12)
--------------------------------
All statistics are computed in **float32 regardless of input dtype**: bf16 is cast
up *first*, because the eigen/SVD used by :func:`collapse_diagnostics` is unstable
in bf16. :func:`collapse_diagnostics` additionally **detaches** its inputs and
returns plain Python floats (JSON-safe, carry no gradient). The regularizer keeps
its graph -- it is a loss.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from fusion_jepa.objectives.base import LossOutput

# A per-dim std below this counts the dimension as "near constant".
NEAR_CONSTANT_EPS = 1.0e-3
# Denominator floor for variance/spectrum ratios. Only ever engaged when the
# denominator is genuinely ~0 (a fully-constant batch), where the numerator is
# also ~0 and the floored ratio is a clean, finite value rather than a NaN.
_EPS = 1.0e-12
# Variance-hinge epsilon inside sqrt(var + eps), matching VICReg's 1e-4. Keeps
# the std gradient finite as a dimension's variance approaches zero.
_VAR_EPS = 1.0e-4


def _fold(z: Tensor) -> Tensor:
    """Fold a latent tensor to ``[N, D]`` (last dim is ``D``); cast/detach to fp32."""
    if z.ndim < 2:
        raise ValueError(
            f"latent tensor must have at least 2 dims ([..., D]); got ndim {z.ndim}"
        )
    folded = z.reshape(-1, z.shape[-1])
    if folded.shape[0] < 2:
        raise ValueError(
            "need at least 2 samples after folding leading dims into [N, D]; "
            f"got N={folded.shape[0]}"
        )
    return folded.detach().to(torch.float32)


def _effective_rank(z_centered: Tensor) -> float:
    """Roy & Vetterli effective rank of a centered latent batch.

    Defined as ``exp(H)`` where ``H`` is the Shannon entropy of the
    L1-normalised singular-value spectrum of the centered batch (the singular
    values of ``z_centered`` are the square roots of the covariance eigenvalues,
    so this is the entropy of the *variance directions*). It ranges in
    ``[1, D]`` for any batch with variance. A fully-constant batch has an
    all-zero spectrum; with the denominator floored this yields the floor value
    ``1.0``, which the default ``effective_rank_min`` threshold flags.
    """
    singular_values = torch.linalg.svdvals(z_centered).clamp_min(0.0)
    total = singular_values.sum()
    probs = singular_values / total.clamp_min(_EPS)
    # p * log(p) -> 0 as p -> 0; clamping only the log argument keeps the exact-0
    # terms at 0 (0 * finite == 0) instead of 0 * -inf == NaN.
    log_probs = probs.clamp_min(_EPS).log()
    entropy = -(probs * log_probs).sum()
    return float(torch.exp(entropy).item())


def collapse_diagnostics(z_target: Tensor, z_pred: Tensor) -> dict[str, float]:
    """Return gradient-free collapse diagnostics for a target/predicted batch.

    Args:
        z_target: target latents ``[..., D]`` (the representation that can
            collapse); all leading dims fold into the sample axis.
        z_pred: predicted latents ``[..., D]`` sharing ``D`` with ``z_target``;
            used only for ``pred_target_variance_ratio``.

    Returns:
        A dict of plain Python floats (JSON-safe). The single-latent metrics are
        computed on ``z_target``:

        * ``per_dim_std_mean`` / ``per_dim_std_min`` -- mean/min over dims of the
          per-dimension population std (``sqrt`` of the variance about the batch
          mean, divided by ``N``). Both -> 0 under constant collapse.
        * ``cov_offdiag_mean_abs`` -- mean absolute off-diagonal of the
          population covariance (``÷N``); large under redundant/dimensional
          collapse. Exactly ``0.0`` when ``D == 1`` (no off-diagonal exists).
        * ``effective_rank`` -- Roy & Vetterli effective rank (see
          :func:`_effective_rank`); -> 1 under rank-one collapse.
        * ``mean_norm`` -- L2 norm of the per-dim mean vector ``||E[z]||``
          (inspection only: it has no single monotone "healthy" direction -- a
          centered healthy batch and a shifted-but-healthy one differ only in
          combination with the std metrics -- so it carries *no* threshold).
        * ``latent_norm_std`` -- sample std of the per-sample L2 norms; -> 0 when
          every embedding has collapsed to the same vector.
        * ``pred_target_variance_ratio`` -- ``sum_d Var(z_pred)_d /
          sum_d Var(z_target)_d`` (total-variance ratio, denominator floored);
          ``<< 1`` when predictions are shrunk toward a constant.
        * ``near_constant_fraction`` -- fraction of dims whose per-dim std is
          below ``NEAR_CONSTANT_EPS`` (``1e-3``); -> 1 under constant collapse.
    """
    z = _fold(z_target)
    n_samples, n_dims = z.shape

    mean_vec = z.mean(dim=0)  # [D]
    z_centered = z - mean_vec  # [N, D]
    per_dim_var = z_centered.pow(2).mean(dim=0)  # [D], population (÷N)
    per_dim_std = per_dim_var.clamp_min(0.0).sqrt()  # [D]

    # Off-diagonal covariance magnitude (population covariance, ÷N).
    if n_dims > 1:
        cov = (z_centered.t() @ z_centered) / n_samples  # [D, D]
        offdiag = cov - torch.diag(torch.diagonal(cov))
        cov_offdiag_mean_abs = offdiag.abs().sum() / (n_dims * (n_dims - 1))
    else:
        cov_offdiag_mean_abs = torch.zeros((), dtype=torch.float32)

    norms = z.norm(dim=1)  # [N] per-sample L2 norm

    z_pred_folded = _fold(z_pred)
    if z_pred_folded.shape[1] != n_dims:
        raise ValueError(
            "z_pred and z_target must share the latent dim D; got "
            f"{z_pred_folded.shape[1]} vs {n_dims}"
        )
    pred_var_sum = z_pred_folded.var(dim=0, correction=0).sum()
    target_var_sum = per_dim_var.sum()

    return {
        "per_dim_std_mean": float(per_dim_std.mean().item()),
        "per_dim_std_min": float(per_dim_std.min().item()),
        "cov_offdiag_mean_abs": float(cov_offdiag_mean_abs.item()),
        "effective_rank": _effective_rank(z_centered),
        "mean_norm": float(mean_vec.norm().item()),
        "latent_norm_std": float(norms.std().item()),
        "pred_target_variance_ratio": float(
            (pred_var_sum / target_var_sum.clamp_min(_EPS)).item()
        ),
        "near_constant_fraction": float(
            (per_dim_std < NEAR_CONSTANT_EPS).to(torch.float32).mean().item()
        ),
    }


@dataclass(frozen=True)
class CollapseThresholds:
    """One threshold per diagnostic that has a monotone healthy direction.

    Every field is overridable. ``*_min`` fields warn when the diagnostic falls
    *below* them; ``*_max`` fields warn when it rises *above* them. Defaults
    assume roughly unit-scale target latents (the regime the
    :class:`VarianceCovarianceRegularizer` steers toward via ``std_target``);
    lower the ``*_std_*`` / ``*_ratio_*`` floors for deliberately small-scale
    embeddings. ``mean_norm`` has no field on purpose (no monotone healthy
    direction; see :func:`collapse_diagnostics`).
    """

    per_dim_std_mean_min: float = 0.1
    per_dim_std_min_min: float = 0.01
    cov_offdiag_mean_abs_max: float = 0.5
    effective_rank_min: float = 1.5
    near_constant_fraction_max: float = 0.1
    pred_target_variance_ratio_min: float = 0.1
    latent_norm_std_min: float = 0.01


def collapse_warnings(
    diagnostics: dict[str, float], thresholds: CollapseThresholds
) -> list[str]:
    """Return one human-readable warning per violated threshold (empty = healthy).

    Warnings are *surfaced, never auto-tuned away*: this function only reports,
    it never mutates thresholds. Each violation maps to exactly one string that
    names the offending diagnostic, so tightening/loosening a single threshold
    adds/removes exactly its warning.
    """
    warnings: list[str] = []

    min_checks = (
        ("per_dim_std_mean", thresholds.per_dim_std_mean_min),
        ("per_dim_std_min", thresholds.per_dim_std_min_min),
        ("effective_rank", thresholds.effective_rank_min),
        ("pred_target_variance_ratio", thresholds.pred_target_variance_ratio_min),
        ("latent_norm_std", thresholds.latent_norm_std_min),
    )
    for key, floor in min_checks:
        value = diagnostics[key]
        if value < floor:
            warnings.append(
                f"{key}={value:.4g} below min threshold {floor:.4g} "
                "(possible representation collapse)"
            )

    max_checks = (
        ("cov_offdiag_mean_abs", thresholds.cov_offdiag_mean_abs_max),
        ("near_constant_fraction", thresholds.near_constant_fraction_max),
    )
    for key, ceiling in max_checks:
        value = diagnostics[key]
        if value > ceiling:
            warnings.append(
                f"{key}={value:.4g} above max threshold {ceiling:.4g} "
                "(possible representation collapse)"
            )

    return warnings


class VarianceCovarianceRegularizer:
    """VICReg-style variance + covariance penalty on a latent batch.

    This is the differentiable collapse-prevention loss of the
    ``END_TO_END_REGULARIZED`` policy. Given a latent batch ``[..., D]`` it
    returns a :class:`LossOutput` whose two ``terms`` are **additive** and sum
    exactly to ``total`` (the reviewed 2.7 contract):

    * ``variance`` = ``var_weight * mean_d relu(std_target - std_d)`` -- a hinge
      that pushes every dimension's std up toward ``std_target``;
    * ``covariance`` = ``cov_weight * (sum of squared off-diagonal covariance) /
      D`` -- decorrelates dimensions.

    Non-additive inspection readouts (``std_mean``, ``cov_offdiag_mean_abs``) go
    in ``diagnostics`` as plain floats. All math runs in float32; unlike
    :func:`collapse_diagnostics` this keeps the autograd graph, so gradients flow
    back to the (possibly bf16) input.

    Args:
        std_target: target per-dimension std the variance hinge steers toward.
        var_weight: weight on the variance term.
        cov_weight: weight on the covariance term.
    """

    def __init__(
        self,
        std_target: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
    ) -> None:
        if std_target <= 0:
            raise ValueError(f"std_target must be positive, got {std_target}")
        if var_weight < 0 or cov_weight < 0:
            raise ValueError(
                "var_weight and cov_weight must be non-negative, got "
                f"{var_weight}, {cov_weight}"
            )
        self.std_target = float(std_target)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)

    def __call__(self, z: Tensor) -> LossOutput:
        """Score a latent batch ``[..., D]`` and return a :class:`LossOutput`."""
        if z.ndim < 2:
            raise ValueError(
                f"latent tensor must have at least 2 dims ([..., D]); got ndim {z.ndim}"
            )
        folded = z.reshape(-1, z.shape[-1]).to(torch.float32)
        n_samples, n_dims = folded.shape
        if n_samples < 2:
            raise ValueError(
                f"need at least 2 samples after folding into [N, D]; got N={n_samples}"
            )

        # Variance hinge: push each dim's std up to std_target (unbiased var).
        std = torch.sqrt(folded.var(dim=0, correction=1) + _VAR_EPS)  # [D]
        var_loss = torch.relu(self.std_target - std).mean()

        # Covariance penalty: decorrelate dims (unbiased covariance, ÷(N-1)).
        centered = folded - folded.mean(dim=0)
        cov = (centered.t() @ centered) / (n_samples - 1)  # [D, D]
        offdiag = cov - torch.diag(torch.diagonal(cov))
        cov_loss = offdiag.pow(2).sum() / n_dims

        var_term = self.var_weight * var_loss
        cov_term = self.cov_weight * cov_loss
        total = var_term + cov_term  # terms sum EXACTLY to total

        diagnostics = {
            "std_mean": float(std.mean().detach().item()),
            "cov_offdiag_mean_abs": float(
                offdiag.abs().sum().detach().item() / max(n_dims * (n_dims - 1), 1)
            ),
        }
        return LossOutput(
            total=total,
            terms={"variance": var_term, "covariance": cov_term},
            diagnostics=diagnostics,
        )
