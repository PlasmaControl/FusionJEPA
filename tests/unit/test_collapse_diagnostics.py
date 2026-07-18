"""Unit tests for collapse diagnostics, thresholds, and the VICReg-style
variance-covariance regularizer (Task 3.4).

The diagnostics are pure functions over latent tensors: :func:`collapse_diagnostics`
returns JSON-safe floats describing whether a batch of target/predicted latents is
collapsing, :class:`CollapseThresholds` + :func:`collapse_warnings` turn those floats
into human-readable warnings (empty when healthy), and
:class:`VarianceCovarianceRegularizer` is the differentiable collapse-prevention loss
returning the shared :class:`~fusion_jepa.objectives.base.LossOutput`.
"""

import math

import pytest
import torch

from fusion_jepa.objectives.base import LossOutput
from fusion_jepa.objectives.collapse_regularizers import (
    CollapseThresholds,
    VarianceCovarianceRegularizer,
    collapse_diagnostics,
    collapse_warnings,
)


def test_constant_latents_flagged():
    """Identical rows -> per-dim std ~0, near_constant_fraction ~1, warnings fire."""
    z = torch.full((256, 16), 3.0)
    diag = collapse_diagnostics(z, z)

    assert diag["per_dim_std_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag["per_dim_std_min"] == pytest.approx(0.0, abs=1e-6)
    assert diag["near_constant_fraction"] == pytest.approx(1.0)

    warnings = collapse_warnings(diag, CollapseThresholds())
    assert warnings  # collapse must never be silent


def test_rank_one_latents_have_effective_rank_near_one():
    """Outer-product (rank-one) latents concentrate all variance in one direction."""
    torch.manual_seed(0)
    a = torch.randn(512)
    b = torch.randn(16)
    z = torch.outer(a, b)  # [512, 16], rank one

    diag = collapse_diagnostics(z, z)
    assert diag["effective_rank"] == pytest.approx(1.0, abs=0.1)


def test_effective_rank_is_singular_value_entropy_not_covariance_eigenvalue():
    """Discriminate the SVD-entropy definition from the covariance-eigen one.

    The rank-one and Gaussian cases cannot tell the implemented
    ``exp(H(singular-value spectrum))`` apart from the plausible-but-wrong
    ``exp(H(covariance-eigenvalue spectrum))`` -- both alternatives agree there.
    Here we build a batch whose *centered* singular values are exactly
    ``[s1, s2] = [3, 1]`` (two orthonormal, zero-mean directions scaled by 3 and
    1), where the two definitions diverge sharply:

    * singular-value spectrum ``[0.75, 0.25]`` -> exp(H) = 1.75477 (implemented)
    * covariance-eigenvalue (s_i**2) spectrum ``[0.9, 0.1]`` -> exp(H) = 1.38415
      (the rejected alternative)

    A ~0.37 gap that pins the implementation to the singular-value definition.
    """
    s1, s2 = 3.0, 1.0
    # Two orthonormal, zero-mean direction vectors in R^N (Hadamard columns
    # tiled). Zero-mean => the batch is already centered, so its singular values
    # survive the diagnostic's internal centering exactly.
    base_a = torch.tensor([1.0, -1.0, 1.0, -1.0])
    base_b = torch.tensor([1.0, 1.0, -1.0, -1.0])
    a = base_a.repeat(16)
    a = a / a.norm()
    b = base_b.repeat(16)
    b = b / b.norm()
    z_centered = torch.stack([s1 * a, s2 * b], dim=1)  # svals exactly [3, 1]
    # A nonzero mean the diagnostic must strip before taking the spectrum.
    z = z_centered + torch.tensor([10.0, -5.0])

    diag = collapse_diagnostics(z, z)

    # Hand-computed exp(entropy) of each candidate normalized spectrum.
    sv_probs = [s1 / (s1 + s2), s2 / (s1 + s2)]  # [0.75, 0.25]
    eff_singular = math.exp(-sum(p * math.log(p) for p in sv_probs))  # 1.75477
    cov_probs = [
        s1**2 / (s1**2 + s2**2),
        s2**2 / (s1**2 + s2**2),
    ]  # [0.9, 0.1]
    eff_covariance = math.exp(-sum(p * math.log(p) for p in cov_probs))  # 1.38415

    assert diag["effective_rank"] == pytest.approx(eff_singular, abs=1e-3)
    # And it is nowhere near the covariance-eigenvalue-entropy alternative.
    assert abs(diag["effective_rank"] - eff_covariance) > 0.3


def test_healthy_gaussian_latents_pass_all_thresholds():
    """Seeded iid standard-normal latents (N >> D) trip nothing on defaults."""
    torch.manual_seed(1)
    z_target = torch.randn(4096, 32)
    z_pred = torch.randn(4096, 32)

    diag = collapse_diagnostics(z_target, z_pred)
    assert collapse_warnings(diag, CollapseThresholds()) == []


def test_variance_ratio_detects_shrunk_predictions():
    """Predictions with 1% of the target variance drive the ratio far below 1."""
    torch.manual_seed(2)
    z_target = torch.randn(4096, 32)
    z_pred = 0.1 * torch.randn(4096, 32)  # independent, shrunk

    diag = collapse_diagnostics(z_target, z_pred)
    assert diag["pred_target_variance_ratio"] < 0.1

    warnings = collapse_warnings(diag, CollapseThresholds())
    assert any("pred_target_variance_ratio" in w for w in warnings)


def test_warnings_list_matches_threshold_config():
    """Tightening/loosening exactly one threshold adds/removes exactly its warning."""
    torch.manual_seed(3)
    z = torch.randn(4096, 32)
    diag = collapse_diagnostics(z, z)

    # Healthy baseline: nothing fires.
    assert collapse_warnings(diag, CollapseThresholds()) == []

    # Tighten a single threshold above the healthy value -> exactly one warning.
    tight = CollapseThresholds(per_dim_std_mean_min=10.0)
    warnings = collapse_warnings(diag, tight)
    assert len(warnings) == 1
    assert "per_dim_std_mean" in warnings[0]

    # Loosening the same threshold back removes it again.
    loose = CollapseThresholds(per_dim_std_mean_min=0.0)
    assert collapse_warnings(diag, loose) == []


def test_regularizer_returns_loss_output():
    """LossOutput contract: additive terms sum to total, grads flow, fp32 out."""
    torch.manual_seed(4)
    z = torch.randn(128, 16, requires_grad=True)

    out = VarianceCovarianceRegularizer()(z)

    assert isinstance(out, LossOutput)
    assert isinstance(out.total, torch.Tensor)
    assert out.total.ndim == 0
    assert out.total.dtype == torch.float32
    assert torch.isfinite(out.total)

    assert set(out.terms) == {"variance", "covariance"}
    for value in out.terms.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert value.dtype == torch.float32
    # Terms are ADDITIVE: they sum exactly to the total.
    term_sum = torch.stack(list(out.terms.values())).sum()
    assert torch.allclose(term_sum, out.total)

    for value in out.diagnostics.values():
        assert isinstance(value, float) and not isinstance(value, bool)
        assert math.isfinite(value)

    # This objective DOES flow gradients into the latent batch.
    out.total.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.any(z.grad != 0)


def test_variance_hinge_fires_on_low_variance_and_is_locked_when_healthy():
    """The variance TERM must actually engage on low variance -- not free-ride cov.

    ``test_regularizer_returns_loss_output`` only checks a *combined* nonzero
    gradient, which a variance hinge that always saturated to zero would still
    pass via the covariance term. This isolates the variance pathway (cov_weight
    = 0) and asserts (1) the hinge term is strictly positive on deliberately
    low-variance NONCONSTANT data, (2) its gradient alone is nonzero, and (3) it
    is direction-locked: ~0 on healthy unit-variance data and exactly 0 on
    over-target variance (relu only penalizes std BELOW std_target).
    """
    # Low-variance but nonconstant: per-dim std ~0.01 << std_target=1.0.
    torch.manual_seed(11)
    z_low = 0.01 * torch.randn(256, 16)
    z_low.requires_grad_(True)

    # cov_weight=0 isolates the variance pathway: total IS the variance term.
    reg_var_only = VarianceCovarianceRegularizer(cov_weight=0.0)
    out_low = reg_var_only(z_low)
    assert float(out_low.terms["covariance"].detach()) == 0.0
    # Hinge is strictly positive (relu(1.0 - ~0.01) per dim, averaged ~0.99).
    assert float(out_low.terms["variance"].detach()) > 0.5

    # Gradient through the variance pathway ALONE is nonzero.
    out_low.total.backward()
    assert z_low.grad is not None
    assert torch.any(z_low.grad != 0)

    # Healthy unit-variance data: the hinge is ~0 (std ~ std_target).
    torch.manual_seed(12)
    z_healthy = torch.randn(8192, 16)
    out_healthy = VarianceCovarianceRegularizer()(z_healthy)
    assert float(out_healthy.terms["variance"]) < 0.05

    # Over-target variance: the hinge is EXACTLY 0 -- relu never fires above
    # std_target, so the hinge only ever pushes variance UP, never down.
    torch.manual_seed(13)
    z_high = 5.0 * torch.randn(4096, 16)
    out_high = VarianceCovarianceRegularizer()(z_high)
    assert float(out_high.terms["variance"]) == pytest.approx(0.0, abs=1e-6)


def test_diagnostics_fp32_under_bf16_input():
    """bf16 inputs are cast to fp32 first (eig/svd is unstable in bf16, R12)."""
    torch.manual_seed(5)
    z_target = torch.randn(1024, 16).to(torch.bfloat16)
    z_pred = torch.randn(1024, 16).to(torch.bfloat16)

    diag = collapse_diagnostics(z_target, z_pred)
    for value in diag.values():
        assert isinstance(value, float) and not isinstance(value, bool)
        assert math.isfinite(value)


def test_regularizer_fp32_under_bf16_input():
    """The regularizer computes in fp32 and stays differentiable under bf16 input."""
    z = torch.randn(128, 16, dtype=torch.bfloat16, requires_grad=True)

    out = VarianceCovarianceRegularizer()(z)
    assert out.total.dtype == torch.float32
    assert torch.isfinite(out.total)

    out.total.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_leading_dims_folding_matches_flattened():
    """[B, K, S, D] folds to [N, D]; results equal the pre-flattened tensor's."""
    torch.manual_seed(6)
    z4 = torch.randn(4, 3, 5, 16)  # [B, K, S, D]
    z_flat = z4.reshape(-1, 16)

    diag_folded = collapse_diagnostics(z4, z4)
    diag_flat = collapse_diagnostics(z_flat, z_flat)

    assert set(diag_folded) == set(diag_flat)
    for key in diag_folded:
        assert diag_folded[key] == pytest.approx(diag_flat[key], rel=1e-5, abs=1e-6)


def test_required_diagnostic_keys_present():
    """The plan's contract keys are all present in the returned dict."""
    torch.manual_seed(7)
    z = torch.randn(256, 8)
    diag = collapse_diagnostics(z, z)
    required = {
        "per_dim_std_mean",
        "per_dim_std_min",
        "cov_offdiag_mean_abs",
        "effective_rank",
        "mean_norm",
        "latent_norm_std",
        "pred_target_variance_ratio",
        "near_constant_fraction",
    }
    assert required <= set(diag)
