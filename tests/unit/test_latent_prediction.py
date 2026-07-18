"""Unit tests for the latent-prediction training objective (Task 3.3).

The JEPA world model predicts a future-state latent ``z_hat`` and encodes the
observed future window into a target latent ``z_target``; the objective scores
the prediction against the target in *latent* space and returns the shared
:class:`~fusion_jepa.objectives.base.LossOutput`. It mirrors the raw-prediction
objective's contract -- mask-aware, NaN-safe, fp32, additive per-term shares --
but the discrepancy lives between latents rather than raw signal values.

Locked behaviours:

* :func:`test_masked_horizons_do_not_contribute` -- cells ``target_valid`` marks
  invalid never influence the loss/terms/gradient, even with a hostile ``NaN``
  latent in the invalid cell;
* :func:`test_horizon_weights_match_hand_computed` -- a K>1 hand-built case pins
  the horizon-weighted batch mean (the reductions are K-generic);
* :func:`test_cosine_vs_smooth_l1_selectable` -- both distances run, differ, and
  each matches a hand-computed toy value;
* :func:`test_per_horizon_terms_present_and_fp32` -- per-horizon terms and
  diagnostics are present, fp32, and the uniform weight default is logged;
* :func:`test_terms_sum_to_total` -- the additive LossOutput contract holds under
  mixed validity and zero weights;
* :func:`test_rejects_target_gradients_by_default_and_allows_when_configured` --
  uses the real JEPA EMA / END_TO_END paths to lock the ``allow_target_gradients``
  reconciliation;
* :func:`test_all_invalid_batch_finite_zero` and
  :func:`test_fp32_under_bf16_inputs` -- the finite limiting cases;
* the normalization-convention locks pin that ``normalize_targets`` normalizes
  BOTH sides for smooth_l1 and that cosine normalizes both intrinsically.
"""

from __future__ import annotations

import pytest
import torch

from fusion_jepa.models.jepa import JEPAOutput, TargetUpdatePolicy
from fusion_jepa.objectives.base import LossOutput
from fusion_jepa.objectives.latent_prediction import LatentPredictionObjective
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_jepa import build_jepa_model

_MODALITIES = ("slow_ts", "profile")


def _out(z_hat, z_target, valid) -> JEPAOutput:
    """Build a :class:`JEPAOutput`, casting list literals to fp32 tensors."""
    if not isinstance(z_hat, torch.Tensor):
        z_hat = torch.tensor(z_hat, dtype=torch.float32)
    if not isinstance(z_target, torch.Tensor):
        z_target = torch.tensor(z_target, dtype=torch.float32)
    return JEPAOutput(
        z_hat=z_hat,
        z_target=z_target,
        target_valid=torch.as_tensor(valid, dtype=torch.bool),
    )


def test_cosine_vs_smooth_l1_selectable():
    # B=1, K=1, S=1, D=2 with already-unit vectors (normalization is a no-op).
    z_hat = [[[[1.0, 0.0]]]]
    z_target = [[[[0.0, 1.0]]]]
    out = _out(z_hat, z_target, [[True]])
    horizon = torch.tensor([[1.0]], dtype=torch.float64)

    cos = LatentPredictionObjective(distance="cosine")(out, horizon)
    sl1 = LatentPredictionObjective(distance="smooth_l1", smooth_l1_beta=1.0)(
        out, horizon
    )

    # cosine distance = 1 - cos([1,0],[0,1]) = 1 - 0 = 1.
    assert cos.total.item() == pytest.approx(1.0)
    # smooth_l1 on unit vecs: |1-0| = 1 >= beta -> 1 - 0.5 = 0.5 per element;
    # mean over (S, D) = 0.5.
    assert sl1.total.item() == pytest.approx(0.5)
    assert not torch.allclose(cos.total, sl1.total)


def test_horizon_weights_match_hand_computed():
    # B=1, K=2, S=1, D=2; smooth_l1 (beta=1), raw latents (no normalization).
    z_hat = torch.tensor([[[[2.0, 4.0]], [[1.0, 3.0]]]])  # [1, 2, 1, 2]
    z_target = torch.zeros(1, 2, 1, 2)
    out = _out(z_hat, z_target, [[True, True]])
    horizon = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    res = LatentPredictionObjective(
        distance="smooth_l1",
        horizon_weights=[1.0, 3.0],
        normalize_targets=False,
        smooth_l1_beta=1.0,
    )(out, horizon)

    # cell(h0) = mean(smooth_l1([2,4],[0,0])) = mean(1.5, 3.5) = 2.5
    # cell(h1) = mean(smooth_l1([1,3],[0,0])) = mean(0.5, 2.5) = 1.5
    # total = (1*2.5 + 3*1.5)/(1+3) = 7/4 = 1.75
    assert res.total.item() == pytest.approx(1.75)
    # Per-horizon additive contributions (weighted) sum to total.
    assert res.terms["latent/h=1"].item() == pytest.approx(0.625)
    assert res.terms["latent/h=2"].item() == pytest.approx(1.125)
    # Per-horizon inspection means are UNWEIGHTED cell means.
    assert res.diagnostics["latent_mean/h=1"] == pytest.approx(2.5)
    assert res.diagnostics["latent_mean/h=2"] == pytest.approx(1.5)
    # A modality-first / flat mean would give a different number: pin the order.
    assert not torch.allclose(res.total, torch.tensor(2.0))


def test_masked_horizons_do_not_contribute():
    # B=2, K=1, S=1, D=2; smooth_l1 raw. cell(b0)=2.5, cell(b1)=1.5.
    z_hat = torch.tensor([[[[2.0, 4.0]]], [[[1.0, 3.0]]]])  # [2, 1, 1, 2]
    z_target = torch.zeros(2, 1, 1, 2)
    horizon = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    obj = LatentPredictionObjective(
        distance="smooth_l1", normalize_targets=False, smooth_l1_beta=1.0
    )

    both = obj(_out(z_hat, z_target, [[True], [True]]), horizon)
    assert both.total.item() == pytest.approx(2.0)  # mean(2.5, 1.5)

    # Flip b1 invalid AND poison its latent with NaN: it must not contribute and
    # the loss must equal the b0-only value (2.5), staying finite.
    poisoned = z_hat.clone()
    poisoned[1] = float("nan")
    masked = obj(_out(poisoned, z_target, [[True], [False]]), horizon)
    assert torch.isfinite(masked.total)
    assert masked.total.item() == pytest.approx(2.5)
    assert masked.terms["latent/h=1"].item() == pytest.approx(2.5)

    # Gradient stays finite despite the NaN in the (excluded) invalid cell.
    grad_hat = z_hat.clone()
    grad_hat[1] = float("nan")
    grad_hat.requires_grad_(True)
    grad_res = obj(_out(grad_hat, z_target, [[True], [False]]), horizon)
    grad_res.total.backward()
    assert grad_hat.grad is not None
    assert torch.isfinite(grad_hat.grad).all()
    # The excluded cell receives an exactly-zero (never NaN) gradient.
    assert torch.all(grad_hat.grad[1] == 0)


def test_terms_sum_to_total():
    torch.manual_seed(0)
    z_hat = torch.randn(2, 2, 3, 4)
    z_target = torch.randn(2, 2, 3, 4)
    valid = [[True, False], [True, True]]
    horizon = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    for obj in (
        LatentPredictionObjective(distance="cosine"),
        LatentPredictionObjective(distance="cosine", horizon_weights=[1.0, 3.0]),
        # A zero weight excludes a horizon; additivity must still hold.
        LatentPredictionObjective(distance="smooth_l1", horizon_weights=[2.0, 0.0]),
    ):
        res = obj(_out(z_hat, z_target, valid), horizon)
        total_from_terms = torch.stack(list(res.terms.values())).sum()
        assert torch.allclose(total_from_terms, res.total, atol=1e-6)


def test_per_horizon_terms_present_and_fp32():
    torch.manual_seed(1)
    z_hat = torch.randn(2, 2, 3, 4)
    z_target = torch.randn(2, 2, 3, 4)
    out = _out(z_hat, z_target, [[True, True], [True, True]])
    horizon = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)

    res = LatentPredictionObjective(distance="cosine")(out, horizon)

    # Additive per-horizon terms keyed by the horizon seconds value.
    assert set(res.terms) == {"latent/h=1", "latent/h=2"}
    for value in res.terms.values():
        assert isinstance(value, torch.Tensor) and value.ndim == 0
        assert value.dtype == torch.float32
    assert res.total.dtype == torch.float32

    # Per-horizon inspection means + uniform-logged weights.
    assert "latent_mean/h=1" in res.diagnostics
    assert "latent_mean/h=2" in res.diagnostics
    assert res.diagnostics["horizon_weights_uniform"] == 1.0
    assert res.diagnostics["horizon_weight/h=1"] == pytest.approx(1.0)
    assert res.diagnostics["horizon_weight/h=2"] == pytest.approx(1.0)
    for value in res.diagnostics.values():
        assert isinstance(value, float) and not isinstance(value, bool)

    # Non-uniform weights are flagged as non-uniform.
    nonuni = LatentPredictionObjective(distance="cosine", horizon_weights=[1.0, 2.0])(
        out, horizon
    )
    assert nonuni.diagnostics["horizon_weights_uniform"] == 0.0
    assert nonuni.diagnostics["horizon_weight/h=2"] == pytest.approx(2.0)


def test_rejects_target_gradients_by_default_and_allows_when_configured():
    batch = make_synthetic_fusion_batch(
        B=2, modalities=_MODALITIES, n_channels=3, T=4, H=3, A=2
    )

    # EMA path: z_target carries no gradient -> the default objective accepts it.
    ema = build_jepa_model(policy=TargetUpdatePolicy.EMA)
    out_ema = ema(batch)
    assert out_ema.z_target.requires_grad is False
    res = LatentPredictionObjective(distance="cosine")(out_ema, batch.horizon_seconds)
    assert isinstance(res, LossOutput)
    assert torch.isfinite(res.total)

    # END_TO_END path: z_target DOES require gradient.
    e2e = build_jepa_model(
        policy=TargetUpdatePolicy.END_TO_END_REGULARIZED,
        collapse_regularizer=object(),
    )
    out_e2e = e2e(batch)
    assert out_e2e.z_target.requires_grad is True
    # The default rejects it with an actionable, policy-naming error.
    with pytest.raises(ValueError, match="allow_target_gradients"):
        LatentPredictionObjective(distance="cosine")(out_e2e, batch.horizon_seconds)
    # Explicitly opting in accepts it and remains backprop-able.
    allowed = LatentPredictionObjective(distance="cosine", allow_target_gradients=True)(
        out_e2e, batch.horizon_seconds
    )
    assert torch.isfinite(allowed.total)
    allowed.total.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in e2e.parameters()
    )


def test_all_invalid_batch_finite_zero():
    z_hat = torch.randn(2, 1, 3, 4, requires_grad=True)
    z_target = torch.randn(2, 1, 3, 4)
    out = _out(z_hat, z_target, [[False], [False]])
    horizon = torch.tensor([[1.0], [1.0]], dtype=torch.float64)

    res = LatentPredictionObjective(distance="cosine")(out, horizon)

    assert torch.isfinite(res.total)
    assert res.total.item() == 0.0
    # The additive terms still sum to (a finite zero) total.
    total_from_terms = torch.stack(list(res.terms.values())).sum()
    assert torch.allclose(total_from_terms, res.total)

    res.total.backward()
    assert z_hat.grad is None or (
        torch.isfinite(z_hat.grad).all() and torch.all(z_hat.grad == 0)
    )


def test_fp32_under_bf16_inputs():
    z_hat = torch.randn(2, 1, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    z_target = torch.randn(2, 1, 3, 4, dtype=torch.bfloat16)
    out = _out(z_hat, z_target, [[True], [True]])
    horizon = torch.tensor([[1.0], [1.0]], dtype=torch.float64)

    for dist in ("cosine", "smooth_l1"):
        res = LatentPredictionObjective(distance=dist)(out, horizon)
        assert res.total.dtype == torch.float32
        assert torch.isfinite(res.total)
        for value in res.terms.values():
            assert value.dtype == torch.float32

    # Backprop from bf16 inputs through the fp32 path stays finite.
    res = LatentPredictionObjective(distance="cosine")(out, horizon)
    res.total.backward()
    assert z_hat.grad is not None
    assert torch.isfinite(z_hat.grad).all()


def test_smooth_l1_normalize_targets_normalizes_both_sides():
    # Non-unit vectors so normalization is observable.
    z_hat = torch.tensor([[[[3.0, 0.0]]]])
    z_target = torch.tensor([[[[0.0, 5.0]]]])
    valid = [[True]]
    horizon = torch.tensor([[1.0]], dtype=torch.float64)

    obj_norm = LatentPredictionObjective(
        distance="smooth_l1", normalize_targets=True, smooth_l1_beta=1.0
    )
    base = obj_norm(_out(z_hat, z_target, valid), horizon).total
    # Scaling EITHER operand by a positive constant leaves the loss unchanged,
    # so BOTH sides are L2-normalized (the disclosed convention).
    scaled_hat = obj_norm(_out(z_hat * 10.0, z_target, valid), horizon).total
    scaled_tgt = obj_norm(_out(z_hat, z_target * 7.0, valid), horizon).total
    assert torch.allclose(base, scaled_hat, atol=1e-6)
    assert torch.allclose(base, scaled_tgt, atol=1e-6)

    # Without normalization, scaling z_hat DOES change the raw Huber loss.
    obj_raw = LatentPredictionObjective(
        distance="smooth_l1", normalize_targets=False, smooth_l1_beta=1.0
    )
    raw = obj_raw(_out(z_hat, z_target, valid), horizon).total
    raw_scaled = obj_raw(_out(z_hat * 10.0, z_target, valid), horizon).total
    assert not torch.allclose(raw, raw_scaled)


def test_cosine_normalizes_both_sides_intrinsically():
    z_hat = torch.tensor([[[[3.0, 0.0]]]])
    z_target = torch.tensor([[[[0.0, 5.0]]]])
    valid = [[True]]
    horizon = torch.tensor([[1.0]], dtype=torch.float64)
    for normalize_targets in (True, False):
        obj = LatentPredictionObjective(
            distance="cosine", normalize_targets=normalize_targets
        )
        base = obj(_out(z_hat, z_target, valid), horizon).total
        scaled = obj(_out(z_hat * 4.0, z_target * 9.0, valid), horizon).total
        # Cosine is scale-invariant on BOTH sides regardless of the flag.
        assert torch.allclose(base, scaled, atol=1e-6)
        assert base.item() == pytest.approx(1.0)  # orthogonal -> distance 1


def test_invalid_distance_rejected():
    with pytest.raises(ValueError, match="distance"):
        LatentPredictionObjective(distance="mse")


def test_smooth_l1_beta_must_be_positive():
    with pytest.raises(ValueError, match="smooth_l1_beta"):
        LatentPredictionObjective(distance="smooth_l1", smooth_l1_beta=0.0)


def test_horizon_weights_validated():
    for bad in (
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, -1.0],
        [1.0, 1.0e-9],
    ):
        with pytest.raises(ValueError):
            LatentPredictionObjective(horizon_weights=bad)
    # Zero excludes a horizon; well-conditioned positives are accepted.
    LatentPredictionObjective(horizon_weights=[0.0, 1.0])


def test_horizon_weights_length_must_match_k():
    z_hat = torch.randn(1, 2, 1, 2)
    z_target = torch.randn(1, 2, 1, 2)
    out = _out(z_hat, z_target, [[True, True]])
    horizon = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="horizon_weights"):
        LatentPredictionObjective(distance="cosine", horizon_weights=[1.0, 2.0, 3.0])(
            out, horizon
        )


def test_horizon_seconds_1d_reshaped_for_k1():
    z_hat = torch.randn(2, 1, 3, 4)
    z_target = torch.randn(2, 1, 3, 4)
    out = _out(z_hat, z_target, [[True], [True]])
    # 1-D horizon_seconds [B] is reshaped to [B, 1] (K == 1).
    res = LatentPredictionObjective(distance="cosine")(
        out, torch.tensor([1.0, 1.0], dtype=torch.float64)
    )
    assert set(res.terms) == {"latent/h=1"}
