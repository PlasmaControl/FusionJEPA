"""Unit tests for the raw-prediction training objective (Task 2.7).

The raw baseline's objective compares the :class:`RawWorldModel` predictions
``{modality: [B, C, T_tgt]}`` against the batch's raw target values under the
observation mask, and reduces them in the spec-fixed order::

    masked (weighted) mean over (channel, frame) within an (example, modality)
        cell -> mean over modalities present -> mean over examples -> scalar

Locked behaviours:

* :func:`test_masked_targets_excluded_from_loss` -- entries the target mask
  marks unobserved never influence the loss, its terms, or its gradient, even
  when their placeholder values are ``NaN``/``inf``;
* :func:`test_reduction_order_matches_hand_computed_toy` -- a tiny hand-built
  case pins the nested mean-of-means order (a flat or modality-first reduction
  produces a different number and fails);
* :func:`test_per_horizon_terms_logged` -- one ``horizon_mean/<t>`` diagnostic
  per target frame (the horizon axis), each an unweighted masked mean the
  horizon weights do not distort (``terms`` itself holds only the additive
  per-modality contributions, per the LossOutput contract --
  :func:`test_terms_sum_to_total`);
* :func:`test_returns_loss_output_contract` -- an end-to-end run on real model
  predictions returns a valid :class:`LossOutput` whose total backpropagates
  finite gradients into the model;
* :func:`test_fully_masked_modality_yields_finite_loss_and_zero_contribution`
  -- a modality whose entire target block is unobserved (a real TokaMark
  window can look like this) stays finite, contributes nothing to the total,
  and receives a zero (never ``NaN``) gradient.

The hand-computed toy cases build tiny dicts directly; the contract test uses
the deterministic synthetic batch and the shared tiny world model.
"""

import pytest
import torch

from fusion_jepa.objectives.base import LossOutput
from fusion_jepa.objectives.raw_prediction import RawPredictionObjective
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_raw_world_model import build_raw_world_model


def test_masked_targets_excluded_from_loss():
    predictions = {
        "a": torch.tensor([[[2.0, 4.0, 6.0]]]),
        "b": torch.tensor([[[1.0, 1.0, 1.0]]]),
    }
    masks = {
        "a": torch.tensor([[[True, False, True]]]),
        "b": torch.tensor([[[False, True, False]]]),
    }
    targets_clean = {
        "a": torch.zeros(1, 1, 3),
        "b": torch.zeros(1, 1, 3),
    }
    # Same observed values, but the *unobserved* entries carry hostile
    # placeholders (large garbage, NaN, inf). A mask-aware objective must be
    # blind to them.
    targets_garbage = {
        "a": torch.tensor([[[0.0, 1.0e6, 0.0]]]),
        "b": torch.tensor([[[float("nan"), 0.0, float("inf")]]]),
    }

    obj = RawPredictionObjective(distance="mse")
    out_clean = obj(predictions, targets_clean, masks)
    out_garbage = obj(predictions, targets_garbage, masks)

    assert torch.isfinite(out_clean.total)
    assert torch.isfinite(out_garbage.total)
    assert torch.allclose(out_clean.total, out_garbage.total)
    assert set(out_clean.terms) == set(out_garbage.terms)
    for key in out_clean.terms:
        assert torch.allclose(out_clean.terms[key], out_garbage.terms[key])
    # Diagnostics (incl. the per-horizon means) must be equally blind to the
    # hostile placeholders -- a NaN leak would surface here.
    assert out_clean.diagnostics == pytest.approx(out_garbage.diagnostics)

    # Gradients stay finite despite NaN/inf in the (excluded) masked targets.
    grad_preds = {
        key: value.clone().requires_grad_(True)
        for key, value in predictions.items()
    }
    out_grad = obj(grad_preds, targets_garbage, masks)
    out_grad.total.backward()
    for value in grad_preds.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_reduction_order_matches_hand_computed_toy():
    # target == 0 everywhere, so the per-element squared error is pred**2.
    predictions = {
        "a": torch.tensor([[[2.0, 4.0]], [[1.0, 1.0]]]),
        "b": torch.tensor([[[3.0, 0.0]], [[2.0, 2.0]]]),
    }
    targets = {"a": torch.zeros(2, 1, 2), "b": torch.zeros(2, 1, 2)}
    masks = {
        "a": torch.tensor([[[True, True]], [[True, False]]]),
        "b": torch.tensor([[[False, True]], [[False, False]]]),
    }
    obj = RawPredictionObjective(distance="mse", horizon_weights=[1.0, 3.0])
    out = obj(predictions, targets, masks)

    # Hand computation (weights w=[1,3] on the frame axis):
    #   cell(a,0) = (1*4 + 3*16)/(1+3) = 13   cell(a,1) = (1*1)/1 = 1
    #   cell(b,0) = (3*0)/3 = 0               cell(b,1) fully masked -> absent
    #   example0 = mean(13, 0) = 6.5          example1 = mean(1) = 1.0
    #   total    = mean(6.5, 1.0) = 3.75
    assert torch.allclose(out.total, torch.tensor(3.75))
    # A flat masked-mean would give 6.625 and a modality-first mean 3.5; both
    # differ from the spec's example-first 3.75, so this pins the order.
    assert not torch.allclose(out.total, torch.tensor(6.625))
    assert not torch.allclose(out.total, torch.tensor(3.5))

    # Per-modality TERMS are additive contributions (LossOutput contract):
    #   term_a = (13/2 + 1/1)/2 = 3.75    term_b = (0/2 + 0)/2 = 0.0
    # and they sum exactly to the total.
    assert torch.allclose(out.terms["modality/a"], torch.tensor(3.75))
    assert torch.allclose(out.terms["modality/b"], torch.tensor(0.0))
    # The per-modality *inspection means* (mean cell loss over the examples
    # where the modality is present) live in diagnostics:
    #   mean_a = (13 + 1)/2 = 7.0         mean_b = 0/1 = 0.0
    assert out.diagnostics["modality_mean/a"] == pytest.approx(7.0)
    assert out.diagnostics["modality_mean/b"] == pytest.approx(0.0)


def test_terms_sum_to_total():
    """`terms` honours the LossOutput contract: the per-modality additive
    contributions sum to `total`, including under mixed modality presence
    and a fully-masked modality."""
    predictions = {
        "a": torch.tensor([[[2.0, 4.0]], [[1.0, 1.0]]]),
        "b": torch.tensor([[[3.0, 0.0]], [[2.0, 2.0]]]),
        "c": torch.tensor([[[9.0, 9.0]], [[9.0, 9.0]]]),
    }
    targets = {key: torch.zeros(2, 1, 2) for key in predictions}
    masks = {
        "a": torch.tensor([[[True, True]], [[True, False]]]),
        "b": torch.tensor([[[False, True]], [[False, False]]]),
        "c": torch.zeros(2, 1, 2, dtype=torch.bool),  # absent everywhere
    }
    out = RawPredictionObjective(distance="mse", horizon_weights=[1.0, 3.0])(
        predictions, targets, masks
    )
    total_from_terms = torch.stack(list(out.terms.values())).sum()
    assert torch.allclose(total_from_terms, out.total)


def test_per_horizon_terms_logged():
    predictions = {"x": torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])}
    targets = {"x": torch.zeros(1, 2, 3)}
    masks = {"x": torch.tensor([[[True, True, False], [True, False, True]]])}

    obj = RawPredictionObjective(distance="mse")
    out = obj(predictions, targets, masks)

    horizon_keys = [
        key for key in out.diagnostics if key.startswith("horizon_mean/")
    ]
    assert len(horizon_keys) == 3
    # Unweighted masked mean over (example, channel) per frame:
    #   t0 = (1 + 16)/2 = 8.5    t1 = 4/1 = 4.0    t2 = 36/1 = 36.0
    assert out.diagnostics["horizon_mean/0"] == pytest.approx(8.5)
    assert out.diagnostics["horizon_mean/1"] == pytest.approx(4.0)
    assert out.diagnostics["horizon_mean/2"] == pytest.approx(36.0)

    # Horizon reporting is unweighted: horizon weights reshape the total but
    # must not distort the per-horizon diagnostics.
    weighted = RawPredictionObjective(
        distance="mse", horizon_weights=[1.0, 5.0, 10.0]
    )
    out_weighted = weighted(predictions, targets, masks)
    for frame in range(3):
        assert out.diagnostics[f"horizon_mean/{frame}"] == pytest.approx(
            out_weighted.diagnostics[f"horizon_mean/{frame}"]
        )


def test_horizon_weights_validated():
    """Weights must be finite and either exactly 0 (frame excluded) or large
    enough (>= 1e-6) not to collide with the internal denominator floor."""
    for bad in (
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, -1.0],
        [1.0, 1.0e-9],
    ):
        with pytest.raises(ValueError):
            RawPredictionObjective(horizon_weights=bad)
    # Zero excludes a frame; well-conditioned positives are fine.
    RawPredictionObjective(horizon_weights=[0.0, 1.0])


def test_all_masked_batch_is_finite_zero():
    """The fully-degenerate case (every modality 100%-masked): the loss is an
    exact, finite 0 and backward stays NaN-free (documented limiting case)."""
    pred = torch.ones(2, 1, 2, requires_grad=True)
    targets = {"a": torch.full((2, 1, 2), float("nan"))}
    masks = {"a": torch.zeros(2, 1, 2, dtype=torch.bool)}

    out = RawPredictionObjective(distance="mse")({"a": pred}, targets, masks)

    assert torch.isfinite(out.total)
    assert out.total.item() == 0.0
    out.total.backward()
    assert pred.grad is None or (
        torch.isfinite(pred.grad).all() and torch.all(pred.grad == 0)
    )


def test_returns_loss_output_contract():
    modalities = ("slow_ts", "profile")
    batch = make_synthetic_fusion_batch(
        B=2,
        modalities=modalities,
        n_channels=3,
        T=4,
        H=3,
        A=2,
        missing_fraction=0.3,
    )
    model = build_raw_world_model(
        modalities=modalities, n_channels=3, n_actuators=2
    )
    preds = model(batch)

    obj = RawPredictionObjective(distance="smooth_l1")
    out = obj(preds, batch.target, batch.target_mask)

    assert isinstance(out, LossOutput)
    assert isinstance(out.total, torch.Tensor) and out.total.ndim == 0
    assert torch.isfinite(out.total)
    assert out.total.requires_grad

    for value in out.terms.values():
        assert isinstance(value, torch.Tensor) and value.ndim == 0
        assert torch.isfinite(value)
    for value in out.diagnostics.values():
        assert isinstance(value, float) and not isinstance(value, bool)

    assert set(out.terms) == {f"modality/{m}" for m in modalities}
    assert sum(key.startswith("horizon_mean/") for key in out.diagnostics) == 3
    # Additivity holds on realistic mixed-missingness model outputs too.
    total_from_terms = torch.stack(list(out.terms.values())).sum()
    assert torch.allclose(total_from_terms, out.total)

    out.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_fully_masked_modality_yields_finite_loss_and_zero_contribution():
    pred_a = torch.tensor([[[2.0, 4.0]], [[1.0, 3.0]]], requires_grad=True)
    pred_b = torch.tensor([[[5.0, 5.0]], [[9.0, 9.0]]], requires_grad=True)
    predictions = {"a": pred_a, "b": pred_b}
    targets = {"a": torch.zeros(2, 1, 2), "b": torch.zeros(2, 1, 2)}
    masks = {
        "a": torch.ones(2, 1, 2, dtype=torch.bool),
        "b": torch.zeros(2, 1, 2, dtype=torch.bool),  # entire block unobserved
    }

    obj = RawPredictionObjective(distance="mse")
    out = obj(predictions, targets, masks)

    assert torch.isfinite(out.total)
    assert torch.allclose(out.terms["modality/b"], torch.tensor(0.0))

    # The fully-masked modality contributes nothing: the total equals the loss
    # computed over the observed modality alone.
    out_a_only = obj({"a": pred_a}, {"a": targets["a"]}, {"a": masks["a"]})
    assert torch.allclose(out.total, out_a_only.total)

    out.total.backward()
    assert torch.isfinite(pred_a.grad).all()
    assert torch.any(pred_a.grad != 0)
    # Gradient-safe: the fully-masked modality receives a zero (never NaN) grad.
    assert pred_b.grad is None or (
        torch.isfinite(pred_b.grad).all() and torch.all(pred_b.grad == 0)
    )
