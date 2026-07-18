"""Unit tests for the persistence (copy-forward) baseline (Task 2.9).

The persistence baseline (experiment ``E00``) is the naive comparison the
learned raw baseline (``E02``) must beat: it predicts every future frame of a
channel by repeating that channel's *last observed* context value. It carries
NO parameters and shares :class:`~fusion_jepa.models.raw_world_model.RawWorldModel`'s
output contract (``{modality: [B, C, T_tgt]}`` float32) so the identical eval
code path runs on both.

Locked behaviours:

* :func:`test_persistence_repeats_last_valid_value` -- with every context frame
  observed, each channel's final context value is copied across all horizons;
* :func:`test_persistence_skips_masked_tail` -- a masked tail (holding a hostile
  ``NaN`` placeholder) is walked back over to the last *observed* frame, and a
  channel with no observed context anywhere predicts a finite ``0.0``;
* :func:`test_persistence_has_zero_parameters` -- the module registers no
  learnable parameters.

Two further tests pin the shared output contract and the disclosed handling of
a target modality that has no context to copy forward.
"""

import torch

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.models.persistence import PersistenceBaseline
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_raw_world_model import build_raw_world_model


def _minimal_batch(
    context: dict[str, torch.Tensor],
    context_mask: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> FusionBatch:
    """Assemble a FusionBatch carrying only the fields persistence reads.

    Persistence consults ``context``/``context_mask`` (to find each channel's
    last observed value) and ``target`` (only for the output shape). The
    remaining fields are irrelevant, so they get well-typed placeholders.
    """
    B = next(iter(target.values())).shape[0]
    dummy_t = torch.zeros(B, 2, dtype=torch.float64)
    return FusionBatch(
        context=context,
        context_mask=context_mask,
        target=target,
        target_mask={
            k: torch.ones_like(v, dtype=torch.bool) for k, v in target.items()
        },
        actions=torch.zeros(B, 2, 1, dtype=torch.float32),
        action_mask=torch.ones(B, 2, dtype=torch.bool),
        context_times=dummy_t,
        target_times=dummy_t,
        action_times=dummy_t,
        horizon_seconds=torch.ones(B, dtype=torch.float64),
        device_id=["MAST"] * B,
        device_context=torch.zeros(B, 3, dtype=torch.float32),
        device_context_mask=torch.ones(B, 3, dtype=torch.bool),
        shot_id=[f"shot-{b}" for b in range(B)],
        window_id=[f"window-{b}" for b in range(B)],
        metadata={},
    )


def test_persistence_repeats_last_valid_value():
    # Two channels, four fully observed context frames; the last frame's value
    # per channel must be copied across all three target frames.
    context = {
        "a": torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]]),
    }
    context_mask = {"a": torch.ones(1, 2, 4, dtype=torch.bool)}
    target = {"a": torch.full((1, 2, 3), -99.0)}  # values irrelevant, shape used
    batch = _minimal_batch(context, context_mask, target)

    preds = PersistenceBaseline()(batch)

    expected = torch.tensor([[[4.0, 4.0, 4.0], [40.0, 40.0, 40.0]]])
    assert preds["a"].shape == (1, 2, 3)
    assert preds["a"].dtype == torch.float32
    assert torch.equal(preds["a"], expected)


def test_persistence_skips_masked_tail():
    # Channel 0: last two frames unobserved and holding NaN -- persistence must
    # walk back to the last OBSERVED frame (value 2.0), never leaking the NaN.
    # Channel 1: never observed anywhere -> a finite 0.0 (disclosed choice).
    nan = float("nan")
    context = {
        "a": torch.tensor([[[1.0, 2.0, nan, nan], [nan, nan, nan, nan]]]),
    }
    context_mask = {
        "a": torch.tensor(
            [[[True, True, False, False], [False, False, False, False]]]
        ),
    }
    target = {"a": torch.zeros(1, 2, 2)}
    batch = _minimal_batch(context, context_mask, target)

    preds = PersistenceBaseline()(batch)

    expected = torch.tensor([[[2.0, 2.0], [0.0, 0.0]]])
    assert torch.isfinite(preds["a"]).all()
    assert torch.equal(preds["a"], expected)


def test_persistence_has_zero_parameters():
    model = PersistenceBaseline()
    assert list(model.parameters()) == []
    assert sum(p.numel() for p in model.parameters()) == 0


def test_persistence_matches_raw_world_model_output_contract():
    # On a fully observed synthetic batch, persistence emits the exact keys,
    # per-modality shapes, and dtype the learned raw model does -- the property
    # that lets one eval loop score both.
    modalities = ("slow_ts", "profile")
    batch = make_synthetic_fusion_batch(
        B=2, modalities=modalities, n_channels=3, T=4, H=3, A=2
    )
    raw_model = build_raw_world_model(
        modalities=modalities, n_channels=3, n_actuators=2
    )

    raw_preds = raw_model(batch)
    persist_preds = PersistenceBaseline()(batch)

    assert set(persist_preds) == set(raw_preds)
    for modality in modalities:
        assert persist_preds[modality].shape == raw_preds[modality].shape
        assert persist_preds[modality].shape == batch.target[modality].shape
        assert persist_preds[modality].dtype == torch.float32


def test_persistence_finite_and_correct_under_missing_context():
    # Heavy missingness leaves NaN placeholders throughout the context; the
    # output must stay NaN-free and every value must equal a genuinely observed
    # context frame (or 0.0 for a channel with no observation at all).
    modalities = ("slow_ts", "profile")
    batch = make_synthetic_fusion_batch(
        B=3, modalities=modalities, n_channels=4, T=5, H=3, missing_fraction=0.6,
        seed=11,
    )

    preds = PersistenceBaseline()(batch)

    for modality in modalities:
        pred = preds[modality]
        assert pred.shape == batch.target[modality].shape
        assert torch.isfinite(pred).all()
        values = batch.context[modality]
        mask = batch.context_mask[modality]
        B, C, _ = pred.shape
        for b in range(B):
            for c in range(C):
                per_frame = pred[b, c]
                # Every horizon carries the identical persisted scalar.
                assert torch.equal(per_frame, per_frame[:1].expand_as(per_frame))
                observed = mask[b, c]
                if observed.any():
                    last_idx = int(torch.nonzero(observed).max())
                    assert per_frame[0] == values[b, c, last_idx]
                else:
                    assert per_frame[0] == 0.0


def test_persistence_zero_for_target_modality_absent_from_context():
    # Disclosed edge: a target modality with no context to copy forward yields a
    # finite all-zero block (there is nothing to persist).
    context = {"a": torch.ones(1, 1, 3)}
    context_mask = {"a": torch.ones(1, 1, 3, dtype=torch.bool)}
    target = {"a": torch.zeros(1, 1, 2), "b": torch.zeros(1, 2, 2)}
    batch = _minimal_batch(context, context_mask, target)

    preds = PersistenceBaseline()(batch)

    assert set(preds) == {"a", "b"}
    assert torch.equal(preds["b"], torch.zeros(1, 2, 2))
    assert torch.isfinite(preds["b"]).all()
