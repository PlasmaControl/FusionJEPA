"""Tests for the canonical Fusion-JEPA batch contract."""

from copy import deepcopy

import pytest
import torch

from fusion_jepa.data.batch import collate_fusion, validate_batch
from tests.fixtures.synthetic import make_ramp_sample


def _valid_batch():
    """Return a valid two-sample batch and its split lookup."""
    first = make_ramp_sample(shot_id="shot-1", window_id="window-1")
    second = make_ramp_sample(
        shot_id="shot-2",
        window_id="window-2",
        time_offset=20.0,
    )
    batch = collate_fusion([first, second])
    return batch, {"shot-1": "train", "shot-2": "train"}


def test_valid_ramp_batch_passes():
    batch, split_lookup = _valid_batch()

    assert validate_batch(batch, split_lookup=split_lookup) == []


def test_nan_under_true_mask_fails():
    batch, split_lookup = _valid_batch()
    batch.context["plasma_current"][0, 0, 0] = torch.nan

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("finite" in violation for violation in violations)
    with pytest.raises(ValueError, match="finite"):
        validate_batch(batch, split_lookup=split_lookup)


def test_nan_under_false_mask_allowed():
    batch, split_lookup = _valid_batch()
    batch.context_mask["plasma_current"][0, 0, 0] = False
    batch.context["plasma_current"][0, 0, 0] = torch.nan

    assert validate_batch(batch, split_lookup=split_lookup) == []


def test_nonmonotonic_times_fail():
    batch, split_lookup = _valid_batch()
    batch.context_times[0, 1] = batch.context_times[0, 0]

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("context_times" in violation for violation in violations)


def test_context_target_overlap_fails():
    batch, split_lookup = _valid_batch()
    batch.target_times[0] -= 3.0

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("overlap" in violation for violation in violations)


def test_horizon_seconds_mismatch_fails():
    batch, split_lookup = _valid_batch()
    batch.horizon_seconds[0] += 1.0

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("horizon_seconds" in violation for violation in violations)


def test_float32_times_reported_not_crash():
    batch, split_lookup = _valid_batch()
    batch.context_times = batch.context_times.float()
    batch.target_times = batch.target_times.float()
    batch.action_times = batch.action_times.float()

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    for name in ("context_times", "target_times", "action_times"):
        assert any(
            name in violation and "float64" in violation
            for violation in violations
        )
    with pytest.raises(ValueError):
        validate_batch(batch, split_lookup=split_lookup)


def test_int_mask_reported_and_finite_check_still_bites():
    batch, split_lookup = _valid_batch()
    batch.context_mask["plasma_current"] = batch.context_mask[
        "plasma_current"
    ].to(torch.int64)
    batch.context["plasma_current"][0, 0, 0] = torch.nan

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any(
        "context_mask" in violation and "bool" in violation
        for violation in violations
    )
    assert any("finite" in violation for violation in violations)


def test_action_window_not_covering_transition_fails():
    batch, split_lookup = _valid_batch()
    batch.action_times[0] += 2.0

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("cover" in violation for violation in violations)


def test_duplicate_window_id_fails():
    batch, split_lookup = _valid_batch()
    batch.window_id[1] = batch.window_id[0]

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("window_id" in violation for violation in violations)


def test_target_beyond_shot_end_fails():
    batch, split_lookup = _valid_batch()
    batch.metadata["shot_time_ranges"]["shot-1"] = (0.0, 5.0)

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("shot_time_ranges" in violation for violation in violations)


def test_split_mismatch_fails():
    batch, split_lookup = _valid_batch()
    split_lookup["shot-1"] = "test"

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("split" in violation for violation in violations)


def test_zero_value_with_true_mask_is_observed_not_missing():
    sample = make_ramp_sample(time_offset=0.0)
    batch = collate_fusion([sample])

    assert batch.context["plasma_current"][0, 0, 0].item() == 0.0
    assert batch.context_mask["plasma_current"][0, 0, 0].item() is True
    assert validate_batch(batch, split_lookup={sample.shot_id: "train"}) == []


def test_collate_stacks_and_preserves_ids():
    first = make_ramp_sample(shot_id="shot-1", window_id="window-1")
    second = make_ramp_sample(
        shot_id="shot-2",
        window_id="window-2",
        time_offset=20.0,
    )

    batch = collate_fusion([first, second])

    assert batch.context["plasma_current"].shape == (2, 1, 4)
    assert batch.target["plasma_current"].shape == (2, 1, 3)
    assert batch.actions.shape == (2, 5, 2)
    assert batch.shot_id == ["shot-1", "shot-2"]
    assert batch.window_id == ["window-1", "window-2"]
    assert batch.device_id == ["MAST", "MAST"]
    assert batch.metadata["shot_time_ranges"] == {
        "shot-1": (0.0, 8.0),
        "shot-2": (20.0, 28.0),
    }
    expected_metadata = deepcopy(first.metadata)
    expected_metadata["shot_time_ranges"].update(
        second.metadata["shot_time_ranges"]
    )
    assert batch.metadata == expected_metadata
    assert batch.metadata is not first.metadata


def test_collate_rejects_mismatched_task_id():
    first = make_ramp_sample(shot_id="shot-1", window_id="window-1")
    second = make_ramp_sample(
        shot_id="shot-2",
        window_id="window-2",
        task_id="reconstruction",
    )

    with pytest.raises(ValueError, match="task_id"):
        collate_fusion([first, second])


def test_missing_units_declaration_fails():
    batch, split_lookup = _valid_batch()
    del batch.metadata["units"]["plasma_current"]

    violations = validate_batch(
        batch,
        split_lookup=split_lookup,
        strict=False,
    )

    assert any("units" in violation for violation in violations)
