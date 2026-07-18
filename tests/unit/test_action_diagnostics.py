"""Tests for evaluation-side action-use diagnostics (Task 3.6).

These exercise the action perturbation modes, the action-use report, and the
finite-difference action-sensitivity probe. None of this is ever wired into a
training loss -- it is strictly evaluation-side instrumentation.
"""

import dataclasses
import json

import pytest
import torch

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.objectives import action_diagnostics as _diag
from fusion_jepa.objectives.action_diagnostics import (
    ActionPerturbation,
    action_sensitivity,
    action_use_report,
    perturb_actions,
)
from tests.fixtures.synthetic import make_synthetic_fusion_batch


def _snapshot(batch: FusionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    return batch.actions.clone(), batch.action_mask.clone()


def _assert_input_unmodified(
    batch: FusionBatch, snapshot: tuple[torch.Tensor, torch.Tensor]
) -> None:
    actions, action_mask = snapshot
    assert torch.equal(batch.actions, actions)
    assert torch.equal(batch.action_mask, action_mask)


def _distinct_row_masks(n_rows: int, n_steps: int) -> torch.Tensor:
    """Return ``n_rows`` mutually distinct boolean masks of width ``n_steps``.

    Each row is the little-endian bit pattern of ``row + 1`` (always nonzero and
    unique), so recovering a per-row permutation from the masks is unambiguous.
    """
    masks = torch.zeros(n_rows, n_steps, dtype=torch.bool)
    for row in range(n_rows):
        bits = row + 1
        for step in range(n_steps):
            masks[row, step] = bool((bits >> step) & 1)
    return masks


def _recover_batch_permutation(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    """Recover the unique B-axis permutation mapping ``after`` back to ``before``.

    Requires each example's action sequence to be distinct so every output row
    matches exactly one input row.
    """
    n_examples = before.shape[0]
    permutation: list[int] = []
    for i in range(n_examples):
        matches = [j for j in range(n_examples) if torch.equal(after[i], before[j])]
        assert len(matches) == 1, (
            f"action sequences must be distinct to recover the permutation; "
            f"output row {i} matched inputs {matches}"
        )
        permutation.append(matches[0])
    return permutation


def _recover_roll_offset(before: torch.Tensor, after: torch.Tensor) -> int:
    """Recover the unique time-axis roll offset mapping ``before`` to ``after``.

    Requires per-timestep values to be distinct so exactly one offset matches.
    """
    n_steps = before.shape[0]
    offsets = [
        s
        for s in range(n_steps)
        if torch.equal(torch.roll(before, shifts=s, dims=0), after)
    ]
    assert len(offsets) == 1, (
        f"per-timestep values must be distinct to recover the roll offset; "
        f"matched offsets {offsets}"
    )
    return offsets[0]


def test_batch_shuffle_preserves_shapes_and_per_channel_marginals() -> None:
    batch = make_synthetic_fusion_batch(B=6, seed=7)
    # Give each example a DISTINCT action mask; the fixture's masks are all-True,
    # so any permutation of them looks identical and could not expose actions and
    # masks being shuffled by different permutations.
    n_examples, n_steps = batch.action_mask.shape
    batch = dataclasses.replace(
        batch, action_mask=_distinct_row_masks(n_examples, n_steps)
    )
    snapshot = _snapshot(batch)
    generator = torch.Generator().manual_seed(11)

    out = perturb_actions(batch, ActionPerturbation.BATCH_SHUFFLE, generator)

    assert out.actions.shape == batch.actions.shape
    assert out.action_mask.shape == batch.action_mask.shape
    # Whole-sequence permutation of the B axis preserves every per-channel
    # value multiset across the batch.
    n_channels = batch.actions.shape[-1]
    for channel in range(n_channels):
        before = torch.sort(batch.actions[..., channel].reshape(-1)).values
        after = torch.sort(out.actions[..., channel].reshape(-1)).values
        assert torch.equal(before, after)
    # Actions and masks must move together under the SAME permutation. Recover
    # the permutation from the (distinct) action sequences, confirm it is a true
    # permutation of range(B), then assert the masks were moved by it too -- this
    # fails if masks were shuffled with a different permutation than the actions.
    permutation = _recover_batch_permutation(batch.actions, out.actions)
    assert sorted(permutation) == list(range(n_examples))
    for position, source in enumerate(permutation):
        assert torch.equal(out.action_mask[position], batch.action_mask[source])
    _assert_input_unmodified(batch, snapshot)


def test_time_shift_never_crosses_shot_boundary() -> None:
    batch = make_synthetic_fusion_batch(B=5, H=6, seed=3)
    # Give each example a NON-roll-invariant action mask (a prefix of Trues,
    # never all-True/all-False); the fixture's all-True masks are identical under
    # every roll, so they could not expose values and masks rolled by different
    # offsets.
    n_examples, n_steps = batch.action_mask.shape
    patterned = torch.zeros(n_examples, n_steps, dtype=torch.bool)
    for b in range(n_examples):
        patterned[b, : (b % (n_steps - 1)) + 1] = True
    batch = dataclasses.replace(batch, action_mask=patterned)
    snapshot = _snapshot(batch)
    generator = torch.Generator().manual_seed(17)

    out = perturb_actions(batch, ActionPerturbation.WITHIN_SHOT_TIME_SHIFT, generator)

    assert out.actions.shape == batch.actions.shape
    for b in range(n_examples):
        # Each example's rolled content is a permutation of that same example's
        # own timesteps -- no value ever leaks in from another shot.
        before = torch.sort(batch.actions[b].reshape(-1)).values
        after = torch.sort(out.actions[b].reshape(-1)).values
        assert torch.equal(before, after)
        # Values and mask must be rolled by the SAME per-example offset. Recover
        # the offset from the (distinct-per-timestep) values, then assert the
        # mask was rolled by it too -- this fails if the mask was rolled by a
        # different offset than the values.
        offset = _recover_roll_offset(batch.actions[b], out.actions[b])
        assert torch.equal(
            torch.roll(batch.action_mask[b], shifts=offset, dims=0),
            out.action_mask[b],
        )
    # The waveform moves relative to true time; action_times are untouched.
    assert torch.equal(out.action_times, batch.action_times)
    _assert_input_unmodified(batch, snapshot)


def test_zero_mode_zeroes_values_preserves_mask() -> None:
    batch = make_synthetic_fusion_batch(B=4, seed=5)
    snapshot = _snapshot(batch)
    generator = torch.Generator().manual_seed(1)

    out = perturb_actions(batch, ActionPerturbation.ZERO, generator)

    assert torch.all(out.actions == 0)
    assert out.actions.shape == batch.actions.shape
    assert torch.equal(out.action_mask, batch.action_mask)
    _assert_input_unmodified(batch, snapshot)


def test_perturbations_deterministic_given_seed() -> None:
    batch = make_synthetic_fusion_batch(B=6, H=6, seed=2)
    stochastic = (
        ActionPerturbation.BATCH_SHUFFLE,
        ActionPerturbation.WITHIN_SHOT_TIME_SHIFT,
    )

    for mode in stochastic:
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        out1 = perturb_actions(batch, mode, g1)
        out2 = perturb_actions(batch, mode, g2)
        assert torch.equal(out1.actions, out2.actions)
        assert torch.equal(out1.action_mask, out2.action_mask)

        # Different seeds must produce variety for the stochastic modes.
        outs = []
        for seed in range(6):
            gen = torch.Generator().manual_seed(seed)
            outs.append(perturb_actions(batch, mode, gen).actions)
        assert any(not torch.equal(outs[0], other) for other in outs[1:])

    # REAL and ZERO ignore the generator entirely.
    for mode in (ActionPerturbation.REAL, ActionPerturbation.ZERO):
        ga = torch.Generator().manual_seed(1)
        gb = torch.Generator().manual_seed(999)
        assert torch.equal(
            perturb_actions(batch, mode, ga).actions,
            perturb_actions(batch, mode, gb).actions,
        )


def test_perturb_actions_accepts_lowercase_string_mode() -> None:
    batch = make_synthetic_fusion_batch(B=3, seed=9)
    generator = torch.Generator().manual_seed(4)

    out = perturb_actions(batch, "zero", generator)

    assert torch.all(out.actions == 0)


def test_sensitivity_nonzero_for_action_dependent_toy_model_and_zero_for_action_blind_model() -> (
    None
):
    batch = make_synthetic_fusion_batch(B=3, seed=8)

    def action_dependent(b: FusionBatch) -> torch.Tensor:
        return b.actions.sum()

    def action_blind(b: FusionBatch) -> torch.Tensor:
        return b.context["slow_ts"].sum()

    dependent = action_sensitivity(
        action_dependent, batch, epsilon=1e-2, n_directions=5, seed=0
    )
    blind = action_sensitivity(
        action_blind, batch, epsilon=1e-2, n_directions=5, seed=0
    )

    assert dependent["sensitivity_mean"] > 0.0
    assert dependent["sensitivity_max"] > 0.0
    assert blind["sensitivity_mean"] == 0.0
    assert blind["sensitivity_max"] == 0.0
    # Sensitivity output must be JSON-safe.
    json.dumps(dependent)
    json.dumps(blind)


def test_sensitivity_deterministic_given_seed() -> None:
    batch = make_synthetic_fusion_batch(B=3, seed=6)

    def forward(b: FusionBatch) -> torch.Tensor:
        return (b.actions**2).sum()

    first = action_sensitivity(forward, batch, epsilon=1e-2, n_directions=4, seed=1)
    second = action_sensitivity(forward, batch, epsilon=1e-2, n_directions=4, seed=1)

    assert first == second


def test_direction_like_matches_actions_device_dtype_and_is_unit_norm() -> None:
    actions = make_synthetic_fusion_batch(B=2, seed=1).actions
    generator = torch.Generator().manual_seed(3)

    direction = _diag._direction_like(actions, generator)

    assert direction.shape == actions.shape
    assert direction.device == actions.device
    assert direction.dtype == actions.dtype
    assert direction.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_direction_like_follows_actions_device_without_gpu() -> None:
    # A meta-device tensor asserts the direction is created on the action
    # tensor's device even with no accelerator present. The pre-fix code always
    # built the direction on CPU, which crashed when added to a non-CPU action
    # tensor in the normal accelerator evaluation path.
    actions = make_synthetic_fusion_batch(B=2, seed=1).actions.to("meta")
    generator = torch.Generator().manual_seed(3)

    direction = _diag._direction_like(actions, generator)

    assert direction.device.type == "meta"
    assert direction.shape == actions.shape
    assert direction.dtype == actions.dtype


def test_direction_like_deterministic_given_seed() -> None:
    actions = make_synthetic_fusion_batch(B=2, seed=1).actions

    same = _diag._direction_like(actions, torch.Generator().manual_seed(7))
    again = _diag._direction_like(actions, torch.Generator().manual_seed(7))
    other = _diag._direction_like(actions, torch.Generator().manual_seed(8))

    assert torch.equal(same, again)
    assert not torch.equal(same, other)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a visible CUDA/ROCm device",
)
def test_action_sensitivity_runs_on_cuda_and_matches_cpu_directions() -> None:
    cpu_batch = make_synthetic_fusion_batch(B=2, seed=4)
    cuda_actions = cpu_batch.actions.to("cuda")

    # Same seed yields seed-identical direction values across devices.
    cpu_dir = _diag._direction_like(cpu_batch.actions, torch.Generator().manual_seed(5))
    cuda_dir = _diag._direction_like(cuda_actions, torch.Generator().manual_seed(5))
    assert cuda_dir.device.type == "cuda"
    assert torch.equal(cuda_dir.cpu(), cpu_dir)

    # The full probe runs end-to-end on an accelerator-resident batch.
    cuda_batch = dataclasses.replace(cpu_batch, actions=cuda_actions)

    def forward(b: FusionBatch) -> torch.Tensor:
        return (b.actions**2).sum()

    report = action_sensitivity(
        forward, cuda_batch, epsilon=1e-2, n_directions=3, seed=0
    )
    assert report["sensitivity_mean"] > 0.0


def test_action_use_report_gain_positive_for_action_dependent_loss() -> None:
    batches = [make_synthetic_fusion_batch(B=6, seed=s) for s in (1, 2, 3)]

    def loss_fn(batch: FusionBatch) -> float:
        # Low when each example's actions match its own target (REAL); higher
        # when the per-example alignment is broken (shuffle) or destroyed (zero).
        action_summary = batch.actions.mean(dim=(1, 2))
        target_summary = batch.target["slow_ts"].mean(dim=(1, 2))
        return float(((action_summary - target_summary) ** 2).mean())

    modes = [
        ActionPerturbation.REAL,
        ActionPerturbation.BATCH_SHUFFLE,
        ActionPerturbation.ZERO,
    ]
    report = action_use_report(loss_fn, batches, modes, seed=123)

    assert report["predictive_gain/batch_shuffle"] > 0.0
    assert report["predictive_gain/zero"] > 0.0
    assert report["predictive_gain/real"] == 0.0
    assert "loss/real" in report
    # Determinism and JSON-safety.
    assert report == action_use_report(loss_fn, batches, modes, seed=123)
    json.dumps(report)
