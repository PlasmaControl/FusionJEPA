"""Tests for evaluation-side action-use diagnostics (Task 3.6).

These exercise the action perturbation modes, the action-use report, and the
finite-difference action-sensitivity probe. None of this is ever wired into a
training loss -- it is strictly evaluation-side instrumentation.
"""

import json

import torch

from fusion_jepa.data.batch import FusionBatch
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


def test_batch_shuffle_preserves_shapes_and_per_channel_marginals() -> None:
    batch = make_synthetic_fusion_batch(B=6, seed=7)
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
    # The mask multiset is preserved too (mask travels with its sequence).
    assert torch.equal(
        torch.sort(batch.action_mask.reshape(-1).int()).values,
        torch.sort(out.action_mask.reshape(-1).int()).values,
    )
    _assert_input_unmodified(batch, snapshot)


def test_time_shift_never_crosses_shot_boundary() -> None:
    batch = make_synthetic_fusion_batch(B=5, H=6, seed=3)
    snapshot = _snapshot(batch)
    generator = torch.Generator().manual_seed(17)

    out = perturb_actions(batch, ActionPerturbation.WITHIN_SHOT_TIME_SHIFT, generator)

    assert out.actions.shape == batch.actions.shape
    # Each example's rolled content is a permutation of that same example's own
    # timesteps -- no value ever leaks in from another shot.
    for b in range(batch.actions.shape[0]):
        before = torch.sort(batch.actions[b].reshape(-1)).values
        after = torch.sort(out.actions[b].reshape(-1)).values
        assert torch.equal(before, after)
        mask_before = torch.sort(batch.action_mask[b].int()).values
        mask_after = torch.sort(out.action_mask[b].int()).values
        assert torch.equal(mask_before, mask_after)
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
