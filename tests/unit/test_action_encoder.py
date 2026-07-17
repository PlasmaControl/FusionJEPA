"""Unit tests for the action encoder (Task 2.4).

Locks the four brief-mandated behaviours: output shapes with per-timestep mask
pass-through, permutation-sensitivity across actuator channels (actuator identity
is positional), shift-sensitivity in time (added Fourier time feature), and
invariance of valid tokens to the values sitting at masked / missing positions --
including finiteness under all-masked and all-``NaN`` inputs, and the learned
per-actuator missing embedding that actually drives a missing channel.
"""

import torch

from fusion_jepa.models.action_encoder import ActionEncoder


def _actions(
    B: int, H: int, A: int, *, offset: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully observed ``[B, H, A]`` actions with distinct per-channel ramps.

    ``actions[b, h, a] == time[h] + 0.1 * a`` so the ``A`` values at each
    timestep are distinct (permutation genuinely changes the input) and times are
    strictly monotone float64.
    """
    times = (
        torch.arange(H, dtype=torch.float64) + offset
    ).unsqueeze(0).expand(B, H).contiguous()
    channel = 0.1 * torch.arange(A, dtype=torch.float32)
    actions = (
        times.to(torch.float32).unsqueeze(-1) + channel.view(1, 1, A)
    ).contiguous()
    mask = torch.ones(B, H, dtype=torch.bool)
    return actions, mask, times


def test_shapes_and_mask():
    B, H, A, D, F = 2, 5, 3, 8, 4
    actions, mask, times = _actions(B, H, A)
    torch.manual_seed(0)
    enc = ActionEncoder(A, D, F)

    u_tokens, u_mask = enc(actions, mask, times)
    assert u_tokens.shape == (B, H, D)
    assert u_tokens.dtype == torch.float32
    assert u_mask.shape == (B, H)
    assert u_mask.dtype == torch.bool
    assert torch.equal(u_mask, mask)  # per-timestep mask pass-through

    # A masked timestep flips u_mask False but keeps its token finite.
    partial = mask.clone()
    partial[:, 0] = False
    u_tokens_p, u_mask_p = enc(actions, partial, times)
    assert torch.equal(u_mask_p, partial)
    assert torch.isfinite(u_tokens_p).all()

    # All-masked + all-NaN robustness: outputs stay finite.
    nan_actions = torch.full((B, H, A), float("nan"), dtype=torch.float32)
    all_masked = torch.zeros(B, H, dtype=torch.bool)
    u_tokens_n, u_mask_n = enc(nan_actions, all_masked, times)
    assert torch.isfinite(u_tokens_n).all()
    assert not u_mask_n.any()


def test_channel_permutation_changes_encoding():
    B, H, A, D, F = 1, 4, 3, 8, 4
    actions, mask, times = _actions(B, H, A)
    torch.manual_seed(0)
    enc = ActionEncoder(A, D, F)

    tokens, _ = enc(actions, mask, times)
    perm = torch.tensor([2, 0, 1])
    tokens_perm, _ = enc(actions[:, :, perm], mask, times)

    # Actuator identity is positional -> permuting channels must move tokens.
    assert not torch.allclose(tokens, tokens_perm)


def test_time_shift_changes_encoding():
    B, H, A, D, F = 1, 4, 3, 8, 4
    actions, mask, times = _actions(B, H, A)
    torch.manual_seed(0)
    enc = ActionEncoder(A, D, F)

    tokens, _ = enc(actions, mask, times)
    # Same actuator values, times shifted by a constant -> isolates the Fourier
    # time feature so any change is attributable to shift-sensitivity alone.
    shifted, _ = enc(actions, mask, times + 0.5)

    assert not torch.allclose(tokens, shifted)


def test_masked_channel_values_are_ignored():
    # Changing the actuator values at MASKED timesteps must not move the tokens
    # at valid timesteps (every timestep is encoded independently).
    B, H, A, D, F = 2, 5, 3, 8, 3
    actions, mask, times = _actions(B, H, A)
    mask[:, 1] = False
    mask[:, 3] = False
    torch.manual_seed(1)
    enc = ActionEncoder(A, D, F)

    tokens1, umask1 = enc(actions, mask, times)

    huge = actions.clone()
    huge[~mask] = 999.0  # mutate only masked-timestep values (finite sentinel)
    nans = actions.clone()
    nans[~mask] = float("nan")  # ... and NaN
    tokens2, umask2 = enc(huge, mask, times)
    tokens3, _ = enc(nans, mask, times)

    valid = mask  # [B, H], True at valid timesteps
    assert torch.equal(tokens1[valid], tokens2[valid])
    assert torch.equal(tokens1[valid], tokens3[valid])  # NaN never leaks in
    assert torch.equal(umask1, umask2)
    assert torch.isfinite(tokens3).all()

    # A per-channel NaN within a VALID timestep flows through the learned
    # per-actuator missing embedding (not a zero-fill): perturbing that
    # parameter must move that token while leaving a fully-observed token equal.
    ch_nan = actions.clone()
    ch_nan[:, 0, 0] = float("nan")  # actuator 0 missing at valid timestep 0
    before, _ = enc(ch_nan, mask, times)
    with torch.no_grad():
        enc.missing_fill[0] += 5.0
    after, _ = enc(ch_nan, mask, times)

    assert not torch.equal(before[:, 0], after[:, 0])  # missing_fill drives it
    assert torch.equal(before[:, 2], after[:, 2])  # observed timestep untouched
    assert torch.isfinite(before).all()
