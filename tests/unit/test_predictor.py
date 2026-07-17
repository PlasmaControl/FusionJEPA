"""Unit tests for the action-conditioned latent predictor (Task 2.5).

Locks the six brief-mandated behaviours of the shared world model:

* direct multi-horizon forward emits ``[B, K, S, d_latent_in]`` float32 latents
  that stay finite even when every action is masked;
* different horizons give different predictions (each horizon adds its own
  Fourier embedding and sees its own causal action subset);
* **the causality invariant** -- a horizon-``h`` query is bit-identical under any
  change to actions whose time exceeds ``h`` (they can never leak in), while an
  in-horizon action genuinely moves it;
* a single-step ``rollout`` equals the equivalent direct single-horizon call
  bit-for-bit (rollout is literally repeated direct calls);
* the ``device_id`` selects a shared embedding row -- switching it moves the
  output without changing the parameter set, and an unknown id is a hard error;
* masked action timesteps are ignored (their values never reach any output).

Forward is deterministic at ``dropout=0.0``; tests seed parameter init with
``torch.manual_seed``.
"""

import pytest
import torch

from fusion_jepa.models.predictor import LatentPredictor


def _make_model(
    *,
    device_vocab=("DIII-D",),
    n_state_tokens=4,
    d_model=16,
    d_latent_in=8,
    d_action=12,
    d_device_context=3,
    n_blocks=2,
    n_heads=4,
    seed=0,
) -> LatentPredictor:
    torch.manual_seed(seed)
    return LatentPredictor(
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        d_latent_in=d_latent_in,
        n_state_tokens=n_state_tokens,
        device_vocab=list(device_vocab),
        d_device_context=d_device_context,
        d_action=d_action,
    )


def _inputs(
    B=2,
    S=4,
    H=6,
    K=3,
    *,
    d_latent_in=8,
    d_action=12,
    Dc=3,
    horizons=None,
    action_times=None,
    device_name="DIII-D",
    seed=1,
) -> dict:
    """Standard forward kwargs; action times/horizons share the context-end base."""
    gen = torch.Generator().manual_seed(seed)
    if action_times is None:
        action_times = (
            0.1 * torch.arange(1, H + 1, dtype=torch.float64)
        ).unsqueeze(0).expand(B, H).contiguous()
    if horizons is None:
        horizons = (
            torch.linspace(0.15, 0.55, K, dtype=torch.float64)
            .unsqueeze(0)
            .expand(B, K)
            .contiguous()
        )
    return {
        "context_latents": torch.randn(B, S, d_latent_in, generator=gen),
        "context_mask": torch.ones(B, S, dtype=torch.bool),
        "action_tokens": torch.randn(B, H, d_action, generator=gen),
        "action_mask": torch.ones(B, H, dtype=torch.bool),
        "action_times": action_times,
        "horizons": horizons,
        "device_id": [device_name] * B,
        "device_context": torch.randn(B, Dc, generator=gen),
        "device_context_mask": torch.ones(B, Dc, dtype=torch.bool),
    }


def test_direct_multi_horizon_output_shape():
    B, S, H, K, d_in = 3, 4, 5, 2, 8
    model = _make_model(n_state_tokens=S, d_latent_in=d_in)
    inp = _inputs(B=B, S=S, H=H, K=K, d_latent_in=d_in)

    z = model(**inp)
    assert z.shape == (B, K, S, d_in)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()

    # All actions masked -> still finite (device + query columns keep every
    # softmax row off the all-`-inf` path).
    all_masked = dict(inp)
    all_masked["action_mask"] = torch.zeros(B, H, dtype=torch.bool)
    z_masked = model(**all_masked)
    assert torch.isfinite(z_masked).all()


def test_different_horizons_give_different_predictions():
    B, S, H = 2, 4, 6
    model = _make_model(n_state_tokens=S)
    horizons = torch.tensor([[0.15, 0.55]], dtype=torch.float64).expand(B, 2)
    inp = _inputs(B=B, S=S, H=H, K=2, horizons=horizons.contiguous())

    z = model(**inp)
    # Distinct horizons carry distinct Fourier embeddings and see distinct
    # causal action subsets -> the two horizon slices must differ.
    assert not torch.allclose(z[:, 0], z[:, 1])


def test_actions_beyond_horizon_do_not_leak():
    # THE scientific invariant: a horizon-h query must not depend on any action
    # whose time exceeds h. Construct actions at times 0.1..0.6 and query at
    # horizons 0.15 / 0.35 / 0.55; changing the values of beyond-horizon actions
    # (finite garbage) must leave that horizon's prediction bit-identical.
    B, S, H, K = 2, 4, 6, 3
    model = _make_model(n_state_tokens=S)
    inp = _inputs(B=B, S=S, H=H, K=K)
    times = inp["action_times"][0]  # [H], shared across batch
    horizons = inp["horizons"][0]  # [K]

    z_ref = model(**inp)

    for k in range(K):
        h = float(horizons[k])
        beyond = times > h  # [H] bool: strictly-later actions
        assert bool(beyond.any())  # the test would be vacuous otherwise
        mutated = inp["action_tokens"].clone()
        mutated[:, beyond, :] = 1.0e6  # huge finite garbage in the future
        z_mut = model(**{**inp, "action_tokens": mutated})
        assert torch.equal(z_ref[:, k], z_mut[:, k])

    # Sanity: an IN-horizon action DOES move the prediction, proving the mask is
    # doing real work rather than the predictor ignoring actions wholesale.
    within = times <= float(horizons[-1])
    moved = inp["action_tokens"].clone()
    moved[:, within, :] += 3.0
    z_moved = model(**{**inp, "action_tokens": moved})
    assert not torch.equal(z_ref[:, -1], z_moved[:, -1])


def test_single_step_rollout_equals_direct_call():
    B, S, H, d_in = 2, 4, 6, 8
    model = _make_model(n_state_tokens=S, d_latent_in=d_in)
    inp = _inputs(B=B, S=S, H=H, K=1, d_latent_in=d_in)
    step = 0.3

    roll = model.rollout(
        inp["context_latents"],
        inp["context_mask"],
        inp["action_tokens"],
        inp["action_mask"],
        inp["action_times"],
        inp["device_id"],
        inp["device_context"],
        inp["device_context_mask"],
        step_seconds=step,
        n_steps=1,
    )
    assert roll.shape == (B, 1, S, d_in)

    direct = model(
        **{
            **inp,
            "horizons": torch.full((B, 1), step, dtype=torch.float64),
        }
    )
    # A 1-step rollout is literally a single-horizon direct call -> bit-exact.
    assert torch.equal(roll, direct)

    # Multi-step rollout has the documented shape and stays finite.
    roll3 = model.rollout(
        inp["context_latents"],
        inp["context_mask"],
        inp["action_tokens"],
        inp["action_mask"],
        inp["action_times"],
        inp["device_id"],
        inp["device_context"],
        inp["device_context_mask"],
        step_seconds=step,
        n_steps=3,
    )
    assert roll3.shape == (B, 3, S, d_in)
    assert torch.isfinite(roll3).all()


def test_device_id_changes_output_not_parameter_set():
    B, S, H = 2, 4, 6
    model = _make_model(device_vocab=("DIII-D", "JET"), n_state_tokens=S)
    inp = _inputs(B=B, S=S, H=H, K=2)

    n_params_before = sum(p.numel() for p in model.parameters())
    z_a = model(**{**inp, "device_id": ["DIII-D"] * B})
    z_b = model(**{**inp, "device_id": ["JET"] * B})
    n_params_after = sum(p.numel() for p in model.parameters())

    # Device conditioning selects an embedding row: the output moves, but the
    # parameter set is a single shared table -- it does not grow per device.
    assert not torch.allclose(z_a, z_b)
    assert n_params_before == n_params_after
    assert model.n_devices == 2

    # An unknown device id is an actionable hard error, not a silent fallback.
    with pytest.raises(ValueError, match="unknown device"):
        model(**{**inp, "device_id": ["ITER"] * B})


def test_masked_action_tokens_ignored():
    B, S, H, K = 2, 4, 6, 3
    model = _make_model(n_state_tokens=S)
    inp = _inputs(B=B, S=S, H=H, K=K)
    action_mask = inp["action_mask"].clone()
    action_mask[:, 1] = False
    action_mask[:, 3] = False
    inp = {**inp, "action_mask": action_mask}

    z_ref = model(**inp)

    # Mutating MASKED timestep values (finite garbage) must not move any output.
    mutated = inp["action_tokens"].clone()
    mutated[:, 1, :] = 1.0e6
    mutated[:, 3, :] = -1.0e6
    z_mut = model(**{**inp, "action_tokens": mutated})
    assert torch.equal(z_ref, z_mut)

    # Sanity: mutating a VALID timestep does move the output.
    valid = inp["action_tokens"].clone()
    valid[:, 0, :] += 3.0  # timestep 0 is observed
    z_valid = model(**{**inp, "action_tokens": valid})
    assert not torch.equal(z_ref, z_valid)
