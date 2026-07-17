"""Unit tests for the shared context encoder (Task 2.3).

Covers the behaviours the task brief locks down for the state-token bottleneck
shared by the raw baseline and the JEPA:

* the encoder always emits a *fixed* ``[B, S, D]`` state slice with an all-True
  mask, independent of how many input tokens it is fed;
* the state latents are exactly invariant to the *values* of input tokens that
  are marked invalid (masked as keys -> zero attention weight), while a valid
  token genuinely moves them;
* a sample whose input tokens are *all* invalid still yields finite latents (the
  all-``-inf`` softmax trap is avoided because state tokens are never padded);
* width (``d_model``) and depth (``n_blocks``) are constructor-configurable.

Forward is deterministic at the default ``dropout=0.0``, so no ``eval()`` dance
is needed; tests seed parameter init with ``torch.manual_seed``.
"""

import torch

from fusion_jepa.models.encoder import ContextEncoder


def _tokens(B: int, N: int, D: int, *, seed: int = 0) -> torch.Tensor:
    """Finite random token payload ``[B, N, D]`` (tokenizer output is finite)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, D, generator=gen)


def test_encoder_output_is_fixed_state_bottleneck():
    B, D, S = 2, 32, 6
    torch.manual_seed(0)
    enc = ContextEncoder(d_model=D, n_heads=4, n_blocks=2, n_state_tokens=S)

    # The state slice is a fixed-size bottleneck: its shape must not depend on
    # the number of input tokens fed in.
    for n in (1, 5, 20):
        tokens = _tokens(B, n, D, seed=n)
        token_mask = torch.ones(B, n, dtype=torch.bool)
        z, z_mask = enc(tokens, token_mask)
        assert z.shape == (B, S, D)
        assert z.dtype == torch.float32
        assert z_mask.shape == (B, S)
        assert z_mask.dtype == torch.bool
        assert z_mask.all()  # state tokens are never padded
        assert torch.isfinite(z).all()


def test_masked_input_tokens_do_not_affect_state_latents():
    B, N, D, S = 2, 8, 32, 4
    torch.manual_seed(1)
    enc = ContextEncoder(d_model=D, n_heads=4, n_blocks=3, n_state_tokens=S)

    tokens = _tokens(B, N, D, seed=7)
    token_mask = torch.ones(B, N, dtype=torch.bool)
    token_mask[0, 2] = False
    token_mask[0, 5] = False
    token_mask[1, 0] = False

    z_ref, _ = enc(tokens, token_mask)

    # Mutating INVALID token values (huge, finite) must not move the latents at
    # all: masked keys carry exactly-zero attention weight, so their value never
    # reaches the state slice, at any depth.
    mutated = tokens.clone()
    mutated[~token_mask] = 1.0e6
    z_hi, _ = enc(mutated, token_mask)
    assert torch.equal(z_ref, z_hi)

    mutated_lo = tokens.clone()
    mutated_lo[~token_mask] = -1.0e6
    z_lo, _ = enc(mutated_lo, token_mask)
    assert torch.equal(z_ref, z_lo)

    # Sanity: mutating a VALID token DOES move the latents, proving the mask is
    # doing real work rather than the encoder ignoring inputs wholesale.
    valid = tokens.clone()
    valid[0, 1] += 3.0  # position (0, 1) is observed
    z_valid, _ = enc(valid, token_mask)
    assert not torch.equal(z_ref, z_valid)


def test_fully_masked_sample_is_finite():
    B, N, D, S = 3, 10, 32, 5
    torch.manual_seed(2)
    enc = ContextEncoder(d_model=D, n_heads=4, n_blocks=3, n_state_tokens=S)

    tokens = _tokens(B, N, D, seed=11)
    token_mask = torch.ones(B, N, dtype=torch.bool)
    token_mask[0] = False  # sample 0: every input token invalid

    z, z_mask = enc(tokens, token_mask)
    assert torch.isfinite(z).all()  # no all-`-inf` softmax NaN
    assert z_mask.all()

    # The extreme case: EVERY sample fully masked still yields finite latents,
    # because state tokens can always attend to one another.
    all_masked = torch.zeros(B, N, dtype=torch.bool)
    z_all, _ = enc(tokens, all_masked)
    assert torch.isfinite(z_all).all()


def test_width_and_depth_configurable():
    B, N = 2, 6
    # Small variant used by the brief: narrow width, shallow depth.
    torch.manual_seed(3)
    small = ContextEncoder(d_model=64, n_heads=8, n_blocks=2, n_state_tokens=4)
    assert small.d_model == 64
    assert small.n_state_tokens == 4
    assert len(small.blocks) == 2

    tokens = _tokens(B, N, 64, seed=5)
    token_mask = torch.ones(B, N, dtype=torch.bool)
    z, z_mask = small(tokens, token_mask)
    assert z.shape == (B, 4, 64)
    assert z_mask.shape == (B, 4)

    # A wider, deeper variant reconfigures cleanly.
    torch.manual_seed(3)
    big = ContextEncoder(d_model=128, n_heads=8, n_blocks=5, n_state_tokens=16)
    assert big.d_model == 128
    assert big.n_state_tokens == 16
    assert len(big.blocks) == 5

    tokens_big = _tokens(B, N, 128, seed=6)
    z_big, _ = big(tokens_big, token_mask)
    assert z_big.shape == (B, 16, 128)
