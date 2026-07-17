"""Unit tests for the continuous modality tokenizers (Task 2.2).

Covers the behaviours mandated by the task brief: output shapes and token-mask
propagation for both tokenizers, invariance of tokens to the values at masked
positions (including NaN placeholders), the learned missing-embedding path that
actually drives masked tokens (not a zero-fill), invalidation of a fully masked
patch while keeping its token finite, and retention of channel/time/coord in the
emitted :class:`TokenMetadata`.
"""

import math

import torch

from fusion_jepa.models.tokenizers import ProfileTokenizer, ScalarSeriesTokenizer
from fusion_jepa.models.types import TokenMetadata


def _scalar_inputs(
    B: int, C: int, T: int, *, offset: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully observed ``[B, C, T]`` ramp with float64 per-sample times."""
    times = (
        torch.arange(T, dtype=torch.float64) + offset
    ).unsqueeze(0).expand(B, T).contiguous()
    channel = 0.1 * torch.arange(C, dtype=torch.float32)
    values = (
        times.to(torch.float32).unsqueeze(1) + channel.view(1, C, 1)
    ).contiguous()
    mask = torch.ones(B, C, T, dtype=torch.bool)
    return values, mask, times


def _profile_inputs(
    B: int, R: int, T: int, *, offset: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully observed ``[B, R, T]`` profile ramp with times and radial coords."""
    times = (
        torch.arange(T, dtype=torch.float64) + offset
    ).unsqueeze(0).expand(B, T).contiguous()
    coords = torch.arange(R, dtype=torch.float32).unsqueeze(0).expand(B, R).contiguous()
    values = (
        times.to(torch.float32).unsqueeze(1) + coords.unsqueeze(-1)
    ).contiguous()
    mask = torch.ones(B, R, T, dtype=torch.bool)
    return values, mask, times, coords


def test_scalar_tokenizer_shape_and_mask_propagation():
    B, C, T, patch_len, d_model, n_freqs = 2, 3, 7, 3, 8, 4
    values, mask, times = _scalar_inputs(B, C, T)
    torch.manual_seed(0)
    tok = ScalarSeriesTokenizer(C, d_model, patch_len, n_freqs)

    n_patches = math.ceil(T / patch_len)
    N = C * n_patches
    tokens, token_mask, meta = tok(values, mask, times)

    assert tokens.shape == (B, N, d_model)
    assert tokens.dtype == torch.float32
    assert token_mask.shape == (B, N)
    assert token_mask.dtype == torch.bool
    assert token_mask.all()  # every patch has observed samples

    assert isinstance(meta, TokenMetadata)
    assert meta.channel_id.shape == (B, N)
    assert meta.time_s.shape == (B, N)
    assert meta.coord.shape == (B, N)
    assert torch.isnan(meta.coord).all()  # scalar signal -> no spatial coord

    # A patch keeps token_mask True as long as one sample survives.
    partial = mask.clone()
    partial[:, 0, 0] = False
    _, token_mask_partial, _ = tok(values, partial, times)
    assert token_mask_partial.all()


def test_profile_tokenizer_shape_and_mask_propagation():
    B, R, T, radial_patch, d_model = 2, 5, 4, 2, 8
    values, mask, times, coords = _profile_inputs(B, R, T)
    torch.manual_seed(0)
    tok = ProfileTokenizer(d_model, radial_patch, R)

    n_patches = math.ceil(R / radial_patch)
    N = T * n_patches
    tokens, token_mask, meta = tok(values, mask, times, coords)

    assert tokens.shape == (B, N, d_model)
    assert tokens.dtype == torch.float32
    assert token_mask.shape == (B, N)
    assert token_mask.dtype == torch.bool
    assert token_mask.all()

    assert meta.coord.shape == (B, N)
    assert torch.isfinite(meta.coord).all()  # profiles carry radial positions
    # coord carries the patch-center radial position (first-time-frame block).
    expected_centers = torch.tensor([0.5, 2.5, 4.0])
    assert torch.allclose(meta.coord[0, :n_patches], expected_centers)


def test_masked_values_do_not_change_tokens():
    # Scalar: mutating values only at masked positions must not move tokens.
    B, C, T, patch_len, d_model, n_freqs = 2, 3, 6, 2, 8, 3
    values, mask, times = _scalar_inputs(B, C, T)
    mask[:, 0, 1] = False
    mask[:, 1, 3] = False
    mask[:, 2, 5] = False
    torch.manual_seed(1)
    tok = ScalarSeriesTokenizer(C, d_model, patch_len, n_freqs)

    tokens1, mask1, _ = tok(values, mask, times)

    huge = values.clone()
    huge[~mask] = 999.0
    nans = values.clone()
    nans[~mask] = float("nan")
    tokens2, mask2, _ = tok(huge, mask, times)
    tokens3, _, _ = tok(nans, mask, times)

    assert torch.equal(tokens1, tokens2)
    assert torch.equal(tokens1, tokens3)  # NaN placeholder never reaches proj
    assert torch.equal(mask1, mask2)

    # Profile: same invariant along the radial axis.
    Bp, Rp, Tp, radial_patch, dm = 2, 4, 3, 2, 8
    pv, pm, pt, pc = _profile_inputs(Bp, Rp, Tp)
    pm[:, 0, 0] = False
    pm[:, 3, 2] = False
    torch.manual_seed(2)
    ptok = ProfileTokenizer(dm, radial_patch, Rp)

    ptokens1, _, _ = ptok(pv, pm, pt, pc)
    pv2 = pv.clone()
    pv2[~pm] = -123.0
    pnan = pv.clone()
    pnan[~pm] = float("nan")
    ptokens2, _, _ = ptok(pv2, pm, pt, pc)
    ptokens3, _, _ = ptok(pnan, pm, pt, pc)
    assert torch.equal(ptokens1, ptokens2)
    assert torch.equal(ptokens1, ptokens3)  # NaN placeholder never reaches proj
    assert torch.isfinite(ptokens3).all()


def test_missing_embedding_drives_masked_tokens():
    # A fully-masked patch's token must flow through the LEARNED per-channel
    # missing-fill parameter, not a zero-fill. Perturbing that parameter must
    # move the fully-masked token while leaving a fully-observed token identical.
    # (Under a `values * mask` zero-fill, missing_fill would be unused and the
    # masked token would NOT move -- the first assertion below would then fail.)
    B, C, T, patch_len, d_model, n_freqs = 1, 2, 6, 3, 8, 2
    values, mask, times = _scalar_inputs(B, C, T)
    mask[:, 0, 0:3] = False  # channel 0, patch 0 fully masked
    torch.manual_seed(3)
    tok = ScalarSeriesTokenizer(C, d_model, patch_len, n_freqs)

    n_patches = math.ceil(T / patch_len)
    masked_tok = 0 * n_patches + 0  # (channel 0, patch 0): fully masked
    observed_tok = 0 * n_patches + 1  # (channel 0, patch 1): fully observed

    tokens_before, _, _ = tok(values, mask, times)

    with torch.no_grad():
        # Distinct new per-channel fill values, far from the ~0.02-std init.
        tok.missing_fill.copy_(torch.tensor([5.0, -5.0]))

    tokens_after, _, _ = tok(values, mask, times)

    # Masked patch flows through the learned missing embedding -> token moves.
    assert not torch.equal(tokens_before[:, masked_tok], tokens_after[:, masked_tok])
    # Observed patch never touches missing_fill -> token is unchanged.
    assert torch.equal(tokens_before[:, observed_tok], tokens_after[:, observed_tok])


def test_fully_masked_patch_yields_invalid_token():
    # Scalar: fully mask channel 1's first patch -> its token is invalid.
    B, C, T, patch_len, d_model, n_freqs = 2, 2, 6, 3, 8, 2
    values, mask, times = _scalar_inputs(B, C, T)
    mask[:, 1, 0:3] = False
    torch.manual_seed(0)
    tok = ScalarSeriesTokenizer(C, d_model, patch_len, n_freqs)

    n_patches = math.ceil(T / patch_len)
    tokens, token_mask, _ = tok(values, mask, times)

    dead = 1 * n_patches + 0  # channel-major index of (channel 1, patch 0)
    assert not token_mask[:, dead].any()
    alive = [i for i in range(C * n_patches) if i != dead]
    assert token_mask[:, alive].all()
    # An invalid token must still be finite (downstream attention safety).
    assert torch.isfinite(tokens).all()

    # Profile: fully mask radial patch 0 at time frame 0 -> its token invalid.
    Bp, Rp, Tp, radial_patch, dm = 2, 4, 3, 2, 8
    pv, pm, pt, pc = _profile_inputs(Bp, Rp, Tp)
    pm[:, 0:2, 0] = False
    torch.manual_seed(0)
    ptok = ProfileTokenizer(dm, radial_patch, Rp)

    npp = math.ceil(Rp / radial_patch)
    ptokens, ptoken_mask, _ = ptok(pv, pm, pt, pc)
    dead_p = 0 * npp + 0  # time-major index of (frame 0, radial patch 0)
    assert not ptoken_mask[:, dead_p].any()
    assert ptoken_mask[:, dead_p + 1].all()  # (frame 0, radial patch 1) survives
    assert torch.isfinite(ptokens).all()


def test_metadata_retains_channel_time_coord():
    # Scalar metadata: channel-major ids, patch-center times, NaN coords.
    B, C, T, patch_len, d_model, n_freqs = 1, 3, 7, 3, 8, 4
    values, mask, times = _scalar_inputs(B, C, T)
    torch.manual_seed(0)
    tok = ScalarSeriesTokenizer(C, d_model, patch_len, n_freqs)
    _, _, meta = tok(values, mask, times)

    assert meta.channel_id.dtype == torch.long
    assert torch.equal(
        meta.channel_id[0], torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    )
    assert meta.time_s.dtype == torch.float64
    assert torch.allclose(
        meta.time_s[0],
        torch.tensor(
            [1.0, 4.0, 6.0, 1.0, 4.0, 6.0, 1.0, 4.0, 6.0], dtype=torch.float64
        ),
    )
    assert meta.coord.dtype == torch.float32
    assert torch.isnan(meta.coord).all()
    assert meta.modality == "scalar_series"

    # Profile metadata: radial-patch ids, frame times, patch-center coords.
    Bp, Rp, Tp, radial_patch, dm = 1, 5, 4, 2, 8
    pv, pm, pt, pc = _profile_inputs(Bp, Rp, Tp)
    torch.manual_seed(0)
    ptok = ProfileTokenizer(dm, radial_patch, Rp)
    _, _, pmeta = ptok(pv, pm, pt, pc)

    assert torch.equal(
        pmeta.channel_id[0],
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    )
    assert torch.allclose(
        pmeta.time_s[0],
        torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.float64
        ),
    )
    assert torch.allclose(
        pmeta.coord[0],
        torch.tensor([0.5, 2.5, 4.0] * Tp, dtype=torch.float32),
    )
    assert pmeta.modality == "profile"
