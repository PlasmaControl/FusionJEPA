"""Continuous modality tokenizers for Fusion-JEPA (M2, Task 2.2).

Two tokenizers turn one signal's raw samples into a flat token sequence for the
shared context encoder:

* :class:`ScalarSeriesTokenizer` -- a multi-channel scalar time series. One token
  per (channel, time-patch); channels are patched independently along time.
* :class:`ProfileTokenizer` -- a radial profile evolving in time. One token per
  (time frame, radial-patch); each time frame's radial profile is chunked into
  radial patches.

Both emit *continuous* tokens (no quantization). A quantized variant is a
separate, comparable alternative integrated later; nothing here builds toward it
beyond keeping the ``(tokens, token_mask, TokenMetadata)`` return contract clean.
Each tokenizer call handles exactly one signal's tensor, so a shared time axis
across signals is never assumed -- the M1 data reality is that context signals
arrive with differing native lengths.

Three invariants hold for both tokenizers:

* **Missing is not zero.** A masked (unobserved) sample is replaced by a learned
  per-channel / per-radial-point fill value *before* the linear projection, never
  by the literal value ``0`` -- so the network can distinguish "unobserved" from
  a true zero reading. Non-finite placeholders (``NaN``) sitting at masked
  positions are neutralized and never reach the projection.
* **A token is observed iff its patch holds at least one observed sample.** The
  returned ``token_mask`` is ``True`` exactly for those tokens.
* **Padding uses missing-marked samples.** When the patched axis length is not a
  multiple of the patch size, the trailing partial patch is padded with samples
  marked unobserved, so the token count is ``ceil(length / patch)`` whole patches
  and padded positions never count toward token validity nor toward the physical
  time / coordinate a token reports.

Token features are ``linear(patch values) + channel/patch embedding +
Fourier(time)`` and, for profiles, additionally ``+ Fourier(radial coord)``.
Physical times are kept as ``float64`` in the emitted :class:`TokenMetadata`;
they are cast to ``float32`` only when computing Fourier features. Forward is
fully deterministic -- all randomness lives in parameter initialization, which
tests seed via ``torch.manual_seed``.
"""

import math

import torch
from torch import Tensor, nn

from fusion_jepa.models.types import TokenMetadata

# Std for the learned embeddings / missing-fill parameters, matching the
# convention used by the legacy tokenizers so the raw-signal projection
# dominates the token at initialization.
_EMBED_INIT_STD = 0.02
# Fourier band count for the ProfileTokenizer, whose constructor takes no
# frequency argument (unlike ScalarSeriesTokenizer's explicit n_time_freqs).
_PROFILE_N_FREQS = 6


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division for positive operands."""
    return -(-numerator // denominator)


class _FourierFeatures(nn.Module):
    """Project a scalar coordinate to ``d_model`` via fixed sinusoidal bands.

    ``n_freqs`` octave-spaced angular frequencies (``pi * 2**k``) are fixed and
    registered as a buffer -- no forward-time randomness and correct device
    placement -- while a learned linear layer maps the ``2 * n_freqs`` sin/cos
    features to ``d_model``.
    """

    def __init__(self, n_freqs: int, d_model: int) -> None:
        super().__init__()
        if n_freqs < 1:
            raise ValueError(f"n_freqs must be >= 1, got {n_freqs}")
        freqs = math.pi * (2.0 ** torch.arange(n_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``x`` (``[...]`` float32) to ``[..., d_model]`` features."""
        angles = x.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feats)


class ScalarSeriesTokenizer(nn.Module):
    """Tokenize a multi-channel scalar time series, one token per (channel, patch).

    Args:
        n_channels: number of channels ``C`` in the input.
        d_model: token embedding dimension ``D``.
        patch_len: samples per time-patch. The time axis is split into
            ``ceil(T / patch_len)`` patches; a trailing partial patch is padded
            with missing-marked samples.
        n_time_freqs: number of Fourier bands for the token time feature.
        modality: modality name recorded in the emitted metadata (each signal
            configures its own instance; forward stays value/mask/time only).

    ``forward(values, value_mask, times)`` takes:

    * ``values`` -- ``[B, C, T]`` float32 raw samples (may hold non-finite
      placeholders at masked positions).
    * ``value_mask`` -- ``[B, C, T]`` bool, ``True`` where a sample is observed.
    * ``times`` -- ``[B, T]`` float64 physical sample times in seconds, shared
      across the channels of this one signal.

    and returns ``(tokens [B, N, D] float32, token_mask [B, N] bool,
    TokenMetadata)`` with ``N = C * ceil(T / patch_len)``. Tokens are ordered
    channel-major: token ``c * n_patches + p`` is channel ``c``'s ``p``-th
    time-patch. ``coord`` is ``NaN`` (scalar signals have no spatial position),
    and each token's ``time_s`` is the mean of the physical times of the real
    (non-padded) samples in its patch.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_len: int,
        n_time_freqs: int,
        *,
        modality: str = "scalar_series",
    ) -> None:
        super().__init__()
        if n_channels < 1 or d_model < 1 or patch_len < 1:
            raise ValueError("n_channels, d_model, patch_len must all be >= 1")
        self.n_channels = n_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.modality = modality

        self.proj = nn.Linear(patch_len, d_model)
        self.channel_embed = nn.Parameter(torch.empty(n_channels, d_model))
        # One learned fill value per channel, substituted for masked samples in
        # value-space before projection (never zero: missing != zero).
        self.missing_fill = nn.Parameter(torch.empty(n_channels))
        self.time_features = _FourierFeatures(n_time_freqs, d_model)

        nn.init.normal_(self.channel_embed, std=_EMBED_INIT_STD)
        nn.init.normal_(self.missing_fill, std=_EMBED_INIT_STD)

    def forward(
        self, values: Tensor, value_mask: Tensor, times: Tensor
    ) -> tuple[Tensor, Tensor, TokenMetadata]:
        if values.ndim != 3:
            raise ValueError(f"values must be [B, C, T], got {tuple(values.shape)}")
        B, C, T = values.shape
        if C != self.n_channels:
            raise ValueError(
                f"values has {C} channels, tokenizer expects {self.n_channels}"
            )
        if values.dtype != torch.float32:
            raise ValueError(f"values must be float32, got {values.dtype}")
        if value_mask.shape != values.shape or value_mask.dtype != torch.bool:
            raise ValueError("value_mask must be a bool tensor matching values")
        if tuple(times.shape) != (B, T) or times.dtype != torch.float64:
            raise ValueError(f"times must be float64 [B, T]=({B}, {T})")

        n_patches = _ceil_div(T, self.patch_len)
        pad = n_patches * self.patch_len - T

        # Neutralize non-finite placeholders, then pad the time axis so it tiles
        # into whole patches. Padded samples are marked unobserved and carry a
        # zero-weight time so they never move a token's reported time.
        safe = torch.nan_to_num(values)
        if pad:
            safe = torch.cat([safe, safe.new_zeros(B, C, pad)], dim=2)
            value_mask = torch.cat(
                [value_mask, value_mask.new_zeros(B, C, pad)], dim=2
            )
            times = torch.cat([times, times.new_zeros(B, pad)], dim=1)
            time_valid = torch.cat(
                [times.new_ones(T), times.new_zeros(pad)], dim=0
            )
        else:
            time_valid = times.new_ones(T)

        # Masked samples -> learned per-channel fill, before projection.
        filled = torch.where(value_mask, safe, self.missing_fill.view(1, C, 1))
        patches = filled.reshape(B, C, n_patches, self.patch_len)
        tokens = self.proj(patches)
        tokens = tokens + self.channel_embed.view(1, C, 1, self.d_model)

        # Token time = mean of the real (non-padded) sample times in each patch.
        weight = time_valid.reshape(1, n_patches, self.patch_len)
        patch_times = times.reshape(B, n_patches, self.patch_len)
        token_time = (patch_times * weight).sum(-1) / weight.sum(-1)
        time_feat = self.time_features(token_time.to(torch.float32))
        tokens = tokens + time_feat.unsqueeze(1)

        # Token valid iff >=1 observed sample in its patch (pad is already False).
        token_valid = value_mask.reshape(
            B, C, n_patches, self.patch_len
        ).any(dim=3)

        N = C * n_patches
        tokens = tokens.reshape(B, N, self.d_model)
        token_mask = token_valid.reshape(B, N)

        channel_id = (
            torch.arange(C, device=values.device)
            .view(C, 1)
            .expand(C, n_patches)
            .reshape(N)
            .unsqueeze(0)
            .expand(B, N)
            .contiguous()
        )
        time_s = (
            token_time.unsqueeze(1).expand(B, C, n_patches).reshape(B, N).contiguous()
        )
        coord = torch.full(
            (B, N), float("nan"), dtype=torch.float32, device=values.device
        )
        metadata = TokenMetadata(
            modality=self.modality,
            channel_id=channel_id,
            time_s=time_s,
            coord=coord,
        )
        return tokens, token_mask, metadata


class ProfileTokenizer(nn.Module):
    """Tokenize a radial profile in time, one token per (time frame, radial-patch).

    Args:
        d_model: token embedding dimension ``D``.
        radial_patch: radial points per patch. The radial axis is split into
            ``ceil(n_radial_points / radial_patch)`` patches; a trailing partial
            patch is padded with missing-marked radial points.
        n_radial_points: number of radial points ``R`` per profile (fixed; the
            radial grid does not vary across calls).
        modality: modality name recorded in the emitted metadata.

    ``forward(values, value_mask, times, coords)`` takes:

    * ``values`` -- ``[B, R, T]`` float32 profile samples (may hold non-finite
      placeholders at masked positions).
    * ``value_mask`` -- ``[B, R, T]`` bool, ``True`` where a sample is observed.
    * ``times`` -- ``[B, T]`` float64 physical frame times in seconds.
    * ``coords`` -- ``[B, R]`` float32 radial positions of the radial grid.

    and returns ``(tokens [B, N, D] float32, token_mask [B, N] bool,
    TokenMetadata)`` with ``N = T * ceil(R / radial_patch)``. Tokens are ordered
    time-major: token ``t * n_patches + p`` is time frame ``t``'s ``p``-th radial
    patch. Each token's ``time_s`` is its frame time, and ``coord`` carries the
    mean radial position of the real (non-padded) points in its radial patch.
    """

    def __init__(
        self,
        d_model: int,
        radial_patch: int,
        n_radial_points: int,
        *,
        modality: str = "profile",
    ) -> None:
        super().__init__()
        if d_model < 1 or radial_patch < 1 or n_radial_points < 1:
            raise ValueError(
                "d_model, radial_patch, n_radial_points must all be >= 1"
            )
        self.d_model = d_model
        self.radial_patch = radial_patch
        self.n_radial_points = n_radial_points
        self.modality = modality
        self.n_patches = _ceil_div(n_radial_points, radial_patch)
        # Padded radial width is fixed at construction (R never varies), so the
        # per-radial-point fill spans the padded grid directly.
        self.radial_padded = self.n_patches * radial_patch

        self.proj = nn.Linear(radial_patch, d_model)
        self.patch_embed = nn.Parameter(torch.empty(self.n_patches, d_model))
        self.missing_fill = nn.Parameter(torch.empty(self.radial_padded))
        self.time_features = _FourierFeatures(_PROFILE_N_FREQS, d_model)
        self.coord_features = _FourierFeatures(_PROFILE_N_FREQS, d_model)

        nn.init.normal_(self.patch_embed, std=_EMBED_INIT_STD)
        nn.init.normal_(self.missing_fill, std=_EMBED_INIT_STD)

        # Per-radial-point weight (1 for real points, 0 for pad) used to average
        # each patch's radial center without contamination from padding.
        radial_valid = torch.zeros(self.radial_padded, dtype=torch.float32)
        radial_valid[:n_radial_points] = 1.0
        self.register_buffer("radial_valid", radial_valid)

    def forward(
        self,
        values: Tensor,
        value_mask: Tensor,
        times: Tensor,
        coords: Tensor,
    ) -> tuple[Tensor, Tensor, TokenMetadata]:
        if values.ndim != 3:
            raise ValueError(f"values must be [B, R, T], got {tuple(values.shape)}")
        B, R, T = values.shape
        if R != self.n_radial_points:
            raise ValueError(
                f"values has {R} radial points, tokenizer expects "
                f"{self.n_radial_points}"
            )
        if values.dtype != torch.float32:
            raise ValueError(f"values must be float32, got {values.dtype}")
        if value_mask.shape != values.shape or value_mask.dtype != torch.bool:
            raise ValueError("value_mask must be a bool tensor matching values")
        if tuple(times.shape) != (B, T) or times.dtype != torch.float64:
            raise ValueError(f"times must be float64 [B, T]=({B}, {T})")
        if tuple(coords.shape) != (B, R) or coords.dtype != torch.float32:
            raise ValueError(f"coords must be float32 [B, R]=({B}, {R})")

        pad = self.radial_padded - R
        safe = torch.nan_to_num(values)
        if pad:
            safe = torch.cat([safe, safe.new_zeros(B, pad, T)], dim=1)
            value_mask = torch.cat(
                [value_mask, value_mask.new_zeros(B, pad, T)], dim=1
            )
            coords = torch.cat([coords, coords.new_zeros(B, pad)], dim=1)

        # Masked samples -> learned per-radial-point fill, before projection.
        filled = torch.where(
            value_mask, safe, self.missing_fill.view(1, self.radial_padded, 1)
        )
        # Group the radial axis into patches, then move to (B, T, n_patches, rp)
        # so the projection consumes one radial patch at one time frame.
        patches = filled.reshape(
            B, self.n_patches, self.radial_patch, T
        ).permute(0, 3, 1, 2)
        tokens = self.proj(patches)
        tokens = tokens + self.patch_embed.view(1, 1, self.n_patches, self.d_model)

        # Fourier(time): one token per frame, so the frame time enters directly.
        time_feat = self.time_features(times.to(torch.float32))
        tokens = tokens + time_feat.unsqueeze(2)

        # Patch-center radial coord = mean of real radial positions in the patch.
        weight = self.radial_valid.reshape(1, self.n_patches, self.radial_patch)
        patch_coords = coords.reshape(B, self.n_patches, self.radial_patch)
        coord_center = (patch_coords * weight).sum(-1) / weight.sum(-1)
        coord_feat = self.coord_features(coord_center)
        tokens = tokens + coord_feat.unsqueeze(1)

        # Token valid iff >=1 observed sample in its radial patch (pad is False).
        token_valid = value_mask.reshape(
            B, self.n_patches, self.radial_patch, T
        ).any(dim=2).permute(0, 2, 1)

        N = T * self.n_patches
        tokens = tokens.reshape(B, N, self.d_model)
        token_mask = token_valid.reshape(B, N)

        channel_id = (
            torch.arange(self.n_patches, device=values.device)
            .view(1, self.n_patches)
            .expand(T, self.n_patches)
            .reshape(N)
            .unsqueeze(0)
            .expand(B, N)
            .contiguous()
        )
        time_s = (
            times.unsqueeze(-1)
            .expand(B, T, self.n_patches)
            .reshape(B, N)
            .contiguous()
        )
        coord = (
            coord_center.unsqueeze(1)
            .expand(B, T, self.n_patches)
            .reshape(B, N)
            .contiguous()
        )
        metadata = TokenMetadata(
            modality=self.modality,
            channel_id=channel_id,
            time_s=time_s,
            coord=coord,
        )
        return tokens, token_mask, metadata
