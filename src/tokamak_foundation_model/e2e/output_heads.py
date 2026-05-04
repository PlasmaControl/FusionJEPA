"""Per-modality output heads.

Each head is an approximate inverse of its sibling tokenizer. They fire only
to compute the training loss against ground-truth raw signals — during
autoregressive rollout the backbone's token output is fed directly to the
next step, bypassing the heads (``ResearchPlan.MD`` §3.5, §3.6, §5.7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowTimeSeriesHead(nn.Module):
    """Linear head reconstructing a slow time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``5`` at 100 Hz).

    Notes
    -----
    Approximate inverse of :class:`SlowTimeSeriesTokenizer`: a single shared
    ``Linear(d_model, window_samples)`` unprojects each per-channel token back
    to raw signal samples.
    """

    def __init__(
        self, d_model: int, n_channels: int, window_samples: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.proj = nn.Linear(d_model, window_samples)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels, d_model)`` — per-channel tokens from the
            backbone for this modality.

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        return self.proj(tokens)


class FastTimeSeriesHead(nn.Module):
    """ConvTranspose1d head reconstructing a fast time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``500`` at 10 kHz).
    patch_size
        Patch length matching the sibling tokenizer (``50`` by default). Must
        divide ``window_samples``.

    Notes
    -----
    Approximate inverse of :class:`FastTimeSeriesTokenizer`. Channels are
    reshaped into the batch axis so a single shared
    ``ConvTranspose1d(in=d_model, out=1, k=s=patch_size)`` unpacks each
    per-channel patch sequence back to raw samples.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        window_samples: int,
        patch_size: int = 50,
    ) -> None:
        super().__init__()
        if window_samples % patch_size != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be a multiple of "
                f"patch_size ({patch_size})"
            )
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size

        self.deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels * n_patches, d_model)`` in channel-major
            order (matching :class:`FastTimeSeriesTokenizer`).

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        batch = tokens.shape[0]
        t = tokens.reshape(batch, self.n_channels, self.n_patches, self.d_model)
        t = t.reshape(batch * self.n_channels, self.n_patches, self.d_model)
        t = t.transpose(1, 2)  # (B*C, d_model, n_patches)
        out = self.deconv(t)  # (B*C, 1, window_samples)
        return out.reshape(batch, self.n_channels, self.window_samples)


class VideoOutputHead(nn.Module):
    """Per-patch reconstruction head — exact inverse of the tube-patch
    :class:`VideoTokenizer`.

    Tokens arrive as ``(B, n_tokens, d_model)`` where
    ``n_tokens = (n_frames / T_p) * (H / H_p) * (W / W_p)``. They are
    reshaped to a 5-D feature volume ``(B, d_model, n_t, n_h, n_w)`` and
    passed through a single ``ConvTranspose3d`` whose kernel and stride
    both equal the patch shape. Each token thus reconstructs its own
    ``(n_channels, T_p, H_p, W_p)`` region without any global mixing.
    Output shape ``(B, n_frames, n_channels, H, W)`` matches the input
    layout permuted from ``(C, T, H, W)`` to ``(T, C, H, W)``.

    Parameters
    ----------
    n_channels : int, optional
        Number of optical filters reconstructed. Default ``7``.
    n_frames : int, optional
        Number of time samples per output window. Default ``3``.
    patch_size : tuple of int, optional
        ``(T_p, H_p, W_p)`` — must match the tokenizer.
        Default ``(3, 12, 12)``.
    d_model : int, optional
        Backbone token dimension. Default ``256``.
    spatial_size : tuple of int, optional
        Output spatial size ``(H, W)``. Default ``(120, 360)``.

    Notes
    -----
    No bilinear upsampling and no MLP. ``ConvTranspose3d`` with
    ``kernel = stride = patch_size`` exactly inverts the tokenizer's
    patch ``Conv3d`` and is the standard ViT/VideoMAE inverse. Param
    count is ``d_model * n_channels * prod(patch_size) + n_channels``,
    e.g. 256 * 7 * 3 * 12 * 12 + 7 ≈ 774 k.
    """

    def __init__(
        self,
        n_channels: int = 7,
        n_frames: int = 3,
        patch_size: tuple[int, int, int] = (3, 12, 12),
        d_model: int = 256,
        spatial_size: tuple[int, int] = (120, 360),
    ) -> None:
        super().__init__()
        T_p, H_p, W_p = (int(p) for p in patch_size)
        H, W = int(spatial_size[0]), int(spatial_size[1])
        if n_frames % T_p:
            raise ValueError(
                f"n_frames={n_frames} must be divisible by patch T_p={T_p}."
            )
        if H % H_p:
            raise ValueError(
                f"spatial H={H} must be divisible by patch H_p={H_p}."
            )
        if W % W_p:
            raise ValueError(
                f"spatial W={W} must be divisible by patch W_p={W_p}."
            )

        self.n_channels = n_channels
        self.n_frames = n_frames
        self.patch_size = (T_p, H_p, W_p)
        self.d_model = d_model
        self.spatial_size = (H, W)
        self.n_h = H // H_p
        self.n_w = W // W_p
        self.n_t = n_frames // T_p

        # Inverse of the tokenizer's patch_embed Conv3d.
        self.patch_unembed = nn.ConvTranspose3d(
            d_model,
            n_channels,
            kernel_size=(T_p, H_p, W_p),
            stride=(T_p, H_p, W_p),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tokens, d_model) -> (B, n_frames, n_channels, H, W)``."""
        B = tokens.shape[0]
        # (B, n_tokens, d_model) -> (B, d_model, n_t, n_h, n_w)
        x = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_t, self.n_h, self.n_w
        )
        out = self.patch_unembed(x)              # (B, n_channels, T, H, W)
        return out.permute(0, 2, 1, 3, 4)        # (B, T, C, H, W)