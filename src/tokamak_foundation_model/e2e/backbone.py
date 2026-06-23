"""Shared Transformer backbone with rollout-step conditioning.

Pre-norm Transformer encoder (LayerNorm → attention → residual, LayerNorm →
MLP → residual), with a Fourier-feature MLP encoding of ``(step_index,
time_offset_s)`` broadcast-added to all tokens before the first block.
See ``ResearchPlan.MD`` §3.4 and §5.6.
"""

import math
from typing import List, Optional, Union, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_ckpt


def _fourier_features(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Map ``x`` of shape ``(B,)`` to ``(B, 2*n_freq)`` sin/cos features."""
    phase = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class StepConditioning(nn.Module):
    """Fourier features of ``(step_index, time_offset_s)`` → ``d_model`` MLP.

    ``step_freqs`` cover typical 0–80-step rollouts; ``time_freqs`` cover
    absolute offsets on the ~0–10 s shot timescale. Frequencies are fixed
    buffers; only the 2-layer MLP is learned.
    """

    def __init__(
        self, d_model: int, n_freq: int = 16, hidden: Optional[int] = None
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = 4 * d_model
        step_freqs = 2 * math.pi * torch.logspace(-3, 0, n_freq)
        time_freqs = 2 * math.pi * torch.logspace(-1, 2, n_freq)
        self.register_buffer("step_freqs", step_freqs)
        self.register_buffer("time_freqs", time_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(4 * n_freq, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Default PyTorch init on the output layer gives embed std ≈ 0.1,
        # too weak to visibly condition the token stream at init (cos_sim
        # between step=0 and step=40 stays > 0.98 through 2 blocks). Scale
        # up so step embed has per-element std ≈ 0.5 at init — same order
        # as post-tokenizer tokens — which is the level §5.6 requires.
        nn.init.normal_(self.mlp[-1].weight, std=0.3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, step_index: torch.Tensor, time_offset_s: torch.Tensor
    ) -> torch.Tensor:
        """Return a per-batch conditioning vector of shape ``(B, d_model)``."""
        step_feats = _fourier_features(
            step_index.float(), cast(torch.Tensor, self.step_freqs)
        )
        time_feats = _fourier_features(
            time_offset_s.float(), cast(torch.Tensor, self.time_freqs)
        )
        return self.mlp(torch.cat([step_feats, time_feats], dim=-1))


class BackboneBlock(nn.Module):
    """Pre-norm Transformer encoder block: norm→attn→residual, norm→MLP→residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SharedBackbone(nn.Module):
    """Stack of :class:`BackboneBlock` with step conditioning.

    Parameters
    ----------
    d_model
        Token embedding dimension (``256`` in the full config, smaller for
        tests).
    n_heads
        Number of attention heads.
    n_layers
        Number of stacked blocks (``8`` in the full config).
    mlp_ratio
        MLP hidden-dim ratio (``4.0``).
    dropout
        Dropout applied inside attention and MLP.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.grad_checkpoint = grad_checkpoint
        self.step_cond = StepConditioning(d_model)
        self.blocks = nn.ModuleList(
            [
                BackboneBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        step_index: torch.Tensor,
        time_offset_s: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run tokens through the stack.

        Parameters
        ----------
        tokens
            Input of shape ``(batch, n_tokens, d_model)``.
        step_index
            Integer-valued tensor of shape ``(batch,)``.
        time_offset_s
            Float tensor of shape ``(batch,)`` with absolute time in seconds.
        return_intermediates
            If ``True``, return a list of length ``n_layers + 2`` containing
            the post-conditioning input, each block's output, and the
            final-norm output (for §5.6 progressive-mixing tests).
        """
        step_embed = self.step_cond(step_index, time_offset_s).unsqueeze(1)
        x = tokens + step_embed
        # Per-block gradient checkpointing: trades ~30% step-time for
        # ~sqrt(n_layers) reduction in activation memory. Required at
        # d_model=1024+ where activations no longer fit per-GCD VRAM
        # without sharding. Skipped when return_intermediates (debug path)
        # or when not training (no grad needed anyway).
        use_ckpt = self.grad_checkpoint and self.training and not return_intermediates
        if return_intermediates:
            intermediates: List[torch.Tensor] = [x]
            for block in self.blocks:
                x = block(x)
                intermediates.append(x)
            intermediates.append(self.final_norm(x))
            return intermediates
        for block in self.blocks:
            if use_ckpt:
                x = torch_ckpt.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.final_norm(x)