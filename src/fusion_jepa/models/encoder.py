"""Shared context encoder with a state-token bottleneck (M2, Task 2.3).

This encoder is the single backbone shared by the raw baseline and the JEPA --
the matched-comparison invariant requires both to see identical representational
capacity, so it lives here once and is reused.

It consumes the flat token sequence a modality tokenizer emits
(``tokens [B, N, D]`` plus a boolean ``token_mask [B, N]`` with ``True == valid``)
and compresses it into a *fixed-size* state bottleneck: ``S`` learned state
tokens are prepended per batch, the whole ``[state; inputs]`` sequence is run
through ``n_blocks`` pre-norm self-attention blocks, and only the ``S`` state
positions are returned as ``z [B, S, D]``. Because ``S`` is fixed and the input
count ``N`` is not, downstream predictor/decoder stages always face the same
latent shape regardless of how many context tokens a shot contributed.

Design notes
------------
**Mask polarity.** The tokenizer's ``token_mask`` uses ``True == valid`` (the
natural "this token was observed" convention). ``torch.nn.MultiheadAttention``'s
``key_padding_mask`` uses the opposite: ``True == ignore this key``. We localize
the single negation in :class:`ContextEncoder`, which builds the combined
``key_padding_mask`` once as ``[state_pad; ~token_mask]`` and threads it through
every block. :class:`MaskedBlock` therefore speaks the torch-native convention
directly (``True == PAD/ignore``) and stays a thin, reusable primitive with no
polarity of its own to reason about.

**All-masked safety.** A softmax over an all-``-inf`` score row returns NaN, so a
query that can attend to *no* valid key would poison the output. The ``S`` state
tokens are *never* padded (their ``key_padding_mask`` entries are always
``False``), so every query -- state or input -- always has at least ``S`` valid
keys to attend to. This structurally prevents the all-``-inf`` trap even when a
sample's input tokens are entirely invalid; no explicit NaN guard is needed, and
:func:`test_fully_masked_sample_is_finite` locks the guarantee. Input token
values are assumed finite (the tokenizers guarantee finite tokens even for
invalid positions), so masked keys contribute an exact ``0 * value == 0`` and
never introduce NaN.

**Masked-input invariance.** An invalid input token is masked as a *key* in every
block, so it receives exactly-zero attention weight and its value never reaches
the state slice -- at any depth. Its own evolving representation is likewise never
read back into the state tokens (it stays a padded key throughout). The returned
``z`` is therefore bit-identical under any change to invalid input token values,
which :func:`test_masked_input_tokens_do_not_affect_state_latents` locks.

**Determinism.** ``dropout`` defaults to ``0.0``, so the forward pass carries no
randomness in either train or eval mode; all randomness lives in parameter init.
If a non-zero dropout is ever configured, call ``.eval()`` for deterministic
inference.

**Throughput fallback (not implemented, YAGNI).** If full self-attention over
``[state; inputs]`` (cost ``O((S + N)^2)``) becomes a throughput bottleneck at
large ``N``, the documented alternative is a Perceiver-style read: let the ``S``
state tokens *cross-attend* into the inputs (cost ``O(S * N)``) instead of
concatenating. It is intentionally not built here until a measured need appears.
"""

import torch
from torch import Tensor, nn

# Std for the learned state-token initialization, matching the tokenizer
# embedding convention so state tokens start at the same scale as input tokens.
_STATE_INIT_STD = 0.02


class MaskedBlock(nn.Module):
    """Pre-norm Transformer block with key-padding-mask support.

    Structure mirrors ``e2e/backbone.BackboneBlock`` (LayerNorm -> attention ->
    residual, LayerNorm -> MLP -> residual) but adds a ``key_padding_mask`` on
    the self-attention so invalid/padded keys can be ignored -- the reason the
    old block cannot be reused verbatim.

    Args:
        d_model: token embedding dimension.
        n_heads: number of attention heads (must divide ``d_model``).
        mlp_ratio: hidden-dim multiplier for the feed-forward MLP.
        dropout: dropout inside attention and MLP; ``0.0`` (default) keeps the
            forward pass deterministic in both train and eval mode.

    ``forward(x, key_padding_mask)`` takes ``x`` of shape ``[B, N, D]`` and an
    optional bool ``key_padding_mask`` of shape ``[B, N]`` using the
    ``torch.nn.MultiheadAttention`` convention: ``True`` marks a key to *ignore*
    (PAD/invalid). Polarity conversion from the tokenizer's ``True == valid``
    mask is the caller's responsibility (see :class:`ContextEncoder`).
    """

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

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Run ``x [B, N, D]`` through pre-norm attention + MLP with residuals."""
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """Encode context tokens into a fixed-size learned state-token bottleneck.

    ``S`` learned state tokens are prepended to the ``N`` input tokens per batch;
    the combined sequence flows through ``n_blocks`` :class:`MaskedBlock` layers
    under a shared key-padding mask, and only the ``S`` state positions are
    returned. The output shape ``[B, S, D]`` is independent of ``N``.

    Args:
        d_model: token / state embedding dimension.
        n_heads: attention heads per block (must divide ``d_model``).
        n_blocks: number of stacked :class:`MaskedBlock` layers.
        n_state_tokens: number of learned state tokens ``S`` (the bottleneck
            width).
        mlp_ratio: MLP hidden-dim multiplier passed to each block.
        dropout: dropout passed to each block; ``0.0`` (default) keeps forward
            deterministic.

    ``forward(tokens, token_mask)`` takes ``tokens [B, N, D]`` (float32) and
    ``token_mask [B, N]`` (bool, ``True == valid``) and returns
    ``(z [B, S, D], z_mask [B, S])`` where ``z_mask`` is all ``True`` -- state
    tokens are always present and never padded.
    """

    def __init__(
        self,
        d_model: int = 320,
        n_heads: int = 8,
        n_blocks: int = 6,
        n_state_tokens: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or n_blocks < 1 or n_state_tokens < 1:
            raise ValueError(
                "d_model, n_heads, n_blocks, n_state_tokens must all be >= 1"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_state_tokens = n_state_tokens

        self.state_tokens = nn.Parameter(torch.empty(n_state_tokens, d_model))
        nn.init.normal_(self.state_tokens, std=_STATE_INIT_STD)

        self.blocks = nn.ModuleList(
            [
                MaskedBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: Tensor, token_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, N, D], got {tuple(tokens.shape)}")
        B, N, D = tokens.shape
        if D != self.d_model:
            raise ValueError(
                f"tokens last dim {D} must equal d_model={self.d_model}"
            )
        if tuple(token_mask.shape) != (B, N) or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be a bool tensor of shape [B, N]")

        state = self.state_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([state, tokens], dim=1)

        # Build the combined key-padding mask once (torch convention: True ==
        # ignore). State tokens are NEVER padded -> always attendable, which is
        # what keeps a fully-masked-input sample off the all-`-inf` softmax path.
        state_pad = torch.zeros(
            B, self.n_state_tokens, dtype=torch.bool, device=tokens.device
        )
        key_padding_mask = torch.cat([state_pad, ~token_mask], dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        z = self.final_norm(x[:, : self.n_state_tokens, :])
        z_mask = torch.ones(
            B, self.n_state_tokens, dtype=torch.bool, device=tokens.device
        )
        return z, z_mask
