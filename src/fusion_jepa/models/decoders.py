"""Shared query-conditioned decoder for Fusion-JEPA (M2, Task 2.6).

The raw baseline reads its predictions out of the predicted future latents with
*one* decoder reused for every modality -- there are no per-modality readout
modules anywhere. This is what makes the raw baseline a *matched* comparison to
the JEPA by construction: both stacks share the same tokenizers, encoder, and
predictor, and the raw baseline adds exactly one shared readout head rather than
a fan-out of modality-specific decoders whose capacity would be impossible to
match.

How one shared head serves every modality
------------------------------------------
A modality's target is a set of scalar readings, each identified by *which*
channel it belongs to, *where* it sits spatially, and *when* it occurs. The
decoder turns each such reading into a **query token** and reads its scalar value
by cross-attending that query onto the ``S`` predicted state latents::

    query(channel c, coord x, horizon t) =
        channel_embedding[c] + Fourier(coord x) + Fourier(horizon t)

Only the ``channel_embedding`` is modality-specific, and it is supplied *from
outside* (see :meth:`QueryConditionedDecoder.build_target_queries`) -- the
codebase reuses each tokenizer's own channel embedding, so no new modality-keyed
parameter is introduced and the decoder itself stays modality-agnostic. The
Fourier(coord)/Fourier(horizon) bands and the whole readout stack are shared.

Readout mechanics
-----------------
Queries are read out *independently*: each query cross-attends onto the state
latents (as keys/values) but never onto other queries. Two consequences follow
directly, and :func:`test_masked_queries_do_not_affect_valid_outputs` locks the
first:

* a masked query cannot influence any valid query's scalar (there is no
  query-to-query path at any depth), and
* masked queries carry no observed target to fit, so their outputs are zeroed via
  ``query_mask`` -- purely cosmetic given independence, but it keeps the returned
  tensor free of predictions at positions the batch never observed.

The ``S`` predicted state latents are always present (the encoder's state
bottleneck is never padded and the predictor emits all ``S``), so the
cross-attention needs no key-padding mask and no all-``-inf`` softmax row can
arise.

Shapes and time base
--------------------
``forward(z, queries, query_mask)`` consumes ``z [B, H, S, d_latent]`` -- the
predictor's ``H`` predicted-latent slices (``H == K``; the raw world model uses
``K == 1``, one latent set for the whole target window) -- together with
``queries [B, H, Q, d_model]`` and ``query_mask [B, H, Q]`` (``True == the
query's target was observed``), and returns one scalar per query, ``[B, H, Q]``.
Query horizons are seconds relative to the context end, matching the predictor's
time base; :meth:`build_target_queries` performs the absolute->relative
conversion from a :class:`~fusion_jepa.data.batch.FusionBatch`.

Determinism
-----------
``dropout`` defaults to ``0.0``, so forward carries no randomness in either mode;
all randomness lives in parameter init. ``float64`` times/coords are cast to
``float32`` only inside the Fourier computation.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.models.tokenizers import _FourierFeatures

# Fourier band counts for the query coordinate / horizon embeddings.
_COORD_N_FREQS = 6
_HORIZON_N_FREQS = 6


class _CrossAttentionBlock(nn.Module):
    """Pre-norm block: queries cross-attend onto fixed latents, then an MLP.

    Mirrors the pre-norm structure of
    :class:`~fusion_jepa.models.encoder.MaskedBlock` but attends the ``Q`` query
    tokens onto separate ``S`` key/value latents (rather than self-attending), so
    a query only ever reads the state bottleneck and never another query. No
    key-padding mask is taken because the state latents are always valid.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
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

    def forward(self, queries: Tensor, latents: Tensor) -> Tensor:
        """Cross-attend ``queries [M, Q, D]`` onto ``latents [M, S, D]``."""
        q = self.norm_q(queries)
        kv = self.norm_kv(latents)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        return queries


class QueryConditionedDecoder(nn.Module):
    """One shared query-conditioned readout head reused for every modality.

    Args:
        d_latent: width of the predicted state latents ``z`` (the shared
            encoder/predictor state width).
        d_model: internal readout width; also the width of the incoming query
            vectors (``Dq == d_model``, since a query is a sum of ``d_model``
            embeddings -- see the module docstring).
        n_heads: attention heads per block (must divide ``d_model``).
        n_blocks: number of stacked cross-attention blocks.
        n_coord_freqs: Fourier bands for the spatial-coordinate query feature.
        n_horizon_freqs: Fourier bands for the horizon-time query feature.
        mlp_ratio: MLP hidden-dim multiplier passed to each block.
        dropout: dropout passed to each block; ``0.0`` (default) keeps forward
            deterministic.

    ``forward`` returns ``[B, H, Q]`` float32 scalar predictions; see the module
    docstring for the query construction, independence guarantee, and time base.
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        n_heads: int,
        n_blocks: int = 2,
        *,
        n_coord_freqs: int = _COORD_N_FREQS,
        n_horizon_freqs: int = _HORIZON_N_FREQS,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_latent < 1 or d_model < 1 or n_heads < 1 or n_blocks < 1:
            raise ValueError(
                "d_latent, d_model, n_heads, n_blocks must all be >= 1"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_latent = d_latent
        self.d_model = d_model

        # Project the predicted latents to the readout width so queries can
        # attend onto them as keys/values.
        self.latent_proj = nn.Linear(d_latent, d_model)
        # Shared (non-modality-keyed) query coordinate / horizon features.
        self.coord_features = _FourierFeatures(n_coord_freqs, d_model)
        self.horizon_features = _FourierFeatures(n_horizon_freqs, d_model)

        self.blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(
        self, z: Tensor, queries: Tensor, query_mask: Tensor
    ) -> Tensor:
        """Read one scalar per query by cross-attending onto the latents.

        Args:
            z: ``[B, H, S, d_latent]`` float32 predicted state latents; ``H`` is
                the predictor horizon count (``K``).
            queries: ``[B, H, Q, d_model]`` float32 query tokens.
            query_mask: ``[B, H, Q]`` bool, ``True`` where the query's target was
                observed. Masked outputs are zeroed.

        Returns:
            ``[B, H, Q]`` float32 scalar predictions.
        """
        if z.ndim != 4:
            raise ValueError(f"z must be [B, H, S, d_latent], got {tuple(z.shape)}")
        B, H, S, Dl = z.shape
        if Dl != self.d_latent:
            raise ValueError(
                f"z last dim {Dl} must equal d_latent={self.d_latent}"
            )
        if queries.ndim != 4 or queries.shape[:2] != (B, H):
            raise ValueError(
                f"queries must be [B, H, Q, d_model], got {tuple(queries.shape)}"
            )
        Q = queries.shape[2]
        if queries.shape[-1] != self.d_model:
            raise ValueError(
                f"queries last dim {queries.shape[-1]} must equal "
                f"d_model={self.d_model}"
            )
        if tuple(query_mask.shape) != (B, H, Q) or query_mask.dtype != torch.bool:
            raise ValueError(
                f"query_mask must be a bool tensor of shape [B, H, Q]="
                f"({B}, {H}, {Q})"
            )

        # Fold the (B, H) horizon slices into the batch axis: each slice is an
        # independent cross-attention (its own S latents, its own Q queries).
        latents = self.latent_proj(z).reshape(B * H, S, self.d_model)
        seq = queries.reshape(B * H, Q, self.d_model)
        for block in self.blocks:
            seq = block(seq, latents)

        out = self.out_proj(self.out_norm(seq)).squeeze(-1)  # [B*H, Q]
        out = out.reshape(B, H, Q)
        # Zero predictions the batch never observed (independence makes this
        # cosmetic for the valid entries, which are untouched).
        return out.masked_fill(~query_mask, 0.0)

    def build_target_queries(
        self,
        batch: FusionBatch,
        registry_embeddings: Mapping[str, Tensor],
    ) -> dict[str, tuple[Tensor, Tensor, tuple[int, int]]]:
        """Assemble per-modality target queries from a :class:`FusionBatch`.

        This is a *pure* builder: it reads only ``batch`` and the supplied
        per-modality channel embeddings and returns freshly built tensors,
        mutating nothing. (It is a method, not a free function, only so it can
        reuse this decoder's shared coord/horizon Fourier bands -- which are the
        same bands the readout is trained with.)

        For a target modality with values ``[B, C, T_tgt]`` it builds one query
        per ``(channel, target frame)``::

            query[b, c, t] = channel_embedding[c]
                             + Fourier(coord_c)          (0 for scalar signals)
                             + Fourier(target_time[b, t] - context_end[b])

        laid out channel-major (``q = c * T_tgt + t``) so the decoder's
        ``[B, 1, Q]`` output reshapes straight back to ``[B, C, T_tgt]``.

        Args:
            batch: the source batch; ``context_times``/``target_times`` are
                absolute float64 seconds. The context end (max valid context
                time per sample) is subtracted to reach the predictor's
                context-end-relative horizon base.
            registry_embeddings: ``{modality: channel_embedding [C, d_model]}``;
                the channel count must match the modality's target channel count.

        Returns:
            ``{modality: (queries [B, Q, d_model], query_mask [B, Q],
            (C, T_tgt))}`` where ``query_mask`` is the flattened target mask and
            ``(C, T_tgt)`` records the shape to reshape the decoder output to.
        """
        context_times = batch.context_times.to(torch.float64)
        target_times = batch.target_times.to(torch.float64)
        # Context end = latest context time per sample (times are monotone, so
        # max == the final frame; max is robust regardless).
        context_end = context_times.max(dim=1).values  # [B]
        horizon_rel = target_times - context_end.unsqueeze(1)  # [B, T_tgt]
        horizon_feat = self.horizon_features(horizon_rel.to(torch.float32))
        # [B, T_tgt, d_model]

        queries_by_modality: dict[str, tuple[Tensor, Tensor, tuple[int, int]]] = {}
        for modality, values in batch.target.items():
            if values.ndim != 3:
                raise ValueError(
                    f"target {modality!r} must be [B, C, T_tgt], got "
                    f"{tuple(values.shape)}"
                )
            B, C, T_tgt = values.shape
            channel_embed = registry_embeddings[modality]
            if channel_embed.shape != (C, self.d_model):
                raise ValueError(
                    f"channel embedding for {modality!r} must be [C, d_model]="
                    f"({C}, {self.d_model}), got {tuple(channel_embed.shape)}"
                )

            # Scalar targets carry no spatial position -> coord 0 (a shared
            # bias). A modality with real coordinates would feed them here.
            coord = torch.zeros(B, C, dtype=torch.float32, device=values.device)
            coord_feat = self.coord_features(coord)  # [B, C, d_model]

            queries = (
                channel_embed.view(1, C, 1, self.d_model)
                + coord_feat.view(B, C, 1, self.d_model)
                + horizon_feat.view(B, 1, T_tgt, self.d_model)
            )  # [B, C, T_tgt, d_model]
            queries = queries.reshape(B, C * T_tgt, self.d_model)
            query_mask = batch.target_mask[modality].reshape(B, C * T_tgt)
            queries_by_modality[modality] = (queries, query_mask, (C, T_tgt))

        return queries_by_modality
