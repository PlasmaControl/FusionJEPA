"""Token-level type contracts shared across Fusion-JEPA model components.

A *token set* is the trio a modality tokenizer emits for one modality: a token
payload tensor ``[B, N, D]``, a boolean observation mask ``[B, N]`` (True means
observed), and the :class:`TokenMetadata` describing each of the ``N`` tokens.
:func:`merge_token_sets` concatenates several such trios along the token axis so
downstream encoders see a single flat token sequence per sample.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TokenMetadata:
    """Per-token descriptors carried alongside a ``[B, N, D]`` token payload.

    Attributes:
        modality: Either a single modality name shared by every token in the
            set, or a per-token ``list[str]`` of length ``N`` aligned to the
            token axis. :func:`merge_token_sets` always emits the per-token
            ``list[str]`` form.
        channel_id: ``[B, N]`` long tensor naming the source channel per token.
        time_s: ``[B, N]`` float64 tensor of physical token times in seconds.
        coord: ``[B, N]`` float32 tensor of a spatial coordinate per token,
            ``NaN`` for scalar signals that have no spatial position.
    """

    modality: list[str] | str
    channel_id: Tensor
    time_s: Tensor
    coord: Tensor


def _expand_modality(modality: list[str] | str, n: int) -> list[str]:
    """Return a per-token modality list of length ``n``.

    A single string is broadcast to every token; an existing list is validated
    to already span the token axis.
    """
    if isinstance(modality, str):
        return [modality] * n
    if len(modality) != n:
        raise ValueError(
            "TokenMetadata.modality list length "
            f"{len(modality)} must equal token count {n}"
        )
    return list(modality)


def merge_token_sets(
    sets: list[tuple[Tensor, Tensor, TokenMetadata]],
) -> tuple[Tensor, Tensor, TokenMetadata]:
    """Concatenate token payloads, masks, and metadata along the token axis.

    Each element of ``sets`` is a ``(tokens, mask, metadata)`` trio where
    ``tokens`` is ``[B, N_i, D]``, ``mask`` is a bool ``[B, N_i]``, and the
    metadata tensors are ``[B, N_i]``. The returned trio stacks them into
    ``[B, sum(N_i), D]`` / ``[B, sum(N_i)]`` in the given order. The merged
    ``TokenMetadata.modality`` is always a per-token ``list[str]`` of length
    ``sum(N_i)``: a set whose ``modality`` is a single string is broadcast
    across its own tokens.

    Raises:
        ValueError: if ``sets`` is empty or the embedding dimension ``D``
            disagrees across sets.
    """
    if not sets:
        raise ValueError("merge_token_sets requires at least one token set")

    embed_dims = {tokens.shape[-1] for tokens, _, _ in sets}
    if len(embed_dims) != 1:
        raise ValueError(
            f"token embedding dim D must agree across sets, got {sorted(embed_dims)}"
        )

    tokens = torch.cat([token_set[0] for token_set in sets], dim=1)
    mask = torch.cat([token_set[1] for token_set in sets], dim=1)
    channel_id = torch.cat([token_set[2].channel_id for token_set in sets], dim=1)
    time_s = torch.cat([token_set[2].time_s for token_set in sets], dim=1)
    coord = torch.cat([token_set[2].coord for token_set in sets], dim=1)

    modality: list[str] = []
    for token_set in sets:
        modality.extend(
            _expand_modality(token_set[2].modality, token_set[0].shape[1])
        )

    return tokens, mask, TokenMetadata(
        modality=modality,
        channel_id=channel_id,
        time_s=time_s,
        coord=coord,
    )
