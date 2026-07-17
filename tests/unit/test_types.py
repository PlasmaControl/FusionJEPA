"""Tests for token/loss type contracts and the M2 synthetic batch fixture."""

import pytest
import torch

from fusion_jepa.data.batch import validate_batch
from fusion_jepa.models.types import TokenMetadata, merge_token_sets
from fusion_jepa.objectives.base import LossOutput
from tests.fixtures.synthetic import make_synthetic_fusion_batch


def _token_set(
    modality: list[str] | str,
    *,
    B: int,
    N: int,
    D: int,
    channel_base: int,
    time_base: float,
) -> tuple[torch.Tensor, torch.Tensor, TokenMetadata]:
    """Build a deterministic (tokens, mask, metadata) triple for merge tests."""
    tokens = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 0] = False
    channel_id = (
        channel_base + torch.arange(N, dtype=torch.long)
    ).unsqueeze(0).expand(B, N).contiguous()
    time_s = (
        time_base + torch.arange(N, dtype=torch.float64)
    ).unsqueeze(0).expand(B, N).contiguous()
    coord = torch.full((B, N), float("nan"), dtype=torch.float32)
    metadata = TokenMetadata(
        modality=modality,
        channel_id=channel_id,
        time_s=time_s,
        coord=coord,
    )
    return tokens, mask, metadata


def test_merge_token_sets_concatenates_masks_and_metadata():
    B, D = 2, 3
    N1, N2 = 4, 5
    first = _token_set(
        "slow_ts", B=B, N=N1, D=D, channel_base=0, time_base=0.0
    )
    second = _token_set(
        ["profile"] * N2, B=B, N=N2, D=D, channel_base=100, time_base=10.0
    )

    tokens, mask, meta = merge_token_sets([first, second])

    assert tokens.shape == (B, N1 + N2, D)
    assert mask.shape == (B, N1 + N2)
    assert mask.dtype == torch.bool
    # Token payloads and masks are concatenated in order along N.
    assert torch.equal(tokens[:, :N1], first[0])
    assert torch.equal(tokens[:, N1:], second[0])
    assert torch.equal(mask[:, :N1], first[1])
    assert torch.equal(mask[:, N1:], second[1])

    # Metadata tensors concatenate along N and keep their dtypes.
    assert meta.channel_id.shape == (B, N1 + N2)
    assert meta.channel_id.dtype == torch.long
    assert meta.time_s.shape == (B, N1 + N2)
    assert meta.time_s.dtype == torch.float64
    assert meta.coord.shape == (B, N1 + N2)
    assert meta.coord.dtype == torch.float32
    assert torch.equal(meta.channel_id[:, :N1], first[2].channel_id)
    assert torch.equal(meta.channel_id[:, N1:], second[2].channel_id)

    # modality is a per-token list of length N aligned to the token axis; a
    # single-string input set is broadcast across its tokens.
    assert meta.modality == ["slow_ts"] * N1 + ["profile"] * N2

    # Disagreeing embedding dim D must raise rather than silently truncate.
    mismatched = _token_set(
        "profile", B=B, N=3, D=D + 1, channel_base=0, time_base=0.0
    )
    with pytest.raises(ValueError, match="[Dd]"):
        merge_token_sets([first, mismatched])


def test_loss_output_terms_are_tensors():
    total = torch.tensor(1.5)
    terms = {"raw_mse": torch.tensor(1.0), "reg": torch.tensor(0.5)}
    diagnostics = {"grad_norm": 0.25, "active_fraction": 1.0}

    out = LossOutput(total=total, terms=terms, diagnostics=diagnostics)

    assert isinstance(out.total, torch.Tensor)
    assert out.total.ndim == 0
    assert set(out.terms) == {"raw_mse", "reg"}
    for value in out.terms.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
    for value in out.diagnostics.values():
        assert isinstance(value, float)


def test_synthetic_batch_passes_validator():
    batch = make_synthetic_fusion_batch(B=3, missing_fraction=0.3, seed=7)
    split_lookup = {
        shot_id: batch.metadata["split"] for shot_id in batch.shot_id
    }

    # strict=True raises on any violation; an empty list means it passed.
    assert validate_batch(batch, split_lookup=split_lookup, strict=True) == []

    # Same arguments reproduce an identical batch (seeded via derive_seed).
    again = make_synthetic_fusion_batch(B=3, missing_fraction=0.3, seed=7)
    for signal in batch.context:
        assert torch.equal(
            batch.context_mask[signal], again.context_mask[signal]
        )
        torch.testing.assert_close(
            batch.context[signal], again.context[signal], equal_nan=True
        )
    assert torch.equal(batch.context_times, again.context_times)
    assert batch.shot_id == again.shot_id
