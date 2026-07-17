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
    coord_base: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, TokenMetadata]:
    """Build a deterministic (tokens, mask, metadata) triple for merge tests.

    ``coord_base`` None yields NaN coords (scalar signal); a float yields finite
    ramped coords so alignment across a merge boundary can be asserted.
    """
    tokens = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 0] = False
    channel_id = (
        channel_base + torch.arange(N, dtype=torch.long)
    ).unsqueeze(0).expand(B, N).contiguous()
    time_s = (
        time_base + torch.arange(N, dtype=torch.float64)
    ).unsqueeze(0).expand(B, N).contiguous()
    if coord_base is None:
        coord = torch.full((B, N), float("nan"), dtype=torch.float32)
    else:
        coord = (
            coord_base + torch.arange(N, dtype=torch.float32)
        ).unsqueeze(0).expand(B, N).contiguous()
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
        ["profile"] * N2,
        B=B,
        N=N2,
        D=D,
        channel_base=100,
        time_base=10.0,
        coord_base=5.0,
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

    # time_s keeps per-set order and is ordered across the concat boundary.
    assert torch.equal(meta.time_s[:, :N1], first[2].time_s)
    assert torch.equal(meta.time_s[:, N1:], second[2].time_s)
    assert bool((meta.time_s[:, N1] > meta.time_s[:, N1 - 1]).all())

    # coord alignment: first set's NaN scalars, then second set's finite coords.
    assert torch.isnan(meta.coord[:, :N1]).all()
    assert torch.equal(meta.coord[:, N1:], second[2].coord)

    # modality is a per-token list of length N aligned to the token axis; a
    # single-string input set is broadcast across its tokens.
    assert meta.modality == ["slow_ts"] * N1 + ["profile"] * N2

    # Disagreeing embedding dim D must raise rather than silently truncate.
    mismatched = _token_set(
        "profile", B=B, N=3, D=D + 1, channel_base=0, time_base=0.0
    )
    with pytest.raises(ValueError, match="[Dd]"):
        merge_token_sets([first, mismatched])


def test_token_metadata_rejects_wrong_dtypes_and_shapes():
    B, N = 2, 3

    # Wrong dtype on each tensor field, named in the error.
    with pytest.raises(ValueError, match="channel_id"):
        TokenMetadata(
            modality="slow_ts",
            channel_id=torch.zeros(B, N, dtype=torch.float32),
            time_s=torch.zeros(B, N, dtype=torch.float64),
            coord=torch.zeros(B, N, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="time_s"):
        TokenMetadata(
            modality="slow_ts",
            channel_id=torch.zeros(B, N, dtype=torch.long),
            time_s=torch.zeros(B, N, dtype=torch.float32),
            coord=torch.zeros(B, N, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="coord"):
        TokenMetadata(
            modality="slow_ts",
            channel_id=torch.zeros(B, N, dtype=torch.long),
            time_s=torch.zeros(B, N, dtype=torch.float64),
            coord=torch.zeros(B, N, dtype=torch.float64),
        )

    # Non-2-D tensor is rejected.
    with pytest.raises(ValueError, match="channel_id"):
        TokenMetadata(
            modality="slow_ts",
            channel_id=torch.zeros(N, dtype=torch.long),
            time_s=torch.zeros(N, dtype=torch.float64),
            coord=torch.zeros(N, dtype=torch.float32),
        )

    # Shape disagreement across fields is rejected.
    with pytest.raises(ValueError, match="time_s"):
        TokenMetadata(
            modality="slow_ts",
            channel_id=torch.zeros(B, N, dtype=torch.long),
            time_s=torch.zeros(B, N + 1, dtype=torch.float64),
            coord=torch.zeros(B, N, dtype=torch.float32),
        )

    # A per-token modality list whose length disagrees with N is rejected.
    with pytest.raises(ValueError, match="modality"):
        TokenMetadata(
            modality=["slow_ts", "profile"],
            channel_id=torch.zeros(B, N, dtype=torch.long),
            time_s=torch.zeros(B, N, dtype=torch.float64),
            coord=torch.zeros(B, N, dtype=torch.float32),
        )

    # A non-str/list modality is rejected.
    with pytest.raises(ValueError, match="modality"):
        TokenMetadata(
            modality=42,  # type: ignore[arg-type]
            channel_id=torch.zeros(B, N, dtype=torch.long),
            time_s=torch.zeros(B, N, dtype=torch.float64),
            coord=torch.zeros(B, N, dtype=torch.float32),
        )


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


def test_loss_output_rejects_nonscalar_and_nontensor_terms():
    scalar = torch.tensor(1.0)

    # total must be a scalar tensor.
    with pytest.raises(ValueError, match="total"):
        LossOutput(total=1.0, terms={}, diagnostics={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total"):
        LossOutput(total=torch.ones(2), terms={}, diagnostics={})

    # A non-tensor term is rejected, naming the offending key.
    with pytest.raises(ValueError, match="reg"):
        LossOutput(
            total=scalar,
            terms={"reg": 0.5},  # type: ignore[dict-item]
            diagnostics={},
        )
    # A non-scalar tensor term is rejected, naming the offending key.
    with pytest.raises(ValueError, match="reg"):
        LossOutput(total=scalar, terms={"reg": torch.ones(3)}, diagnostics={})

    # Non-float diagnostics (tensor, int, bool) are rejected, naming the key.
    with pytest.raises(ValueError, match="grad_norm"):
        LossOutput(
            total=scalar,
            terms={},
            diagnostics={"grad_norm": scalar},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="steps"):
        LossOutput(
            total=scalar,
            terms={},
            diagnostics={"steps": 3},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="converged"):
        LossOutput(
            total=scalar,
            terms={},
            diagnostics={"converged": True},  # type: ignore[dict-item]
        )


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
        assert torch.equal(
            batch.target_mask[signal], again.target_mask[signal]
        )
        torch.testing.assert_close(
            batch.context[signal], again.context[signal], equal_nan=True
        )
        torch.testing.assert_close(
            batch.target[signal], again.target[signal], equal_nan=True
        )
    assert torch.equal(batch.actions, again.actions)
    assert torch.equal(batch.action_mask, again.action_mask)
    assert torch.equal(batch.context_times, again.context_times)
    assert torch.equal(batch.target_times, again.target_times)
    assert torch.equal(batch.action_times, again.action_times)
    assert torch.equal(batch.horizon_seconds, again.horizon_seconds)
    assert batch.shot_id == again.shot_id
    assert batch.window_id == again.window_id
    assert batch.metadata == again.metadata
