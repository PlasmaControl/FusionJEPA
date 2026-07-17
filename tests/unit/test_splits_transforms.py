"""Tests for leakage-safe data splits and signal normalization."""

from dataclasses import asdict

import pytest
import torch

from fusion_jepa.data.splits import SplitManifest
from fusion_jepa.data.transforms import (
    NormalizationStats,
    Standardize,
    fit_normalization,
)
from fusion_jepa.utils.manifests import manifest_hash
from tests.fixtures.synthetic import make_ramp_sample


def _manifest() -> SplitManifest:
    return SplitManifest(
        name="official",
        source="tokamark_official",
        source_hash="source-sha256",
        splits={"train": ["shot-1"], "val": ["shot-2"], "test": ["shot-3"]},
    )


def test_disjoint_manifest_passes() -> None:
    manifest = _manifest()

    manifest.assert_disjoint()
    assert manifest.split_of("shot-2") == "val"
    with pytest.raises(KeyError, match="unknown-shot"):
        manifest.split_of("unknown-shot")


def test_shot_in_two_splits_raises() -> None:
    with pytest.raises(ValueError, match="shot-1"):
        SplitManifest(
            name="invalid",
            source="test",
            source_hash="hash",
            splits={"train": ["shot-1"], "test": ["shot-1"]},
        )


def test_round_trip_preserves_hash(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "splits.yaml"
    hash_before_save = manifest_hash(asdict(manifest))

    manifest.save(path)
    loaded = SplitManifest.load(path)

    assert loaded == manifest
    assert manifest_hash(asdict(loaded)) == hash_before_save


def test_normalization_stats_save_load_round_trip(tmp_path) -> None:
    stats = NormalizationStats(
        per_signal={"plasma_current": (2.5, 1.25), "density": (4.0, 0.5)},
        fit_split="train",
        split_manifest_hash="manifest-sha256",
    )
    path = tmp_path / "normalization.yaml"

    stats.save(path)
    loaded = NormalizationStats.load(path)

    assert loaded.per_signal == stats.per_signal
    assert loaded.fit_split == stats.fit_split
    assert loaded.split_manifest_hash == stats.split_manifest_hash


def test_fit_ignores_masked_values() -> None:
    sample = make_ramp_sample()
    sample.context["plasma_current"][0, 0] = 1.0e20
    sample.context_mask["plasma_current"][0, 0] = False

    stats = fit_normalization([sample], split="train", manifest=_manifest())

    mean, _ = stats.per_signal["plasma_current"]
    assert mean == pytest.approx(3.5)


def test_fit_refuses_non_train_split() -> None:
    with pytest.raises(ValueError, match="train.*leakage"):
        fit_normalization(
            [make_ramp_sample()], split="val", manifest=_manifest()
        )


def test_fit_refuses_sample_from_other_split() -> None:
    manifest = SplitManifest(
        name="official",
        source="tokamark_official",
        source_hash="source-sha256",
        splits={"train": ["shot-train"], "val": ["shot-1"]},
    )

    with pytest.raises(ValueError, match="shot-1"):
        fit_normalization(
            [make_ramp_sample(shot_id="shot-1")],
            split="train",
            manifest=manifest,
        )


def test_standardize_inverse_round_trip() -> None:
    sample = make_ramp_sample(signals=("plasma_current", "density"))
    stats = fit_normalization([sample], split="train", manifest=_manifest())

    standardized = Standardize(stats)(sample)
    restored = Standardize(stats).inverse(standardized)

    assert standardized is not sample
    for signal in sample.context:
        assert torch.allclose(restored.context[signal], sample.context[signal])
        assert torch.allclose(restored.target[signal], sample.target[signal])
        assert standardized.context[signal] is not sample.context[signal]


def test_stats_record_split_manifest_hash() -> None:
    manifest = _manifest()

    stats = fit_normalization(
        [make_ramp_sample()], split="train", manifest=manifest
    )

    assert stats.fit_split == "train"
    assert stats.split_manifest_hash == manifest_hash(asdict(manifest))
