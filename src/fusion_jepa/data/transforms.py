"""Leakage-safe signal normalization transforms."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from fusion_jepa.data.batch import FusionSample
from fusion_jepa.data.splits import SplitManifest
from fusion_jepa.utils.manifests import manifest_hash, read_manifest, write_manifest

_STD_FLOOR = 1e-8


@dataclass
class NormalizationStats:
    """Per-signal scalar statistics fitted on a declared split."""

    per_signal: dict[str, tuple[float, float]]
    fit_split: str
    split_manifest_hash: str

    def save(self, path: str | Path) -> None:
        """Persist these statistics as YAML."""
        write_manifest(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        """Load normalization statistics from YAML."""
        data = read_manifest(path)
        data["per_signal"] = {
            signal: tuple(values)
            for signal, values in data["per_signal"].items()
        }
        return cls(**data)


def _observed(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return values[mask.expand_as(values)]


def fit_normalization(
    samples: Iterable[FusionSample],
    *,
    split: str,
    manifest: SplitManifest,
) -> NormalizationStats:
    """Fit scalar signal statistics using observed train values only."""
    if split != "train":
        raise ValueError(
            "normalization statistics must only ever be fit on the train split "
            "to avoid data leakage"
        )

    values_by_signal: dict[str, list[Tensor]] = {}
    for sample in samples:
        actual_split = manifest.split_of(sample.shot_id)
        if actual_split != split:
            raise ValueError(
                f"sample shot_id={sample.shot_id!r} belongs to "
                f"split={actual_split!r} per the manifest, but fit_normalization "
                f"was called with split={split!r}; normalization statistics must "
                "only be fit on matching-split samples to avoid leakage"
            )
        for values, masks in (
            (sample.context, sample.context_mask),
            (sample.target, sample.target_mask),
        ):
            for signal, tensor in values.items():
                observed = _observed(tensor, masks[signal])
                if observed.numel():
                    values_by_signal.setdefault(signal, []).append(observed)

    per_signal: dict[str, tuple[float, float]] = {}
    for signal, chunks in values_by_signal.items():
        values = torch.cat([chunk.reshape(-1) for chunk in chunks]).double()
        mean = values.mean().item()
        std = max(values.std(unbiased=False).item(), _STD_FLOOR)
        per_signal[signal] = (mean, std)

    return NormalizationStats(
        per_signal=per_signal,
        fit_split=split,
        split_manifest_hash=manifest_hash(asdict(manifest)),
    )


class Standardize:
    """Apply and invert per-signal scalar standardization."""

    def __init__(self, stats: NormalizationStats) -> None:
        self.stats = stats

    def _transform(self, sample: FusionSample, *, inverse: bool) -> FusionSample:
        signals = set(sample.context) | set(sample.target)
        missing = sorted(signals - self.stats.per_signal.keys())
        if missing:
            raise KeyError(
                f"normalization statistics are missing signal(s): {missing}; "
                "fit stats on training samples containing every signal"
            )

        transformed = deepcopy(sample)
        for values in (transformed.context, transformed.target):
            for signal, tensor in values.items():
                mean, std = self.stats.per_signal[signal]
                if inverse:
                    values[signal] = tensor * std + mean
                else:
                    values[signal] = (tensor - mean) / std
        return transformed

    def __call__(self, sample: FusionSample) -> FusionSample:
        """Return a standardized copy of ``sample``."""
        return self._transform(sample, inverse=False)

    def inverse(self, sample: FusionSample) -> FusionSample:
        """Return a copy with standardization undone."""
        return self._transform(sample, inverse=True)
