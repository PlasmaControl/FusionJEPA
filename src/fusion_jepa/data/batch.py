"""Canonical sample and batch contracts for Fusion-JEPA."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

_TIME_TOLERANCE = 1e-6


@dataclass
class FusionSample:
    """One unbatched multimodal fusion window."""

    context: dict[str, Tensor]
    context_mask: dict[str, Tensor]
    target: dict[str, Tensor]
    target_mask: dict[str, Tensor]
    actions: Tensor
    action_mask: Tensor
    context_times: Tensor
    target_times: Tensor
    action_times: Tensor
    horizon_seconds: Tensor
    device_id: str
    device_context: Tensor
    device_context_mask: Tensor
    shot_id: str
    window_id: str
    metadata: dict[str, Any]


@dataclass
class FusionBatch:
    """A batched multimodal fusion window with explicit observation masks."""

    context: dict[str, Tensor]
    context_mask: dict[str, Tensor]
    target: dict[str, Tensor]
    target_mask: dict[str, Tensor]
    actions: Tensor
    action_mask: Tensor
    context_times: Tensor
    target_times: Tensor
    action_times: Tensor
    horizon_seconds: Tensor
    device_id: list[str]
    device_context: Tensor
    device_context_mask: Tensor
    shot_id: list[str]
    window_id: list[str]
    metadata: dict[str, Any]


def _matching_metadata(samples: list[FusionSample], key: str) -> Any:
    """Return common metadata or raise when samples disagree."""
    value = samples[0].metadata[key]
    if any(sample.metadata[key] != value for sample in samples[1:]):
        raise ValueError(f"metadata {key!r} must agree across samples")
    return deepcopy(value)


def collate_fusion(samples: list[FusionSample]) -> FusionBatch:
    """Stack compatible samples into the canonical batched representation."""
    if not samples:
        raise ValueError("collate_fusion requires at least one sample")

    context_keys = samples[0].context.keys()
    target_keys = samples[0].target.keys()
    if any(sample.context.keys() != context_keys for sample in samples[1:]):
        raise ValueError("context signal keys must agree across samples")
    if any(sample.target.keys() != target_keys for sample in samples[1:]):
        raise ValueError("target signal keys must agree across samples")

    shot_time_ranges: dict[str, tuple[float, float]] = {}
    for sample in samples:
        for shot_id, time_range in sample.metadata["shot_time_ranges"].items():
            if shot_id in shot_time_ranges and shot_time_ranges[shot_id] != time_range:
                raise ValueError(
                    "metadata 'shot_time_ranges' entries must agree across samples"
                )
            shot_time_ranges[shot_id] = time_range

    metadata = {
        "units": _matching_metadata(samples, "units"),
        "canonical_names": _matching_metadata(samples, "canonical_names"),
        "task_id": _matching_metadata(samples, "task_id"),
        "split": _matching_metadata(samples, "split"),
        "shot_time_ranges": shot_time_ranges,
    }
    return FusionBatch(
        context={
            key: torch.stack([sample.context[key] for sample in samples])
            for key in context_keys
        },
        context_mask={
            key: torch.stack([sample.context_mask[key] for sample in samples])
            for key in context_keys
        },
        target={
            key: torch.stack([sample.target[key] for sample in samples])
            for key in target_keys
        },
        target_mask={
            key: torch.stack([sample.target_mask[key] for sample in samples])
            for key in target_keys
        },
        actions=torch.stack([sample.actions for sample in samples]),
        action_mask=torch.stack([sample.action_mask for sample in samples]),
        context_times=torch.stack([sample.context_times for sample in samples]),
        target_times=torch.stack([sample.target_times for sample in samples]),
        action_times=torch.stack([sample.action_times for sample in samples]),
        horizon_seconds=torch.stack(
            [sample.horizon_seconds for sample in samples]
        ),
        device_id=[sample.device_id for sample in samples],
        device_context=torch.stack(
            [sample.device_context for sample in samples]
        ),
        device_context_mask=torch.stack(
            [sample.device_context_mask for sample in samples]
        ),
        shot_id=[sample.shot_id for sample in samples],
        window_id=[sample.window_id for sample in samples],
        metadata=metadata,
    )


def _masked_values_are_finite(values: Tensor, mask: Tensor) -> bool:
    """Return whether all observed values are finite."""
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return bool(torch.isfinite(values).logical_or(~mask).all())


def validate_batch(
    batch: FusionBatch,
    *,
    split_lookup: Mapping[str, str],
    strict: bool = True,
) -> list[str]:
    """Validate scientific invariants and return all violation descriptions."""
    violations: list[str] = []

    for group_name, values_by_signal, masks_by_signal in (
        ("context", batch.context, batch.context_mask),
        ("target", batch.target, batch.target_mask),
    ):
        for signal, values in values_by_signal.items():
            if not _masked_values_are_finite(values, masks_by_signal[signal]):
                violations.append(
                    f"{group_name} signal {signal!r} must be finite where its "
                    "mask is True"
                )
    if not _masked_values_are_finite(batch.actions, batch.action_mask):
        violations.append("actions must be finite where action_mask is True")
    if not _masked_values_are_finite(
        batch.device_context,
        batch.device_context_mask,
    ):
        violations.append(
            "device_context must be finite where device_context_mask is True"
        )

    for name, times in (
        ("context_times", batch.context_times),
        ("target_times", batch.target_times),
        ("action_times", batch.action_times),
    ):
        if not bool((times[:, 1:] > times[:, :-1]).all()):
            violations.append(f"{name} must be strictly monotone increasing")

    context_end = batch.context_times[:, -1]
    target_start = batch.target_times[:, 0]
    target_end = batch.target_times[:, -1]
    if not bool((context_end < target_start).all()):
        violations.append(
            "context and target time ranges must not overlap"
        )

    expected_horizon = target_end - context_end
    if not bool(
        torch.isclose(
            batch.horizon_seconds,
            expected_horizon,
            atol=_TIME_TOLERANCE,
            rtol=0.0,
        ).all()
    ):
        violations.append(
            "horizon_seconds must equal target end time minus context end time"
        )

    covers_start = batch.action_times[:, 0] <= context_end + _TIME_TOLERANCE
    covers_end = batch.action_times[:, -1] >= target_end - _TIME_TOLERANCE
    if not bool((covers_start & covers_end).all()):
        violations.append(
            "action_times must cover the interval from context end to target end"
        )

    signal_keys = set(batch.context) | set(batch.target)
    units = batch.metadata.get("units", {})
    canonical_names = batch.metadata.get("canonical_names", {})
    missing_units = sorted(signal_keys - units.keys())
    if missing_units:
        violations.append(
            f"metadata units must declare every signal key: {missing_units}"
        )
    missing_names = sorted(signal_keys - canonical_names.keys())
    if missing_names:
        violations.append(
            "metadata canonical_names must declare every signal key: "
            f"{missing_names}"
        )

    if len(set(batch.window_id)) != len(batch.window_id):
        violations.append("window_ids must be unique within the batch")

    shot_ranges = batch.metadata.get("shot_time_ranges", {})
    for index, shot_id in enumerate(batch.shot_id):
        if shot_id not in shot_ranges:
            violations.append(
                f"metadata shot_time_ranges must contain shot_id {shot_id!r}"
            )
        else:
            shot_start, shot_end = shot_ranges[shot_id]
            in_range = bool(
                (batch.target_times[index] >= shot_start).all()
                and (batch.target_times[index] <= shot_end).all()
            )
            if not in_range:
                violations.append(
                    f"target times for {shot_id!r} must fall within metadata "
                    "shot_time_ranges"
                )

    expected_split = batch.metadata.get("split")
    for shot_id in batch.shot_id:
        if split_lookup.get(shot_id) != expected_split:
            violations.append(
                f"split_lookup for {shot_id!r} must equal metadata split "
                f"{expected_split!r}"
            )

    if strict and violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise ValueError(f"FusionBatch validation failed:\n{details}")
    return violations
