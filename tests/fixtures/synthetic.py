"""Deterministic synthetic fixtures for Fusion-JEPA tests."""

from collections.abc import Sequence

import torch

from fusion_jepa.data.batch import FusionBatch, FusionSample, collate_fusion
from fusion_jepa.utils.reproducibility import derive_seed

_CHANNEL_STEP = 0.1
_ACTION_CHANNEL_STEP = 0.1
_DEVICE_CONTEXT_DIM = 3


def make_ramp_sample(
    signals: Sequence[str] = ("plasma_current",),
    T_ctx: int = 4,
    T_tgt: int = 3,
    H: int = 5,
    *,
    channels: int = 1,
    action_dim: int = 2,
    device_context_dim: int = 3,
    time_offset: float = 0.0,
    shot_id: str = "shot-1",
    window_id: str = "window-1",
    device_id: str = "MAST",
    task_id: str = "forecast",
    split: str = "train",
) -> FusionSample:
    """Create a sample whose signal values equal their physical times."""
    context_times = torch.arange(
        T_ctx,
        dtype=torch.float64,
    ) + time_offset
    target_times = torch.arange(
        T_ctx,
        T_ctx + T_tgt,
        dtype=torch.float64,
    ) + time_offset
    action_times = torch.linspace(
        context_times[-1],
        target_times[-1],
        H,
        dtype=torch.float64,
    )
    context_ramp = context_times.to(torch.float32).repeat(channels, 1)
    target_ramp = target_times.to(torch.float32).repeat(channels, 1)
    context = {signal: context_ramp.clone() for signal in signals}
    target = {signal: target_ramp.clone() for signal in signals}
    context_mask = {
        signal: torch.ones_like(values, dtype=torch.bool)
        for signal, values in context.items()
    }
    target_mask = {
        signal: torch.ones_like(values, dtype=torch.bool)
        for signal, values in target.items()
    }
    units = {signal: "s" for signal in signals}
    canonical_names = {signal: signal for signal in signals}
    shot_end = time_offset + T_ctx + T_tgt + 1.0

    return FusionSample(
        context=context,
        context_mask=context_mask,
        target=target,
        target_mask=target_mask,
        actions=torch.zeros((H, action_dim), dtype=torch.float32),
        action_mask=torch.ones(H, dtype=torch.bool),
        context_times=context_times,
        target_times=target_times,
        action_times=action_times,
        horizon_seconds=target_times[-1] - context_times[-1],
        device_id=device_id,
        device_context=torch.zeros(device_context_dim, dtype=torch.float32),
        device_context_mask=torch.ones(device_context_dim, dtype=torch.bool),
        shot_id=shot_id,
        window_id=window_id,
        metadata={
            "units": units,
            "canonical_names": canonical_names,
            "task_id": task_id,
            "split": split,
            "shot_time_ranges": {shot_id: (time_offset, shot_end)},
        },
    )


def _ramp_with_missing(
    times: torch.Tensor,
    n_channels: int,
    *,
    seed: int,
    missing_fraction: float,
    modality: str,
    group: str,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a ``[C, T]`` ramp and its bool mask with seeded missingness.

    ``value[c, t] == times[t] + c * _CHANNEL_STEP``. A deterministic fraction of
    the entries are marked unobserved (mask ``False``) and their values replaced
    with ``NaN`` -- a placeholder that never violates the finite-where-observed
    invariant because the corresponding mask entry is ``False``.
    """
    channel_offset = _CHANNEL_STEP * torch.arange(n_channels, dtype=torch.float64)
    ramp = (times.unsqueeze(0) + channel_offset.unsqueeze(1)).to(torch.float32)

    generator = torch.Generator().manual_seed(
        derive_seed(seed, "missing", modality, group, sample_index)
    )
    observed = (
        torch.rand(ramp.shape, generator=generator) >= missing_fraction
    )
    values = ramp.clone()
    values[~observed] = torch.nan
    return values, observed


def make_synthetic_fusion_batch(
    B: int,
    modalities: Sequence[str] = ("slow_ts", "profile"),
    *,
    n_channels: int = 3,
    T: int = 4,
    H: int = 3,
    A: int = 2,
    seed: int = 0,
    missing_fraction: float = 0.0,
) -> FusionBatch:
    """Build a deterministic multimodal batch that passes ``validate_batch``.

    Dimensions:
        B: number of samples in the batch (each a distinct synthetic shot).
        modalities: signal names present in both context and target.
        n_channels: channels per signal (shared across modalities).
        T: number of context time frames.
        H: number of target (prediction-horizon) time frames.
        A: number of actuator channels (the action feature dimension).

    Every signal value is a ramp equal to its physical time plus a per-channel
    offset. ``missing_fraction`` of each signal's entries are marked unobserved
    (mask ``False`` with a ``NaN`` placeholder); actions and device context are
    always fully observed so horizon/action coverage stays consistent. Times are
    strictly monotone float64 and each shot's range contains its target window.
    Seeding is derived via :func:`derive_seed`, so identical arguments always
    reproduce an identical batch.
    """
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must lie in [0, 1]")

    n_action_steps = max(2, H + 1)
    span = float(T + H + 1)
    samples: list[FusionSample] = []
    for b in range(B):
        offset = float(b) * span
        context_times = torch.arange(T, dtype=torch.float64) + offset
        target_times = torch.arange(T, T + H, dtype=torch.float64) + offset
        action_times = torch.linspace(
            context_times[-1],
            target_times[-1],
            n_action_steps,
            dtype=torch.float64,
        )

        context: dict[str, torch.Tensor] = {}
        context_mask: dict[str, torch.Tensor] = {}
        target: dict[str, torch.Tensor] = {}
        target_mask: dict[str, torch.Tensor] = {}
        for modality in modalities:
            context[modality], context_mask[modality] = _ramp_with_missing(
                context_times,
                n_channels,
                seed=seed,
                missing_fraction=missing_fraction,
                modality=modality,
                group="context",
                sample_index=b,
            )
            target[modality], target_mask[modality] = _ramp_with_missing(
                target_times,
                n_channels,
                seed=seed,
                missing_fraction=missing_fraction,
                modality=modality,
                group="target",
                sample_index=b,
            )

        action_offset = _ACTION_CHANNEL_STEP * torch.arange(
            A, dtype=torch.float64
        )
        actions = (
            action_times.unsqueeze(1) + action_offset.unsqueeze(0)
        ).to(torch.float32)

        shot_id = f"synthetic-shot-{b}"
        samples.append(
            FusionSample(
                context=context,
                context_mask=context_mask,
                target=target,
                target_mask=target_mask,
                actions=actions,
                action_mask=torch.ones(n_action_steps, dtype=torch.bool),
                context_times=context_times,
                target_times=target_times,
                action_times=action_times,
                horizon_seconds=target_times[-1] - context_times[-1],
                device_id="MAST",
                device_context=torch.zeros(
                    _DEVICE_CONTEXT_DIM, dtype=torch.float32
                ),
                device_context_mask=torch.ones(
                    _DEVICE_CONTEXT_DIM, dtype=torch.bool
                ),
                shot_id=shot_id,
                window_id=f"synthetic-window-{b}",
                metadata={
                    "units": {modality: "s" for modality in modalities},
                    "canonical_names": {
                        modality: modality for modality in modalities
                    },
                    "task_id": "forecast",
                    "split": "train",
                    "shot_time_ranges": {shot_id: (offset, offset + T + H)},
                },
            )
        )

    return collate_fusion(samples)
