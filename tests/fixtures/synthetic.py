"""Deterministic synthetic fixtures for Fusion-JEPA tests."""

from collections.abc import Sequence

import torch

from fusion_jepa.data.batch import FusionSample


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
