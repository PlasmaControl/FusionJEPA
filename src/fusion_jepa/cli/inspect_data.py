"""M1 data-inspection command: pull one real batch, validate, summarize, plot.

This is the M1 acceptance artifact. ``experiment=mast_smoke`` streams a single
batch of raw (un-normalized) windows through the whole data stack -- upstream
``tokamark`` -> :func:`fusion_jepa.data.tokamark.make_dataset` ->
:func:`~fusion_jepa.data.batch.collate_fusion` -> the scientific validator --
and emits a per-modality summary plus per-modality trace plots so a human can
eyeball what actually comes off the store.

The two pure functions :func:`summarize_batch` and :func:`plot_batch` are kept
free of any I/O beyond writing the requested PNGs, so the offline unit tests
exercise them against the synthetic ramp fixture with no network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from fusion_jepa.config import ConfigError
from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.utils.run_artifacts import create_run_dir, write_completion

from ._common import (
    add_experiment_name,
    dry_run_report,
    parse_argv,
    report_config_error,
    resolve_cli_config,
)

# A handful of windows is plenty to eyeball the shapes/units/masks; keeping it
# small keeps the remote S3 pull cheap.
_MAX_SAMPLES = 4
# Cap the number of channels drawn per signal so wide profile diagnostics
# (e.g. Thomson scattering, tens of positions) stay legible.
_MAX_PLOT_CHANNELS = 8

# The run config labels the held-out split ``validation`` (RunSettings default)
# while TokaMark's official split CSV names it ``val``; bridge the two here.
_SPLIT_ALIASES = {"validation": "val"}


def _official_split_name(split: str) -> str:
    """Map a run-config split label to the upstream official split name."""
    return _SPLIT_ALIASES.get(split, split)


# ----------------------------------------------------------------------------
# Pure summary
# ----------------------------------------------------------------------------
def _range_seconds(times: torch.Tensor) -> tuple[float, float]:
    values = times.to(torch.float64)
    return float(values.min().item()), float(values.max().item())


def _fill_percent(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return 100.0 * float(mask.to(torch.float64).mean().item())


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _signal_table(
    values_by_signal: dict[str, torch.Tensor],
    masks_by_signal: dict[str, torch.Tensor],
    units: dict[str, str],
) -> list[str]:
    """Render an aligned per-signal shape/dtype/units/fill table."""
    header = ("signal", "shape", "dtype", "units", "fill")
    rows = [header]
    for name, values in values_by_signal.items():
        rows.append(
            (
                name,
                str(tuple(values.shape)),
                _dtype_name(values),
                units.get(name, "?"),
                f"{_fill_percent(masks_by_signal[name]):.1f}%",
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]


def _action_channel_names(batch: FusionBatch) -> list[str]:
    """Actuator signal names: metadata entries that are neither context nor target."""
    units = batch.metadata.get("units", {})
    context_keys = set(batch.context)
    target_keys = set(batch.target)
    return sorted(set(units) - context_keys - target_keys)


def summarize_batch(
    batch: FusionBatch,
    problems: Sequence[str] | None = None,
) -> str:
    """Return a human-readable per-modality summary of ``batch``.

    Covers, per modality, tensor shape/dtype, registry units, and mask
    fill-rate, plus the batch-level context/target/action time ranges (seconds),
    the forecast horizon (milliseconds), and the fused action channel names. Any
    ``problems`` reported by the validator are appended verbatim -- this is an
    inspection tool, so validator findings are surfaced, not fatal.
    """
    units = batch.metadata.get("units", {})
    lines: list[str] = []
    lines.append("Fusion-JEPA batch inspection")
    lines.append(
        f"  task_id: {batch.metadata.get('task_id')}    "
        f"split: {batch.metadata.get('split')}    "
        f"batch_size: {len(batch.window_id)}"
    )

    ctx_lo, ctx_hi = _range_seconds(batch.context_times)
    tgt_lo, tgt_hi = _range_seconds(batch.target_times)
    act_lo, act_hi = _range_seconds(batch.action_times)
    horizon_ms = batch.horizon_seconds.to(torch.float64) * 1000.0
    lines.append(f"  context time range: [{ctx_lo:.6g}, {ctx_hi:.6g}] s")
    lines.append(f"  target time range:  [{tgt_lo:.6g}, {tgt_hi:.6g}] s")
    lines.append(f"  action time range:  [{act_lo:.6g}, {act_hi:.6g}] s")
    lines.append(
        f"  horizon: [{horizon_ms.min().item():.4g}, "
        f"{horizon_ms.max().item():.4g}] ms"
    )

    lines.append("")
    lines.append("context signals (units, shape, dtype, mask fill%):")
    lines.extend(
        f"  {row}"
        for row in _signal_table(batch.context, batch.context_mask, units)
    )

    lines.append("")
    lines.append("target signals (units, shape, dtype, mask fill%):")
    lines.extend(
        f"  {row}"
        for row in _signal_table(batch.target, batch.target_mask, units)
    )

    lines.append("")
    action_names = _action_channel_names(batch)
    lines.append(
        f"actions: shape {tuple(batch.actions.shape)}, "
        f"dtype {_dtype_name(batch.actions)}, "
        f"mask fill {_fill_percent(batch.action_mask):.1f}%"
    )
    lines.append(
        "  action channels: "
        + (", ".join(action_names) if action_names else "(none named)")
    )

    lines.append("")
    if problems:
        lines.append(f"validator problems ({len(problems)}, non-fatal):")
        lines.extend(f"  - {problem}" for problem in problems)
    else:
        lines.append("validator problems: none")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Pure plotting
# ----------------------------------------------------------------------------
def _safe_name(signal: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in signal)


def _channel_time_view(values: torch.Tensor) -> Any:
    """Flatten a per-sample tensor to ``(channels, time)`` numpy."""
    array = values.detach().cpu().numpy()
    return array.reshape(-1, array.shape[-1])


def _shade_masked(ax: Any, times: Any, observed: Any) -> None:
    import numpy as np

    missing = ~np.asarray(observed, dtype=bool)
    if not missing.any():
        return
    ax.fill_between(
        times,
        0,
        1,
        where=missing,
        transform=ax.get_xaxis_transform(),
        color="tab:red",
        alpha=0.15,
        step="mid",
        label="masked",
    )


def _plot_trace(
    ax: Any,
    axis_times: Any,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    label: str,
) -> bool:
    import numpy as np

    view = _channel_time_view(values)
    mask_view = _channel_time_view(mask).astype(bool)
    n_time = view.shape[-1]
    # Non-reference signals keep their native sampling rate, so a signal's time
    # length may not match the batch reference axis; fall back to sample index.
    if axis_times is not None and len(axis_times) == n_time:
        xs = np.asarray(axis_times)
        x_is_time = True
    else:
        xs = np.arange(n_time)
        x_is_time = False
    n_channels = min(view.shape[0], _MAX_PLOT_CHANNELS)
    for channel in range(n_channels):
        suffix = "" if view.shape[0] == 1 else f" ch{channel}"
        ax.plot(xs, view[channel], label=f"{label}{suffix}")
    _shade_masked(ax, xs, mask_view[:n_channels].all(axis=0))
    return x_is_time


def plot_batch(batch: FusionBatch, out_dir: str | Path) -> list[Path]:
    """Write one trace PNG per modality (each signal, plus actions) to ``out_dir``.

    Sample 0 of the batch is drawn. For every signal that appears in the context
    and/or target, its context and target traces share one figure (with masked
    timesteps shaded); the fused action channels get their own figure. Returns
    the list of written paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    units = batch.metadata.get("units", {})
    paths: list[Path] = []

    ctx_times = batch.context_times[0].detach().cpu().numpy()
    tgt_times = batch.target_times[0].detach().cpu().numpy()

    signal_keys = list(dict.fromkeys(list(batch.context) + list(batch.target)))
    for signal in signal_keys:
        fig, ax = plt.subplots(figsize=(9, 4))
        any_time_axis = False
        if signal in batch.context:
            any_time_axis |= bool(
                _plot_trace(
                    ax,
                    ctx_times,
                    batch.context[signal][0],
                    batch.context_mask[signal][0],
                    label="context",
                )
            )
        if signal in batch.target:
            any_time_axis |= bool(
                _plot_trace(
                    ax,
                    tgt_times,
                    batch.target[signal][0],
                    batch.target_mask[signal][0],
                    label="target",
                )
            )
        ax.set_title(f"{signal} ({units.get(signal, '?')})")
        ax.set_xlabel("time (s)" if any_time_axis else "sample index")
        ax.set_ylabel(units.get(signal, ""))
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        path = out / f"{_safe_name(signal)}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)

    if batch.actions.shape[-1] > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        act_times = batch.action_times[0].detach().cpu().numpy()
        actions = batch.actions[0].detach().cpu().numpy()  # (H, action_dim)
        names = _action_channel_names(batch)
        for channel in range(min(actions.shape[1], _MAX_PLOT_CHANNELS)):
            label = names[channel] if channel < len(names) else f"action[{channel}]"
            ax.plot(act_times, actions[:, channel], label=label)
        action_mask = batch.action_mask[0].detach().cpu().numpy().astype(bool)
        _shade_masked(ax, act_times, np.asarray(action_mask))
        ax.set_title("actions")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("actuator value")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        path = out / "actions.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)

    return paths


# ----------------------------------------------------------------------------
# Command flow
# ----------------------------------------------------------------------------
def _take_samples(dataset: Iterable, count: int) -> list:
    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= count:
            break
    return samples


def _inspect(cfg, run_dir: Path) -> tuple[str, list[str], list[Path]]:
    """Stream one batch, validate it, and produce the summary + plots."""
    from fusion_jepa.data.batch import collate_fusion, validate_batch
    from fusion_jepa.data.tokamark import make_dataset, official_split

    split = _official_split_name(str(cfg.split))
    dataset = make_dataset(
        cfg.data.task_id,
        split,
        cluster=cfg.cluster,
        data_cfg=cfg.data,
        normalization=None,  # raw inspection: never standardize
    )
    samples = _take_samples(dataset, _MAX_SAMPLES)
    if not samples:
        raise RuntimeError(
            f"dataset for task {cfg.data.task_id!r} split {split!r} yielded no "
            "windows to inspect"
        )
    batch = collate_fusion(samples)

    manifest = official_split()
    split_lookup = {shot: manifest.split_of(shot) for shot in batch.shot_id}
    problems = validate_batch(batch, split_lookup=split_lookup, strict=False)

    summary = summarize_batch(batch, problems=problems)
    plots = plot_batch(batch, run_dir / "artifacts" / "inspect")
    return summary, problems, plots


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data-inspection CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_argv(arguments)
        cfg = resolve_cli_config(parsed)
    except ConfigError as exc:
        return report_config_error(exc)

    if parsed.dry_run:
        return dry_run_report(cfg)

    add_experiment_name(cfg, parsed.dotlist)
    started_at = datetime.now(timezone.utc).isoformat()
    context = create_run_dir(cfg, arguments, base=cfg.cluster.runs_root)

    try:
        summary, problems, plots = _inspect(cfg, context.run_dir)
    except Exception as exc:  # surface adapter bugs, but record the failed run
        write_completion(
            context.run_dir,
            status="failed",
            started_at=started_at,
            warnings=[],
            failure_reason=repr(exc),
        )
        raise

    print(summary)
    for path in plots:
        print(f"wrote {path}")

    write_completion(
        context.run_dir,
        status="succeeded",
        started_at=started_at,
        warnings=list(problems),
        failure_reason=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
