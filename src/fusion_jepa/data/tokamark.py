"""Adapter between the pinned upstream ``tokamark`` benchmark and Fusion-JEPA.

This module is the single seam between the frozen TokaMark/MAST_tools benchmark
(datasets, task configs, official splits, metrics) and the canonical
:class:`~fusion_jepa.data.batch.FusionSample`/``FusionBatch`` contract that
Fusion-JEPA trains and evaluates against.

Design rules (see ``docs/decisions/0002-tokamark-adapter-semantics.md``):

* **Single upstream shim.** Every touch of ``tokamark``/``MAST_tools`` goes
  through the module-level :data:`_upstream` object. Unit tests monkeypatch its
  attributes with fakes, so the adapter is testable offline and survives
  upstream API drift.
* **NaN sentinels -> explicit masks.** Upstream exposes no boolean mask; missing
  data is ``NaN``. The adapter derives ``mask = isfinite(values)`` and replaces
  the missing entries with ``0.0`` so ``FusionBatch``'s finite-where-observed
  invariant holds while imputed/missing stays distinguishable via the mask.
* **Seconds, everywhere.** Upstream times and task-config lengths are already in
  seconds; the adapter applies no unit rescaling. ``FusionBatch`` times/horizons
  are ``float64`` seconds.
* **No standardization on the eval path.** Datasets are built with
  ``use_std_scaling=False``; our :class:`~fusion_jepa.data.transforms.Standardize`
  is applied only when a ``normalization`` transform is passed in (pretraining),
  never inside the official-metric path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from fusion_jepa.data.batch import FusionSample
from fusion_jepa.data.registry import SignalSpec, load_registry
from fusion_jepa.data.splits import SplitManifest
from fusion_jepa.utils.manifests import read_manifest, write_manifest


class TokamarkAdapterError(RuntimeError):
    """Raised when an upstream window cannot be translated to a FusionSample."""


# ============================================================================
# Upstream shim -- the ONLY place upstream symbols are imported.
# ============================================================================
class _UpstreamShim:
    """Lazily-imported, monkeypatchable view of the upstream API.

    Each attribute resolves (and caches) a symbol from the installed
    ``tokamark``/``MAST_tools`` distribution on first access. Tests replace
    attributes wholesale (``monkeypatch.setattr(_upstream, "name", fake)``);
    a set instance attribute shadows :meth:`__getattr__`, so no real import
    happens on the faked path.
    """

    _TARGETS = {
        "get_task_config": ("tokamark.tasks", "get_task_config"),
        "get_task_metadata": ("tokamark.tasks", "get_task_metadata"),
        "TASKS_CONFIGS_MAP": ("tokamark.tasks", "TASKS_CONFIGS_MAP"),
        "GROUP_TASKS": ("tokamark.tasks", "GROUP_TASKS"),
        "TASKS_CONFIGS_DIR": ("tokamark.tools.path", "TASKS_CONFIGS_DIR"),
        "split_csv_path": (
            "tokamark.tools.path",
            "RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE",
        ),
        "initialize_MAST_dataset": ("tokamark.data", "initialize_MAST_dataset"),
        "initialize_TokaMark_dataset": (
            "tokamark.data",
            "initialize_TokaMark_dataset",
        ),
        "WindowMetricsAccumulator": (
            "tokamark.evaluator",
            "WindowMetricsAccumulator",
        ),
        "compute_metrics": ("tokamark.evaluator", "compute_metrics"),
    }

    def __getattr__(self, name: str) -> Any:
        try:
            module_path, symbol = self._TARGETS[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        value = getattr(importlib.import_module(module_path), symbol)
        # Cache on the instance so subsequent access (and monkeypatching)
        # bypasses __getattr__.
        object.__setattr__(self, name, value)
        return value


_upstream = _UpstreamShim()


# ============================================================================
# Pin verification
# ============================================================================
def _repo_root() -> Path:
    """Locate the repo root (holds ``manifests/upstream.yaml``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "manifests" / "upstream.yaml").exists():
            return parent
    raise TokamarkAdapterError(
        "Could not locate manifests/upstream.yaml by walking up from "
        f"{Path(__file__).resolve()}"
    )


def assert_pinned_upstream() -> None:
    """Fail closed if the installed ``tokamark`` has drifted from the pin.

    Compares the commit recorded in the installed distribution's PEP 610
    ``direct_url.json`` against ``manifests/upstream.yaml`` (Task 0.8's pin,
    the single source of truth). Raises an actionable error on any mismatch or
    missing provenance.
    """
    import importlib.metadata as importlib_metadata

    pinned = read_manifest(_repo_root() / "manifests" / "upstream.yaml")
    expected = pinned["tokamark"]["commit"]

    try:
        dist = importlib_metadata.distribution("tokamark")
    except importlib_metadata.PackageNotFoundError as exc:
        raise TokamarkAdapterError(
            "tokamark is not installed; run `pixi install` for the 'frontier' "
            "or 'default' environment (see docs/decisions/0001-tokamark-pin.md)."
        ) from exc

    raw = dist.read_text("direct_url.json")
    if not raw:
        raise TokamarkAdapterError(
            "Installed tokamark has no direct_url.json provenance; cannot verify "
            f"it matches the pinned commit {expected!r}. Reinstall from the "
            "pinned git rev via `pixi install`."
        )

    installed = json.loads(raw).get("vcs_info", {}).get("commit_id")
    if installed != expected:
        raise TokamarkAdapterError(
            "Upstream tokamark drift detected: installed commit "
            f"{installed!r} != pinned {expected!r} (manifests/upstream.yaml). "
            "Sync the pin in pyproject.toml and run `pixi install`."
        )


# ============================================================================
# Task configuration (deep copy + provenance)
# ============================================================================
@dataclass
class TaskConfig:
    """A deep-copied task config plus the provenance of its packaged YAML."""

    task_id: str
    config: dict[str, Any]
    source_path: str
    source_sha256: str

    def provenance(self) -> dict[str, str]:
        """Return a plain-dict provenance record for per-run saving."""
        return {
            "task_id": self.task_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }

    def save(self, path: str | Path) -> None:
        """Persist the provenance record (not the config body) as YAML."""
        write_manifest(self.provenance(), path)


def load_task_config(task_id: str) -> TaskConfig:
    """Load one benchmark task config, deep-copied, with a recorded YAML hash.

    The returned :class:`TaskConfig` owns an independent deep copy of the
    upstream config dict (mutating it never affects a re-load) and records the
    packaged YAML's path and sha256 so a run can pin exactly which task
    definition it evaluated against.
    """
    from copy import deepcopy

    config = deepcopy(_upstream.get_task_config(task_id))

    source = Path(_upstream.TASKS_CONFIGS_DIR) / _upstream.TASKS_CONFIGS_MAP[task_id]
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    return TaskConfig(
        task_id=task_id,
        config=config,
        source_path=str(source),
        source_sha256=source_sha256,
    )


# ============================================================================
# Official split
# ============================================================================
def official_split(name: str = "tokamark_official") -> SplitManifest:
    """Build a :class:`SplitManifest` from the packaged official split CSV.

    Reads the ``train``/``val``/``test`` boolean columns of TokaMark's random
    split CSV (a packaged resource -- no network). ``source_hash`` is the
    sha256 of that CSV file, and disjointness is enforced at construction.
    """
    import pandas as pd

    csv_path = Path(_upstream.split_csv_path)
    frame = pd.read_csv(csv_path)

    splits: dict[str, list[str]] = {}
    for split, column in (("train", "train"), ("val", "val"), ("test", "test")):
        selected = frame.loc[frame[column] == True, "shot_id"]  # noqa: E712
        splits[split] = [str(int(shot)) for shot in selected.tolist()]

    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return SplitManifest(
        name=name,
        source=csv_path.name,
        source_hash=source_hash,
        splits=splits,
    )


# ============================================================================
# Pure translation: upstream window -> FusionSample
# ============================================================================
def _config_dict(task_cfg: Any) -> dict[str, Any]:
    return task_cfg.config if isinstance(task_cfg, TaskConfig) else task_cfg


def _group_keys(config: Mapping[str, Any], group: str) -> list[str]:
    """Return the ``"{source}-{signal}"`` keys for a segmenter group."""
    pairs = config["task_window_segmenter"][f"{group}_keys"] or []
    return [f"{source}-{signal}" for source, signal in pairs]


def _canonical_lookup(registry: Mapping[str, SignalSpec]) -> dict[str, str]:
    """Map upstream ``source_name`` -> canonical name."""
    return {spec.source_name: name for name, spec in registry.items()}


def _require_canonical(canonical: Mapping[str, str], key: str) -> str:
    try:
        return canonical[key]
    except KeyError as exc:
        raise TokamarkAdapterError(
            f"upstream signal {key!r} has no entry in the signal registry; "
            "add it to signal_registry/mast.yaml (source_name must equal the "
            "upstream '{source}-{signal}' key)"
        ) from exc


def _reference_geometry(group: Mapping[str, Any], ref_key: str) -> tuple[float, int]:
    """Return ``(dt, n)`` for a group's fixed reference signal.

    The group's first key is the deterministic reference (all windows of a task
    share it, so axis lengths are collate-stable). ``dt`` is the median sample
    interval of the reference window; ``n`` its length. Raises if the reference
    times are missing/degenerate -- the dataset turns that into a skipped
    window.

    The reference axes are rebuilt from ``dt``/``n`` anchored at ``t_cut``
    (see :func:`to_fusion_sample`) rather than reused verbatim, so that the
    forecast boundary is a single clean split. Coarse-rate targets whose
    nearest sample rounds across ``t_cut`` would otherwise overlap the context
    axis and fail ``FusionBatch``'s non-overlap invariant.
    """
    entry = group[ref_key]
    times = np.asarray(entry["time"], dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise TokamarkAdapterError(
            f"reference signal {ref_key!r} has an unusable time axis "
            f"(shape {times.shape}); cannot align the window"
        )
    if not np.all(np.isfinite(times)):
        raise TokamarkAdapterError(
            f"reference signal {ref_key!r} has non-finite times; the reference "
            "diagnostic is missing for this window"
        )
    dt = float(np.median(np.diff(times)))
    if not dt > 0:
        raise TokamarkAdapterError(
            f"reference signal {ref_key!r} has a non-positive sampling interval"
        )
    return dt, int(times.size)


def _as_channel_time(values: np.ndarray) -> np.ndarray:
    """Normalize upstream values to a channel-first, time-last float array.

    Upstream squeezes single-channel windows to 1-D ``(T,)``; promote those to
    ``(1, T)`` to match the ``(channels, ..., time)`` batch convention.
    Multi-channel / profile shapes are kept as-is.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim <= 1:
        values = values.reshape(1, -1)
    return values


def _signal_tensors(
    group: Mapping[str, Any],
    keys: Iterable[str],
    canonical: Mapping[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Translate a window group into per-signal value/mask tensors."""
    values_out: dict[str, torch.Tensor] = {}
    masks_out: dict[str, torch.Tensor] = {}
    for key in keys:
        name = _require_canonical(canonical, key)
        values = _as_channel_time(group[key]["values"])
        mask = np.isfinite(values)
        filled = np.where(mask, values, 0.0).astype(np.float32)
        values_out[name] = torch.from_numpy(filled)
        masks_out[name] = torch.from_numpy(mask)
    return values_out, masks_out


def _build_actions(
    group: Mapping[str, Any],
    keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse actuator signals into an ``(H, action_dim)`` tensor + ``(H,)`` mask.

    All actuator signals in the pinned benchmark share ``dt = 0.00025 s`` (and
    thus an identical window length ``H``); this is asserted, not assumed.
    The per-timestep mask is True only where every actuator channel is observed
    -- a conservative summary (per-channel action missingness is not
    representable in the single ``(H,)`` mask).
    """
    columns: list[np.ndarray] = []
    timestep_masks: list[np.ndarray] = []
    horizon: int | None = None
    for key in keys:
        values = _as_channel_time(group[key]["values"])
        flat = values.reshape(-1, values.shape[-1])  # (channels, H)
        by_time = flat.T  # (H, channels)
        if horizon is None:
            horizon = by_time.shape[0]
        elif by_time.shape[0] != horizon:
            raise TokamarkAdapterError(
                f"actuator {key!r} has length {by_time.shape[0]} but the "
                f"reference actuator has {horizon}; the adapter requires a "
                "shared actuator time grid"
            )
        finite = np.isfinite(by_time)
        columns.append(np.where(finite, by_time, 0.0))
        timestep_masks.append(finite.all(axis=1))

    actions = np.concatenate(columns, axis=1).astype(np.float32)
    action_mask = np.logical_and.reduce(timestep_masks)
    return torch.from_numpy(actions), torch.from_numpy(action_mask)


def _valid_time_mask(values: np.ndarray) -> np.ndarray:
    """Per-timestep validity mask (True where any channel is non-NaN)."""
    values = np.asarray(values)
    if values.ndim <= 1:
        return ~np.isnan(values)
    return ~np.all(np.isnan(values), axis=tuple(range(values.ndim - 1)))


def _shot_time_bounds(
    raw_shot: Mapping[str, Any],
    start_keys: Iterable[str],
    end_keys: Iterable[str],
) -> Optional[tuple[float, float]]:
    """Compute a shot's global [start, end] the way upstream ``_process_shot`` does.

    ``start`` is the earliest valid input/actuator time; ``end`` is the latest
    valid output time. Returns ``None`` if the shot has no usable times.
    """
    starts: list[float] = []
    for key in start_keys:
        entry = raw_shot.get(key)
        if entry is None:
            continue
        times = np.asarray(entry["time"], dtype=np.float64)
        values = np.asarray(entry["values"])
        if times.size == 0 or values.size == 0:
            continue
        mask = _valid_time_mask(values)
        if not np.any(mask):
            continue
        starts.append(float(times[mask][0]))

    ends: list[float] = []
    for key in end_keys:
        entry = raw_shot.get(key)
        if entry is None:
            continue
        times = np.asarray(entry["time"], dtype=np.float64)
        values = np.asarray(entry["values"])
        if times.size == 0 or values.size == 0:
            continue
        mask = _valid_time_mask(values)
        if not np.any(mask):
            continue
        ends.append(float(times[mask][-1]))

    if not starts or not ends:
        return None
    return (min(starts), max(ends))


def to_fusion_sample(
    upstream_item: Mapping[str, Any],
    task_cfg: Any,
    split: str,
    registry: Mapping[str, SignalSpec],
    *,
    shot_time_range: Optional[tuple[float, float]] = None,
) -> FusionSample:
    """Translate one upstream window dict into a :class:`FusionSample`.

    Pure function -- no upstream imports, no I/O -- so it is fully unit-testable
    with fake windows. ``input`` -> context, ``output`` -> target, ``actuator``
    -> fused actions. Time axes come from each group's first (reference) key as
    ``float64`` seconds; per-signal values keep their native resolution.

    ``shot_time_range`` is the shot-level ``[start, end]`` recorded in metadata
    (identical across every window of a shot, as ``FusionBatch`` collation
    requires). When omitted, this window's own extent is used -- adequate for a
    lone window, but the dataset supplies the true shot bounds so same-shot
    windows collate.
    """
    config = _config_dict(task_cfg)
    task_id = config.get("task_name")
    canonical = _canonical_lookup(registry)

    input_keys = _group_keys(config, "input")
    actuator_keys = _group_keys(config, "actuator")
    output_keys = _group_keys(config, "output")

    inputs = upstream_item["input"]
    actuators = upstream_item["actuator"]
    outputs = upstream_item["output"]

    context, context_mask = _signal_tensors(inputs, input_keys, canonical)
    target, target_mask = _signal_tensors(outputs, output_keys, canonical)

    segmenter = config["task_window_segmenter"]
    t_cut = float(upstream_item["t_cut"])
    input_length = float(segmenter["input_length"])
    delta = float(segmenter["delta"])

    # Reference axes are geometry anchored at t_cut: context is the pre-t_cut
    # interval (ends one step before t_cut), target the [t_cut + delta, ...]
    # future interval. This makes the forecast boundary a single clean split
    # regardless of per-signal sampling rates.
    dt_ctx, n_ctx = _reference_geometry(inputs, input_keys[0])
    context_times = torch.from_numpy(
        t_cut - dt_ctx * (n_ctx - np.arange(n_ctx, dtype=np.float64))
    )
    dt_tgt, n_tgt = _reference_geometry(outputs, output_keys[0])
    target_times = torch.from_numpy(
        (t_cut + delta) + dt_tgt * np.arange(n_tgt, dtype=np.float64)
    )

    if actuator_keys:
        dt_act, n_act = _reference_geometry(actuators, actuator_keys[0])
        action_times = torch.from_numpy(
            (t_cut - input_length) + dt_act * np.arange(n_act, dtype=np.float64)
        )
        actions, action_mask = _build_actions(actuators, actuator_keys)
    else:
        # Reconstruction-style tasks carry no actuators; synthesize a minimal
        # covering axis so the covers-transition invariant is well-defined.
        action_times = torch.tensor(
            [context_times[0].item(), target_times[-1].item()],
            dtype=torch.float64,
        )
        actions = torch.zeros((2, 0), dtype=torch.float32)
        action_mask = torch.ones(2, dtype=torch.bool)

    horizon_seconds = (target_times[-1] - context_times[-1]).to(torch.float64)

    shot_id = str(upstream_item["shot_id"])
    window_index = int(upstream_item["window_index"])
    window_id = f"{task_id}:{shot_id}:{window_index}"

    units: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for key in input_keys + actuator_keys + output_keys:
        name = _require_canonical(canonical, key)
        units[name] = registry[name].units
        canonical_names[name] = name

    device_id = registry[canonical[input_keys[0]]].device

    if shot_time_range is not None:
        lo, hi = float(shot_time_range[0]), float(shot_time_range[1])
    else:
        lo = min(
            context_times[0].item(),
            target_times[0].item(),
            action_times[0].item(),
        )
        hi = max(
            context_times[-1].item(),
            target_times[-1].item(),
            action_times[-1].item(),
        )
    metadata = {
        "units": units,
        "canonical_names": canonical_names,
        "task_id": task_id,
        "split": split,
        "shot_time_ranges": {shot_id: (float(lo), float(hi))},
    }

    return FusionSample(
        context=context,
        context_mask=context_mask,
        target=target,
        target_mask=target_mask,
        actions=actions,
        action_mask=action_mask,
        context_times=context_times,
        target_times=target_times,
        action_times=action_times,
        horizon_seconds=horizon_seconds,
        device_id=device_id,
        device_context=torch.zeros(0, dtype=torch.float32),
        device_context_mask=torch.zeros(0, dtype=torch.bool),
        shot_id=shot_id,
        window_id=window_id,
        metadata=metadata,
    )


# ============================================================================
# Dataset: stream upstream windows as FusionSamples (Risk R2 isolation)
# ============================================================================
class TokamarkWindowDataset(IterableDataset):
    """Wrap an upstream iterable window stream as a FusionSample stream.

    Isolates the map(shot)-vs-iterable(window) mismatch (Risk R2): we never
    index by sample, we iterate and key on the upstream-yielded ``shot_id``.
    A window whose reference diagnostic is missing is skipped (counted in
    :attr:`skipped`); a window whose shot is not in the requested split is a
    fail-closed error (leakage must be impossible).
    """

    def __init__(
        self,
        upstream_dataset: Optional[Iterable[Mapping[str, Any]]],
        *,
        task_cfg: TaskConfig,
        split: str,
        allowed_shots: Iterable[str],
        registry: Mapping[str, SignalSpec],
        normalization: Any = None,
        base_dataset: Any = None,
    ) -> None:
        self._upstream_dataset = upstream_dataset
        self.task_cfg = task_cfg
        self.split = split
        self.allowed_shots = {str(shot) for shot in allowed_shots}
        self.registry = registry
        self.normalization = normalization
        self._base_dataset = base_dataset
        self._shot_bounds: dict[str, Optional[tuple[float, float]]] = {}
        config = task_cfg.config
        self._start_keys = _group_keys(config, "input") + _group_keys(
            config, "actuator"
        )
        self._end_keys = _group_keys(config, "output")
        self.skipped = 0

    def _shot_time_range(self, shot_id: str) -> Optional[tuple[float, float]]:
        """Return the cached shot-level time bounds, or None if unavailable.

        Bounds are computed once per shot from the base MAST dataset so every
        window of a shot reports byte-identical bounds (a ``collate_fusion``
        requirement). On Frontier the store is local (cheap); on the S3 dev
        path this is one extra per-shot read -- a documented cost.
        """
        if shot_id in self._shot_bounds:
            return self._shot_bounds[shot_id]

        bounds: Optional[tuple[float, float]] = None
        base = self._base_dataset
        if base is not None:
            shots_list = getattr(base, "shots_list", None)
            index = {str(shot): i for i, shot in enumerate(shots_list or [])}
            position = index.get(shot_id)
            if position is not None:
                raw = base[position]
                bounds = _shot_time_bounds(raw, self._start_keys, self._end_keys)
        self._shot_bounds[shot_id] = bounds
        return bounds

    def __iter__(self) -> Iterator[FusionSample]:
        if self._upstream_dataset is None:
            return
        for item in self._upstream_dataset:
            shot_id = str(item["shot_id"])
            if shot_id not in self.allowed_shots:
                raise ValueError(
                    f"upstream yielded shot {shot_id!r} which is not in the "
                    f"{self.split!r} split; refusing to leak across splits"
                )
            try:
                sample = to_fusion_sample(
                    item,
                    self.task_cfg,
                    self.split,
                    self.registry,
                    shot_time_range=self._shot_time_range(shot_id),
                )
            except TokamarkAdapterError:
                self.skipped += 1
                continue
            if self.normalization is not None:
                sample = self.normalization(sample)
            yield sample


def _default_registry() -> dict[str, SignalSpec]:
    return load_registry(_repo_root() / "signal_registry" / "mast.yaml")


def _storage_from_cluster(cluster: Any) -> tuple[bool, dict[str, Any]]:
    """Translate a cluster profile into (local_flag, store_manager_settings)."""
    root = str(cluster.tokamark_root)
    options = cluster.tokamark_storage_options or {}
    # OmegaConf containers -> plain dict.
    options = dict(options)

    if root.startswith("s3://"):
        rest = root[len("s3://"):]
        endpoint = (
            dict(options.get("client_kwargs", {})).get("endpoint_url")
            or "https://s3.echo.stfc.ac.uk"
        )
        settings = {
            "s3_mast_dataset_path": "/" + rest.strip("/"),
            "s3_endpoint_url": endpoint,
            "target_fsspec_protocol": "s3",
        }
        return False, settings

    return True, {"base_local_zarr_path": root}


def make_dataset(
    task_id: str,
    split: str,
    *,
    cluster: Any,
    data_cfg: Any,
    normalization: Any = None,
) -> TokamarkWindowDataset:
    """Build a :class:`TokamarkWindowDataset` for one task and split.

    Storage (local Lustre vs anonymous S3) is derived from the ``cluster``
    profile. Datasets are constructed with ``use_std_scaling=False`` and
    ``use_nan_filling=False`` so the official-metric path is neither
    standardized nor zero-imputed; pass ``normalization`` (our
    :class:`~fusion_jepa.data.transforms.Standardize`) only for pretraining
    pipelines.
    """
    task_cfg = load_task_config(task_id)
    manifest = official_split()
    if split not in manifest.splits:
        raise ValueError(
            f"unknown split {split!r}; official split has "
            f"{sorted(manifest.splits)}"
        )

    allowed = list(manifest.splits[split])
    limit = getattr(data_cfg, "limit_shots", None)
    if limit:
        allowed = allowed[:limit]
    shots = [int(shot) for shot in allowed]

    local_flag, store_settings = _storage_from_cluster(cluster)
    config = task_cfg.config
    task_metadata = _upstream.get_task_metadata(config)

    base_dataset = _upstream.initialize_MAST_dataset(
        config_task=config,
        shots_list=shots,
        local_flag=local_flag,
        use_std_scaling=False,
        use_nan_filling=False,
        store_manager_settings=store_settings,
    )
    window_dataset = _upstream.initialize_TokaMark_dataset(
        base_dataset,
        task_metadata,
        config,
        custom_transform=None,
        test_mode=False,
        shuffle_windows=False,
    )

    return TokamarkWindowDataset(
        window_dataset,
        task_cfg=task_cfg,
        split=split,
        allowed_shots=allowed,
        registry=_default_registry(),
        normalization=normalization,
        base_dataset=base_dataset,
    )


# ============================================================================
# Official metrics adapter
# ============================================================================
@dataclass
class OfficialMetrics:
    """Thin wrapper over upstream ``WindowMetricsAccumulator``/``compute_metrics``.

    Accumulate per-window errors one feature at a time (matching the upstream
    ``add_batch`` contract), then :meth:`compute` the official task metrics
    dataframe (and optionally write windows/shots/task CSVs).
    """

    task_id: str
    _accumulator: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._accumulator = _upstream.WindowMetricsAccumulator(self.task_id)

    def add_batch(
        self,
        *,
        y_target: np.ndarray,
        y_pred: np.ndarray,
        shot_ids: Any,
        window_indices: Any,
        feature_name: str,
    ) -> None:
        """Add one feature's per-window errors for a model batch."""
        self._accumulator.add_batch(
            y_target=y_target,
            y_pred=y_pred,
            shot_ids=shot_ids,
            window_indices=window_indices,
            feature_name=feature_name,
        )

    def is_empty(self) -> bool:
        """Return True when nothing has been accumulated yet."""
        return self._accumulator.is_empty()

    def compute(
        self,
        output_dir: str | Path,
        *,
        save_windows_metrics: bool = False,
        save_shot_metrics: bool = False,
        save_task_metrics: bool = True,
    ) -> Any:
        """Compute the official task-metrics dataframe (indexed by feature)."""
        return _upstream.compute_metrics(
            self.task_id,
            output_dir,
            self._accumulator,
            save_windows_metrics=save_windows_metrics,
            save_shot_metrics=save_shot_metrics,
            save_task_metrics=save_task_metrics,
        )
