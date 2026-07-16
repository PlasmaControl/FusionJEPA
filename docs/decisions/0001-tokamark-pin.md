# 0001 — Pin + install upstream `tokamark`; API notes for the Task 1.5 adapter

**Status:** accepted. **Date:** 2026-07-16. **Task:** 1.1.

## Decision

`tokamark` (UKAEA-IBM-STFC-Fusion-FMs/tokamark, tag `OS_v2.0`, commit
`1a200f4588addaad2ae3d53d438d9e1946c09294` — the pin `manifests/upstream.yaml`
already recorded from Task 0.8) is installed as a pixi **git pypi-dependency**,
scoped to a new `[tool.pixi.feature.data]` feature rather than the shared
`[tool.pixi.pypi-dependencies]` table:

```toml
[tool.pixi.feature.data.pypi-dependencies]
tokamark = { git = "https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark", rev = "1a200f4588addaad2ae3d53d438d9e1946c09294" }
```

`[tool.pixi.environments]` now reads:

```toml
default = ["cuda", "data"]
fdp = ["fdp", "cuda"]
frontier = ["frontier", "data"]
```

`tokamark` is importable from `default` and `frontier`; **not** from `fdp`.

### Why scoped (Risk R1 resolution)

A plain top-level add (`[tool.pixi.pypi-dependencies].tokamark = {...}`)
conflicts only in the `fdp` environment:

```
Because tokamark==0.1.0 depends on jupyterlab-widgets==3.0.15 and
jupyterlab-widgets==3.0.16, we can conclude that tokamark==0.1.0 cannot
be used.
And because only tokamark==0.1.0 is available and you require tokamark,
we can conclude that your requirements are unsatisfiable.
help: The following PyPI packages have been pinned by the conda solve...
      jupyterlab_widgets==3.0.16
```

Root cause: tokamark's own `pyproject.toml` pins `jupyterlab-widgets==3.0.15`
exactly (alongside `ipywidgets==8.1.8`). The `fdp` environment's ga-fdp
`toksearch`/`toksearch_d3d` conda packages pull in the conda-forge `jupyter`
meta-package, which fixes `jupyterlab_widgets` at `3.0.16` via the conda
solve — a pin that then binds the co-located PyPI solve and conflicts with
tokamark's own exact pin. This isn't a pin we control on either side (not
ours to relax, and vendoring/patching upstream is out of scope per the
brief), so per the R1 escalation ladder it was scoped to a dedicated
`data` feature shared by `default`/`frontier` only. `default` and `frontier`
solved and installed cleanly on the first attempt with `tokamark` present;
`fdp` was re-verified to still solve/install cleanly with `tokamark` absent.
No pins of ours were relaxed; nothing was vendored.

### Reference clone (not a dependency)

`tokamark_baseline` (same org, tag `OS_v2.0`, commit
`1e7c489653716a60ffe0b4c2a6a2e51883228695`) is cloned read-only, checked out
at that pinned commit, at:

```
/lustre/orion/fus187/scratch/nchen/reference/tokamark_baseline
```

This is reference material for M2 (baseline comparison), not installed by
pixi and not imported by any Fusion-JEPA code.

## API notes for Task 1.5's adapter

All paths below are inside the installed `tokamark` distribution at the
pinned commit (`site-packages/{tokamark,MAST_tools}/...`, mirroring
`src/tokamark/...` and `src/MAST_tools/...` in the upstream repo). The
brief's guessed layout (`utils/store_utils.py`, `MAST_tools/MAST_dataset.py`)
is *close but not exact* — the real split is `tokamark/` (benchmark-facing:
tasks, windowing, evaluation) built on top of `MAST_tools/` (MAST-database
access: raw per-shot loading, storage backends). Both ship as separate
top-level packages from the one `tokamark` sdist/wheel (confirmed by
building the wheel locally: `tokamark-*.whl` contains both `tokamark/` and
`MAST_tools/` trees, including all `tasks_configs/*.yaml` and
`metadata/*.yaml|*.csv` data files — modern `setuptools` (83.0.0 in the
build check) includes them automatically even though the repo's own
`MANIFEST.in` is stale/wrong for this project, e.g. `recursive-include
src/benchmark/task_configs *` references a `benchmark` dir that doesn't
exist. Not a Task 1.1 blocker; flagging in case a future upstream bump
changes packaging behavior).

### Dataset classes: two layers, map-style wrapped by iterable

- **`MAST_tools/MAST_dataset.py::MastDataset`** — `torch.utils.data.Dataset`
  (**map-style**), indexed by **shot** (not window/sample). `__len__` is the
  number of shots; `__getitem__(idx)` returns one shot's raw signals as a
  dict (schema below). Construction:

  ```python
  MastDataset(
      local: bool,
      shots_list: list[int],
      source_signal_list: list[list[str] | tuple[str]],   # [[source, signal], ...]
      signal_level_transform_map: Optional[Mapping[str, Callable]] = None,
      remove_outliers: bool = False,
      outlier_metadata_file: str = RANDOM_SPLIT_OUTLIER_METADATA_FILE,
      remove_bad_efit_rating: bool = False,
      store_manager_settings: StoreManagerParametersType | None = None,
      verbose: bool = False,
  )
  ```

- **`tokamark/tools/TokaMark_dataset.py::TokaMarkDataset`** —
  `torch.utils.data.IterableDataset` (**iterable**, streaming), wraps a
  `MastDataset` instance and yields **windows**, not shots. Worker-safe:
  `__iter__` shards shots across `torch.utils.data.get_worker_info()`
  workers, then optionally streams through a shuffle buffer
  (`_shuffle_buffer`, default `shuffle_buffer_size=512`). One shot can yield
  zero, one, or many window dicts (a shot with too-short a usable time range
  yields none — `_process_shot` returns early). Construction:

  ```python
  TokaMarkDataset(
      base_dataset: MastDataset,
      task_metadata: Mapping[str, Any],       # from tokamark.tasks.get_task_metadata
      config_metadata: Mapping[str, Any],     # the raw task config dict (task_type, task_window_segmenter, ...)
      custom_transform: Callable | None = None,
      test_mode: bool = False,
      shuffle_windows: bool = False,
      shuffle_buffer_size: int = 512,
      verbose: bool = False,
  )
  ```

  Both dataset classes are exercised together via
  `tokamark/data.py::initialize_MAST_dataset(...) -> MastDataset` and
  `tokamark/data.py::initialize_TokaMark_dataset(dataset, task_metadata,
  config_metadata, ...) -> TokaMarkDataset | None`.

### Sample dict schema

**`MastDataset.__getitem__`** returns `dict[str, dict[str, np.ndarray]]`,
keyed `"{source}-{signal}"` (e.g. `"summary-ip"`), each value:

```python
{"time": np.ndarray, "values": np.ndarray}
```

`values` is `(1, T)` or `(C, T)` (channel-first, time last); `time` is
`(T,)`. A signal missing entirely for a shot (outlier-removed, absent
source, or fetch error) becomes `{"time": np.array([]), "values":
np.array([])}` — empty arrays, not `None` and not NaN-filled at this layer.

**`TokaMarkDataset`** (via `_process_shot`) yields, per window:

```python
{
    "shot_id": int,
    "window_index": int,
    "input": {key: {"time": np.ndarray, "values": np.ndarray}, ...},
    "actuator": {key: {...}, ...},
    "output": {key: {...}, ...},
    "t_cut": float,
}
```
(`custom_transform`, if given, can replace the `input`/`actuator`/`output`
sub-dicts entirely before the final yield — `tokamark/tools/
TokaMark_dataset.py:471-475`.) `input`/`actuator`/`output` key sets come from
the task config's `task_window_segmenter.{input,actuator,output}_keys`.

### Mask / imputation semantics: NaN sentinel, no separate boolean mask

There is **no boolean mask array** anywhere in the sample schema. Missing
data is represented purely by sentinel values:

- Whole-signal-missing (shot level): empty `time`/`values` arrays (see
  above), which `TokaMarkDataset._build_window`
  (`tokamark/tools/TokaMark_dataset.py:525-527`) turns into
  `np.full(..., np.nan)`-filled windows of the expected shape.
- Partial-missing (interval padding): `_pad_time_series_to_interval`
  (`tokamark/tools/TokaMark_dataset.py:576-635`) left/right-pads a signal's
  time series to the shot's global `[t_start, t_end]` window with
  `np.full(pad_shape, np.nan)`.
- `MastDataset.__getitem__`'s EFIT-quality gate
  (`MAST_tools/MAST_dataset.py:271-318`) also injects NaN via
  `np.where(mask_expanded, np.nan, shot_vals)` for timesteps with a bad
  `ip_rating`.
- Optional imputation is a **transform**, not a dataset feature:
  `tokamark/tools/transforms/fill_profile_with_zeros_imputer_transform.py::
  FillProfileWithZerosTransform` zero-fills NaN entries, but explicitly
  *skips* columns that are all-NaN (whole-channel-missing) — those stay NaN.
  Composed into per-signal transforms via
  `tokamark/tools/MAST_composite_transform.py::build_common_signal_transform_map(
  source_signal_list, use_std_scaling=True, use_nan_filling=True,
  stats_metadata_file_path=...)`.
- Test-mode window filtering (`_process_shot`, same file, ~455-465) checks
  NaN fractions directly via `_all_vars_have_all_nans` /
  `_any_vars_have_any_nans` helpers (module-level, same file) — again no
  mask array, just `np.isnan(...)` on `values`.
- `tokamark/evaluator.py::compute_windows_metrics` computes RMSE/MAE with
  `np.nanmean`, and separately reports `nan_fraction` per window
  (`nan_mask = np.isnan(y_target)`), i.e. the evaluator treats NaN as the
  mask at metric-computation time too.

**Adapter implication for Task 1.5:** if Fusion-JEPA's `FusionBatch` wants an
explicit boolean mask tensor, it must be *derived* (`~np.isnan(values)` or
equivalent) at adapter time — tokamark does not hand one over.

### Local vs. S3 store selection

`MAST_tools/utils/store_utils.py::MASTStorageManager` is the single
selection point. Constructor:

```python
MASTStorageManager(
    base_fsspec_protocol: str = "simplecache",
    target_fsspec_protocol: str = "s3",
    s3_endpoint_url: str = "https://s3.echo.stfc.ac.uk",
    s3_mast_dataset_path: str = "/mast/tokamark/v1",
    base_local_zarr_path: str | None = "/mast/tokamark/v1",  # placeholder; real installs override this
)
```

Selection is **per-call**, via a `local: bool` flag threaded through
`ShotInfo` (`MAST_tools/utils/data_utils.py::ShotInfo` — a `TypedDict` with
`shot_id: int` and `local: NotRequired[bool]`, default `False` = remote) —
not a single mode fixed at manager-construction time:

```python
make_shot_store(shot_info: ShotInfoType, verbose: bool = False) -> ZarrStoreType
```
(`MAST_tools/utils/store_utils.py:658-694`)
- `local=True` → `zarr.storage.LocalStore(root=f"{base_local_zarr_path}/{shot_id}.zarr")`.
- `local=False` (default) → `zarr.storage.FsspecStore(fs=self.fs_remote_s3fs,
  read_only=True, path=f"{s3_mast_dataset_path}/{shot_id}.zarr")`, where
  `fs_remote_s3fs = s3fs.S3FileSystem(anon=True, endpoint_url=s3_endpoint_url,
  asynchronous=True)` — anonymous access, matching `manifests/upstream.yaml`'s
  `tokamark_dataset.s3_endpoint` (`https://s3.echo.stfc.ac.uk`).

`MastDataset.__init__` takes its own top-level `local: bool` (constant for
the whole dataset instance) and passes `ShotInfo(shot_id=..., local=self.local)`
per `__getitem__` call (`MAST_tools/MAST_dataset.py:237-239`) — so in
practice one `MastDataset` is homogeneously local-or-remote, but the
underlying storage manager supports per-shot overrides if needed.
`store_manager_settings` (a `StoreManagerParametersType` = `Mapping[str,
Any]`, validated as "only recognized keys applied") flows from
`initialize_MAST_dataset(...)` all the way down to `MASTStorageManager.__init__`
kwargs, so all five constructor fields above (endpoint URL, dataset path,
local root, fsspec protocols) are adapter-overridable without subclassing.

### `WindowMetricsAccumulator` — brief's method name is wrong; real name is `add_batch`

The brief anticipated a `.update(...)` method; the actual API
(`tokamark/evaluator.py:122-213`) is:

```python
class WindowMetricsAccumulator:
    def __init__(self, task: str) -> None: ...
    def add_batch(
        self,
        y_target: np.ndarray,
        y_pred: np.ndarray,
        shot_ids: Union[np.ndarray, torch.Tensor],
        window_indices: Union[np.ndarray, torch.Tensor],
        feature_name: str,
    ) -> None: ...
    def is_empty(self) -> bool: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

One `add_batch` call is **one feature at a time** (`feature_name` is a
single string, not a batch of names) — the caller loops over features per
model batch. `add_batch` appends a chunk computed by module-level
`compute_windows_metrics(y_target, y_pred, shot_ids, window_indices,
feature_name) -> pd.DataFrame` (RMSE/MAE via `np.nanmean`, plus
`nan_fraction`), so nothing is aggregated until `to_dataframe()`/
`compute_metrics()` is called.

**`compute_metrics` flow** (`tokamark/evaluator.py:541-642`):

```python
compute_metrics(
    task: str,
    output_dir: Union[Path, str],
    window_metrics_accumulator: WindowMetricsAccumulator,
    save_windows_metrics: bool = False,
    save_shot_metrics: bool = False,
    save_task_metrics: bool = True,
) -> pd.DataFrame   # indexed by feature_name, includes a `task`-named summary row
```

1. Validates `window_metrics_accumulator` is a non-`None`
   `WindowMetricsAccumulator` whose `.task` matches the `task` arg (raises
   `TypeError`/`ValueError` otherwise), and that its `.to_dataframe()` has
   the required columns (`shot_id`, `window_index`, `feature_name`, `RMSE`,
   `MAE`, `nan_fraction`) and is non-empty.
2. `aggregate_windows_metrics(df)` — window rows → per-(shot, feature) means
   (RMSE via mean-of-MSE then sqrt) → per-feature signal rows (mean + pop-std
   across shots, normalized to `NRMSE`/`NMAE` by dividing by each signal's
   `std` from `get_signals_metadata()`) → per-shot task rows (mean across
   features within a shot).
3. `_build_task_metrics_df` appends one task-summary row (named after
   `task`) to the per-feature signal rows.
4. Conditionally writes `windows_metrics.csv` / `shots_metrics.csv` /
   `task_metrics.csv` under `<output_dir>/<task>/`.
5. Separately, `compute_summary_metrics(output_dir, source="task_metrics"|
   "windows_metrics"|"shots_metrics")` rolls every task in
   `tokamark.tasks.GROUP_TASKS` up into `signals_metrics.csv` /
   `groups_metrics.csv` (equal-weight mean of task means/stds per group).

### The 14 task configs

From `tokamark/tasks.py::GROUP_TASKS` / `TASKS_CONFIGS_MAP` (verified count:
`len(TASKS_CONFIGS_MAP) == 14`, loaded per-task by
`test_all_14_task_configs_loadable`):

| Group | Tasks |
|---|---|
| `group_1_reconstruction` | `task_1-1`, `task_1-2`, `task_1-3` |
| `group_2_magnetics_dynamics` | `task_2-1`, `task_2-2`, `task_2-3` |
| `group_3_profiles_dynamics` | `task_3-1`, `task_3-2`, `task_3-3` |
| `group_4_mhd_activity` | `task_4-1`, `task_4-2`, `task_4-3`, `task_4-4`, `task_4-5` |

Each config YAML (e.g. `tokamark/tasks_configs/group_1_reconstruction/
task_1-1.yaml`) has top-level keys `task_name`, `task_type` (`"markovian"` |
presumably `"non_markovian"` — both branches exist in
`TokaMarkDataset._build_window`), `sources_and_signals` (`input_name`/
`actuator_name`/`output_name`, each a list of `[source, signal]` pairs or
`null`), `task_window_segmenter` (`input_keys`/`actuator_keys`/`output_keys`
YAML-anchored back to the same lists, plus `input_length`/`output_length`/
`delta` in seconds), and `stride_window` (seconds). `get_task_config(task_name)`
(`tokamark/tasks.py:43-62`) just resolves `TASKS_CONFIGS_MAP[task_name]`
against `tokamark.tools.path.TASKS_CONFIGS_DIR` and loads the YAML — pure
package-resource read, no data/network access, which is why
`test_all_14_task_configs_loadable` can run as a unit test.

## Everything is provisional until Task 1.5

Per the brief: all of the above is read from the pinned source to unblock
Task 1.5's adapter design, not to lock in Fusion-JEPA's own `FusionBatch`
schema. Signatures are quoted verbatim with file paths so Task 1.5 can be
reviewed against this record.
