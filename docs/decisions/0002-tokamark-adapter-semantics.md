# 0002 — `data/tokamark.py` adapter semantics; R2/R3 outcomes

**Status:** accepted. **Date:** 2026-07-17. **Task:** 1.5.

## Decision

`src/fusion_jepa/data/tokamark.py` is the single seam between the pinned
upstream `tokamark`/`MAST_tools` benchmark (Task 1.1, commit
`1a200f4588addaad2ae3d53d438d9e1946c09294`) and Fusion-JEPA's
`FusionSample`/`FusionBatch` contract (Task 1.2). All upstream touches go
through one module-level shim, `_upstream` (a `_UpstreamShim` instance whose
`__getattr__` lazily imports and caches real symbols, letting tests
monkeypatch attributes wholesale with fakes). Public surface, matching the
Task 1.5 brief:

- `assert_pinned_upstream()` — fail-closed drift check against
  `manifests/upstream.yaml`, comparing the installed dist's PEP 610
  `direct_url.json` commit.
- `load_task_config(task_id) -> TaskConfig` — deep-copied config +
  `source_path`/`source_sha256` provenance of the packaged task YAML.
- `official_split(name="tokamark_official") -> SplitManifest` — built from
  `tokamark.tools.path.RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE` (packaged
  `TokaMark_data_splits.csv`), `source_hash` = sha256 of that CSV.
- `to_fusion_sample(upstream_item, task_cfg, split, registry, *,
  shot_time_range=None) -> FusionSample` — pure translation, no I/O.
- `make_dataset(task_id, split, *, cluster, data_cfg, normalization=None) ->
  TokamarkWindowDataset` — builds `MastDataset` + `TokaMarkDataset` via
  `tokamark.data.initialize_MAST_dataset`/`initialize_TokaMark_dataset`,
  `use_std_scaling=False`, `use_nan_filling=False` always (see R3 below);
  storage (local Lustre vs. anonymous S3) is derived from the `cluster`
  profile's `tokamark_root`/`tokamark_storage_options`.
- `OfficialMetrics` — thin wrapper over `tokamark.evaluator
  .WindowMetricsAccumulator`/`compute_metrics`.

## Upstream API verification (against the installed source, this task)

Every symbol the shim resolves was re-checked against the pinned installed
tree (`.pixi/envs/frontier/lib/python3.11/site-packages/{tokamark,
MAST_tools}/`), beyond what 0001 already verified:

- `tokamark/data.py::initialize_MAST_dataset(config_task, shots_list,
  local_flag=True, use_std_scaling=True, stats_metadata_file_path=...,
  use_nan_filling=False, remove_outliers=True, outlier_metadata_file=...,
  remove_bad_efit_rating=True, *, store_manager_settings=None,
  verbose=False) -> MastDataset` — the adapter's keyword call
  (`config_task=`, `shots_list=`, `local_flag=`, `use_std_scaling=False`,
  `use_nan_filling=False`, `store_manager_settings=`) matches exactly.
- `tokamark/data.py::initialize_TokaMark_dataset(dataset, task_metadata,
  config_metadata, custom_transform=None, test_mode=False,
  shuffle_windows=True, shuffle_buffer_size=512, *, verbose=False) ->
  TokaMarkDataset | None` — adapter passes `shuffle_windows=False`
  explicitly (deterministic order for eval reproducibility; upstream's own
  default is `True`).
- `tokamark/tasks.py::get_task_config`/`TASKS_CONFIGS_MAP`/`GROUP_TASKS`/
  `get_task_metadata(config_task, verbose=False)` — signatures match.
- `tokamark/tools/path.py::TASKS_CONFIGS_DIR`,
  `RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE` — both are plain
  `os.path.join(PACKAGE_ROOT_DIR, ...)` constants, package-resource reads,
  no network.
- `tokamark/data_split.py::read_data_split_csv` reads exactly the
  `shot_id`/`train`/`val`/`test` boolean columns `official_split()` assumes.
- `tokamark/evaluator.py::WindowMetricsAccumulator.add_batch(y_target,
  y_pred, shot_ids, window_indices, feature_name)`,
  `.is_empty()`, `.to_dataframe()`, and `compute_metrics(task, output_dir,
  window_metrics_accumulator, save_windows_metrics=False,
  save_shot_metrics=False, save_task_metrics=True) -> pd.DataFrame` (indexed
  by `feature_name`, includes `NRMSE_mean`/`NMAE_mean`/... columns and a
  task-named summary row) — all confirmed exactly as 0001 recorded.
- `MAST_tools/utils/store_utils.py::MASTStorageManager.__init__` accepts
  `base_fsspec_protocol`, `target_fsspec_protocol`, `s3_endpoint_url`,
  `s3_mast_dataset_path`, `base_local_zarr_path` — the keys
  `_storage_from_cluster()` emits into `store_manager_settings` are a subset
  of these, confirmed against both `configs/cluster/local.yaml` (S3, anon)
  and `configs/cluster/frontier.yaml` (local Lustre mirror path).

No drift from 0001's record was found; no shim assumption needed correction.

## R2 — map-vs-iterable: confirmed, isolated as designed

`MastDataset` (map-style, indexed by shot) is wrapped by `TokaMarkDataset`
(`IterableDataset`, yields windows, worker-sharded). `TokamarkWindowDataset`
(this adapter) never indexes by sample position — it iterates the upstream
window stream and keys everything on the `shot_id` upstream yields per
window. Split-leakage is a **fail-closed error**, not a silent filter: any
`shot_id` outside `allowed_shots` raises `ValueError` immediately (loud
break, matching FusionBatch's leakage-must-be-impossible design goal from
0001/Task 1.2). This was verified as load-bearing, not just plausible, by a
mutation test this task (see task-1.5-report.md's TDD Evidence section):
disabling the guard made `test_requesting_shot_outside_split_impossible`
fail exactly as expected, then re-enabling restored green.

Per-shot time bounds (`shot_time_ranges`, required by `FusionBatch`
collation so every window of a shot reports byte-identical bounds) are
computed once per shot from the underlying `MastDataset` (map-style, cheap
random access) and cached — a documented one-extra-read-per-shot cost on the
S3 dev path, free on Frontier's local mirror.

## R3 — imputation vs. missing: escalated, not solved (per brief)

0001 already established there is no upstream boolean mask; everything is
NaN sentinels, and any zero-imputation is a **transform**, not a dataset
feature (`FillProfileWithZerosTransform`, gated by
`use_nan_filling`). This task's adapter always calls
`initialize_MAST_dataset(..., use_nan_filling=False, use_std_scaling=False,
...)`, so:

- `FillProfileWithZerosTransform` is never composed into the per-signal
  transform map — confirmed by reading
  `tokamark/tools/MAST_composite_transform.py::build_common_signal_transform_map`:
  `maybe_nan_filling()` returns `[]` when `use_nan_filling=False`. Partial
  per-timestep NaNs in `magnetics-*`/`thomson_scattering-*` profiles
  therefore reach the adapter as real NaN, not silently zero-filled.
- `StdScalingTransform` is likewise never composed
  (`use_std_scaling=False`), so no rescaling happens before our own
  optional `Standardize` (pretraining-only, per Task 1.3/1.5 design rule).

**New finding this task (not in 0001): two transforms are unconditionally
applied regardless of `use_nan_filling`/`use_std_scaling`**, discovered by
reading `MAST_composite_transform.py`'s per-variable overrides:

- `ClipXPointTransform` (`equilibrium-x_point_r`, `equilibrium-x_point_z`):
  replaces values `> 2` or `< -2` with `NaN` unconditionally. This is
  benign for the adapter — it *produces* NaN (which our
  `isfinite`-derived mask correctly turns into `False`), it does not
  fabricate an observed-looking value.
- `ReshapeLcfsTransform` (`equilibrium-lcfs_r`, `equilibrium-lcfs_z`):
  unconditionally strips NaNs per timestep and **resamples the remaining
  valid channels to a fixed 170-length profile via linear interpolation**
  (`scipy.ndimage.zoom`, `order=1`). This means the values our adapter
  receives for these two signals are never "raw" in the sense of untouched
  sensor readings — they are already a fixed-length interpolated
  reconstruction, and genuinely-missing vs. interpolated-filler is
  **not distinguishable post-transform** for `lcfs_r`/`lcfs_z` specifically
  (a fully-missing timestep still yields an all-NaN 170-vector, but a
  partially-missing timestep's gaps are invisibly interpolated over before
  the adapter ever sees them). Both signals are registered
  (`mast.equilibrium.lcfs_r`/`lcfs_z`) but are not part of the group-2/3
  task configs the adapter is exercised against in this task's tests; flag
  for whoever first builds a task/dataset config that references them.

**The genuinely open R3 risk remains unresolved, as the brief anticipated**:
whether the raw MAST zarr store itself (populated by an ingestion pipeline
outside the pinned `tokamark`/`MAST_tools` Python source tree — the
data-generation process that built `s3://mast/tokamark/v1`) already
contains imputed/interpolated values indistinguishable from true sensor
readings, is **not answerable by reading the pinned benchmark code**; it
would require provenance from the dataset publishers themselves. This
adapter does not attempt to solve it — consistent with the brief's
instruction to escalate rather than force a false resolution. Downstream
consumers (M2+) should treat every `mask=True` entry as "not NaN in the
pinned MAST zarr store", not as a hard guarantee of true sensor origin.

## Time units: seconds throughout, no rescaling

Task-config lengths (`input_length`, `output_length`, `delta`,
`stride_window`) and all upstream `time` arrays are already **seconds**
(verified against `tokamark/tasks.py::get_task_metadata` and
`TokaMark_dataset.py`'s `t_cut`-relative window construction — `t_cut_start
= global_start_time + input_length`, `t_cut_stop = global_end_time - delta -
output_length`, i.e. plain second-denominated arithmetic, no `ms`
conversion anywhere in the pinned source). The adapter applies **no unit
rescaling**; `FusionSample.horizon_seconds = target_times[-1] -
context_times[-1]`, `float64` seconds, exactly matching `delta +
output_length` from the task YAML. `test_horizons_reported_in_seconds_match_task_yaml_ms`
(brief-mandated name; the "_ms" is a misnomer given the lengths were never
ms) asserts this, and a mutation test this task (scaling by 1000×)
confirmed the test fails hard on a ms/s confusion.

## Reference-axis reconstruction (not verbatim upstream times)

Rather than reusing each window's per-signal `time` array verbatim, the
adapter rebuilds context/target/action axes from `(dt, n)` — the reference
signal's (first key in each `input`/`actuator`/`output` group) median
sample interval and length — anchored at `t_cut`. This keeps the
context/target boundary a single clean split (`context ends at t_cut`,
`target starts at t_cut + delta`) regardless of a coarse-rate target's
nearest-sample rounding, which could otherwise nudge a target sample to
land before `t_cut` and fail `FusionBatch`'s non-overlap invariant. This was
exercised, not just designed: `test_ramp_alignment_context_action_target`
checks float64 agreement to `atol=1e-12` against an independently
constructed ramp fixture, and the remote test
`test_real_batch_passes_dev_validator` runs the reconstructed axes from a
real S3 window through `validate_batch` end to end.

## Review adjudications (Task 1.5 review, controller-overruled findings)

Codex's Task 1.5 review raised four findings. Two ("assert_pinned_upstream
untested and bypasses the shim"; "blanket TokamarkAdapterError suppression
in `TokamarkWindowDataset.__iter__`") were accepted and fixed — see the
shim's `installed_commit` method and the `_UnusableReferenceWindowError`
subtype introduced above. The other two were overruled by the controller,
with the reasoning recorded here rather than silently dropped:

1. **"No ms→seconds conversion" — CORRECT as implemented, not a defect.**
   The review's premise was that the task YAML's `input_length`/
   `output_length`/`stride_window` might be milliseconds needing a ×1e-3
   (or the adapter needing a ×1e3) conversion. They are not: the pinned
   `task_2-3` YAML's actual values (`input_length: 0.005`,
   `output_length: 0.025`, `stride_window: 0.001`) are physically sensible
   only as **seconds** — a MAST discharge lasts on the order of
   0.01-1 s total, so a 0.005 ms (5 microsecond) input window or a 0.001 ms
   stride would be nonsensically fine-grained relative to any diagnostic's
   sample rate, whereas 5 ms/25 ms/1 ms are exactly the kind of windowing
   parameters this benchmark's magnetics/profile-dynamics tasks use. This
   is also independently confirmed by reading `TokaMark_dataset.py`'s
   `t_cut`-relative arithmetic directly (see "Time units" above) — no `ms`
   constant or `/1000`/`*1000` conversion exists anywhere in the pinned
   window-construction code. The existing
   `test_horizons_reported_in_seconds_match_task_yaml_ms` (asserting
   `horizon_seconds == delta + output_length` with **no** rescaling, plus
   the explicit `< 1.0` sanity bound) is the correct, intended oracle; no
   code change was made in response to this finding.
2. **`isfinite`-based mask derivation RETAINED** (supersedes 0001/this
   doc's earlier "`mask = ~isnan(values)`" wording, which described the
   pre-adapter upstream sentinel convention, not a normative statement
   about what the adapter itself must do). Marking `+-inf` as "observed"
   would violate `FusionBatch.validate_batch`'s finite-where-masked
   invariant the moment such a value reached a loss function — `inf` is
   therefore treated as missing **by design**, the same as `NaN`, not
   because upstream happens to only ever emit `NaN`. Locked by the new
   `test_infinity_treated_as_missing_like_nan` (an `+inf`/`-inf` window
   value produces `mask=False` + a finite placeholder, mirroring
   `test_upstream_missing_channel_becomes_false_mask_not_zero`'s NaN case).

**Standardize-never-in-official-eval remains a caller convention at this
layer, not a structural guarantee.** `make_dataset` always passes
`use_std_scaling=False` to the upstream constructors (structural, enforced
here), but whether a caller subsequently applies our own `Standardize`
transform on an official-metric path is presently only prevented by
`normalization=None` being the caller's responsibility to leave unset for
eval — `TokamarkWindowDataset` itself has no way to know "this call is for
official evaluation" and refuse a `normalization` argument on that basis.
Structural enforcement (e.g. an eval CLI that never accepts/threads a
`normalization` argument at all, or a dataset-level `official_eval: bool`
flag that hard-disables it) is deferred to whichever M2 task builds the
evaluate CLI; this adapter only documents the convention.

## Everything above is provisional until a downstream task exercises it further

Group-2/3 tasks (magnetics/profiles dynamics) are what this task's registry
(Task 1.6, 37 entries) and tests cover. The 8 group-1/4 task configs
(reconstruction, MHD activity) are loadable (`load_task_config`) but not
yet exercised end-to-end through `make_dataset`/`to_fusion_sample` against
real data — a gap for whichever M1/M2 task first trains against them.
