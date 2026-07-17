# M1 Closure — Public Data Contract (TokaMark)

Date: 2026-07-17. Branch: dev-peter.

## Acceptance status

| Item | Status |
|---|---|
| Upstream pinned + API decision record (0001) | Done — tokamark @ `1a200f45…` (OS_v2.0), reviewer-verified against installed source |
| FusionBatch contract + validator | Done — dtype-checked, mask-robust, ramp fixture |
| Split integrity + train-only normalization | Done — manifest-membership enforced in `fit_normalization` |
| Acquire script + full dataset pull | Script done; 564 GB pull RUNNING to `/lustre/orion/fus187/proj-shared/mast/tokamark/v1` |
| TokaMark adapter (official split, metrics, leakage guard) | Done — `manifests/splits/tokamark_official.yaml` committed (8963/1115/1110) |
| Signal registry v1 (37 group-2/3 signals) | Done — all `pending_physics_review`, none `shared_core` |
| Dev-subset builder | Done — real build is an operator step after the pull lands |
| `inspect_data` end-to-end (M1 acceptance artifact) | PASSED live: real S3 batch → validated → summarized → 10 plots → completion succeeded |

## Open escalations

1. **Imputation semantics (R3, unresolved upstream):** tokamark does not distinguish imputed from observed values; `use_nan_filling=False` disables its one imputing transform, but `ReshapeLcfsTransform` unconditionally interpolates `equilibrium-lcfs_r/z`, and pre-ingestion imputation of the Zarr store is unknowable from source. Details + adjudications: `docs/decisions/0002-tokamark-adapter-semantics.md`. Any task consuming LCFS signals must treat them as partially synthetic.
2. **Frontier I/O benchmark (remote-vs-local Zarr throughput):** deferred to the M2 boundary per plan; the training loop (Task 2.12) should carry a data-wait logging hook so the comparison is measurable once local data lands.
3. **Physics review of the signal registry:** external gate; all entries remain `pending_physics_review`.

## Pending (gated on the running download)

- `manifests/datasets/tokamark_v1.yaml` — written by the acquire run on completion; commit then.
- Dev-subset build + its manifest — run `scripts/build_dev_subset.py` against the local store; commit manifest.

## Data realities discovered (carry into M2)

- Some validation windows have 100%-masked targets (`equilibrium.psi`) and a 100%-masked fused action channel — objectives must handle zero-observed targets without NaN losses.
- Context signals arrive at differing native lengths — tokenizers must not assume a shared time axis.
