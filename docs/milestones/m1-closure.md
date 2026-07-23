# M1 Closure — Public Data Contract (TokaMark)

Date: 2026-07-17 (updated 2026-07-22). Branch: main.

## Acceptance status

| Item | Status |
|---|---|
| Upstream pinned + API decision record (0001) | Done — tokamark @ `1a200f45…` (OS_v2.0), reviewer-verified against installed source |
| FusionBatch contract + validator | Done — dtype-checked, mask-robust, ramp fixture |
| Split integrity + train-only normalization | Done — manifest-membership enforced in `fit_normalization` |
| Acquire script + full dataset pull | Done — 564 GB pull complete to `/lustre/orion/fus187/proj-shared/mast/tokamark/v1` (11,573/11,573 shots, 100% coverage), retrieved 2026-07-19; `manifests/datasets/tokamark_v1.yaml` committed |
| TokaMark adapter (official split, metrics, leakage guard) | Done — `manifests/splits/tokamark_official.yaml` committed (8963/1115/1110) |
| Signal registry v1 (37 group-2/3 signals) | Done — all `pending_physics_review`, none `shared_core` |
| Dev-subset builder | Done — verified dev subset built and committed as `manifests/datasets/dev_subset_v1.yaml` (train 92 / val 45 / test 47 shots) |
| `inspect_data` end-to-end (M1 acceptance artifact) | PASSED live: real S3 batch → validated → summarized → 10 plots → completion succeeded |

## Open escalations

1. **Imputation semantics (R3, unresolved upstream):** tokamark does not distinguish imputed from observed values; `use_nan_filling=False` disables its one imputing transform, but `ReshapeLcfsTransform` unconditionally interpolates `equilibrium-lcfs_r/z`, and pre-ingestion imputation of the Zarr store is unknowable from source. Details + adjudications: `docs/decisions/0002-tokamark-adapter-semantics.md`. Any task consuming LCFS signals must treat them as partially synthetic.
2. **Frontier I/O benchmark (remote-vs-local Zarr throughput):** deferred to the M2 boundary per plan; the training loop (Task 2.12) should carry a data-wait logging hook so the comparison is measurable now that the local store is complete.
3. **Physics review of the signal registry:** external gate; all entries remain `pending_physics_review`.

## Post-closure follow-ups (resolved)

- `manifests/datasets/tokamark_v1.yaml` — committed on pull completion (100% coverage, retrieved 2026-07-19).
- Dev-subset build + its manifest — built and committed as `manifests/datasets/dev_subset_v1.yaml` (train 92 / val 45 / test 47 shots).

## Data realities discovered (carry into M2)

- Some validation windows have 100%-masked targets (`equilibrium.psi`) and a 100%-masked fused action channel — objectives must handle zero-observed targets without NaN losses.
- Context signals arrive at differing native lengths — tokenizers must not assume a shared time axis.
