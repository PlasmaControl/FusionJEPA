#!/bin/bash
# Fusion-JEPA M0 smoke test (minimal scaffold-validation job, Task 0.9).
#
# Proves, on an actual Frontier compute node, that: (1) the pixi `frontier`
# env activates and the unit test suite is green, and (2) the CLI config
# resolution path works end to end via a dry-run (no real training, no
# GPUs touched). This is deliberately NOT the full 8-GPU rank-uniqueness
# smoke described in CODING_AGENT_PLAN.md -- that lands with M2's Task 2.13
# once there is an actual model/training loop to exercise.
#
# The account is intentionally NOT hard-coded below (never commit a real
# project account -- see scripts/slurm_frontier/_frontier_common.sh and
# configs/cluster/*.yaml for the same rule). The operator supplies it via
# env var at submit time:
#
#   sbatch -A "$SBATCH_ACCOUNT" slurm/smoke.slurm
#
#SBATCH -J fusion_jepa_smoke
#SBATCH -o logs/%j_smoke.out
#SBATCH -e logs/%j_smoke.err
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
set -euo pipefail

# SLURM stages the submit script under /var/spool/slurmd/..., so
# BASH_SOURCE can't be used to locate the repo -- use SLURM_SUBMIT_DIR
# instead (same pattern as scripts/slurm_frontier/build_dataset_cache.sh
# and make_processing_stats.sh). Submit from the repo root.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionJEPA repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

echo "[smoke] step 1/2: unit test suite (pixi run -e frontier test-unit)"
pixi run -e frontier test-unit

# NOTE on cluster selection: mast_smoke.yaml points at `cluster: local` by
# default. A bare `cluster=frontier` dotlist override does NOT work here --
# resolve_config only special-cases the `experiment=` key for a full
# profile swap (config.py's merge order); `cluster` is already a resolved
# ClusterConfig by the time CLI dotlist overrides are applied, so trying to
# merge the string "frontier" into it fails
# (Merge error: str is not a subclass of ClusterConfig). Verified on the
# login node 2026-07-16. Overriding every `cluster.<field>` individually
# would work but duplicates configs/cluster/frontier.yaml and can drift, so
# this smoke step intentionally runs the dry-run against the default
# `local` profile -- it only needs to prove the CLI/config plumbing works,
# not exercise Frontier-specific paths. A `cluster=<name>` profile-swap
# override is a config.py enhancement to consider for a later milestone.
echo "[smoke] step 2/2: CLI config dry-run (inspect_data, mast_smoke experiment)"
python -m fusion_jepa.cli.inspect_data experiment=mast_smoke --dry-run

echo "[smoke] all checks passed"
