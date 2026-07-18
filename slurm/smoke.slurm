#!/bin/bash
# Fusion-JEPA Frontier DDP smoke test (Task 2.13).
#
# Runs on ONE Frontier node across all 8 GCDs and proves the real distributed
# training path end to end: (1) the pixi `frontier` env activates and the unit
# suite is green, then (2) an 8-rank srun job that verifies rank/device
# uniqueness, an all-reduce checksum, and a ~20-step Trainer.fit() of the
# `raw_predictor_small` model on synthetic data (see tests/frontier/smoke_ddp.py).
# This replaces the M0 scaffold-only smoke now that M2 has a model + training loop.
#
# The account is intentionally NOT hard-coded (never commit a real project
# account -- see scripts/slurm_frontier/_frontier_common.sh and
# configs/cluster/*.yaml for the same rule). The operator supplies it at submit
# time from the repo root:
#
#   sbatch -A "$SBATCH_ACCOUNT" slurm/smoke.slurm
#
#SBATCH -J fusion_jepa_smoke
#SBATCH -o logs/%j_smoke.out
#SBATCH -e logs/%j_smoke.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
# Deliver SIGUSR1 to the BATCH shell (the `B:` prefix) 300s before the walltime
# limit, so the trap below can forward it to the srun step; the Trainer's
# SIGUSR1 handler then saves latest.pt + completion.json and reports "preempted".
#SBATCH --signal=B:USR1@300
set -uo pipefail

# SLURM stages the submit script under /var/spool/slurmd/..., so BASH_SOURCE
# can't locate the repo -- use SLURM_SUBMIT_DIR instead (same pattern as
# scripts/slurm_frontier/build_dataset_cache.sh). Submit from the repo root.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionJEPA repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
# SLURM opens logs/%j_smoke.{out,err} BEFORE this script runs, so `logs/` must
# already exist at submit time; recreate it here too for a manual `bash` run.
mkdir -p logs

# Set MASTER_PORT before sourcing _frontier_common.sh (it only fills in if unset).
export MASTER_PORT="${MASTER_PORT:-29500}"

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

# The srun payload imports `tests.fixtures` and `fusion_jepa`; running a bare
# `python tests/frontier/smoke_ddp.py` puts tests/frontier (not the repo root)
# on sys.path, so add the repo root explicitly. (smoke_ddp.py also self-inserts
# it, but exporting here keeps any sub-import robust.)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── Resource shape (from the SLURM allocation, never hard-coded) ────────────
NODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 8))}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-7}"

echo "[smoke] step 1/2: unit test suite (pixi run -e frontier --frozen test-unit)"
# --frozen trusts pixi.lock and skips the metadata refresh that otherwise hits
# the rattler cache on Lustre (slow/contended, and a source of env-activation
# hangs under concurrent SLURM jobs). All packages are already installed under
# .pixi/envs/frontier/, so the refresh adds nothing at job runtime.
#
# This shell runs with `set -uo pipefail` but NOT `-e` (the trap-safe wait loop
# below needs to survive `wait` being interrupted), so a red preamble would NOT
# abort on its own -- capture its real exit code explicitly and fail the job here
# rather than marching on into the DDP payload with a broken env/suite.
pixi run -e frontier --frozen test-unit
PREAMBLE_RC=$?
if [ "${PREAMBLE_RC}" -ne 0 ]; then
    echo "[smoke] unit-suite preamble FAILED (exit ${PREAMBLE_RC}); aborting" \
         "before the DDP payload." >&2
    exit "${PREAMBLE_RC}"
fi

echo "[smoke] step 2/2: 8-rank DDP payload (${TOTAL_RANKS} ranks on ${NODES} node)"
echo "[smoke] master=${MASTER_ADDR}:${MASTER_PORT}"

# Forward SIGUSR1 (from the #SBATCH --signal above, or `scancel --signal=USR1
# <jobid>`) to the srun step so its ranks checkpoint-and-exit. NOTE: reliable
# signal delivery to srun-launched ranks is a KNOWN VERIFICATION ITEM on this
# stack; if the forwarded USR1 does not reach the ranks, the manual fallback is
# `scancel --signal=USR1 <jobid>` (which srun relays to the tasks directly).
SRUN_PID=""
_forward_usr1() {
    echo "[smoke] received SIGUSR1; forwarding to srun step (pid=${SRUN_PID:-none})"
    [ -n "${SRUN_PID}" ] && kill -USR1 "${SRUN_PID}" 2>/dev/null || true
}
trap _forward_usr1 USR1

# Run srun in the background so the batch shell stays responsive to the trap,
# then wait for it -- re-waiting if `wait` is interrupted by the trap so we
# still capture srun's real exit code.
srun -N "${NODES}" -n "${TOTAL_RANKS}" -c "${CPUS_PER_TASK}" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     tests/frontier/smoke_ddp.py &
SRUN_PID=$!

RC=0
while true; do
    if wait "${SRUN_PID}"; then
        RC=0
        break
    fi
    RC=$?
    # `wait` returns >128 when interrupted by a trap; keep waiting while the
    # step is still alive so the recorded RC is srun's actual exit code.
    if [ "${RC}" -le 128 ] || ! kill -0 "${SRUN_PID}" 2>/dev/null; then
        break
    fi
done

if [ "${RC}" -ne 0 ]; then
    echo "[smoke] DDP payload FAILED (srun exit ${RC})" >&2
    exit "${RC}"
fi

echo "[smoke] all checks passed"
