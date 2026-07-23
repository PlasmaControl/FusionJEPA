#!/bin/bash
# E01 -- FULL paper-scale reproduction of the official TokaMark CNN baseline.
#
# Runs the PINNED, UNMODIFIED tokamark_baseline (OS_v2.0, commit 1e7c489) stock
# entry points -- run_training.py then run_evaluation.py -- at production scale
# (config_e01_production.yaml = the baseline's own config_model.yaml with only
# the data + output paths repointed at this machine: all shots, batch 512,
# max_steps 1e7 / patience 10). This is what a real "reproduce the paper's
# headline CNN numbers" run requires; the login-node run_e01_scaled.py is only a
# bounded interim. It became runnable once the local mirror hit 100% coverage
# (the stock fixed-split shot ordering starts on shots that were downloading).
#
# NOT a DDP job. The CNN baseline's BatchStepTrainer is single-device, so scale
# comes from running many independent per-task jobs in parallel, not one big
# distributed job. This is a SLURM ARRAY: one array element per task, one GCD
# used per element (7 idle -- the operator may pack tasks per node as a
# refinement). Multiple seeds = resubmit with SEEDS overridden.
#
# The account is intentionally NOT hard-coded (never commit a real project
# account -- same rule as slurm/smoke.slurm and configs/cluster/*.yaml). Submit
# from the repo root, supplying account + array range explicitly:
#
#   # all 14 tasks, seed 23, VAL eval (test-split NOT touched):
#   sbatch -A "$SBATCH_ACCOUNT" --array=0-13 slurm/e01_baseline.slurm
#
#   # official TEST-split numbers (the paper protocol) -- deliberate + audited:
#   ALLOW_TEST=1 sbatch -A "$SBATCH_ACCOUNT" --array=0-13 slurm/e01_baseline.slurm
#
#   # a single task (index into the TASKS list below), e.g. task_2-3 = index 5:
#   sbatch -A "$SBATCH_ACCOUNT" --array=5 slurm/e01_baseline.slurm
#
# Env knobs: SEED (default 23), ALLOW_TEST (default 0 -> train only; the stock
# run_evaluation.py evaluates the TEST split, so it runs ONLY when ALLOW_TEST=1),
# BASELINE_ROOT, E01_CONFIG (override config path).
#
#SBATCH -J e01_baseline
#SBATCH -o logs/%A_%a_e01.out
#SBATCH -e logs/%A_%a_e01.err
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionJEPA repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# The 14 official TokaMark tasks, indexed by SLURM_ARRAY_TASK_ID.
TASKS=(task_1-1 task_1-2 task_1-3 \
       task_2-1 task_2-2 task_2-3 \
       task_3-1 task_3-2 task_3-3 \
       task_4-1 task_4-2 task_4-3 task_4-4 task_4-5)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#TASKS[@]}" ]; then
    echo "ERROR: array index ${IDX} out of range 0-$(( ${#TASKS[@]} - 1 ))." >&2
    exit 1
fi
TASK="${TASKS[$IDX]}"
SEED="${SEED:-23}"
ALLOW_TEST="${ALLOW_TEST:-0}"

BASELINE_ROOT="${BASELINE_ROOT:-/lustre/orion/fus187/scratch/nchen/reference/tokamark_baseline}"
E01_CONFIG="${E01_CONFIG:-/lustre/orion/fus187/proj-shared/mast/e01_baseline_repro/config/config_e01_production.yaml}"
if [ ! -f "${BASELINE_ROOT}/run_training.py" ] || [ ! -f "${E01_CONFIG}" ]; then
    echo "ERROR: baseline clone or config not found (BASELINE_ROOT=${BASELINE_ROOT}, E01_CONFIG=${E01_CONFIG})." >&2
    exit 1
fi

echo "[e01] task=${TASK} seed=${SEED} allow_test=${ALLOW_TEST}"
echo "[e01] baseline=${BASELINE_ROOT}"
echo "[e01] config=${E01_CONFIG}"

# Frontier module loads (rocm/PrgEnv/RCCL etc.); harmless for a single-GPU job.
# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

# The stock scripts do `import globals` / `from src... import ...` relative to
# the clone root, and read data via the pinned tokamark package -> run under the
# repo's pixi `frontier` env (torch 2.10 + rocm7.1 + installed tokamark) with the
# clone root on PYTHONPATH so its own modules resolve regardless of CWD.
export PYTHONPATH="${BASELINE_ROOT}:${PYTHONPATH:-}"

echo "[e01] === TRAINING (stock run_training.py, unmodified) ==="
pixi run -e frontier --frozen python "${BASELINE_ROOT}/run_training.py" \
    --task "${TASK}" --config "${E01_CONFIG}" --seed "${SEED}" \
    --model cnn --split random
TRAIN_RC=$?
if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "[e01] training FAILED (exit ${TRAIN_RC})" >&2
    exit "${TRAIN_RC}"
fi
echo "[e01] training complete."

# The stock run_evaluation.py evaluates the TEST split (the paper's official
# protocol). Per the project test-split rule this runs ONLY on explicit opt-in,
# so test access stays deliberate and auditable (M4 acceptance criterion).
if [ "${ALLOW_TEST}" = "1" ]; then
    echo "[e01] === EVALUATION (stock run_evaluation.py, TEST split) ==="
    pixi run -e frontier --frozen python "${BASELINE_ROOT}/run_evaluation.py" \
        --task "${TASK}" --config "${E01_CONFIG}" --seed "${SEED}" \
        --model cnn --split random
    EVAL_RC=$?
    if [ "${EVAL_RC}" -ne 0 ]; then
        echo "[e01] evaluation FAILED (exit ${EVAL_RC})" >&2
        exit "${EVAL_RC}"
    fi
    echo "[e01] TEST-split evaluation complete (official metrics written)."
else
    echo "[e01] ALLOW_TEST!=1 -> skipping the TEST-split evaluation." \
         "Trained checkpoint is written; rerun with ALLOW_TEST=1 for the" \
         "official test numbers, or evaluate VAL via run_e01_scaled.py."
fi

echo "[e01] task=${TASK} seed=${SEED} DONE."
