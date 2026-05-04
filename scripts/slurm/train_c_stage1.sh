#!/bin/bash
#SBATCH --job-name=c_stage1
#SBATCH --output=logs/%j_c_stage1.out
#SBATCH --error=logs/%j_c_stage1.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Phase C Stage 1 — single-step pretraining of TS + tangtv video.
#
# Mirror of train_e2e_stage1.sh with three additions:
#   --use_video tangtv       — adds the 300-token tangtv diagnostic in
#                              the diagnostic prefix
#   --init_checkpoint <Phase A best>
#                            — warm-starts TS+actuator weights from
#                              e2e_stage1_best.pt (Phase A Stage 1).
#                              Video tokenizer + head init from
#                              scratch (allowed_missing_prefixes
#                              accepts "diag_tokenizers.tangtv." and
#                              "diag_heads.tangtv.").
#   --freeze_backbone_steps 5000
#                            — backbone + TS modules + actuator
#                              tokenizers held fixed for 5 k steps so
#                              the freshly-initialised video tokenizer
#                              + head can find their feet without
#                              perturbing the Phase A-trained
#                              backbone. After 5 k steps the freeze
#                              releases and all params train.
#
# Same modality table as Phase A Stage 1 (8 diag + 9 actuator).
# Step budget: 336,000 steps = 10 epochs at batch 256. At 0.97 s/step
# (memory benchmark §17), wall ≈ 3.7 days, ~5 chained 24 h jobs.
#
# Output: runs/c_stage1/. Does not touch runs/e2e_stage1/, so the
# Phase A pipeline (Stage 2b chain + Stage 2 Extended) is unaffected.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot Phase A Stage 1 best ──────────────────────────────────
# Snapshotted at job start so a future Phase A retraining cannot
# silently change what this Phase C run warm-started from.
PHASE_A_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
SNAPSHOT="runs/e2e_stage1/e2e_stage1_best_c_stage1_init.${SLURM_JOB_ID}.pt"

if [ ! -f "$PHASE_A_BEST" ]; then
    echo "ERROR: $PHASE_A_BEST does not exist." >&2
    echo "Phase A Stage 1 must produce a best checkpoint first." >&2
    exit 1
fi
cp "$PHASE_A_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

# ── Auto-resume across 24 h walls ─────────────────────────────────
# If a *_latest.pt exists in the C-Stage 1 checkpoint dir from a
# previous submission, resume from it; the trainer's resume path
# overrides --init_checkpoint, so passing both unconditionally is safe.
# train_e2e_stage1.py hardcodes the basename "e2e_stage1_latest.pt" —
# under --checkpoint_dir runs/c_stage1 that lands at the path below,
# even though we'd nominally call this run "c_stage1".
LATEST="runs/c_stage1/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
fi

srun pixi run python ../training/train_e2e_stage1.py \
    $RESUME_FLAG \
    --init_checkpoint "$SNAPSHOT" \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/c_stage1 \
    --val_fraction 0.1 \
    --seed 42 \
    \
    --chunk_duration_s 0.05 \
    --prediction_horizon_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    \
    --d_model 256 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 256 \
    --num_workers 8 \
    --max_steps 336000 \
    --log_every 50 \
    --val_every 2000 \
    --val_max_batches 50 \
    \
    --use_video tangtv \
    --freeze_backbone_steps 5000