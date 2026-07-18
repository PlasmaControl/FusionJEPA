"""Unit tests for the synthetic-smoke experiment config (Task 2.13).

`configs/experiment/synthetic_smoke.yaml` is the profile the Frontier DDP payload
(`tests/frontier/smoke_ddp.py`) runs. These lock:

* the file loads on a CPU login node and its `cluster`/`data`/`model` pointers
  reference committed profiles;
* its `training:` block carries every field the landed Trainer consumes, and the
  effective-batch math divides cleanly for BOTH a single process AND the 8-GCD
  Frontier run;
* the block actually DRIVES a short `Trainer.fit()` of the referenced
  `raw_predictor_small` model to completion on CPU (the payload's Trainer step,
  single-process); and
* the documented loading constraint: because the block carries `training:` (and
  `model:`) keys the landed `resolve_config` schema does not model, the file is
  loaded DIRECTLY via `OmegaConf.load`, and `resolve_config` deliberately refuses
  it.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fusion_jepa.config import ConfigError, resolve_config
from fusion_jepa.models.build import build_raw_world_model
from fusion_jepa.objectives.raw_prediction import RawPredictionObjective
from fusion_jepa.training.distributed import DistributedManager
from fusion_jepa.training.loop import Trainer, should_use_bf16_autocast
from tests.fixtures.synthetic import make_synthetic_fusion_batch

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"
EXPERIMENT = CONFIGS_ROOT / "experiment" / "synthetic_smoke.yaml"

# Fields the landed Trainer reads from cfg.training (training/loop.py).
_TRAINER_TRAINING_FIELDS = (
    "lr",
    "min_lr",
    "weight_decay",
    "warmup_steps",
    "warmup_start_factor",
    "total_steps",
    "micro_batch_samples",
    "effective_batch_samples",
    "grad_clip_norm",
    "bf16",
    "log_every",
    "val_every_steps",
    "val_max_batches",
)


def _load() -> OmegaConf:
    return OmegaConf.load(EXPERIMENT)


def test_yaml_loads_and_pointers_reference_committed_profiles():
    cfg = _load()
    assert (CONFIGS_ROOT / "cluster" / f"{cfg.cluster}.yaml").exists()
    assert (CONFIGS_ROOT / "data" / f"{cfg.data}.yaml").exists()
    assert (CONFIGS_ROOT / "model" / f"{cfg.model}.yaml").exists()
    # The smoke targets the Frontier cluster profile.
    assert cfg.cluster == "frontier"
    assert cfg.model == "raw_predictor_small"


def test_training_block_has_every_trainer_consumed_field():
    cfg = _load()
    assert "seed" in cfg
    for field in _TRAINER_TRAINING_FIELDS:
        assert field in cfg.training, f"missing training.{field}"
    # ~20-step smoke with a real warmup that ends before the run does.
    assert 10 <= cfg.training.total_steps <= 100
    assert 0 < cfg.training.warmup_steps < cfg.training.total_steps


@pytest.mark.parametrize("world_size", [1, 8])
def test_effective_batch_divides_for_single_process_and_frontier(world_size):
    cfg = _load()
    micro = int(cfg.training.micro_batch_samples)
    effective = int(cfg.training.effective_batch_samples)
    denom = micro * world_size
    assert effective % denom == 0, (
        f"effective_batch_samples={effective} must divide by "
        f"micro_batch_samples*world_size={denom} for world_size={world_size}"
    )


def test_bf16_is_disabled_pending_autocast_safe_model():
    # bf16 is intentionally OFF (Task 2.13 report disclosure #1): the landed model
    # classes strictly validate float32 at internal boundaries and crash under the
    # loop's bf16 autocast wrap on GPU. The loop gates bf16 to CUDA regardless, so
    # either way a CPU dry-run runs fp32.
    cfg = _load()
    assert cfg.training.bf16 is False
    assert should_use_bf16_autocast(torch.device("cpu"), cfg.training.bf16) is False
    assert should_use_bf16_autocast(torch.device("cuda", 0), cfg.training.bf16) is False


def test_resolve_config_deliberately_refuses_the_training_block():
    # The training:/model: keys are not in the ExperimentConfig schema, so
    # resolve_config refuses them (naming the offending key) -- the file is
    # loaded directly via OmegaConf.load instead.
    with pytest.raises(ConfigError, match="ExperimentConfig"):
        resolve_config(["experiment=synthetic_smoke"], config_root=CONFIGS_ROOT)


def test_block_drives_a_short_trainer_run_to_completion_on_cpu(tmp_path):
    """The single-process (CPU) analogue of the Frontier payload: build the
    referenced model, hand the loaded config to the Trainer, and run a short
    fit() to completion on synthetic data. total_steps is reduced so the CPU
    lock stays fast; every other training field is the committed one."""
    cfg = _load()
    cfg.training.total_steps = 2
    cfg.training.val_every_steps = 2
    cfg.training.val_max_batches = 1
    cfg.training.warmup_steps = 1

    model = build_raw_world_model(CONFIGS_ROOT / "model" / f"{cfg.model}.yaml")
    objective = RawPredictionObjective(distance="smooth_l1", smooth_l1_beta=1.0)
    dm = DistributedManager()

    def _batches(n, seed):
        return [
            make_synthetic_fusion_batch(
                B=2,
                modalities=("slow_ts", "profile"),
                n_channels=8,
                T=8,
                H=4,
                A=4,
                seed=seed + i,
                missing_fraction=0.1,
            )
            for i in range(n)
        ]

    trainer = Trainer(
        cfg=cfg,
        model=model,
        objective=objective,
        dm=dm,
        train_loader=_batches(8, seed=0),
        val_loader=_batches(2, seed=500),
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )
    result = trainer.fit()
    dm.close()

    assert result.status == "completed"
    assert result.final_step == 2
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "completion.json").exists()
