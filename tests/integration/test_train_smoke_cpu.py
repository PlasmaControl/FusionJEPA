"""End-to-end CPU smoke test for the M2 training loop (Task 2.12).

A tiny raw-baseline run (~20 optimizer steps on deterministic synthetic data)
must train -- loss finite and decreasing -- and leave a complete run directory
(``metrics.jsonl``, ``completion.json`` status ``completed``, ``best.pt``,
``latest.pt``). Everything is CPU-only and finishes in well under 90 seconds.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from fusion_jepa.objectives.raw_prediction import RawPredictionObjective
from fusion_jepa.training.distributed import DistributedManager
from fusion_jepa.training.loop import Trainer
from fusion_jepa.utils.logging import read_metrics
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_raw_world_model import build_raw_world_model

_MODALITIES = ("slow_ts", "profile")


def _smoke_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        experiment_name="smoke",
        training=SimpleNamespace(
            lr=5e-3,
            weight_decay=0.0,
            effective_batch_samples=2,
            micro_batch_samples=2,
            total_steps=20,
            warmup_steps=3,
            min_lr=1e-5,
            val_every_steps=5,
            val_max_batches=2,
            log_every=1,
            grad_clip_norm=1.0,
            bf16=False,
        ),
    )


def test_raw_baseline_tiny_run_trains_and_writes_artifacts(tmp_path):
    torch.manual_seed(0)
    batches = [
        make_synthetic_fusion_batch(
            B=2, modalities=_MODALITIES, n_channels=3, T=4, H=3, A=2, seed=s
        )
        for s in range(2)
    ]
    model = build_raw_world_model(
        modalities=_MODALITIES, n_channels=3, n_actuators=2
    )
    objective = RawPredictionObjective(distance="mse")
    dm = DistributedManager()

    trainer = Trainer(
        cfg=_smoke_cfg(),
        model=model,
        objective=objective,
        dm=dm,
        train_loader=batches,
        val_loader=batches,
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )
    result = trainer.fit()
    dm.close()

    assert result.status == "completed"
    assert result.final_step == 20

    # Run directory is complete.
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["status"] == "completed"

    # Loss is finite for every step and decreases over the run.
    records = read_metrics(tmp_path / "metrics.jsonl")
    losses = [r["loss"] for r in records if r.get("event") == "train_step"]
    assert len(losses) == 20
    assert all(loss is not None and math.isfinite(loss) for loss in losses)
    first_five = sum(losses[:5]) / 5.0
    last_five = sum(losses[-5:]) / 5.0
    assert last_five < first_five
