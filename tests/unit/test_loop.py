"""Unit tests for the M2 training loop (Task 2.12).

These lock the ten training-loop contracts against tiny *real* components:
the raw world model (``build_raw_world_model``), the mask-aware
:class:`RawPredictionObjective`, a single-process CPU
:class:`DistributedManager`, and deterministic synthetic
:class:`~fusion_jepa.data.batch.FusionBatch` batches. Every test forces
``device=cpu`` so the suite is independent of whether the host exposes a GPU.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
from types import SimpleNamespace

import pytest
import torch

from fusion_jepa.objectives.raw_prediction import RawPredictionObjective
from fusion_jepa.training.checkpoint import CHECKPOINT_KEYS, load_checkpoint
from fusion_jepa.training.distributed import DistributedManager
from fusion_jepa.training.loop import (
    ResumableSampler,
    RunResult,
    Trainer,
    resolve_autocast,
    should_use_bf16_autocast,
)
from fusion_jepa.utils.logging import read_metrics
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_raw_world_model import build_raw_world_model

_MODALITIES = ("slow_ts", "profile")
_CPU = torch.device("cpu")


def _make_batches(n: int = 3, B: int = 2, seed: int = 0) -> list:
    """A plain list of distinct synthetic batches (an iterable of FusionBatch)."""
    return [
        make_synthetic_fusion_batch(
            B=B, modalities=_MODALITIES, n_channels=3, T=4, H=3, A=2, seed=seed + i
        )
        for i in range(n)
    ]


def _make_cfg(**over) -> SimpleNamespace:
    defaults = dict(
        lr=3e-3,
        weight_decay=0.0,
        effective_batch_samples=2,
        micro_batch_samples=2,
        total_steps=4,
        warmup_steps=1,
        min_lr=0.0,
        val_every_steps=2,
        val_max_batches=1,
        log_every=1,
        grad_clip_norm=1.0,
        bf16=False,
    )
    defaults.update(over)
    return SimpleNamespace(
        seed=0, experiment_name="test", training=SimpleNamespace(**defaults)
    )


def _make_trainer(
    run_dir,
    *,
    cfg=None,
    train_loader=None,
    val_loader=None,
    objective=None,
    **tkw,
):
    if cfg is None:
        cfg = _make_cfg()
    model = build_raw_world_model(
        modalities=_MODALITIES, n_channels=3, n_actuators=2
    )
    if objective is None:
        objective = RawPredictionObjective(distance="mse")
    dm = DistributedManager()
    if train_loader is None:
        train_loader = _make_batches(3)
    if val_loader is None:
        val_loader = _make_batches(1)
    trainer = Trainer(
        cfg=cfg,
        model=model,
        objective=objective,
        dm=dm,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        device=_CPU,
        **tkw,
    )
    return trainer, dm


# ── Contract 1: grad accumulation math + effective-batch logging ──────────


def test_accumulation_math_and_effective_batch_logged(tmp_path):
    cfg = _make_cfg(
        effective_batch_samples=8,
        micro_batch_samples=2,
        total_steps=2,
        warmup_steps=1,
        val_every_steps=100,
    )
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, train_loader=_make_batches(4))
    # effective / (micro * world_size) == 8 / (2 * 1) == 4
    assert trainer.accumulation_steps == 4
    assert trainer.effective_batch_samples == 8

    result = trainer.fit()
    dm.close()

    assert result.status == "completed"
    assert result.final_step == 2

    records = read_metrics(tmp_path / "metrics.jsonl")
    cfg_recs = [r for r in records if r.get("event") == "train_config"]
    assert cfg_recs, "the effective batch must be logged once"
    assert cfg_recs[0]["effective_batch_samples"] == 8
    assert cfg_recs[0]["accumulation_steps"] == 4
    assert cfg_recs[0]["micro_batch_samples"] == 2
    assert cfg_recs[0]["world_size"] == 1

    # optimizer.step() fires exactly once per full accumulation window.
    step_recs = [r for r in records if r.get("event") == "train_step"]
    assert len(step_recs) == 2


def test_accumulation_rejects_indivisible_effective_batch(tmp_path):
    cfg = _make_cfg(effective_batch_samples=7, micro_batch_samples=2)
    with pytest.raises(ValueError, match="divisible"):
        _make_trainer(tmp_path, cfg=cfg)


# ── Contract 2: fixed-step validation ─────────────────────────────────────


def test_validation_fires_at_fixed_step_intervals(tmp_path):
    cfg = _make_cfg(total_steps=6, val_every_steps=2)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg)
    trainer.fit()
    dm.close()

    records = read_metrics(tmp_path / "metrics.jsonl")
    val_steps = [r["step"] for r in records if r.get("event") == "val"]
    assert val_steps == [2, 4, 6]


# ── Contract 3: best.pt / latest.pt are distinct, full-payload checkpoints ──


def test_best_and_latest_checkpoints_distinct(tmp_path):
    cfg = _make_cfg(total_steps=6, val_every_steps=2)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg)
    trainer.fit()
    dm.close()

    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    assert best.exists() and latest.exists()
    assert best != latest

    best_payload = load_checkpoint(best)
    latest_payload = load_checkpoint(latest)
    assert set(best_payload) == set(CHECKPOINT_KEYS)
    assert set(latest_payload) == set(CHECKPOINT_KEYS)
    # Raw baseline disclosures: no EMA target encoder, no grad scaler.
    assert latest_payload["target_encoder"] is None
    assert latest_payload["scaler"] is None

    records = read_metrics(tmp_path / "metrics.jsonl")
    val_losses = [r["val_loss"] for r in records if r.get("event") == "val"]
    assert best_payload["best_metric"] == pytest.approx(min(val_losses))
    assert latest_payload["step"] == 6


# ── Contract 4: test-split refusal at construction ────────────────────────


def test_refuses_test_split(tmp_path):
    with pytest.raises(ValueError, match="test"):
        _make_trainer(tmp_path, val_loader=SimpleNamespace(split="test"))

    # A held-out validation split constructs cleanly (no iteration needed).
    trainer, dm = _make_trainer(
        tmp_path / "ok", val_loader=SimpleNamespace(split="validation")
    )
    dm.close()
    assert trainer is not None


# ── Contract 5: SIGUSR1 preemption ────────────────────────────────────────


class _SignalOnFirstCall:
    """Objective wrapper that raises a signal to *itself* after the first call.

    Lets a single-process test drive the preemption path deterministically:
    the first optimizer step's forward completes, the signal is delivered, the
    loop finishes that step, saves ``latest.pt`` and returns ``preempted``.
    """

    def __init__(self, inner, sig) -> None:
        self._inner = inner
        self._sig = sig
        self.calls = 0

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        self.calls += 1
        if self.calls == 1:
            os.kill(os.getpid(), self._sig)
        return out


def test_sigusr1_saves_latest_and_reports_preempted(tmp_path):
    prior = signal.getsignal(signal.SIGUSR1)
    obj = _SignalOnFirstCall(
        RawPredictionObjective(distance="mse"), signal.SIGUSR1
    )
    cfg = _make_cfg(total_steps=50, val_every_steps=100)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, objective=obj)

    result = trainer.fit()
    dm.close()

    assert result.status == "preempted"
    assert 1 <= result.final_step < 50
    assert (tmp_path / "latest.pt").exists()

    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["status"] == "preempted"

    # Prior handler restored on exit from fit().
    assert signal.getsignal(signal.SIGUSR1) == prior


def test_sigterm_reports_preempted(tmp_path):
    prior = signal.getsignal(signal.SIGTERM)
    obj = _SignalOnFirstCall(
        RawPredictionObjective(distance="mse"), signal.SIGTERM
    )
    cfg = _make_cfg(total_steps=50, val_every_steps=100)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, objective=obj)

    result = trainer.fit()
    dm.close()

    assert result.status == "preempted"
    assert (tmp_path / "latest.pt").exists()
    assert signal.getsignal(signal.SIGTERM) == prior


# ── Contract 6: resume restores step/opt/sched/RNG/sampler and continues ──


def test_resume_restores_step_and_continues(tmp_path):
    cfg = _make_cfg(total_steps=4, val_every_steps=2)
    t1, dm1 = _make_trainer(tmp_path, cfg=cfg)
    r1 = t1.fit()
    dm1.close()
    assert r1.final_step == 4

    cfg2 = _make_cfg(total_steps=6, val_every_steps=2)
    t2, dm2 = _make_trainer(tmp_path, cfg=cfg2, resume_from="auto")
    # Restored from latest.pt at construction.
    assert t2.step == 4
    r2 = t2.fit()
    dm2.close()
    assert r2.final_step == 6


def test_resume_from_explicit_path(tmp_path):
    cfg = _make_cfg(total_steps=4, val_every_steps=2)
    t1, dm1 = _make_trainer(tmp_path, cfg=cfg)
    t1.fit()
    dm1.close()

    cfg2 = _make_cfg(total_steps=8, val_every_steps=2)
    t2, dm2 = _make_trainer(
        tmp_path, cfg=cfg2, resume_from=str(tmp_path / "latest.pt")
    )
    assert t2.step == 4
    dm2.close()


# ── Contract 7: JSONL metric fields ───────────────────────────────────────


def test_jsonl_metrics_include_required_fields(tmp_path):
    cfg = _make_cfg(total_steps=3, val_every_steps=100, grad_clip_norm=1.0)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg)
    trainer.fit()
    dm.close()

    records = read_metrics(tmp_path / "metrics.jsonl")
    step_recs = [r for r in records if r.get("event") == "train_step"]
    assert step_recs
    record = step_recs[0]
    for key in (
        "loss",
        "grad_norm",
        "lr",
        "tokens_per_s",
        "samples_per_s",
        "data_wait_s",
        "wall_s",
        "epoch",
    ):
        assert key in record, f"missing metric {key!r}"
    assert any(k.startswith("loss_term/") for k in record)


# ── Contract 8: bf16 autocast is CUDA-only ────────────────────────────────


def test_bf16_autocast_is_cuda_only():
    assert should_use_bf16_autocast(torch.device("cpu"), True) is False
    assert should_use_bf16_autocast(torch.device("cpu"), False) is False
    assert should_use_bf16_autocast(torch.device("cuda", 0), True) is True
    assert should_use_bf16_autocast(torch.device("cuda", 0), False) is False

    # On CPU the resolved context is a genuine no-op even when bf16 is asked.
    ctx = resolve_autocast(torch.device("cpu"), True)
    assert isinstance(ctx, contextlib.nullcontext)


# ── Contract 9: ResumableSampler state_dict round-trip ────────────────────


def test_resumable_sampler_state_dict_round_trip():
    order = list(ResumableSampler(10, root_seed=0, shuffle=True))
    assert sorted(order) == list(range(10))

    # Deterministic per (root_seed, epoch).
    assert list(ResumableSampler(10, root_seed=0, shuffle=True)) == order

    # Resume mid-epoch: cursor carried in the state_dict.
    s = ResumableSampler(10, root_seed=0, shuffle=True)
    it = iter(s)
    consumed = [next(it) for _ in range(3)]
    state = s.state_dict()
    assert state == {"epoch": 0, "cursor": 3}

    resumed = ResumableSampler(10, root_seed=0, shuffle=True)
    resumed.load_state_dict(state)
    rest = list(resumed)
    assert consumed + rest == order

    # The following epoch reshuffles under a new derived seed.
    assert list(resumed) != order


def test_resumable_sampler_sequential_when_not_shuffled():
    s = ResumableSampler(5, root_seed=1, shuffle=False)
    assert list(s) == [0, 1, 2, 3, 4]


# ── Contract 10: failure records completion.json and re-raises ────────────


def test_failure_writes_completion_and_reraises(tmp_path):
    class _Boom:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("objective exploded")

    cfg = _make_cfg(total_steps=4, val_every_steps=2)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, objective=_Boom())
    with pytest.raises(RuntimeError, match="objective exploded"):
        trainer.fit()
    dm.close()

    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["status"] == "failed"
    assert "objective exploded" in completion["failure_reason"]


def test_failure_returns_result_when_not_reraising(tmp_path):
    class _Boom:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("objective exploded")

    cfg = _make_cfg(total_steps=4, val_every_steps=2)
    trainer, dm = _make_trainer(
        tmp_path, cfg=cfg, objective=_Boom(), reraise_failures=False
    )
    result = trainer.fit()
    dm.close()
    assert isinstance(result, RunResult)
    assert result.status == "failed"
    assert "objective exploded" in result.failure_reason
