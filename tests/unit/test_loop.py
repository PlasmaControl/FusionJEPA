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


class _RecordingObjective:
    """Delegates to a real objective, recording the *consumed* train-batch ids.

    On CPU the loop hands the objective the batch's own ``target`` dict object
    (``_move_batch`` is a no-op), so ``id(target)`` uniquely identifies the
    batch. Validation batches (distinct objects) are absent from ``id_by_target``
    and never logged, so the log is exactly the consumed *training* order --
    which lets a resume test assert that resumed data order continues rather than
    replaying already-seen batches.
    """

    def __init__(self, inner, id_by_target, log):
        self._inner = inner
        self._id_by_target = id_by_target
        self._log = log

    def __call__(self, predictions, targets, target_masks):
        key = id(targets)
        if key in self._id_by_target:
            self._log.append(self._id_by_target[key])
        return self._inner(predictions, targets, target_masks)


class _DecreasingValObjective:
    """Real objective while training; a scripted loss during validation.

    Validation runs under ``torch.no_grad()``, so ``torch.is_grad_enabled()`` is
    ``False`` there and ``True`` during training. This lets a test drive a
    deterministic validation-loss *sequence* (whose final value is the best)
    while training still produces a genuine differentiable loss to backprop.
    """

    def __init__(self, inner, val_losses):
        self._inner = inner
        self._val_losses = list(val_losses)
        self._val_idx = 0
        self.val_returned: list[float] = []

    def __call__(self, predictions, targets, target_masks):
        if not torch.is_grad_enabled():
            value = self._val_losses[self._val_idx]
            self._val_idx += 1
            self.val_returned.append(value)
            return SimpleNamespace(total=torch.tensor(value, dtype=torch.float32))
        return self._inner(predictions, targets, target_masks)


def _clone_params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _adam_moment_buffers(optimizer) -> tuple[list, list]:
    """Return per-parameter ``(exp_avg, exp_avg_sq)`` clones in parameter order."""
    exp_avg, exp_avg_sq = [], []
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            ea = state.get("exp_avg")
            eas = state.get("exp_avg_sq")
            exp_avg.append(ea.detach().clone() if ea is not None else None)
            exp_avg_sq.append(eas.detach().clone() if eas is not None else None)
    return exp_avg, exp_avg_sq


def _assert_tensor_lists_equal(a_list, b_list, msg):
    assert len(a_list) == len(b_list), f"{msg}: length mismatch"
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if a is None or b is None:
            assert a is None and b is None, f"{msg}: one buffer missing at {i}"
            continue
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), f"{msg}: mismatch at {i}"


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
    # A strictly decreasing validation-loss sequence whose FINAL validation is
    # the best. latest.pt (written at the last save point, step 6) must then
    # record the NEW best -- not the stale prior best from step 4. A plain
    # "check min of the logged losses" would miss stale latest metadata, so we
    # script the sequence and assert latest.pt's best_metric directly (finding 6,
    # discriminating finding 2's save-before-update ordering bug).
    val_losses = [0.5, 0.4, 0.3]
    obj = _DecreasingValObjective(RawPredictionObjective(distance="mse"), val_losses)
    cfg = _make_cfg(total_steps=6, val_every_steps=2, val_max_batches=1)
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, objective=obj)
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
    assert latest_payload["step"] == 6

    # Validation fired at steps 2, 4, 6 with the scripted decreasing losses.
    assert obj.val_returned == val_losses
    # Both checkpoints record the FINAL (best) metric. The load-bearing check is
    # latest.pt: a save-before-update ordering would leave it holding 0.4.
    assert best_payload["best_metric"] == pytest.approx(min(val_losses))
    assert latest_payload["best_metric"] == pytest.approx(min(val_losses))
    assert latest_payload["best_metric"] == pytest.approx(val_losses[-1])


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


# ── Resume fidelity: resumed run == uninterrupted run (finding 3, finding 1) ──


class _RecordThenPreempt:
    """Record consumed train-batch ids and raise a signal after ``fire_after``
    training calls, so the loop finishes that step, saves latest.pt and stops.

    ``total_steps`` stays identical to the uninterrupted run (so the warmup->
    cosine schedule is the same curve); only the *interruption* is different.
    """

    def __init__(self, inner, id_by_target, log, *, fire_after, sig):
        self._inner = inner
        self._id_by_target = id_by_target
        self._log = log
        self._fire_after = fire_after
        self._sig = sig
        self._train_calls = 0

    def __call__(self, predictions, targets, target_masks):
        out = self._inner(predictions, targets, target_masks)
        key = id(targets)
        if key in self._id_by_target and torch.is_grad_enabled():
            self._log.append(self._id_by_target[key])
            self._train_calls += 1
            if self._train_calls == self._fire_after:
                os.kill(os.getpid(), self._sig)
        return out


def test_resume_matches_uninterrupted_run(tmp_path):
    """A save/kill/resume run must be indistinguishable from a straight-through
    run: identical consumed batch order, identical final weights, and identical
    AdamW moment buffers. A plain list loader (no stateful sampler) forces the
    loop's own cursor-skip to do the resuming -- so this fails against the
    pre-fix loop, which rebuilt the iterator at its start and replayed batches.

    The first leg runs with the SAME ``total_steps`` as the reference and is cut
    short by a signal at the interruption step (a realistic SLURM preemption),
    so the LR schedule and optimizer trajectory up to that point are identical.
    """
    total, interrupt = 6, 4
    common = dict(
        total_steps=total,
        val_every_steps=100,  # no validation; latest.pt comes from preemption
        val_max_batches=1,
        effective_batch_samples=2,
        micro_batch_samples=2,  # accumulation_steps == 1: one batch per step
        warmup_steps=1,
        grad_clip_norm=1.0,
    )

    # (1) Uninterrupted reference run to `total` steps.
    straight_batches = _make_batches(3, seed=100)
    straight_ids = {id(b.target): i for i, b in enumerate(straight_batches)}
    straight_log: list[int] = []
    straight_obj = _RecordingObjective(
        RawPredictionObjective(distance="mse"), straight_ids, straight_log
    )
    t_full, dm_full = _make_trainer(
        tmp_path / "full",
        cfg=_make_cfg(**common),
        train_loader=straight_batches,
        val_loader=_make_batches(1, seed=900),
        objective=straight_obj,
    )
    t_full.fit()
    full_weights = _clone_params(dm_full.unwrap(t_full.model))
    full_exp_avg, full_exp_avg_sq = _adam_moment_buffers(t_full.optimizer)
    dm_full.close()

    # (2a) First leg: same config, preempted after `interrupt` steps.
    resume_batches = _make_batches(3, seed=100)
    resume_ids = {id(b.target): i for i, b in enumerate(resume_batches)}
    first_log: list[int] = []
    first_obj = _RecordThenPreempt(
        RawPredictionObjective(distance="mse"), resume_ids, first_log,
        fire_after=interrupt, sig=signal.SIGUSR1,
    )
    t1, dm1 = _make_trainer(
        tmp_path / "resume",
        cfg=_make_cfg(**common),
        train_loader=resume_batches,
        val_loader=_make_batches(1, seed=900),
        objective=first_obj,
    )
    r1 = t1.fit()
    dm1.close()
    assert r1.status == "preempted"
    assert r1.final_step == interrupt

    # (2b) Second leg: resume from latest.pt and run to `total`.
    resumed_log: list[int] = []
    resumed_obj = _RecordingObjective(
        RawPredictionObjective(distance="mse"), resume_ids, resumed_log
    )
    t2, dm2 = _make_trainer(
        tmp_path / "resume",
        cfg=_make_cfg(**common),
        train_loader=resume_batches,
        val_loader=_make_batches(1, seed=900),
        objective=resumed_obj,
        resume_from="auto",
    )
    assert t2.step == interrupt
    t2.fit()
    resumed_weights = _clone_params(dm2.unwrap(t2.model))
    resumed_exp_avg, resumed_exp_avg_sq = _adam_moment_buffers(t2.optimizer)
    dm2.close()

    # Consumed batch order: the two legs concatenated equal the straight run.
    assert first_log == [0, 1, 2, 0]
    assert first_log + resumed_log == straight_log
    # And resume genuinely continued (did not replay the interrupted epoch).
    assert resumed_log and resumed_log != straight_log[:len(resumed_log)]

    _assert_tensor_lists_equal(full_weights, resumed_weights, "final weights")
    _assert_tensor_lists_equal(full_exp_avg, resumed_exp_avg, "exp_avg")
    _assert_tensor_lists_equal(full_exp_avg_sq, resumed_exp_avg_sq, "exp_avg_sq")


# ── Accumulation: gradient == full-batch mean, optimizer.step() once ──────────


def test_accumulation_grad_matches_full_batch_and_steps_once(tmp_path):
    """The accumulated gradient over K micro-steps must equal the gradient of the
    mean loss over the effective batch (the ``1/accumulation_steps`` scaling is
    load-bearing), and ``optimizer.step()`` must fire exactly once per optimizer
    step -- not once per micro-step. Grad clipping is disabled so the gradient at
    ``optimizer.step()`` time is the raw accumulation, comparable to a reference.
    """
    cfg = _make_cfg(
        effective_batch_samples=6,
        micro_batch_samples=2,
        total_steps=1,
        warmup_steps=1,
        val_every_steps=100,
        grad_clip_norm=0.0,  # -> no clipping; grads unmodified at step time
        log_every=1,
    )
    train_batches = _make_batches(3, seed=7)  # exactly K=3 micro-batches
    trainer, dm = _make_trainer(tmp_path, cfg=cfg, train_loader=train_batches)
    assert trainer.accumulation_steps == 3
    model = dm.unwrap(trainer.model)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Spy on optimizer.step(): count calls and snapshot grads at the first call.
    orig_step = trainer.optimizer.step
    call_count = {"n": 0}
    captured: dict[str, list] = {}

    def spy_step(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            captured["grads"] = [
                (name, None if p.grad is None else p.grad.detach().clone())
                for name, p in model.named_parameters()
            ]
        return orig_step(*args, **kwargs)

    trainer.optimizer.step = spy_step
    trainer.fit()
    dm.close()

    # optimizer.step() fired once for one optimizer step over 3 micro-batches.
    assert call_count["n"] == 1

    # Reference: same init, mean of per-micro losses (1/K each) -> full-batch
    # mean gradient. This is what the loop must have accumulated.
    ref = build_raw_world_model(modalities=_MODALITIES, n_channels=3, n_actuators=2)
    ref.load_state_dict(init_state)
    ref.train()
    ref_obj = RawPredictionObjective(distance="mse")
    ref.zero_grad(set_to_none=True)
    k = 3
    for batch in train_batches:
        out = ref_obj(ref(batch), batch.target, batch.target_mask)
        (out.total / k).backward()
    ref_grads = {name: p.grad for name, p in ref.named_parameters()}

    nonzero = 0
    for name, grad in captured["grads"]:
        expected = ref_grads[name]
        if expected is None:
            assert grad is None, f"{name}: unexpected grad"
            continue
        assert grad is not None, f"{name}: missing accumulated grad"
        assert torch.allclose(grad, expected, atol=1e-6, rtol=1e-5), name
        if expected.abs().sum() > 0:
            nonzero += 1
    assert nonzero > 0, "reference gradient must be non-trivial to discriminate"


# ── Scheduler cadence: warmup->cosine, stepped once per optimizer step ────────


def _cosine_ref_lrs(base_lr, warmup_steps, total_steps, min_lr, n, start_factor):
    """Hand-built LinearLR->Cosine LR sequence, stepped once per optimizer step."""
    ref_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        ref_opt, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr
    )
    if warmup_steps <= 0:
        sched = cosine
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(
            ref_opt, start_factor=start_factor, end_factor=1.0,
            total_iters=warmup_steps,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            ref_opt, [warmup, cosine], milestones=[warmup_steps]
        )
    lrs = []
    for _ in range(n):
        ref_opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    return lrs


def test_scheduler_steps_once_per_optimizer_step(tmp_path):
    cfg = _make_cfg(
        lr=0.01,
        effective_batch_samples=4,
        micro_batch_samples=2,  # K=2 micro-steps per optimizer step
        total_steps=6,
        warmup_steps=2,
        min_lr=1e-4,
        val_every_steps=100,
        log_every=1,
    )
    trainer, dm = _make_trainer(
        tmp_path, cfg=cfg, train_loader=_make_batches(2, seed=3)
    )
    assert trainer.accumulation_steps == 2
    trainer.fit()
    dm.close()

    records = read_metrics(tmp_path / "metrics.jsonl")
    logged = [r["lr"] for r in records if r.get("event") == "train_step"]
    assert len(logged) == 6

    # Reference stepped once per OPTIMIZER step. If the loop stepped the
    # scheduler per micro-step it would advance twice as fast and diverge.
    ref = _cosine_ref_lrs(
        base_lr=0.01, warmup_steps=2, total_steps=6, min_lr=1e-4, n=6,
        start_factor=1e-3,
    )
    assert logged == pytest.approx(ref, rel=1e-6, abs=1e-12)
    # If the loop stepped the scheduler per MICRO-step (K=2), then at optimizer
    # step s the scheduler would have advanced 2*s times, so the logged sequence
    # would be the LRs after 2, 4, ..., 12 steps -- a genuinely different curve.
    full = _cosine_ref_lrs(
        base_lr=0.01, warmup_steps=2, total_steps=6, min_lr=1e-4, n=12,
        start_factor=1e-3,
    )
    per_micro_cadence = [full[2 * s - 1] for s in range(1, 7)]
    assert logged != pytest.approx(per_micro_cadence, rel=1e-6, abs=1e-12)


def test_zero_warmup_is_pure_cosine_no_linear_phase(tmp_path):
    """warmup_steps=0 must mean NO LinearLR phase: pure cosine from step 0.

    Discriminates the ``max(warmup_steps, 1)`` floor (finding 7), which injected
    a one-step LinearLR warmup and started the run at ~start_factor*lr.
    """
    cfg = _make_cfg(
        lr=0.01,
        effective_batch_samples=2,
        micro_batch_samples=2,
        total_steps=4,
        warmup_steps=0,
        min_lr=0.0,
        val_every_steps=100,
        log_every=1,
    )
    trainer, dm = _make_trainer(
        tmp_path, cfg=cfg, train_loader=_make_batches(3, seed=5)
    )
    trainer.fit()
    dm.close()

    records = read_metrics(tmp_path / "metrics.jsonl")
    logged = [r["lr"] for r in records if r.get("event") == "train_step"]
    assert len(logged) == 4

    ref = _cosine_ref_lrs(
        base_lr=0.01, warmup_steps=0, total_steps=4, min_lr=0.0, n=4,
        start_factor=1e-3,
    )
    assert logged == pytest.approx(ref, rel=1e-6, abs=1e-12)
    # Pure cosine over T_max == total_steps reaches eta_min (0.0) at the final
    # step. A max(warmup, 1) floor injects a warmup step, shifting the cosine so
    # the last LR never lands on eta_min -- a clean discriminator for finding 7.
    assert logged[-1] == pytest.approx(0.0, abs=1e-9)


# ── EMA cadence: exactly one update per optimizer step (M3 contract) ──────────


def test_ema_updater_called_once_per_optimizer_step(tmp_path):
    class _CountingEMA:
        def __init__(self):
            self.calls = 0

        def update(self, target):
            self.calls += 1

    ema = _CountingEMA()
    cfg = _make_cfg(
        effective_batch_samples=6,
        micro_batch_samples=2,  # K=3 micro-steps per optimizer step
        total_steps=4,
        warmup_steps=1,
        val_every_steps=100,
    )
    trainer, dm = _make_trainer(
        tmp_path, cfg=cfg, train_loader=_make_batches(3, seed=1), ema_updater=ema
    )
    assert trainer.accumulation_steps == 3
    trainer.fit()
    dm.close()

    # Exactly one EMA update per OPTIMIZER step (4), not per micro-step (12).
    assert ema.calls == 4
    assert ema.calls != 4 * 3
