"""Checkpoint-resume reproducibility for the JEPA + EMA composition (Task 3.7).

This integration test proves that four landed Milestone-2/3 components *compose*
into a bit-for-bit resumable run:

* the atomic checkpoint I/O + full-state capture of
  :mod:`fusion_jepa.training.checkpoint` (Task 2.11),
* the :class:`~fusion_jepa.training.loop.Trainer` (Task 2.12),
* the standalone :class:`~fusion_jepa.training.ema.EmaUpdater` (Task 3.1), and
* the :class:`~fusion_jepa.models.jepa.JEPAModel` with its EMA target trunk
  (Task 3.2).

It is deliberately TEST-ONLY: no source module is modified. Everything runs
CPU-only, deterministically, and in well under a minute.

Disclosure 1 -- the objective adapter
-------------------------------------
The Trainer calls its objective in the RAW mask-aware form
``objective(model(batch), batch.target, batch.target_mask)`` (loop.py:8-9),
but the JEPA's :class:`~fusion_jepa.objectives.latent_prediction.LatentPredictionObjective`
is natively called ``obj(jepa_output, horizon_seconds)``. Wiring that native form
into the Trainer is Task 3.8's scope; here a tiny test-local adapter
(:class:`_LatentObjectiveAdapter`) bridges the two: it receives the Trainer's
``(preds, target, target_mask)``, ignores ``target``/``target_mask`` (the
JEPAOutput already carries ``z_hat``/``z_target``/``target_valid``), and calls the
real objective with a constant ``horizon_seconds`` of the right ``[B, K]`` shape.
``horizon_seconds`` is labeling-only in the objective -- it never enters the loss
value (see the objective's module docstring) -- so a constant is exactly correct.

Disclosure 2 -- the EMA counter is NOT restored on resume (composition gap)
--------------------------------------------------------------------------
The Trainer *saves* the EMA state -- ``_build_payload`` stores
``target_encoder=ema_updater.state_dict()`` (loop.py:565) -- but ``_restore``
(loop.py:407-419) never loads ``payload["target_encoder"]`` back. A resumed run's
``EmaUpdater`` therefore counts only its post-resume optimizer steps, so
``num_updates`` diverges from an uninterrupted run. Per the Task 3.7 brief's
STOP-and-report rule this gap is reported, NOT patched (test-only task). It is
benign for weight reproducibility: the EMA nudge ``target <- decay*target +
(1-decay)*online`` uses a constant weight ``1 - decay`` that is
``num_updates``-independent, and the target weights themselves live in
``JEPAModel.state_dict()`` (restored via the model payload). The tests below prove
the online AND target weights reproduce EXACTLY, and assert the landed
``num_updates`` values so the gap is documented rather than hidden.

Disclosure 3 -- PATH A realized as an uninterrupted reference run
----------------------------------------------------------------
The brief frames PATH A as "the same Trainer continues one more step". Re-calling
``fit()`` on a cold-started (never-resumed) Trainer would re-seed every RNG
(loop.py:774-775 seeds only when ``not self._resumed``), corrupting the
continuation. PATH A is therefore realized as a single uninterrupted straight run
to the same total step count -- the ground-truth continuation -- exactly as the
landed unit test ``test_resume_matches_uninterrupted_run`` (tests/unit/test_loop.py)
does. The checkpoint PATH B resumes from is written by a SIGUSR1 preemption (never
by validation, whose forward passes would advance the global RNG before the state
is captured).
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import torch

from fusion_jepa.models.jepa import TargetUpdatePolicy
from fusion_jepa.objectives.latent_prediction import LatentPredictionObjective
from fusion_jepa.training.distributed import DistributedManager
from fusion_jepa.training.ema import EmaUpdater
from fusion_jepa.training.loop import Trainer
from tests.fixtures.synthetic import make_synthetic_fusion_batch
from tests.unit.test_jepa import build_jepa_model

_MODALITIES = ("slow_ts", "profile")
_CPU = torch.device("cpu")
_EMA_DECAY = 0.9


# ── objective adapter + recording wrappers (disclosure 1) ─────────────────────


class _LatentObjectiveAdapter:
    """Bridge the Trainer's raw call form to the JEPA latent-prediction objective.

    The Trainer calls ``self(preds, batch.target, batch.target_mask)``; the JEPA
    objective wants ``obj(jepa_output, horizon_seconds)``. ``target``/
    ``target_mask`` are ignored (the ``JEPAOutput`` already carries everything the
    latent loss needs), and ``horizon_seconds`` -- labeling-only, never part of the
    loss value -- is a constant of the right ``[B, K]`` shape derived from ``z_hat``.
    """

    def __init__(self, distance: str = "cosine") -> None:
        self._objective = LatentPredictionObjective(distance=distance)

    def _loss(self, preds):
        batch, n_horizons = preds.z_hat.shape[0], preds.z_hat.shape[1]
        horizon_seconds = torch.ones(batch, n_horizons, dtype=torch.float64)
        return self._objective(preds, horizon_seconds)

    def __call__(self, preds, target, target_mask):
        return self._loss(preds)


class _RecordingAdapter(_LatentObjectiveAdapter):
    """Adapter that also records the consumed *training* batch order.

    On CPU ``_move_batch`` is a no-op, so the ``target`` dict the Trainer hands the
    objective is the batch's own object -- ``id(target)`` uniquely identifies the
    training batch. Validation batches (distinct objects, and seen under
    ``no_grad``) are excluded, so ``log`` is exactly the consumed training order.
    """

    def __init__(self, id_by_target, log, distance: str = "cosine") -> None:
        super().__init__(distance)
        self._id_by_target = id_by_target
        self._log = log

    def __call__(self, preds, target, target_mask):
        if id(target) in self._id_by_target and torch.is_grad_enabled():
            self._log.append(self._id_by_target[id(target)])
        return self._loss(preds)


class _RecordThenPreemptAdapter(_RecordingAdapter):
    """Recording adapter that raises ``sig`` to itself after ``fire_after`` steps.

    The loss is computed first (the forward completes), then the signal is
    delivered; the loop finishes the in-flight optimizer step, saves ``latest.pt``,
    and returns ``preempted`` -- a faithful SLURM-style preemption at a known step.
    """

    def __init__(
        self, id_by_target, log, *, fire_after: int, sig: int, distance="cosine"
    ) -> None:
        super().__init__(id_by_target, log, distance)
        self._fire_after = fire_after
        self._sig = sig
        self._train_calls = 0

    def __call__(self, preds, target, target_mask):
        out = self._loss(preds)
        # Count every *training* call (grad enabled; validation is no_grad),
        # independently of the id-map so preemption fires even when batch-order
        # recording is not requested (an empty id-map). Recording stays gated on
        # the map.
        if torch.is_grad_enabled():
            self._train_calls += 1
            if id(target) in self._id_by_target:
                self._log.append(self._id_by_target[id(target)])
            if self._train_calls == self._fire_after:
                os.kill(os.getpid(), self._sig)
        return out


# ── config / data / trainer plumbing (imitates test_loop.py) ──────────────────


def _make_cfg(total_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        experiment_name="resume-repro",
        training=SimpleNamespace(
            lr=3e-3,
            weight_decay=0.0,
            effective_batch_samples=2,
            micro_batch_samples=2,  # accumulation_steps == 1: one batch per step
            total_steps=total_steps,
            warmup_steps=1,
            min_lr=0.0,
            val_every_steps=100,  # no periodic validation; latest.pt via preemption
            val_max_batches=1,
            log_every=1,
            grad_clip_norm=1.0,
            bf16=False,
        ),
    )


def _make_batches(n: int, seed: int) -> list:
    return [
        make_synthetic_fusion_batch(
            B=2, modalities=_MODALITIES, n_channels=3, T=4, H=3, A=2, seed=seed + i
        )
        for i in range(n)
    ]


def _make_trainer(run_dir, *, total_steps, objective, train_loader, resume_from=None):
    """Build a fresh identically-seeded JEPA + EmaUpdater + Trainer on CPU."""
    model = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=0)
    ema_updater = EmaUpdater(model.target_encoder_pairs(), decay=_EMA_DECAY)
    dm = DistributedManager()
    trainer = Trainer(
        cfg=_make_cfg(total_steps),
        model=model,
        objective=objective,
        dm=dm,
        train_loader=train_loader,
        val_loader=_make_batches(1, seed=900),
        run_dir=run_dir,
        ema_updater=ema_updater,
        device=_CPU,
        resume_from=resume_from,
    )
    return trainer, dm, ema_updater


# ── state-capture helpers ─────────────────────────────────────────────────────


def _online_params(model) -> list[torch.Tensor]:
    """Trainable (online) parameters; the frozen EMA target params are excluded."""
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _target_params(model) -> list[torch.Tensor]:
    """The frozen EMA target-trunk parameters (tokenizers + encoder twins)."""
    return [
        p.detach().clone()
        for _online, target_module in model.target_encoder_pairs()
        for p in target_module.parameters()
    ]


def _adam_moments(optimizer) -> tuple[list, list]:
    """Per-parameter ``(exp_avg, exp_avg_sq)`` clones in optimizer-param order."""
    exp_avg, exp_avg_sq = [], []
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            avg = state.get("exp_avg")
            avg_sq = state.get("exp_avg_sq")
            exp_avg.append(avg.detach().clone() if avg is not None else None)
            exp_avg_sq.append(avg_sq.detach().clone() if avg_sq is not None else None)
    return exp_avg, exp_avg_sq


def _assert_exactly_equal(reference, resumed, label: str) -> None:
    assert len(reference) == len(resumed), f"{label}: length mismatch"
    for index, (a, b) in enumerate(zip(reference, resumed)):
        if a is None or b is None:
            assert a is None and b is None, f"{label}: one buffer missing at {index}"
            continue
        assert torch.equal(a, b), f"{label}: mismatch at {index}"


def _any_differs(a_list, b_list) -> bool:
    return any(not torch.equal(a, b) for a, b in zip(a_list, b_list))


# ── Test 1: resume reproduces the next optimization step exactly ──────────────


def test_resume_reproduces_next_optimization_step(tmp_path):
    """Resuming from a step-``k`` checkpoint and taking one step must match an
    uninterrupted run to step ``k + 1`` -- exactly, across the FULL training state:
    every online parameter, every EMA target parameter, and both AdamW moment
    buffers (``torch.equal``). The EMA ``num_updates`` counter is asserted at its
    landed values to document the resume gap (disclosure 2).
    """
    k, total = 3, 4  # checkpoint after k steps; one more step reaches `total`

    # Sanity anchor: an untrained model, so the equalities below are non-trivial.
    init_online = _online_params(
        build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=0)
    )

    # PATH A -- uninterrupted reference run to `total` (disclosure 3).
    ref_batches = _make_batches(3, seed=100)
    t_ref, dm_ref, ema_ref = _make_trainer(
        tmp_path / "reference",
        total_steps=total,
        objective=_LatentObjectiveAdapter(),
        train_loader=ref_batches,
    )
    t_ref.fit()
    ref_model = dm_ref.unwrap(t_ref.model)
    ref_online = _online_params(ref_model)
    ref_target = _target_params(ref_model)
    ref_exp_avg, ref_exp_avg_sq = _adam_moments(t_ref.optimizer)
    ref_num_updates = ema_ref.num_updates
    dm_ref.close()

    # Training genuinely moved the online weights (equality is not vacuous).
    assert _any_differs(init_online, ref_online)

    # PATH B leg 1 -- preempt exactly at step k, writing latest.pt (no validation).
    resume_batches = _make_batches(3, seed=100)
    first_obj = _RecordThenPreemptAdapter({}, [], fire_after=k, sig=signal.SIGUSR1)
    t_first, dm_first, _ = _make_trainer(
        tmp_path / "resume",
        total_steps=total,
        objective=first_obj,
        train_loader=resume_batches,
    )
    r_first = t_first.fit()
    dm_first.close()
    assert r_first.status == "preempted"
    assert r_first.final_step == k

    # PATH B leg 2 -- fresh Trainer + fresh identically-seeded model, resume + 1 step.
    t_resume, dm_resume, ema_resume = _make_trainer(
        tmp_path / "resume",
        total_steps=total,
        objective=_LatentObjectiveAdapter(),
        train_loader=resume_batches,
        resume_from="auto",
    )
    assert t_resume.step == k  # restored the step counter from latest.pt
    assert ema_resume.num_updates == 0  # disclosure 2: EMA counter NOT restored
    t_resume.fit()
    resume_model = dm_resume.unwrap(t_resume.model)
    resume_online = _online_params(resume_model)
    resume_target = _target_params(resume_model)
    resume_exp_avg, resume_exp_avg_sq = _adam_moments(t_resume.optimizer)
    resume_num_updates = ema_resume.num_updates
    dm_resume.close()

    # Exact reproduction of every weight and optimizer moment buffer.
    _assert_exactly_equal(ref_online, resume_online, "online params")
    _assert_exactly_equal(ref_target, resume_target, "target (EMA) params")
    _assert_exactly_equal(ref_exp_avg, resume_exp_avg, "AdamW exp_avg")
    _assert_exactly_equal(ref_exp_avg_sq, resume_exp_avg_sq, "AdamW exp_avg_sq")

    # EMA num_updates (disclosure 2): the uninterrupted run counts one update per
    # optimizer step (== total); the resumed updater counts ONLY its post-resume
    # steps because _restore never loads payload["target_encoder"]. The target
    # WEIGHTS above still match exactly (constant, num_updates-independent nudge).
    assert ref_num_updates == total
    assert resume_num_updates == total - k
    assert ref_num_updates != resume_num_updates


# ── Test 2: resume reproduces final weights + batch order after an interrupt ──


def test_resume_reproduces_final_weights_after_interrupt(tmp_path):
    """A straight run to ``n`` steps must be indistinguishable from a run preempted
    at ``m`` and resumed to ``n``: identical final online AND target weights, and
    identical consumed training-batch order. A plain-list loader (no stateful
    sampler) forces the loop's own cursor-skip to resume the data stream.
    """
    total, interrupt = 6, 4

    # Straight reference run; record its consumed training-batch order.
    straight_batches = _make_batches(3, seed=100)
    straight_ids = {id(b.target): i for i, b in enumerate(straight_batches)}
    straight_log: list[int] = []
    t_straight, dm_straight, _ = _make_trainer(
        tmp_path / "straight",
        total_steps=total,
        objective=_RecordingAdapter(straight_ids, straight_log),
        train_loader=straight_batches,
    )
    t_straight.fit()
    straight_model = dm_straight.unwrap(t_straight.model)
    straight_online = _online_params(straight_model)
    straight_target = _target_params(straight_model)
    dm_straight.close()

    # Leg 1: preempt after `interrupt` steps.
    resume_batches = _make_batches(3, seed=100)
    resume_ids = {id(b.target): i for i, b in enumerate(resume_batches)}
    first_log: list[int] = []
    first_obj = _RecordThenPreemptAdapter(
        resume_ids, first_log, fire_after=interrupt, sig=signal.SIGUSR1
    )
    t_first, dm_first, _ = _make_trainer(
        tmp_path / "interrupted",
        total_steps=total,
        objective=first_obj,
        train_loader=resume_batches,
    )
    r_first = t_first.fit()
    dm_first.close()
    assert r_first.status == "preempted"
    assert r_first.final_step == interrupt

    # Leg 2: resume from latest.pt and run to `total`.
    resumed_log: list[int] = []
    t_resume, dm_resume, _ = _make_trainer(
        tmp_path / "interrupted",
        total_steps=total,
        objective=_RecordingAdapter(resume_ids, resumed_log),
        train_loader=resume_batches,
        resume_from="auto",
    )
    assert t_resume.step == interrupt
    t_resume.fit()
    resume_model = dm_resume.unwrap(t_resume.model)
    resume_online = _online_params(resume_model)
    resume_target = _target_params(resume_model)
    dm_resume.close()

    # Consumed batch order: the two legs concatenated equal the straight run, and
    # the resume genuinely continued (did not replay the interrupted epoch).
    assert first_log == [0, 1, 2, 0]
    assert first_log + resumed_log == straight_log
    assert resumed_log and resumed_log != straight_log[: len(resumed_log)]

    # Identical final online AND target (EMA) weights.
    _assert_exactly_equal(straight_online, resume_online, "final online params")
    _assert_exactly_equal(straight_target, resume_target, "final target params")
