"""The M2 training loop -- the milestone merge point (Task 2.12).

:class:`Trainer` composes every landed Milestone-2 component into one
resumable, preemptible training run:

* the raw world model (:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`)
  called as ``model(batch) -> {modality: preds}``;
* the mask-aware objective, called as
  ``objective(predictions, batch.target, batch.target_mask) -> LossOutput``;
* :class:`~fusion_jepa.training.distributed.DistributedManager` (``dm``) for the
  device, rank bookkeeping and cross-rank metric reduction;
* the atomic checkpoint I/O and full-state capture of
  :mod:`fusion_jepa.training.checkpoint` (the ``CHECKPOINT_KEYS`` payload
  contract);
* :class:`~fusion_jepa.utils.logging.MetricsLogger` (JSONL per-step metrics),
  :func:`~fusion_jepa.utils.reproducibility.derive_seed` (the single seed
  source), :func:`~fusion_jepa.utils.accounting.token_throughput_summary`, and
  :func:`~fusion_jepa.utils.run_artifacts.write_completion`.

The ten contracts locked by ``tests/unit/test_loop.py`` are documented at each
method. See ``.superpowers/sdd/task-2.12-report.md`` for every design
disclosure (RunResult fields, resume auto-detection, failure re-raise policy,
loader/split convention, the CPU-safe ``device`` override, and the
``target_encoder``/``scaler``/``upstream_manifest`` payload placeholders).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fusion_jepa.training.checkpoint import (
    CHECKPOINT_KEYS,
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from fusion_jepa.utils.accounting import token_throughput_summary
from fusion_jepa.utils.logging import MetricsLogger
from fusion_jepa.utils.reproducibility import derive_seed, seed_everything
from fusion_jepa.utils.run_artifacts import write_completion

_UNSET = object()
_PREEMPT_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)


# ── Autocast decision (bf16 is CUDA-only) ─────────────────────────────────


def should_use_bf16_autocast(device: torch.device | str, bf16_enabled: bool) -> bool:
    """Return whether bf16 autocast should wrap the forward pass.

    bf16 autocast engages *only* when the run device is CUDA (which
    :class:`DistributedManager` selects only when a device is actually
    available) **and** the config enables it. On CPU it is always refused, so
    the metric-accumulation-in-fp32 invariant holds trivially there.
    """
    return bool(bf16_enabled) and torch.device(device).type == "cuda"


def resolve_autocast(device: torch.device | str, bf16_enabled: bool):
    """Return the autocast context manager for the forward pass.

    A real :func:`torch.amp.autocast` (bf16) on CUDA when enabled, otherwise a
    :class:`contextlib.nullcontext` -- so the CPU path runs in fp32.
    """
    if should_use_bf16_autocast(device, bf16_enabled):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ── Minimal resumable sampler ─────────────────────────────────────────────


class ResumableSampler:
    """A minimal index sampler carrying ``(epoch, cursor)`` resume state.

    This is the explicit coordination point with the TokaMark loader: the loop
    checkpoints ``sampler.state_dict()`` and restores it on resume, so a run
    continues from the exact within-epoch position it was interrupted at. Per-
    epoch shuffling is seeded via ``derive_seed(root_seed, "sampler", epoch)``,
    so the order is a deterministic function of the root seed and epoch and is
    identical across processes and restarts.

    Args:
        data_source: an ``int`` length, or any sized object (``len(...)``).
        root_seed: root seed threaded through :func:`derive_seed`.
        shuffle: per-epoch permutation when ``True``, identity order otherwise.
    """

    def __init__(
        self,
        data_source: int | Sized,
        root_seed: int = 0,
        *,
        shuffle: bool = True,
    ) -> None:
        self._n = int(data_source) if isinstance(data_source, int) else len(data_source)
        if self._n < 0:
            raise ValueError(f"data_source length must be >= 0, got {self._n}")
        self.root_seed = int(root_seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.cursor = 0

    def _order(self) -> list[int]:
        if not self.shuffle:
            return list(range(self._n))
        generator = torch.Generator()
        generator.manual_seed(derive_seed(self.root_seed, "sampler", self.epoch))
        return torch.randperm(self._n, generator=generator).tolist()

    def __iter__(self):
        order = self._order()
        while self.cursor < self._n:
            index = order[self.cursor]
            self.cursor += 1
            yield index
        # Epoch exhausted: advance to the next epoch and reset the cursor so a
        # fresh ``iter(sampler)`` reshuffles under the next derived seed.
        self.epoch += 1
        self.cursor = 0

    def __len__(self) -> int:
        return self._n

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])


# ── Run result ────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Terminal summary of a :meth:`Trainer.fit` call.

    Attributes:
        status: ``"completed"``, ``"preempted"``, or ``"failed"``.
        final_step: optimizer steps completed when fit returned.
        best_val_loss: lowest validation loss seen (``None`` if never validated).
        best_step: optimizer step at which ``best_val_loss`` was recorded.
        last_val_loss: most recent validation loss (``None`` if never validated).
        run_dir: the run directory path (as a string).
        failure_reason: ``repr`` of the exception on ``"failed"`` runs, else
            ``None``.
    """

    status: str
    final_step: int
    best_val_loss: float | None
    best_step: int
    last_val_loss: float | None
    run_dir: str
    failure_reason: str | None = None


# ── Config access helper ──────────────────────────────────────────────────


def _cfg_get(cfg: Any, path: str, default: Any = _UNSET) -> Any:
    """Read a dotted ``path`` from ``cfg`` (mapping/attr/OmegaConf agnostic).

    An explicit ``None`` value is treated as absent: numeric training settings
    are never legitimately ``None``, and ``grad_clip_norm=None`` means "no
    clip", which the caller supplies via ``default=None``.
    """
    node = cfg
    for key in path.split("."):
        if node is None:
            break
        try:
            node = node[key]
        except (TypeError, KeyError, IndexError):
            node = getattr(node, key, None)
    if node is None:
        if default is _UNSET:
            raise ValueError(f"cfg is missing required training setting {path!r}")
        return default
    return node


def _git_commit() -> str | None:
    """Best-effort short-lived lookup of the repo HEAD commit, else ``None``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
        )
        return out.stdout.decode().strip()
    except Exception:
        return None


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


# ── Trainer ───────────────────────────────────────────────────────────────


class Trainer:
    """Composes the M2 components into a resumable, preemptible training run.

    Args:
        cfg: run config; the loop reads ``cfg.seed`` and a ``cfg.training``
            block (``lr``, ``weight_decay``, ``effective_batch_samples``,
            ``micro_batch_samples``, ``total_steps``, ``warmup_steps``,
            ``min_lr``, ``val_every_steps``, ``val_max_batches``, ``log_every``,
            ``grad_clip_norm``, ``bf16``). Mapping, attribute, and OmegaConf
            configs are all supported.
        model: an ``nn.Module`` called as ``model(batch) -> {modality: preds}``.
        objective: called as ``objective(preds, batch.target, batch.target_mask)``
            returning a :class:`~fusion_jepa.objectives.base.LossOutput`.
        dm: a :class:`DistributedManager` (single-process works CPU-only).
        train_loader: an iterable of ``FusionBatch`` (a ``DataLoader`` or a list
            of prebuilt batches); it is cycled across epochs to ``total_steps``.
        val_loader: an iterable of ``FusionBatch`` for fixed-step validation.
        run_dir: directory receiving ``metrics.jsonl``, ``best.pt``,
            ``latest.pt`` and ``completion.json``.
        ema_updater: optional EMA target-encoder updater (JEPA, M3); ``None``
            for the raw baseline.
        device: device override (defaults to ``dm.device``) -- lets the CPU-only
            suite run regardless of host GPU visibility.
        resume_from: ``None`` (cold start), ``"auto"``/``True`` (resume from
            ``run_dir/latest.pt`` when present), or a checkpoint path.
        val_split: explicit split label for the refusal check; falls back to
            ``val_loader.split`` then ``val_loader.dataset.split``.
        reraise_failures: when ``True`` (default) an unhandled exception is
            re-raised after ``completion.json`` is written (fail-loud); when
            ``False``, ``fit`` returns ``RunResult(status="failed")`` instead.
    """

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        objective: Any,
        dm: Any,
        train_loader: Iterable,
        val_loader: Iterable,
        run_dir: str | os.PathLike[str],
        ema_updater: Any = None,
        *,
        device: torch.device | None = None,
        resume_from: Any = None,
        val_split: str | None = None,
        reraise_failures: bool = True,
    ) -> None:
        self.cfg = cfg
        self.objective = objective
        self.dm = dm
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.ema_updater = ema_updater
        self.device = device if device is not None else dm.device
        self.reraise_failures = bool(reraise_failures)

        # Contract 4: refuse a test-split val loader at construction.
        self._refuse_test_split(val_split)

        # Training hyperparameters.
        self.root_seed = int(_cfg_get(cfg, "seed", 0))
        self.lr = float(_cfg_get(cfg, "training.lr"))
        self.weight_decay = float(_cfg_get(cfg, "training.weight_decay", 0.0))
        self.effective_batch_samples = int(
            _cfg_get(cfg, "training.effective_batch_samples")
        )
        self.micro_batch_samples = int(
            _cfg_get(cfg, "training.micro_batch_samples")
        )
        self.total_steps = int(_cfg_get(cfg, "training.total_steps"))
        self.warmup_steps = int(_cfg_get(cfg, "training.warmup_steps", 0))
        self.warmup_start_factor = float(
            _cfg_get(cfg, "training.warmup_start_factor", 1e-3)
        )
        self.min_lr = float(_cfg_get(cfg, "training.min_lr", 0.0))
        self.val_every_steps = int(_cfg_get(cfg, "training.val_every_steps"))
        val_max = _cfg_get(cfg, "training.val_max_batches", None)
        self.val_max_batches = int(val_max) if val_max is not None else None
        self.log_every = int(_cfg_get(cfg, "training.log_every", 1))
        clip = _cfg_get(cfg, "training.grad_clip_norm", None)
        self.grad_clip_norm = float(clip) if clip is not None and clip > 0 else None
        self._bf16 = bool(_cfg_get(cfg, "training.bf16", False))

        # Contract 1: grad accumulation math with a divisibility assert.
        denom = self.micro_batch_samples * self.dm.world_size
        if denom <= 0:
            raise ValueError(
                "micro_batch_samples * world_size must be positive, got "
                f"{denom}"
            )
        if self.effective_batch_samples % denom != 0:
            raise ValueError(
                f"effective_batch_samples ({self.effective_batch_samples}) must "
                f"be divisible by micro_batch_samples * world_size ({denom})"
            )
        self.accumulation_steps = self.effective_batch_samples // denom

        # Model, optimizer, scheduler. The optimizer is built over the pre-wrap
        # parameters (DDP shares the same tensors), so its state indices are
        # stable across resume.
        model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.model = self.dm.wrap(model)

        # Run state.
        self.step = 0
        self.epoch = 0
        self._epoch_cursor = 0
        self._best_val_loss = float("inf")
        self._best_step = 0
        self._last_val_step = -1
        self._last_val_loss: float | None = None
        self._train_iter = None
        self._resumed = False
        self._preempt_signal: int | None = None
        self._started_at: str | None = None
        self._git_commit = _git_commit()

        resume_path = self._resolve_resume_path(resume_from)
        if resume_path is not None:
            self._restore(resume_path)

    # ── construction helpers ──────────────────────────────────────────────

    def _refuse_test_split(self, val_split: str | None) -> None:
        split = val_split
        if split is None:
            split = getattr(self.val_loader, "split", None)
        if split is None:
            split = getattr(getattr(self.val_loader, "dataset", None), "split", None)
        if split is not None and str(split) == "test":
            raise ValueError(
                "Trainer refuses a test-split val_loader: evaluating on the "
                "held-out test split during training is not permitted "
                "(got split='test')."
            )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Linear warmup then cosine decay, stepped once per optimizer step.

        Mirrors ``scripts/training/train_e2e_stage1.py:_build_scheduler``: a
        :class:`~torch.optim.lr_scheduler.LinearLR` warmup chained to a
        :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` via
        :class:`~torch.optim.lr_scheduler.SequentialLR`.
        """
        warmup_iters = max(self.warmup_steps, 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        cosine_steps = max(self.total_steps - self.warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_steps, eta_min=self.min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[warmup_iters]
        )

    def _resolve_resume_path(self, resume_from: Any) -> Path | None:
        if resume_from is None or resume_from is False:
            return None
        if resume_from is True or resume_from == "auto":
            latest = self.run_dir / "latest.pt"
            return latest if latest.exists() else None
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path

    def _restore(self, path: Path) -> None:
        """Restore model/optimizer/scheduler/RNG/sampler/step from a checkpoint."""
        payload = load_checkpoint(path, map_location=self.device)
        self.dm.unwrap(self.model).load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        restore_rng_states(payload["rng_states"])
        self.step = int(payload["step"])
        self.epoch = int(payload["epoch"])
        best = payload["best_metric"]
        self._best_val_loss = float(best) if best is not None else float("inf")
        self._restore_sampler(payload.get("sampler_state"))
        self._resumed = True

    def _restore_sampler(self, sampler_state: Any) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if (
            sampler is not None
            and hasattr(sampler, "load_state_dict")
            and isinstance(sampler_state, Mapping)
        ):
            sampler.load_state_dict(sampler_state)
        if isinstance(sampler_state, Mapping):
            self.epoch = int(sampler_state.get("epoch", self.epoch))
            self._epoch_cursor = int(sampler_state.get("cursor", 0))

    # ── device / batch helpers ────────────────────────────────────────────

    def _move_batch(self, batch):
        """Move a ``FusionBatch``'s tensors to ``self.device`` (no-op on CPU)."""
        if self.device.type == "cpu":
            return batch

        def move(value):
            return value.to(self.device) if isinstance(value, torch.Tensor) else value

        def move_dict(mapping):
            return {key: move(value) for key, value in mapping.items()}

        return dataclasses.replace(
            batch,
            context=move_dict(batch.context),
            context_mask=move_dict(batch.context_mask),
            target=move_dict(batch.target),
            target_mask=move_dict(batch.target_mask),
            actions=move(batch.actions),
            action_mask=move(batch.action_mask),
            context_times=move(batch.context_times),
            target_times=move(batch.target_times),
            action_times=move(batch.action_times),
            horizon_seconds=move(batch.horizon_seconds),
            device_context=move(batch.device_context),
            device_context_mask=move(batch.device_context_mask),
        )

    @staticmethod
    def _count_tokens(batch) -> int:
        """Context elements fed to the tokenizers -- a real throughput proxy."""
        return sum(int(values.numel()) for values in batch.context.values())

    @staticmethod
    def _count_samples(batch) -> int:
        return int(batch.actions.shape[0])

    def _autocast(self):
        return resolve_autocast(self.device, self._bf16)

    def _maybe_no_sync(self, micro_index: int):
        """DDP gradient-sync suppression on non-final accumulation micro-steps."""
        is_final = micro_index == self.accumulation_steps - 1
        if not is_final and hasattr(self.model, "no_sync"):
            return self.model.no_sync()
        return contextlib.nullcontext()

    def _next_batch(self):
        """Pull the next train batch, cycling epochs; return ``(batch, wait_s)``."""
        start = time.perf_counter()
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self.epoch += 1
            self._epoch_cursor = 0
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        self._epoch_cursor += 1
        return batch, time.perf_counter() - start

    # ── signal handling ───────────────────────────────────────────────────

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Install SIGUSR1/SIGTERM handlers; return the previous handlers.

        No-op (returns empty) when not on the main thread, where
        :func:`signal.signal` is unavailable.
        """
        if threading.current_thread() is not threading.main_thread():
            return {}

        def handler(signum, _frame):
            self._preempt_signal = signum

        prior: dict[int, Any] = {}
        for sig in _PREEMPT_SIGNALS:
            prior[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        return prior

    @staticmethod
    def _restore_signal_handlers(prior: dict[int, Any]) -> None:
        for sig, previous in prior.items():
            signal.signal(sig, previous)

    # ── checkpoint payload ────────────────────────────────────────────────

    def _sampler_state(self) -> dict[str, int]:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "state_dict"):
            return sampler.state_dict()
        return {"epoch": self.epoch, "cursor": self._epoch_cursor}

    def _ema_state_dict(self) -> Any:
        if self.ema_updater is not None and hasattr(self.ema_updater, "state_dict"):
            return self.ema_updater.state_dict()
        return None

    def _build_payload(self) -> dict[str, Any]:
        payload = {
            "model": self.dm.unwrap(self.model).state_dict(),
            "target_encoder": self._ema_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": None,
            "step": self.step,
            "epoch": self.epoch,
            "best_metric": _finite_or_none(self._best_val_loss),
            "rng_states": capture_rng_states(),
            "sampler_state": self._sampler_state(),
            "resolved_config": self.cfg,
            "git_commit": self._git_commit,
            "upstream_manifest": None,
        }
        assert set(payload) == set(CHECKPOINT_KEYS)
        return payload

    def _save_checkpoint(self, *, is_best: bool) -> None:
        if not self.dm.is_main:
            return
        name = "best.pt" if is_best else "latest.pt"
        save_checkpoint(self.run_dir / name, self._build_payload())

    def _write_completion(self, status: str, failure_reason: str | None = None) -> None:
        if not self.dm.is_main:
            return
        write_completion(
            self.run_dir,
            status=status,
            started_at=self._started_at,
            warnings=[],
            failure_reason=failure_reason,
            final_step=self.step,
            best_step=self._best_step,
            best_val_loss=_finite_or_none(self._best_val_loss),
            last_val_loss=self._last_val_loss,
        )

    def _result(self, status: str, failure_reason: str | None = None) -> RunResult:
        return RunResult(
            status=status,
            final_step=self.step,
            best_val_loss=_finite_or_none(self._best_val_loss),
            best_step=self._best_step,
            last_val_loss=self._last_val_loss,
            run_dir=str(self.run_dir),
            failure_reason=failure_reason,
        )

    # ── validation ────────────────────────────────────────────────────────

    def _validate(self) -> float:
        """Fixed-step validation over up to ``val_max_batches`` batches (fp32)."""
        self.model.eval()
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        count = 0
        with torch.no_grad():
            for index, batch in enumerate(self.val_loader):
                if self.val_max_batches is not None and index >= self.val_max_batches:
                    break
                batch = self._move_batch(batch)
                with self._autocast():
                    preds = self.model(batch)
                    loss_out = self.objective(
                        preds, batch.target, batch.target_mask
                    )
                total = total + loss_out.total.float()
                count += 1
        self.model.train()
        local = total / max(count, 1)
        reduced = self.dm.all_reduce_mean(local)
        return float(reduced.item())

    def _validate_and_checkpoint(self, logger: MetricsLogger | None) -> None:
        val_loss = self._validate()
        self._last_val_step = self.step
        self._last_val_loss = val_loss
        if logger is not None:
            logger.log(step=self.step, event="val", val_loss=val_loss)
        # ``latest.pt`` is written at every save point; ``best.pt`` only when the
        # validation loss improves. They are always distinct files.
        self._save_checkpoint(is_best=False)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_step = self.step
            self._save_checkpoint(is_best=True)
        self.dm.barrier()

    # ── optimizer step ────────────────────────────────────────────────────

    def _clip_grads(self) -> float:
        max_norm = self.grad_clip_norm if self.grad_clip_norm is not None else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        return float(grad_norm)

    def _run_optimizer_step(self, logger: MetricsLogger | None) -> None:
        step_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        term_sums: dict[str, float] = {}
        data_wait = 0.0
        tokens = 0
        samples = 0

        for micro in range(self.accumulation_steps):
            batch, wait = self._next_batch()
            data_wait += wait
            batch = self._move_batch(batch)
            with self._maybe_no_sync(micro):
                with self._autocast():
                    preds = self.model(batch)
                    loss_out = self.objective(
                        preds, batch.target, batch.target_mask
                    )
                # Scale by 1/accum so the accumulated gradient equals the mean
                # over the effective batch; optimizer.step() runs once below.
                (loss_out.total / self.accumulation_steps).backward()
            total_loss += float(loss_out.total.detach().item())
            for name, value in loss_out.terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + float(
                    value.detach().item()
                )
            tokens += self._count_tokens(batch)
            samples += self._count_samples(batch)

        grad_norm = self._clip_grads()
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()

        step_wall = time.perf_counter() - step_start
        if logger is not None and self.step % self.log_every == 0:
            self._log_step(
                logger,
                total_loss=total_loss,
                term_sums=term_sums,
                grad_norm=grad_norm,
                data_wait=data_wait,
                tokens=tokens,
                samples=samples,
                step_wall=step_wall,
            )

    def _update_ema(self) -> None:
        if self.ema_updater is None:
            return
        target = self.dm.unwrap(self.model)
        if hasattr(self.ema_updater, "update"):
            self.ema_updater.update(target)
        elif callable(self.ema_updater):
            self.ema_updater(target)

    def _log_step(
        self,
        logger: MetricsLogger,
        *,
        total_loss: float,
        term_sums: Mapping[str, float],
        grad_norm: float,
        data_wait: float,
        tokens: int,
        samples: int,
        step_wall: float,
    ) -> None:
        wall = max(step_wall, 1e-9)
        world = self.dm.world_size
        throughput = token_throughput_summary(tokens * world, wall, world)
        metrics: dict[str, Any] = {
            "event": "train_step",
            "epoch": self.epoch,
            "loss": total_loss / self.accumulation_steps,
            "grad_norm": grad_norm,
            "lr": float(self.scheduler.get_last_lr()[0]),
            "tokens_per_s": throughput["tokens_per_s"],
            "samples_per_s": samples * world / wall,
            "data_wait_s": data_wait,
            "wall_s": step_wall,
        }
        for name, value in term_sums.items():
            metrics[f"loss_term/{name}"] = value / self.accumulation_steps
        if self.device.type == "cuda":
            metrics["gpu_mem_bytes"] = float(
                torch.cuda.max_memory_allocated(self.device)
            )
        logger.log(step=self.step, **metrics)

    # ── public API ────────────────────────────────────────────────────────

    def fit(self) -> RunResult:
        """Run training to ``total_steps`` (or until preempted/failed)."""
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._preempt_signal = None
        prior_handlers = self._install_signal_handlers()

        logger: MetricsLogger | None = None
        if self.dm.is_main:
            logger = MetricsLogger(self.run_dir / "metrics.jsonl")

        # Cold start seeds every RNG from the single derive_seed source; resume
        # restores RNG state instead, so it must not reseed.
        if not self._resumed:
            seed_everything(derive_seed(self.root_seed, "trainer", self.dm.rank))

        self.model.train()

        try:
            if logger is not None:
                logger.log(
                    step=self.step,
                    event="train_config",
                    effective_batch_samples=self.effective_batch_samples,
                    micro_batch_samples=self.micro_batch_samples,
                    world_size=self.dm.world_size,
                    accumulation_steps=self.accumulation_steps,
                    total_steps=self.total_steps,
                )
            return self._train(logger)
        except BaseException as exc:  # noqa: BLE001 - record then re-raise/return
            self._write_completion("failed", failure_reason=repr(exc))
            if self.reraise_failures:
                raise
            return self._result("failed", failure_reason=repr(exc))
        finally:
            self._restore_signal_handlers(prior_handlers)
            if logger is not None:
                logger.close()

    def _train(self, logger: MetricsLogger | None) -> RunResult:
        while self.step < self.total_steps:
            self._run_optimizer_step(logger)

            if self._preempt_signal is not None:
                # Finish-the-current-step preemption: save latest, record the
                # preempted status, and hand back to the CLI (which exits 0).
                self._save_checkpoint(is_best=False)
                self._write_completion("preempted")
                return self._result("preempted")

            if self.step % self.val_every_steps == 0:
                self._validate_and_checkpoint(logger)

        # Final validation + checkpoint if the last step was not a val step.
        if self._last_val_step != self.step:
            self._validate_and_checkpoint(logger)

        self._write_completion("completed")
        return self._result("completed")
