"""Unit tests for ``fusion_jepa.training.distributed`` (Task 2.10).

Covers the single-process defaults, the non-distributed identity paths
(``wrap`` / ``all_reduce_mean``), the idempotent-and-safe ``close()`` that
replaces the fragile ``__del__`` destructor, the device-availability
backend selection, and a real two-process gloo integration test that
verifies ``all_reduce_mean`` reproduces the single-process global mean.

Everything runs offline and CPU-only. The integration test uses
``torch.multiprocessing.spawn`` with a ``file://`` rendezvous under
``tmp_path`` (never a fixed TCP port -- login/CI nodes collide) and gloo,
so it needs no GPU. It is kept small (8 scalars over 2 ranks) and
deterministic so it finishes well under a second.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from fusion_jepa.training.distributed import (
    DistributedManager,
    _default_backend,
)


def _clear_dist_env(monkeypatch) -> None:
    """Force the plain single-process path regardless of the launcher env.

    A stray ``MASTER_PORT`` / ``WORLD_SIZE`` (e.g. an interactive srun
    allocation) would otherwise push ``DistributedManager`` down its
    distributed branch and try to init a process group.
    """
    for var in ("MASTER_PORT", "MASTER_ADDR", "WORLD_SIZE", "RANK",
                "LOCAL_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(var, raising=False)


def test_single_process_defaults_world_size_one(monkeypatch) -> None:
    """With no launcher env, the manager is a rank-0 world-size-1 no-op."""
    _clear_dist_env(monkeypatch)

    dm = DistributedManager()
    try:
        assert dm.world_size == 1
        assert dm.rank == 0
        assert dm.local_rank == 0
        assert dm.distributed is False
        assert dm.is_main is True
    finally:
        dm.close()


def test_wrap_is_identity_when_not_distributed(monkeypatch) -> None:
    """``wrap`` returns the exact same module object outside DDP."""
    _clear_dist_env(monkeypatch)

    dm = DistributedManager()
    try:
        model = nn.Linear(3, 2)
        wrapped = dm.wrap(model)
        assert wrapped is model
        # ``unwrap`` is the inverse and equally a no-op here.
        assert dm.unwrap(wrapped) is model
    finally:
        dm.close()


def test_all_reduce_mean_identity_when_not_distributed(monkeypatch) -> None:
    """Outside a process group, ``all_reduce_mean`` returns the input."""
    _clear_dist_env(monkeypatch)

    dm = DistributedManager()
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        out = dm.all_reduce_mean(x)
        assert torch.equal(out, x)
    finally:
        dm.close()


def test_close_is_idempotent_and_safe_when_never_initialized(
    monkeypatch,
) -> None:
    """``close()`` never raises and is safe to call repeatedly.

    The single-process manager never created a process group, so
    ``close()`` must be a pure no-op even when called twice.
    """
    _clear_dist_env(monkeypatch)

    dm = DistributedManager()
    dm.close()
    dm.close()  # second call must not raise


def test_default_backend_follows_device_availability(monkeypatch) -> None:
    """``nccl`` when a CUDA/ROCm device is visible, ``gloo`` otherwise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _default_backend() == "nccl"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _default_backend() == "gloo"


def test_barrier_and_gather_concat_are_noops_when_not_distributed(
    monkeypatch,
) -> None:
    """Collectives degrade to no-ops / identity in a single process."""
    _clear_dist_env(monkeypatch)

    dm = DistributedManager()
    try:
        dm.barrier()  # must not raise
        x = torch.arange(4)
        assert torch.equal(dm.gather_concat(x), x)
    finally:
        dm.close()


def _reduce_worker(
    rank: int,
    world_size: int,
    init_file: str,
    shards: list[list[float]],
    result_path: str,
) -> None:
    """Rank body for the two-process gloo integration test.

    Must live at module scope so ``mp.spawn`` (spawn start method) can
    pickle it. Each rank averages the square of its own shard, then
    ``all_reduce_mean`` combines the equal-sized per-rank means into the
    global mean; rank 0 writes it out for the parent to compare.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    dm = DistributedManager(backend="gloo", init_method=f"file://{init_file}")
    try:
        shard = torch.tensor(shards[rank], dtype=torch.float64)
        local_loss = shard.pow(2).mean()
        global_loss = dm.all_reduce_mean(local_loss)
        assert dm.world_size == world_size
        assert dm.distributed is True
        if dm.is_main:
            torch.save(global_loss.detach().cpu(), result_path)
    finally:
        dm.close()


def test_two_process_gloo_global_loss_matches_single_process(
    tmp_path,
) -> None:
    """A 2-rank gloo ``all_reduce_mean`` equals the full-data mean.

    Equal-sized shards make mean-of-per-rank-means identical to the mean
    over the concatenated data, so the distributed reduction must match
    the single-process reference exactly.
    """
    world_size = 2
    data = torch.arange(8, dtype=torch.float64)
    shards = [data[:4].tolist(), data[4:].tolist()]
    expected = data.pow(2).mean().item()

    init_file = tmp_path / "rendezvous"
    result_path = tmp_path / "global_loss.pt"

    mp.spawn(
        _reduce_worker,
        args=(world_size, str(init_file), shards, str(result_path)),
        nprocs=world_size,
        join=True,
    )

    got = torch.load(result_path).item()
    assert got == pytest.approx(expected)


def _wrap_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_path: str,
) -> None:
    """Rank body: wrap a CPU model in an explicit-gloo group and train it."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    dm = DistributedManager(backend="gloo", init_method=f"file://{init_file}")
    try:
        wrapped = dm.wrap(nn.Linear(3, 2))
        assert isinstance(wrapped, DistributedDataParallel)
        # The explicit gloo override is a CPU group: no GPU device_ids may
        # be attached, however many devices the host exposes (the reviewed
        # regression keyed device_ids on torch.cuda.is_available(), which
        # misbinds a CPU module on a GPU-visible host).
        assert not wrapped.device_ids
        out = wrapped(torch.ones(4, 3))
        out.sum().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in wrapped.parameters()
        )
        if dm.is_main:
            Path(result_path).write_text("ok", encoding="utf-8")
    finally:
        dm.close()


def test_two_process_gloo_wrap_produces_cpu_ddp(tmp_path) -> None:
    """``wrap`` keys ``device_ids`` on the RESOLVED backend, not on device
    visibility: a CPU model in an explicit-gloo group must become a working
    CPU DDP module with no GPU device_ids (Codex 2.10 review finding)."""
    init_file = tmp_path / "wrap_rendezvous"
    result_path = tmp_path / "wrap_ok"

    mp.spawn(
        _wrap_worker,
        args=(2, str(init_file), str(result_path)),
        nprocs=2,
        join=True,
    )

    assert result_path.read_text(encoding="utf-8") == "ok"
