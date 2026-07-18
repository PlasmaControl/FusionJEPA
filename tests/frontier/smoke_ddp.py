"""Frontier DDP smoke payload (Task 2.13).

NOT a pytest test. This file has no ``test_`` prefix on purpose, so pytest's
default collection skips it; it is executed directly, one process per GCD, by
``slurm/smoke.slurm`` via the per-rank wrapper::

    srun ... scripts/slurm_frontier/_srun_rank_wrapper.sh tests/frontier/smoke_ddp.py

(the wrapper already prepends ``python -u``, so the script path is passed WITHOUT
a leading ``python``). It proves, on real Frontier hardware:

1. per-rank identity -- RANK, SLURM_PROCID, LOCAL_RANK, ROCR_VISIBLE_DEVICES,
   and ``torch.cuda.get_device_name``;
2. rank uniqueness (STRICT: the gathered ranks are exactly ``0..world_size-1``)
   plus GPU-binding distinctness (HARD-FAIL): a device-uuid collision fails the
   job; when uuids are unavailable (e.g. ROCm) but ROCR_VISIBLE_DEVICES is set, a
   ``(hostname, ROCR_VISIBLE_DEVICES)`` collision across ranks also fails the job
   (under ``--gpus-per-task=1`` each rank's mask must be distinct per node). Only
   when uuids are unavailable AND ROCR_VISIBLE_DEVICES is entirely unset does the
   check fall back to a WARNING that the operator must verify uniqueness manually;
3. an all-reduce checksum: each rank contributes ``rank + 1`` and every rank
   asserts the SUM equals ``world_size * (world_size + 1) / 2``;
4. a ~20-step :meth:`Trainer.fit` of the reference ``raw_predictor_small`` model
   on synthetic :class:`FusionBatch` data, run on ``dm.device`` in fp32 (the
   experiment ``synthetic_smoke.yaml`` sets ``bf16: false`` -- an accepted
   deviation: the loop's bf16 autocast makes ``nn.Linear`` outputs bf16, which
   ``LatentPredictor``'s float32-only validation rejects; the bf16 fix is a
   separately tracked M2-exit item -- see report disclosure #1), that returns
   ``status == "completed"``; rank 0 then reads tokens/s, data-wait, and GPU
   memory back from the Trainer's ``metrics.jsonl`` (not reimplemented here).

Any failure prints an unambiguous ``SMOKE FAILED`` line and exits nonzero, so the
job log is decisive. Env overrides: ``SMOKE_TOTAL_STEPS`` (shorten the run, e.g.
for a single-process CPU dry-run) and ``SMOKE_RUN_DIR`` (run directory).
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from pathlib import Path

import torch

# The wrapper runs ``python tests/frontier/smoke_ddp.py`` with CWD at the repo
# root, which puts tests/frontier -- NOT the repo root -- on sys.path[0]. Add the
# repo root so ``fusion_jepa`` and ``tests.fixtures`` both import. (The brief is
# explicit: do NOT move the fixture; make the script self-sufficient instead.)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.distributed as dist  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from fusion_jepa.models.build import build_raw_world_model  # noqa: E402
from fusion_jepa.objectives.raw_prediction import RawPredictionObjective  # noqa: E402
from fusion_jepa.training.distributed import DistributedManager  # noqa: E402
from fusion_jepa.training.loop import Trainer  # noqa: E402
from fusion_jepa.utils.logging import read_metrics  # noqa: E402
from tests.fixtures.synthetic import make_synthetic_fusion_batch  # noqa: E402

_CONFIGS = _REPO_ROOT / "configs"
_MODEL_CONFIG = _CONFIGS / "model" / "raw_predictor_small.yaml"
_EXPERIMENT = _CONFIGS / "experiment" / "synthetic_smoke.yaml"


def _device_name(dm: DistributedManager) -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(dm.device_index)
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            return f"<cuda name unavailable: {exc!r}>"
    return "cpu"


def _print_rank_identity(dm: DistributedManager) -> None:
    print(
        f"[rank {dm.rank}/{dm.world_size}] host={socket.gethostname()} "
        f"SLURM_PROCID={os.environ.get('SLURM_PROCID')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', dm.local_rank)} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
        f"device={_device_name(dm)}",
        flush=True,
    )


def _device_identity(dm: DistributedManager) -> dict:
    """Per-rank identity: a real GPU uuid when available, plus the ROCR binding."""
    uuid = None
    if torch.cuda.is_available():
        try:
            uuid = str(torch.cuda.get_device_properties(dm.device_index).uuid)
        except Exception:  # noqa: BLE001 - uuid is optional on ROCm
            uuid = None
    return {
        "rank": dm.rank,
        "uuid": uuid,
        "rocr": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "procid": os.environ.get("SLURM_PROCID", str(dm.rank)),
        "host": socket.gethostname(),
    }


def _check_uniqueness(dm: DistributedManager) -> None:
    info = _device_identity(dm)
    if dm.distributed:
        gathered: list = [None] * dm.world_size
        dist.all_gather_object(gathered, info)
    else:
        gathered = [info]

    # STRICT: the gathered ranks must be exactly {0 .. world_size-1}.
    ranks = sorted(g["rank"] for g in gathered)
    assert ranks == list(range(dm.world_size)), (
        f"rank set is not unique/complete: {ranks}"
    )

    if not dm.is_main:
        return

    print("[smoke] gathered rank/device identities:", flush=True)
    for g in sorted(gathered, key=lambda x: x["rank"]):
        print(
            f"    rank {g['rank']}: host={g['host']} rocr={g['rocr']} "
            f"procid={g['procid']} uuid={g['uuid']}",
            flush=True,
        )

    # GPU-binding distinctness. Two ranks bound to the same GCD MUST fail the job
    # (a silent collision would leave 8 ranks on one GPU while the log stays
    # green), so this is HARD-FAIL wherever we have a real identity to compare:
    #   * device uuids when available -- a uuid collision means two ranks share
    #     one physical GPU;
    #   * else, when ROCR_VISIBLE_DEVICES is set (e.g. ROCm builds where the uuid
    #     is unavailable), the (host, ROCR_VISIBLE_DEVICES) mask -- under
    #     --gpus-per-task=1 each rank's mask must be distinct per node.
    # The ONLY warning-only case is uuids unavailable AND ROCR entirely unset:
    # then there is nothing to compare and the operator must verify manually.
    uuids = [g["uuid"] for g in gathered]
    rocr_identities = [(g["host"], g["rocr"]) for g in gathered]
    rocr_entirely_unset = all(g["rocr"] == "" for g in gathered)

    if all(u is not None for u in uuids):
        if len(set(uuids)) != dm.world_size:
            raise AssertionError(
                "GPU uuids are NOT distinct across ranks "
                f"({uuids}); two or more ranks are bound to the same physical "
                "GPU. Check --gpu-bind=closest and --gpus-per-task=1."
            )
        print(
            f"[smoke] device distinctness OK via GPU uuid ({dm.world_size} distinct)",
            flush=True,
        )
    elif not rocr_entirely_unset:
        if len(set(rocr_identities)) != dm.world_size:
            raise AssertionError(
                "device binding is NOT unique: (host, ROCR_VISIBLE_DEVICES) "
                f"collides across ranks ({rocr_identities}). Under "
                "--gpus-per-task=1 each rank must see a distinct GCD; check "
                "--gpu-bind=closest and the allocation shape."
            )
        print(
            f"[smoke] device distinctness OK via (host, ROCR_VISIBLE_DEVICES) "
            f"({dm.world_size} distinct)",
            flush=True,
        )
    else:
        print(
            "[smoke] WARNING: no GPU uuid AND ROCR_VISIBLE_DEVICES is entirely "
            "unset, so device-binding uniqueness cannot be checked here -- the "
            "operator MUST verify --gpu-bind pinned distinct GCDs manually.",
            flush=True,
        )


def _check_all_reduce(dm: DistributedManager) -> None:
    contribution = torch.tensor(
        float(dm.rank + 1), dtype=torch.float64, device=dm.device
    )
    if dm.distributed:
        dist.all_reduce(contribution, op=dist.ReduceOp.SUM)
    expected = dm.world_size * (dm.world_size + 1) / 2.0
    got = float(contribution.item())
    assert abs(got - expected) < 1e-6, (
        f"all-reduce checksum {got} != expected {expected}"
    )
    print(
        f"[rank {dm.rank}] all-reduce checksum OK: {got} == {expected}", flush=True
    )


def _synthetic_loader(cfg, dm: DistributedManager, n: int, seed0: int) -> list:
    """A cycled list of synthetic batches whose shapes match the model config."""
    model_cfg = OmegaConf.to_container(OmegaConf.load(_MODEL_CONFIG), resolve=True)
    n_channels = int(model_cfg["modalities"]["slow_ts"]["n_channels"])
    n_actuators = int(model_cfg["action_encoder"]["n_actuators"])
    micro = int(cfg.training.micro_batch_samples)
    return [
        make_synthetic_fusion_batch(
            B=micro,
            modalities=("slow_ts", "profile"),
            n_channels=n_channels,
            T=8,
            H=4,
            A=n_actuators,
            # Distinct per-rank data so DDP is genuinely data-parallel.
            seed=seed0 + dm.rank * 1000 + i,
            missing_fraction=0.1,
        )
        for i in range(n)
    ]


def _report_metrics(run_dir: Path, dm: DistributedManager) -> None:
    records = read_metrics(run_dir / "metrics.jsonl")
    steps = [r for r in records if r.get("event") == "train_step"]
    if not steps:
        print("[smoke] WARNING: no train_step metrics were logged", flush=True)
        return
    last = steps[-1]
    print(
        f"[smoke] final train_step={last.get('step')} "
        f"tokens_per_s={last.get('tokens_per_s'):.1f} "
        f"data_wait_s={last.get('data_wait_s'):.4f} "
        f"gpu_mem_bytes={last.get('gpu_mem_bytes')}",
        flush=True,
    )
    tps = [r.get("tokens_per_s", 0.0) for r in steps]
    print(
        f"[smoke] tokens/s over {len(steps)} logged steps: "
        f"min={min(tps):.1f} max={max(tps):.1f}",
        flush=True,
    )


def _run_training(dm: DistributedManager) -> None:
    cfg = OmegaConf.load(_EXPERIMENT)

    override = os.environ.get("SMOKE_TOTAL_STEPS")
    if override:
        steps = int(override)
        cfg.training.total_steps = steps
        cfg.training.val_every_steps = steps
        cfg.training.warmup_steps = 1

    model = build_raw_world_model(_MODEL_CONFIG)
    objective = RawPredictionObjective(distance="smooth_l1", smooth_l1_beta=1.0)

    job = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = Path(
        os.environ.get("SMOKE_RUN_DIR", str(_REPO_ROOT / "runs" / f"frontier_smoke_{job}"))
    )

    train_loader = _synthetic_loader(cfg, dm, n=8, seed0=0)
    val_loader = _synthetic_loader(
        cfg, dm, n=int(cfg.training.val_max_batches), seed0=500
    )

    # The Trainer owns dm.wrap() internally (self.model = self.dm.wrap(model)) and
    # defaults its device to dm.device, so it is handed the RAW model -- pre-
    # wrapping here would double-wrap in DDP.
    trainer = Trainer(
        cfg=cfg,
        model=model,
        objective=objective,
        dm=dm,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
    )
    result = trainer.fit()
    dm.barrier()

    assert result.status == "completed", (
        f"training did not complete: status={result.status} "
        f"reason={result.failure_reason}"
    )
    if dm.is_main:
        print(
            f"[smoke] Trainer.fit completed: final_step={result.final_step} "
            f"best_val_loss={result.best_val_loss}",
            flush=True,
        )
        _report_metrics(run_dir, dm)


def main() -> int:
    dm = DistributedManager()
    try:
        _print_rank_identity(dm)
        dm.barrier()
        _check_uniqueness(dm)
        _check_all_reduce(dm)
        dm.barrier()
        _run_training(dm)
        dm.barrier()
        if dm.is_main:
            print("[smoke] ALL CHECKS PASSED", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - report then exit nonzero
        print(f"[rank {dm.rank}] SMOKE FAILED: {exc!r}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        dm.close()


if __name__ == "__main__":
    sys.exit(main())
