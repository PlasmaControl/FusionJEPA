"""Distributed-training helper for Fusion-JEPA (Task 2.10).

Ported from ``tokamak_foundation_model.utils.distributed.DistributedManager``,
keeping the Frontier-hardened behaviours verified on that stack:

* the ``ROCR_VISIBLE_DEVICES`` device-index clamp (each rank launched with
  ``--gpus-per-task=1`` sees exactly one GPU at index 0 regardless of
  ``LOCAL_RANK``),
* the 30-minute collective timeout (the default 10-minute watchdog trips
  during the long K=80 validation phase's post-val barrier), and
* ``wrap`` defaulting to ``find_unused_parameters=False`` with its RCCL
  static-bucket rationale.

Adaptations vs. the source (see task-2.10 report for the full disclosure):

1. Backend is chosen by device availability -- ``"nccl"`` when a CUDA/ROCm
   device is visible (RCCL on Frontier), ``"gloo"`` otherwise -- so CPU-only
   tests and login-node runs work. An explicit ``backend`` override is
   accepted.
2. ``all_reduce_mean`` is added (SUM all-reduce divided by ``world_size``;
   identity when not distributed).
3. The fragile ``__del__`` destructor is replaced by an explicit,
   idempotent ``close()`` that is safe when no group was ever created.
4. An optional ``init_method`` is accepted so a ``file://`` rendezvous can
   be used (env:// via ``MASTER_PORT`` remains the default production path).
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _default_backend() -> str:
    """Pick the collective backend from visible device type.

    ``torch.cuda.is_available()`` is ``True`` under ROCm as well (the HIP
    runtime masquerades as CUDA), so this returns ``"nccl"`` -- which maps
    to RCCL on Frontier -- whenever a GPU is present, and ``"gloo"`` on a
    CPU-only host.
    """
    return "nccl" if torch.cuda.is_available() else "gloo"


class DistributedManager:
    """Owns process-group setup/teardown and the common collectives.

    Construct with no arguments under a torchrun/srun launch (rank info is
    read from the environment) or in a plain single process (becomes a
    rank-0, world-size-1 no-op). Pass ``init_method`` to use a ``file://``
    rendezvous instead of the env:// ``MASTER_PORT`` path, and ``backend``
    to override the device-based backend choice.
    """

    def __init__(
        self,
        backend: str | None = None,
        init_method: str | None = None,
    ) -> None:
        # Set first so ``close()`` is safe even if ``__init__`` raises later.
        self._owns_process_group = False

        # Distributed either when an explicit rendezvous file is given
        # (tests / file://) or when a launcher has set MASTER_PORT (env://).
        distributed = init_method is not None or bool(
            os.environ.get("MASTER_PORT")
        )

        if distributed:
            # torchrun sets RANK; fall back to SLURM_PROCID for srun launches.
            self.rank = int(
                os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))
            )
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ["WORLD_SIZE"])

            # On Frontier with --gpus-per-task=1, each rank sees exactly one
            # GPU (masked by ROCR_VISIBLE_DEVICES); the device index is 0
            # regardless of LOCAL_RANK. Clamp accordingly.
            visible = torch.cuda.device_count()
            self.device_index = self.local_rank if visible > 1 else 0

            self.distributed = True
            backend = backend or _default_backend()

            # 30-min collective timeout (default is 10 min) -- the
            # extended Stage 2 K=80 val phase can have rank-skew
            # exceeding the default watchdog, especially at the
            # post-val dm.barrier() where rank 0 finishes its
            # Tier 4 metric polling + checkpoint save. See
            # smoke 4793237 (val timeout at step 40, K=80).
            init_kwargs = {
                "backend": backend,
                "rank": self.rank,
                "world_size": self.world_size,
                "timeout": timedelta(minutes=30),
            }
            if init_method is not None:
                init_kwargs["init_method"] = init_method
            # ``device_id`` binds the group to a CUDA/ROCm device and is
            # invalid for the gloo/CPU backend.
            if backend == "nccl":
                init_kwargs["device_id"] = torch.device(
                    "cuda", self.device_index
                )

            dist.init_process_group(**init_kwargs)
            self._owns_process_group = True

            if backend == "nccl" and torch.cuda.is_available():
                torch.cuda.set_device(self.device_index)
        else:
            # Single-process (plain python).
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.device_index = 0
            self.distributed = False
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_index)

        self.barrier()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.device_index)
        return torch.device("cpu")

    def wrap(
        self, model: torch.nn.Module, find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        """Wrap model with DDP if distributed, otherwise return as-is.

        Default ``find_unused_parameters=False`` relies on every parameter
        being touched in every step. The video / spectrogram tokenizers
        always run ``_encode`` and reference ``missing_token`` regardless
        of the per-batch validity mask, so the autograd graph is
        data-independent and DDP's reducer can use static buckets. This
        avoids RCCL bucket-rebuild faults observed on Frontier.

        Override to ``True`` only as a debugging escape hatch -- it incurs
        a per-step unused-param scan and was previously observed to
        trigger GPU memory faults via RCCL on this stack.
        """
        if self.distributed:
            # ``device_ids`` is a GPU-only affinity hint; on the gloo/CPU
            # backend it must be omitted (kept as the ported [device_index]
            # on GPU where every production run wraps).
            device_ids = (
                [self.device_index] if torch.cuda.is_available() else None
            )
            return DistributedDataParallel(
                model,
                device_ids=device_ids,
                find_unused_parameters=find_unused_parameters,
            )
        return model

    def unwrap(self, model: torch.nn.Module):
        if self.distributed and hasattr(model, "module"):
            return model.module
        return model

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list)

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the mean of ``tensor`` across all ranks.

        SUM all-reduce divided by ``world_size``. The input is cloned so
        the caller's tensor is never mutated in place. Identity (returns
        the input unchanged) when not distributed.
        """
        if not self.distributed:
            return tensor
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        return reduced

    def close(self) -> None:
        """Tear down the process group if this manager created one.

        Explicit replacement for the source's ``__del__`` (destructors are
        fragile under pytest / interpreter shutdown). Idempotent and safe
        to call when no group was ever initialised.
        """
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group = False
