"""Atomic checkpoint I/O and full-state capture for training resume.

This module owns three concerns that together make a run bit-for-bit
resumable (Task 3.7):

* **Atomic writes.** :func:`save_checkpoint` serialises to a temp file in
  the *same directory* as the target and promotes it with a single
  ``os.replace``. A failed save therefore never touches the committed
  path -- readers (and the previous checkpoint) never see a torn write.
  This mirrors the repo's established convention in
  ``scripts/acquire_tokamark.py``.
* **RNG state.** :func:`capture_rng_states` / :func:`restore_rng_states`
  round-trip the python, numpy, torch-CPU and torch-CUDA generators so a
  resumed run draws the same random numbers as an uninterrupted one. The
  CUDA generators are captured only when a device is present, and restore
  is a no-op for absent devices -- the unit suite runs CPU-only.
* **Explicit loading.** :func:`load_state_dict_explicit` is a strict-key
  loader (raises on unexpected keys; raises on missing keys outside an
  allow-list) ported verbatim from the e2e trainer.

:data:`CHECKPOINT_KEYS` is the canonical payload contract consumed by the
Trainer (Task 2.12) and the resume test (Task 3.7).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

# Canonical top-level keys of a checkpoint payload. This is a contract,
# not an enforced schema: ``save_checkpoint`` serialises whatever dict it
# is given, but the Trainer builds its payload from exactly these keys so
# resume (Task 3.7) has a stable surface to read back.
CHECKPOINT_KEYS = (
    "model",
    "target_encoder",
    "optimizer",
    "scheduler",
    "scaler",
    "step",
    "epoch",
    "best_metric",
    "rng_states",
    "sampler_state",
    "resolved_config",
    "git_commit",
    "upstream_manifest",
)


def capture_rng_states() -> dict[str, Any]:
    """Snapshot every RNG a resumed run depends on.

    Returns a dict with the python ``random``, numpy, and torch-CPU
    states always present, plus a ``"cuda"`` entry (the per-device torch
    CUDA states) only when a CUDA device is available. The returned
    objects are plain picklable values, safe to hand to
    :func:`save_checkpoint`.
    """
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Mapping[str, Any]) -> None:
    """Restore RNGs previously captured by :func:`capture_rng_states`.

    Restoring the CUDA generators is skipped when no device is present, so
    a checkpoint saved on a GPU box loads cleanly on a CPU-only machine
    (the unit suite). The python / numpy / torch-CPU generators are always
    restored.
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    cuda_states = states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> None:
    """Atomically serialise ``payload`` to ``path`` with ``torch.save``.

    Writes to a temp file in the same directory, then promotes it with a
    single ``os.replace``. If serialisation raises, the temp file is
    removed and the committed ``path`` is left byte-identical to whatever
    was there before.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: with a fixed staging name, two concurrent savers
    # (e.g. two runs sharing a runs dir) could truncate or promote each
    # other's half-written temp. A PID suffix makes promotion strictly
    # last-writer-wins (same convention as scripts/acquire_tokamark.py).
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
    except BaseException:
        # Never leave a torn temp file behind after a failed write.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is passed explicitly: our payload holds far
    more than tensors -- optimizer / scheduler state dicts, RNG states
    (numpy / python tuples), and resolved-config objects. torch >= 2.6
    defaults ``weights_only=True``, which would reject those, so we opt
    out deliberately (we only ever load our own checkpoints).

    ``map_location`` defaults to ``"cpu"`` so a GPU-saved checkpoint loads
    on the CPU-only unit machine; callers that resume onto a device move
    tensors afterwards.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_explicit(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    allowed_missing_prefixes: Sequence[str] = (),
) -> None:
    """Load ``state_dict`` into ``model`` with explicit key checks.

    Parameters
    ----------
    model : nn.Module
        Target model. Must already have its final architecture (e.g.
        already include video modules if a TS+video state is loaded).
    state_dict : mapping
        Dict of ``name -> Tensor`` to load.
    allowed_missing_prefixes : sequence of str
        If non-empty, missing keys are allowed only when their name
        starts with one of these prefixes. Use this to permit fresh
        init of new modules that didn't exist in the saved state.

    Raises
    ------
    RuntimeError
        If the state contains any unexpected keys, or if any missing
        key falls outside ``allowed_missing_prefixes``.
    """
    result = model.load_state_dict(state_dict, strict=False)

    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in checkpoint (state contains keys the "
            f"model does not have): {result.unexpected_keys}"
        )

    disallowed_missing = [
        k
        for k in result.missing_keys
        if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Missing keys in checkpoint not covered by "
            f"allowed_missing_prefixes={tuple(allowed_missing_prefixes)}: "
            f"{disallowed_missing}"
        )
