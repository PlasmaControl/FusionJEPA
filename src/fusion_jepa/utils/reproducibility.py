"""Helpers for deterministic random-number generation."""

import hashlib
import json
import random

import numpy as np
import torch


def derive_seed(root_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from ordered components."""
    payload = json.dumps([root_seed, *components], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(
    root_seed: int,
    *,
    rank: int,
    worker_id: int,
    epoch: int,
) -> int:
    """Derive a seed for one worker at one distributed rank and epoch."""
    return derive_seed(root_seed, "worker", rank, worker_id, epoch)
