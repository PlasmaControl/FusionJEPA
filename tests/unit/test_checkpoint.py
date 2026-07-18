"""Unit tests for ``fusion_jepa.training.checkpoint`` (Task 2.11).

Covers the atomic save/load contract, full RNG-state capture/restore
(python / numpy / torch-CPU / torch-CUDA, with CPU-only safety), the
canonical ``CHECKPOINT_KEYS`` payload contract, and the explicit
strict-key loader ported from the e2e trainer.

Everything here runs offline and with no GPU: the CUDA code paths are
driven deterministically by monkeypatching ``torch.cuda.is_available``
so the CPU-only no-op behaviour is exercised regardless of the machine
the suite happens to run on.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from fusion_jepa.training.checkpoint import (
    CHECKPOINT_KEYS,
    capture_rng_states,
    load_checkpoint,
    load_state_dict_explicit,
    restore_rng_states,
    save_checkpoint,
)


def test_atomic_write_preserves_previous_checkpoint_on_failure(
    tmp_path, monkeypatch
) -> None:
    """A failed save must leave the prior checkpoint byte-identical.

    The atomicity convention is temp-in-same-dir + ``os.replace``: the
    final path is only ever touched by the rename, which never runs if
    ``torch.save`` raises. We simulate a torn write (partial bytes to the
    temp target, then an exception) and assert the committed file is
    untouched.
    """
    path = tmp_path / "latest.pt"
    save_checkpoint(path, {"step": 1, "payload": "original"})
    original_bytes = path.read_bytes()

    def exploding_save(obj, f, *args, **kwargs):
        # Write partial garbage to the temp target, then fail mid-write.
        with open(f, "wb") as handle:
            handle.write(b"torn partial write")
        raise RuntimeError("disk full mid-write")

    monkeypatch.setattr(torch, "save", exploding_save)

    with pytest.raises(RuntimeError, match="disk full mid-write"):
        save_checkpoint(path, {"step": 2, "payload": "should-not-land"})

    assert path.read_bytes() == original_bytes


def test_rng_round_trip_reproduces_next_randoms() -> None:
    """Capture -> draw -> restore -> draw must reproduce the draws.

    Exercises python ``random``, numpy, and torch-CPU generators, the
    ones a resumed run depends on for exact reproduction (Task 3.7).
    """
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Advance every generator off its seeded start so the captured state
    # is non-trivial.
    random.random()
    np.random.rand(5)
    torch.rand(5)

    states = capture_rng_states()

    expected_py = [random.random() for _ in range(3)]
    expected_np = np.random.rand(4)
    expected_torch = torch.rand(6)

    restore_rng_states(states)

    got_py = [random.random() for _ in range(3)]
    got_np = np.random.rand(4)
    got_torch = torch.rand(6)

    assert got_py == expected_py
    assert np.array_equal(got_np, expected_np)
    assert torch.equal(got_torch, expected_torch)


def test_save_load_round_trip_equality(tmp_path) -> None:
    """A full CHECKPOINT_KEYS payload survives save/load unchanged.

    Weights, optimizer buffers, scheduler state, scalar bookkeeping,
    nested config dicts, and RNG states must all come back equal, and the
    restored RNG state must reproduce the next draws.
    """
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    target_encoder = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Take a real step so the optimizer accumulates non-trivial buffers.
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    payload = {
        "model": model.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None,
        "step": 7,
        "epoch": 2,
        "best_metric": 0.1234,
        "rng_states": capture_rng_states(),
        "sampler_state": {"epoch": 2, "index": 17},
        "resolved_config": {"lr": 1e-3, "model": {"dim": 4}},
        "git_commit": "abcdef0",
        "upstream_manifest": {"tokamark": "1a200f4"},
    }
    assert set(payload) == set(CHECKPOINT_KEYS)

    path = tmp_path / "best.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)

    # Scalars / plain containers.
    assert loaded["step"] == 7
    assert loaded["epoch"] == 2
    assert loaded["best_metric"] == pytest.approx(0.1234)
    assert loaded["scaler"] is None
    assert loaded["sampler_state"] == {"epoch": 2, "index": 17}
    assert loaded["resolved_config"] == {"lr": 1e-3, "model": {"dim": 4}}
    assert loaded["git_commit"] == "abcdef0"
    assert loaded["upstream_manifest"] == {"tokamark": "1a200f4"}

    # Model + target-encoder weights.
    for key, value in model.state_dict().items():
        assert torch.equal(loaded["model"][key], value)
    for key, value in target_encoder.state_dict().items():
        assert torch.equal(loaded["target_encoder"][key], value)

    # Optimizer buffers (exp_avg / exp_avg_sq / step).
    orig_opt = optimizer.state_dict()
    assert loaded["optimizer"]["param_groups"] == orig_opt["param_groups"]
    for pid, buffers in orig_opt["state"].items():
        for name, value in buffers.items():
            restored = loaded["optimizer"]["state"][pid][name]
            if torch.is_tensor(value):
                assert torch.equal(restored, value)
            else:
                assert restored == value

    # Scheduler state.
    assert loaded["scheduler"] == scheduler.state_dict()

    # Restored RNG reproduces the next draws.
    expected = torch.rand(4)
    restore_rng_states(loaded["rng_states"])
    assert torch.equal(torch.rand(4), expected)


def test_explicit_load_raises_on_unexpected_and_missing_keys() -> None:
    """``load_state_dict_explicit`` is strict but honours allowed prefixes."""
    model = nn.Linear(4, 3)
    good_state = model.state_dict()

    # Unexpected key -> always raises.
    with_extra = dict(good_state)
    with_extra["ghost.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        load_state_dict_explicit(model, with_extra)

    # Missing key with no allowance -> raises.
    without_bias = {k: v for k, v in good_state.items() if k != "bias"}
    with pytest.raises(RuntimeError, match="Missing keys"):
        load_state_dict_explicit(model, without_bias)

    # Missing key covered by an allowed prefix -> no raise.
    load_state_dict_explicit(model, without_bias, allowed_missing_prefixes=("bias",))


def test_checkpoint_keys_is_the_canonical_contract() -> None:
    """The payload contract is frozen for the Trainer (2.12) / resume (3.7)."""
    assert CHECKPOINT_KEYS == (
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
    assert isinstance(CHECKPOINT_KEYS, tuple)


def test_rng_capture_and_restore_are_cpu_only_safe(monkeypatch) -> None:
    """With CUDA reported absent, capture omits it and restore is a no-op.

    Forces the CPU-only path deterministically so the behaviour is
    verified even on a machine that does expose a device.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    states = capture_rng_states()
    assert "cuda" not in states

    # A payload carrying CUDA state (e.g. saved on a GPU box) must restore
    # cleanly on a CPU-only machine -- the CUDA entry is skipped, not fed to
    # ``set_rng_state_all`` (which would crash on this bogus value).
    states["cuda"] = ["bogus-cuda-state-that-would-crash-set_rng_state_all"]
    restore_rng_states(states)  # must not raise
