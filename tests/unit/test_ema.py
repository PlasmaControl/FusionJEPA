"""Unit tests for ``fusion_jepa.training.ema`` (Task 3.1).

The :class:`~fusion_jepa.training.ema.EmaUpdater` is the standalone,
model-agnostic extraction of the legacy in-model EMA loop at
``tokamak_foundation_model/models/latent_feature_space/foundation_model.py``
(``update_ema``): every target parameter is nudged toward its online
counterpart with ``target.lerp_(online, 1 - decay)``.

All arithmetic here is exact-in-float32 on purpose (decays and values are
powers of two), so equality is asserted with ``torch.equal`` rather than a
tolerance -- the EMA step is a single fused multiply-add whose result is
representable exactly, and any drift would signal a real bug.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn as nn

from fusion_jepa.training.ema import EmaUpdater


def _one_param(value: float) -> nn.Linear:
    """A one-parameter module (``nn.Linear(1, 1, bias=False)``)."""
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(value)
    return module


def test_ema_update_matches_hand_computed_step() -> None:
    """One step must equal ``decay*target0 + (1 - decay)*online`` exactly.

    With ``decay=0.5``, ``target0=2.0`` and ``online=4.0`` the update is the
    exact midpoint ``3.0`` -- fully representable in float32, so ``torch.equal``
    is the right assertion (no tolerance needed).
    """
    online = _one_param(4.0)
    target = _one_param(2.0)
    target0 = target.weight.detach().clone()

    updater = EmaUpdater([(online.weight, target.weight)], decay=0.5)
    updater.step()

    expected = 0.5 * target0 + 0.5 * online.weight.detach()
    assert torch.equal(target.weight.detach(), expected)
    assert torch.equal(target.weight.detach(), torch.tensor([[3.0]]))
    # The online parameter is never touched.
    assert torch.equal(online.weight.detach(), torch.tensor([[4.0]]))


def test_target_parameters_never_require_grad() -> None:
    """Targets are frozen at construction and stay frozen across steps."""
    online = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    target = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    assert all(p.requires_grad for p in target.parameters())

    updater = EmaUpdater((online, target), decay=0.9)
    assert all(not p.requires_grad for p in target.parameters())
    # Online parameters are left alone -- they still train.
    assert all(p.requires_grad for p in online.parameters())

    updater.step()
    updater.step()
    assert all(not p.requires_grad for p in target.parameters())


def test_updater_fires_only_when_called() -> None:
    """Construction moves nothing; N calls apply exactly N steps."""
    online = _one_param(4.0)
    target = _one_param(2.0)
    before = target.weight.detach().clone()

    updater = EmaUpdater([(online.weight, target.weight)], decay=0.5)
    # Merely constructing the updater must not move the target.
    assert torch.equal(target.weight.detach(), before)
    assert updater.num_updates == 0

    for expected_count in range(1, 6):
        updater.step()
        assert updater.num_updates == expected_count

    # After 5 steps of t <- 0.5*t + 0.5*4 from t0=2: 4 - 2*0.5**5 = 3.9375,
    # exact in float32 (all powers of two).
    assert torch.equal(target.weight.detach(), torch.tensor([[3.9375]]))


def test_update_alias_performs_one_step_and_ignores_argument() -> None:
    """``update(model)`` is the Trainer hook: one step, argument ignored.

    The Trainer calls ``ema_updater.update(unwrapped_model)`` once per
    optimizer step. The updater already holds bound parameter references, so
    the passed object is irrelevant -- passing an unrelated module (or nothing)
    must still update the originally-bound target and nothing else.
    """
    online = _one_param(4.0)
    target = _one_param(2.0)
    updater = EmaUpdater([(online.weight, target.weight)], decay=0.5)

    unrelated = _one_param(999.0)
    updater.update(unrelated)  # argument ignored
    assert updater.num_updates == 1
    assert torch.equal(target.weight.detach(), torch.tensor([[3.0]]))
    # The unrelated module handed in was not touched.
    assert torch.equal(unrelated.weight.detach(), torch.tensor([[999.0]]))

    updater.update()  # no-argument form also works
    assert updater.num_updates == 2
    # Second step: 0.5*3.0 + 0.5*4.0 = 3.5.
    assert torch.equal(target.weight.detach(), torch.tensor([[3.5]]))


def test_state_dict_round_trip() -> None:
    """save/load restores ``num_updates``/``decay`` and keeps stepping right."""
    online = _one_param(4.0)
    target = _one_param(2.0)
    updater = EmaUpdater([(online.weight, target.weight)], decay=0.5)
    updater.step()
    updater.step()

    state = updater.state_dict()
    assert state == {"num_updates": 2, "decay": 0.5}
    # Must be plain-Python / JSON-safe for the checkpoint payload.
    assert json.loads(json.dumps(state)) == state

    # A fresh updater built with a *different* decay, then reloaded.
    online2 = _one_param(4.0)
    target2 = _one_param(2.0)
    fresh = EmaUpdater([(online2.weight, target2.weight)], decay=0.9)
    fresh.load_state_dict(state)
    assert fresh.num_updates == 2
    assert fresh.decay == 0.5

    # Continues correctly with the *restored* decay (0.5, not 0.9): a step on
    # the untouched target2 gives the 0.5 midpoint 3.0. Had decay stayed 0.9
    # the result would be 2.2, so this pins the restored value.
    fresh.step()
    assert fresh.num_updates == 3
    assert torch.equal(target2.weight.detach(), torch.tensor([[3.0]]))


def test_module_pair_form_matches_explicit_param_pairs() -> None:
    """``(online_module, target_module)`` == the zipped explicit param pairs."""
    torch.manual_seed(0)
    online = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    target = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    online_copy = copy.deepcopy(online)
    target_copy = copy.deepcopy(target)

    explicit = EmaUpdater(
        list(zip(online.parameters(), target.parameters(), strict=True)),
        decay=0.7,
    )
    module_form = EmaUpdater((online_copy, target_copy), decay=0.7)

    explicit.step()
    module_form.step()

    for p_explicit, p_module in zip(
        target.parameters(), target_copy.parameters(), strict=True
    ):
        assert torch.equal(p_explicit, p_module)


def test_bad_decay_raises_actionable_error() -> None:
    """``decay`` must live in ``[0.0, 1.0)`` with a message naming the bound."""
    online = _one_param(1.0)
    target = _one_param(1.0)
    pairs = [(online.weight, target.weight)]

    with pytest.raises(ValueError, match="decay"):
        EmaUpdater(pairs, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        EmaUpdater(pairs, decay=-0.1)
    # A valid boundary value is accepted.
    EmaUpdater(pairs, decay=0.0)


def test_shape_mismatch_raises_actionable_error() -> None:
    """Same parameter count but mismatched shapes must be rejected."""
    online = nn.Linear(2, 3)  # weight (3, 2), bias (3,)
    target = nn.Linear(2, 4)  # weight (4, 2), bias (4,)
    with pytest.raises(ValueError, match="shape"):
        EmaUpdater((online, target), decay=0.9)


def test_parameter_count_mismatch_raises_actionable_error() -> None:
    """A differing number of parameters between modules must be rejected."""
    online = nn.Linear(2, 3)  # weight + bias -> 2 params
    target = nn.Linear(2, 3, bias=False)  # weight only -> 1 param
    with pytest.raises(ValueError, match="parameter"):
        EmaUpdater((online, target), decay=0.9)
