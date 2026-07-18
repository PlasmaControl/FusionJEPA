"""Action-use diagnostics and perturbation modes for Fusion-JEPA.

EVALUATION-SIDE ONLY. Everything in this module is instrumentation for probing
*how much a trained model actually relies on the actuator (action) channel*.
None of it is -- or may ever be -- wired into a training loss: the perturbations
destroy or corrupt actions on purpose, and the report/sensitivity helpers only
*measure* a model, they never contribute gradient. Keep it out of every
objective's ``forward``/``LossOutput`` path.

The workhorse is :func:`perturb_actions`, a pure function that returns a *new*
:class:`~fusion_jepa.data.batch.FusionBatch` (via :func:`dataclasses.replace`)
with only the ``actions``/``action_mask`` fields altered; the input batch is
never mutated and every other field (times, context, targets, metadata, ...) is
carried through unchanged. Given a fixed ``torch.Generator`` state each mode is
fully deterministic.

Perturbation modes (:class:`ActionPerturbation`)
------------------------------------------------
* ``REAL`` -- identity. Returns a *copy* (a fresh ``FusionBatch`` whose
  ``actions``/``action_mask`` are cloned; all other fields are shared
  references), so callers can treat every mode uniformly.
* ``BATCH_SHUFFLE`` -- apply one random permutation of the batch (``B``) axis to
  ``actions`` and ``action_mask`` *together*. Whole action sequences are swapped
  between examples, which preserves every per-channel value marginal by
  construction while destroying the pairing between an example's actions and its
  context/target. A derangement is not required (the identity permutation is a
  legal, if unlikely, draw).
* ``WITHIN_SHOT_TIME_SHIFT`` -- cyclically roll ``actions`` and ``action_mask``
  together along the time axis by an independent per-example shift. Because a
  roll only ever reindexes an example's own timesteps, the waveform can never
  cross a shot boundary or leak in another shot's data. ``action_times`` are
  deliberately left UNCHANGED: the waveform is moved relative to true time.
* ``ZERO`` -- set the action *values* to zero while leaving ``action_mask``
  unchanged. The mask asserts *state availability*; zeros are a value choice, so
  a channel that was observed stays observed (its value is simply zero).

Reports
-------
:func:`action_use_report` scores a model+objective (closed over into a single
``loss_fn: Callable[[FusionBatch], float]``) under each mode and reports the
per-mode mean loss plus ``predictive_gain/<mode> = loss(mode) - loss(REAL)``; a
positive gain means corrupting the actions *raised* the loss, i.e. the actions
were being used. :func:`action_sensitivity` is a finite-difference probe of a
``forward_fn: Callable[[FusionBatch], Tensor]`` along random unit directions of
the action tensor.
"""

import dataclasses
from collections.abc import Callable, Sequence
from enum import Enum

import torch
from torch import Tensor

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.utils.reproducibility import derive_seed

__all__ = [
    "ActionPerturbation",
    "perturb_actions",
    "action_use_report",
    "action_sensitivity",
]


class ActionPerturbation(str, Enum):
    """How to corrupt (or preserve) a batch's actions for a diagnostic pass."""

    REAL = "real"
    BATCH_SHUFFLE = "batch_shuffle"
    WITHIN_SHOT_TIME_SHIFT = "within_shot_time_shift"
    ZERO = "zero"


def _coerce_mode(mode: "ActionPerturbation | str") -> ActionPerturbation:
    """Accept an :class:`ActionPerturbation` or its lowercase string value."""
    if isinstance(mode, ActionPerturbation):
        return mode
    if isinstance(mode, str):
        try:
            return ActionPerturbation(mode.lower())
        except ValueError as exc:
            valid = ", ".join(member.value for member in ActionPerturbation)
            raise ValueError(
                f"unknown action perturbation mode {mode!r}; expected one of: {valid}"
            ) from exc
    raise TypeError(
        f"mode must be an ActionPerturbation or str, got {type(mode).__name__}"
    )


def _batch_shuffle(batch: FusionBatch, generator: torch.Generator) -> FusionBatch:
    """Permute whole action sequences across the batch axis."""
    permutation = torch.randperm(batch.actions.shape[0], generator=generator)
    # Advanced indexing already returns fresh tensors (no aliasing with input).
    return dataclasses.replace(
        batch,
        actions=batch.actions[permutation],
        action_mask=batch.action_mask[permutation],
    )


def _within_shot_time_shift(
    batch: FusionBatch, generator: torch.Generator
) -> FusionBatch:
    """Roll each example's actions/mask along time by an independent shift."""
    actions = batch.actions
    action_mask = batch.action_mask
    n_examples, n_steps = actions.shape[0], actions.shape[1]
    shifts = torch.randint(0, n_steps, (n_examples,), generator=generator)

    new_actions = torch.empty_like(actions)
    new_mask = torch.empty_like(action_mask)
    for index in range(n_examples):
        shift = int(shifts[index])
        new_actions[index] = torch.roll(actions[index], shifts=shift, dims=0)
        new_mask[index] = torch.roll(action_mask[index], shifts=shift, dims=0)
    return dataclasses.replace(batch, actions=new_actions, action_mask=new_mask)


def perturb_actions(
    batch: FusionBatch,
    mode: "ActionPerturbation | str",
    generator: torch.Generator,
) -> FusionBatch:
    """Return a new batch with its actions perturbed under ``mode``.

    Pure and deterministic: the input ``batch`` is never mutated, and the result
    is fully determined by ``mode`` and the ``generator`` state. Only
    ``actions``/``action_mask`` change; all other fields are carried through.
    """
    mode = _coerce_mode(mode)

    if mode is ActionPerturbation.REAL:
        return dataclasses.replace(
            batch,
            actions=batch.actions.clone(),
            action_mask=batch.action_mask.clone(),
        )
    if mode is ActionPerturbation.ZERO:
        return dataclasses.replace(
            batch,
            actions=torch.zeros_like(batch.actions),
            action_mask=batch.action_mask.clone(),
        )
    if mode is ActionPerturbation.BATCH_SHUFFLE:
        return _batch_shuffle(batch, generator)
    if mode is ActionPerturbation.WITHIN_SHOT_TIME_SHIFT:
        return _within_shot_time_shift(batch, generator)

    raise AssertionError(f"unhandled perturbation mode {mode!r}")  # pragma: no cover


def action_use_report(
    loss_fn: Callable[[FusionBatch], float],
    batches: Sequence[FusionBatch],
    modes: Sequence["ActionPerturbation | str"],
    seed: int,
) -> dict[str, float]:
    """Score ``loss_fn`` under each perturbation mode and report predictive gain.

    ``loss_fn`` is model+objective agnostic on purpose: the caller closes over
    whichever model/objective pair it wants (e.g. ``JEPAModel`` +
    ``LatentPredictionObjective`` or ``RawWorldModel`` +
    ``RawPredictionObjective``) into a single ``FusionBatch -> float`` callable,
    which keeps this diagnostic decoupled from any objective's call form.

    For each mode the mean loss over ``batches`` is reported as ``loss/<mode>``;
    ``predictive_gain/<mode> = loss(mode) - loss(REAL)`` is reported for every
    requested mode (positive => corrupting actions raised the loss => the model
    was using the actions). ``REAL`` is always evaluated as the baseline even if
    it is not among ``modes``. Determinism comes from a per-(mode, batch-index)
    generator seeded via :func:`~fusion_jepa.utils.reproducibility.derive_seed`.
    """
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("action_use_report requires at least one batch")

    requested = [_coerce_mode(mode) for mode in modes]
    # REAL first so the baseline is always available for the gain computation.
    evaluated: list[ActionPerturbation] = list(
        dict.fromkeys([ActionPerturbation.REAL, *requested])
    )

    mean_loss: dict[ActionPerturbation, float] = {}
    for mode in evaluated:
        total = 0.0
        for index, batch in enumerate(batch_list):
            generator = torch.Generator().manual_seed(
                derive_seed(seed, mode.value, index)
            )
            perturbed = perturb_actions(batch, mode, generator)
            total += float(loss_fn(perturbed))
        mean_loss[mode] = total / len(batch_list)

    report: dict[str, float] = {}
    for mode in evaluated:
        report[f"loss/{mode.value}"] = float(mean_loss[mode])
    baseline = mean_loss[ActionPerturbation.REAL]
    for mode in requested:
        report[f"predictive_gain/{mode.value}"] = float(mean_loss[mode] - baseline)
    return report


def _direction_like(actions: Tensor, generator: torch.Generator) -> Tensor:
    """Return a unit-norm action-perturbation direction on ``actions``' device.

    The random draw and its L2 normalization run on CPU in float32 with the
    seeded ``generator``, so the direction *values* are identical for a given
    seed regardless of which device ``actions`` lives on (a CUDA generator would
    draw a different stream). The finished unit vector is then moved onto
    ``actions.device`` and cast to ``actions.dtype`` so it can be added to an
    accelerator-resident action tensor without a cross-device error. A
    degenerate zero-norm draw is returned unnormalized (all zeros); the caller
    treats such a direction as a null perturbation (the zero-norm guard).
    """
    direction = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
    norm = direction.norm()
    if float(norm) != 0.0:
        direction = direction / norm
    return direction.to(device=actions.device, dtype=actions.dtype)


def action_sensitivity(
    forward_fn: Callable[[FusionBatch], Tensor],
    batch: FusionBatch,
    epsilon: float,
    n_directions: int,
    seed: int,
) -> dict[str, float]:
    """Finite-difference sensitivity of ``forward_fn`` to the action tensor.

    For each of ``n_directions`` random *unit* perturbation directions of the
    (whole) action tensor, we form ``batch + epsilon * direction`` and measure
    the mean absolute change in ``forward_fn`` output divided by ``epsilon`` --
    a directional finite-difference derivative magnitude. The mean and max of
    those per-direction magnitudes are returned as JSON-safe floats.

    Deterministic given ``seed`` (each direction is drawn from a generator
    seeded via :func:`~fusion_jepa.utils.reproducibility.derive_seed`). Each
    direction is built by :func:`_direction_like` on the CPU generator and then
    moved onto ``batch.actions.device``, so the probe runs unchanged on an
    accelerator-resident batch and stays seed-identical across devices. All
    reduction/accumulation runs in float32.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if n_directions < 1:
        raise ValueError("n_directions must be at least 1")

    actions = batch.actions
    baseline = forward_fn(batch).detach().to(torch.float32)

    magnitudes = torch.zeros(n_directions, dtype=torch.float32)
    for index in range(n_directions):
        generator = torch.Generator().manual_seed(derive_seed(seed, "direction", index))
        direction = _direction_like(actions, generator)
        if float(direction.norm()) == 0.0:  # pragma: no cover - unlikely
            continue
        perturbed = dataclasses.replace(batch, actions=actions + epsilon * direction)
        output = forward_fn(perturbed).detach().to(torch.float32)
        difference = (output - baseline).abs()
        per_direction = difference.mean().to(device="cpu", dtype=torch.float32)
        magnitudes[index] = per_direction / epsilon

    return {
        "sensitivity_mean": float(magnitudes.mean()),
        "sensitivity_max": float(magnitudes.max()),
        "n_directions": int(n_directions),
        "epsilon": float(epsilon),
    }
