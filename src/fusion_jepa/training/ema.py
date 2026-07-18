"""Standalone exponential-moving-average (EMA) target-parameter updater.

:class:`EmaUpdater` is the model-agnostic extraction of the legacy in-model EMA
loop that lived at
``tokamak_foundation_model/models/latent_feature_space/foundation_model.py``
(``update_ema``, lines 208-227). That loop hard-coded which sub-modules to
update; here the target/online correspondence is passed in explicitly at
construction, so any JEPA-style model (M3's ``JEPAModel`` and beyond) can reuse
the exact same, tested nudge:

    target.data.lerp_(online.data, 1 - decay)

which is algebraically ``target <- decay * target + (1 - decay) * online``.

The updater holds *bound references* to the target and online parameters. It
never re-derives them from a model, which is what lets it slot cleanly behind
the M2 Trainer hook (see :meth:`update`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["EmaUpdater"]

# A single (online, target) tuple of two ``nn.Module`` objects, or an iterable
# of (online, target) pairs whose elements are either modules or parameters.
PairsInput = Any


class EmaUpdater:
    """Nudge frozen *target* parameters toward trainable *online* parameters.

    Parameters
    ----------
    pairs:
        Either

        * a single ``(online_module, target_module)`` tuple of two
          :class:`torch.nn.Module` objects -- their ``.parameters()`` are
          zipped in definition order (equal count and matching shapes are
          validated with an actionable error), or
        * an iterable of ``(online, target)`` pairs, where each element of a
          pair is itself either a :class:`torch.nn.Module` (its parameters are
          zipped in) or a :class:`torch.nn.Parameter` / :class:`torch.Tensor`
          used directly.

        A bare ``(online_param, target_param)`` tuple is therefore passed as a
        one-element iterable, e.g. ``[(p_online, p_target)]``; the two-tuple
        form is reserved for the module-pair shorthand.
    decay:
        The EMA decay ``tau`` in ``[0.0, 1.0)``. Each step moves the target a
        fraction ``1 - decay`` of the way toward the online value; larger decay
        means a slower-moving target.
    """

    def __init__(self, pairs: PairsInput, decay: float = 0.996) -> None:
        self.decay = _validate_decay(decay)
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = _build_pairs(pairs)
        self.num_updates: int = 0
        # Enforce the frozen-target invariant idempotently. M3's JEPAModel also
        # freezes these at construction; doing it here too is belt-and-braces
        # and keeps the updater correct even when handed raw parameters.
        for _online, target in self._pairs:
            target.requires_grad_(False)

    @torch.no_grad()
    def step(self) -> None:
        """Apply one EMA step to every target parameter, in place.

        Fires only when called -- there are no hooks and no implicit updates.
        """
        weight = 1.0 - self.decay
        for online, target in self._pairs:
            target.data.lerp_(online.data, weight)
        self.num_updates += 1

    def update(self, model: Any = None) -> None:
        """Trainer-hook alias for :meth:`step`; the argument is ignored.

        The M2 Trainer calls ``ema_updater.update(unwrapped_model)`` exactly
        once per successful optimizer step, passing the unwrapped online model.
        This updater already holds bound references to the online parameters, so
        the passed model is redundant and deliberately ignored -- crucially,
        those references stay valid across DDP wrapping because DDP wraps a
        module *by reference* rather than copying its parameters. We therefore
        do **not** re-derive pairs from ``model``; doing so would risk binding
        to the wrong (wrapped, or freshly rebuilt) parameters.
        """
        self.step()

    def state_dict(self) -> dict[str, Any]:
        """Return plain-Python, JSON-safe EMA bookkeeping.

        Only ``num_updates`` and ``decay`` are carried (the latter enables
        future decay schedules). Target *weights* are not the updater's job --
        in M3 they live inside ``JEPAModel.state_dict()`` and are checkpointed
        there. The checkpoint layer best-effort saves this dict under the
        ``target_encoder`` payload key.
        """
        return {"num_updates": int(self.num_updates), "decay": float(self.decay)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore ``num_updates`` and ``decay`` from :meth:`state_dict`."""
        self.num_updates = int(state["num_updates"])
        self.decay = _validate_decay(float(state["decay"]))


def _validate_decay(decay: float) -> float:
    """Validate ``0.0 <= decay < 1.0`` (also rejects NaN)."""
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must satisfy 0.0 <= decay < 1.0, got {decay!r}")
    return decay


def _is_single_module_pair(pairs: PairsInput) -> bool:
    """True iff ``pairs`` is a ``(module, module)`` two-tuple shorthand."""
    return (
        isinstance(pairs, (tuple, list))
        and len(pairs) == 2
        and isinstance(pairs[0], nn.Module)
        and isinstance(pairs[1], nn.Module)
    )


def _build_pairs(pairs: PairsInput) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalise any accepted input into a flat list of ``(online, target)``."""
    flat: list[tuple[torch.Tensor, torch.Tensor]] = []
    if _is_single_module_pair(pairs):
        flat.extend(_zip_modules(pairs[0], pairs[1]))
    else:
        for index, pair in enumerate(pairs):
            online, target = pair
            if isinstance(online, nn.Module) and isinstance(target, nn.Module):
                flat.extend(_zip_modules(online, target))
            elif isinstance(online, nn.Module) or isinstance(target, nn.Module):
                raise ValueError(
                    f"EMA pair {index}: online and target must both be "
                    "Modules or both be Parameters/Tensors, not a mix"
                )
            else:
                _check_shapes(online, target, index)
                flat.append((online, target))
    if not flat:
        raise ValueError(
            "EmaUpdater requires at least one (online, target) parameter pair"
        )
    return flat


def _zip_modules(
    online: nn.Module, target: nn.Module
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Zip two modules' parameters in order, validating count and shapes."""
    online_params = list(online.parameters())
    target_params = list(target.parameters())
    if len(online_params) != len(target_params):
        raise ValueError(
            f"online module has {len(online_params)} parameters but target "
            f"module has {len(target_params)}; they must correspond one-to-one "
            "for EMA (same architecture, same parameter order)"
        )
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (online_param, target_param) in enumerate(
        zip(online_params, target_params, strict=True)
    ):
        _check_shapes(online_param, target_param, index)
        result.append((online_param, target_param))
    return result


def _check_shapes(online: torch.Tensor, target: torch.Tensor, index: int) -> None:
    """Raise an actionable error if paired tensors differ in shape."""
    if online.shape != target.shape:
        raise ValueError(
            f"EMA pair {index}: online shape {tuple(online.shape)} != target "
            f"shape {tuple(target.shape)}; paired parameters must match shape"
        )
