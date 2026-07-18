"""Objective-call-form bridge for wiring the JEPA into the M2 Trainer (Task 3.8).

The :class:`~fusion_jepa.training.loop.Trainer` calls its objective in the RAW,
mask-aware form ``objective(model(batch), batch.target, batch.target_mask)`` (the
form the raw-prediction objective wants), but the JEPA's
:class:`~fusion_jepa.objectives.latent_prediction.LatentPredictionObjective` is
natively ``obj(jepa_output, horizon_seconds)``. Rather than change the Trainer's
call form (a landed, review-closed contract), this adapter bridges the two:

* ``target`` / ``target_mask`` are deliberately UNUSED -- the JEPA's target
  signal is the model's OWN target-encoder output, already carried in the
  :class:`~fusion_jepa.models.jepa.JEPAOutput` as ``z_target`` (alongside
  ``z_hat`` and ``target_valid``). There is no external raw target to compare
  against, so the Trainer's ``batch.target`` / ``batch.target_mask`` are the wrong
  signal for a latent objective and are ignored on purpose.
* ``horizon_seconds`` is read from the model output (``JEPAOutput.horizon_seconds``,
  populated by :meth:`JEPAModel.forward` from ``batch.horizon_seconds``), so the
  adapter needs nothing but the forward result. ``horizon_seconds`` is
  labeling-only inside the objective and never enters the loss value.
"""

from __future__ import annotations

from fusion_jepa.models.jepa import JEPAOutput
from fusion_jepa.objectives.base import LossOutput
from fusion_jepa.objectives.latent_prediction import LatentPredictionObjective

__all__ = ["JepaObjectiveAdapter"]


class JepaObjectiveAdapter:
    """Adapt a :class:`LatentPredictionObjective` to the Trainer's raw call form.

    Args:
        latent_objective: the wrapped :class:`LatentPredictionObjective` (or any
            callable with the same ``obj(out, horizon_seconds) -> LossOutput``
            signature).
    """

    def __init__(self, latent_objective: LatentPredictionObjective) -> None:
        self._objective = latent_objective

    def __call__(
        self, out: JEPAOutput, target: object = None, target_mask: object = None
    ) -> LossOutput:
        """Score ``out`` with the wrapped latent objective.

        ``target`` / ``target_mask`` are accepted (so the Trainer's raw call form
        works unchanged) but ignored -- the latent objective's target signal is
        ``out.z_target``, the model's own target-encoder output.
        """
        if out.horizon_seconds is None:
            raise ValueError(
                "JepaObjectiveAdapter requires JEPAOutput.horizon_seconds to be "
                "populated (JEPAModel.forward sets it from batch.horizon_seconds); "
                "got None. Score the objective directly if you must pass a custom "
                "horizon."
            )
        return self._objective(out, out.horizon_seconds)
