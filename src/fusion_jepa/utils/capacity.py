"""Matched-capacity verification: raw baseline vs. JEPA (M3, Task 3.5).

The raw baseline (:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`) and
the JEPA (:class:`~fusion_jepa.models.jepa.JEPAModel`) are a *matched* comparison
by construction: they share the IDENTICAL trunk -- per-modality tokenizers, the
context :class:`~fusion_jepa.models.encoder.ContextEncoder`, the
:class:`~fusion_jepa.models.action_encoder.ActionEncoder`, and the
:class:`~fusion_jepa.models.predictor.LatentPredictor`. This module makes that
match *checkable* and produces a compact, JSON-serializable report of where the
two models' capacity actually differs.

:func:`verify_matched_capacity` reuses the landed
:func:`~fusion_jepa.utils.accounting.assert_matched_backbones` for the trunk check
(exact equality by default; a relative tolerance is available) and adds the two
objective-specific pieces that are deliberately *excluded* from that trunk:

* the raw baseline's **decoder** (raw-only: the JEPA has none); and
* the JEPA's **EMA target copy** (a training-only twin of the tokenizers +
  encoder). Whether that copy exists depends on the JEPA's
  :class:`~fusion_jepa.models.jepa.TargetUpdatePolicy`:

  - under ``EMA`` the target trunk is a frozen ``deepcopy`` of the online trunk,
    so its parameters are *distinct* tensors and must be counted; while
  - under ``SHARED_STOPGRAD`` / ``END_TO_END_REGULARIZED`` the target modules
    ARE the online modules (same objects, shared weights), so the "copy" adds
    zero parameters and must never be double-counted.

The distinction is detected by **parameter identity** (via
:func:`~fusion_jepa.utils.accounting.count_parameters`, which dedupes by
``id``), not by the policy string alone -- so a shared-trunk JEPA reports
``ema_copy_params == 0`` regardless of what its ``policy`` attribute says. The
report records *both* the identity-derived count and the declared policy string,
so any disagreement between them is visible in the report itself (under the landed
:class:`~fusion_jepa.models.jepa.JEPAModel` they cannot disagree: ``EMA`` always
deepcopies and the other policies always alias).

The **deployed** totals answer "how big is the model you actually ship": the raw
baseline deploys its whole self (trunk + decoder), whereas the JEPA deploys only
its online trunk -- the EMA target copy is a training-time artefact and is
excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

from fusion_jepa.utils.accounting import assert_matched_backbones, count_parameters

__all__ = ["MatchReport", "verify_matched_capacity"]


@dataclass(frozen=True)
class MatchReport:
    """Matched-capacity accounting for a raw baseline / JEPA pair.

    Attributes:
        trunk_components: ``{component: parameter_count}`` for the shared trunk
            (tokenizers, encoder, action_encoder, predictor). These counts are
            equal across the two models by definition -- the pair would not have
            passed :func:`verify_matched_capacity` otherwise.
        trunk_total: the sum of ``trunk_components`` -- the shared-trunk capacity.
        decoder_params: parameters of the raw baseline's decoder (raw-only; the
            JEPA has none, so this is reported separately from the trunk).
        ema_copy_params: parameters of the JEPA's training-only EMA target copy,
            detected by parameter identity: nonzero only when the target trunk is
            a distinct deepcopy (the ``EMA`` policy), and ``0`` when the target
            shares the online modules (``SHARED_STOPGRAD`` /
            ``END_TO_END_REGULARIZED``).
        deployed_raw_total: parameters actually deployed for the raw baseline
            (trunk + decoder) -- its whole self.
        deployed_jepa_total: parameters actually deployed for the JEPA -- its
            online trunk, EXCLUDING the training-only EMA copy.
        policy: the JEPA target-update policy as its lowercase string name; recorded
            alongside the identity-derived ``ema_copy_params`` so the two can be
            cross-checked.
    """

    trunk_components: dict[str, int]
    trunk_total: int
    decoder_params: int
    ema_copy_params: int
    deployed_raw_total: int
    deployed_jepa_total: int
    policy: str

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable ``dict`` (ints / strs only)."""
        return {
            "trunk_components": {
                str(name): int(count) for name, count in self.trunk_components.items()
            },
            "trunk_total": int(self.trunk_total),
            "decoder_params": int(self.decoder_params),
            "ema_copy_params": int(self.ema_copy_params),
            "deployed_raw_total": int(self.deployed_raw_total),
            "deployed_jepa_total": int(self.deployed_jepa_total),
            "policy": str(self.policy),
        }


def _policy_name(jepa_model: object) -> str:
    """The JEPA's target-update policy as a lowercase string (``TargetUpdatePolicy``
    exposes it via ``.value``); fall back to ``str`` / ``"unknown"`` defensively."""
    policy = getattr(jepa_model, "policy", None)
    if policy is None:
        return "unknown"
    return getattr(policy, "value", str(policy))


def _ema_copy_params(jepa_model: object) -> int:
    """Parameters the EMA target copy adds *on top of* the online trunk.

    Computed by parameter identity, not by the policy string:
    :func:`count_parameters` dedupes by tensor ``id``, so counting the online
    trunk together with the target trunk and subtracting the online-only count
    yields exactly the parameters unique to the target copy. Under the ``EMA``
    policy the target is a distinct deepcopy (nonzero); under the shared-trunk
    policies the target IS the online trunk (same objects) and the difference is
    ``0``, so shared weights are never double-counted.
    """
    online = [
        jepa_model.tokenizers,
        jepa_model.encoder,
        jepa_model.action_encoder,
        jepa_model.predictor,
    ]
    online_only = count_parameters(online)
    online_plus_target = count_parameters(
        [*online, jepa_model.target_tokenizers, jepa_model.target_encoder]
    )
    return online_plus_target - online_only


def verify_matched_capacity(
    raw_model: object,
    jepa_model: object,
    rel_tol: float = 0.0,
) -> MatchReport:
    """Verify the raw baseline and JEPA share a matched trunk; report capacity.

    The shared trunk (tokenizers, encoder, action_encoder, predictor) is checked
    with :func:`~fusion_jepa.utils.accounting.assert_matched_backbones`. With the
    default ``rel_tol=0.0`` the per-component counts must match EXACTLY; a
    positive ``rel_tol`` permits a relative discrepancy (forwarded verbatim). On a
    mismatch, that helper's actionable ``ValueError`` -- naming the first
    offending component and both counts -- propagates unchanged.

    Args:
        raw_model: the raw baseline; must expose ``.decoder`` (raw-only).
        jepa_model: the JEPA; must expose the online trunk plus
            ``.target_tokenizers`` / ``.target_encoder`` and (for the report's
            ``policy`` field) ``.policy``.
        rel_tol: relative tolerance for the trunk check (``0.0`` = exact).

    Returns:
        A :class:`MatchReport`: the matched trunk counts, the raw-only decoder
        count, the identity-detected EMA copy count, the two deployed totals, and
        the JEPA policy string.

    Raises:
        ValueError: if the trunks do not match within ``rel_tol``.
    """
    # Trunk check first: on mismatch, assert_matched_backbones raises before any
    # of the objective-specific accounting runs.
    trunk_components = assert_matched_backbones(raw_model, jepa_model, rel_tol=rel_tol)
    trunk_total = sum(trunk_components.values())

    decoder_params = count_parameters(raw_model.decoder)
    ema_copy_params = _ema_copy_params(jepa_model)

    # Deployed capacity: the raw baseline ships its whole self (trunk + decoder);
    # the JEPA ships only its online trunk -- the EMA copy is training-only.
    deployed_raw_total = count_parameters(raw_model)
    deployed_jepa_total = count_parameters(jepa_model) - ema_copy_params

    return MatchReport(
        trunk_components=trunk_components,
        trunk_total=trunk_total,
        decoder_params=decoder_params,
        ema_copy_params=ema_copy_params,
        deployed_raw_total=deployed_raw_total,
        deployed_jepa_total=deployed_jepa_total,
        policy=_policy_name(jepa_model),
    )
