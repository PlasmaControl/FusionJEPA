"""Unit tests for matched-capacity verification (Task 3.5).

:func:`~fusion_jepa.utils.capacity.verify_matched_capacity` turns the
raw-baseline-vs-JEPA *matched trunk* contract into a single, JSON-serializable
:class:`~fusion_jepa.utils.capacity.MatchReport`. It layers on the landed
:func:`~fusion_jepa.utils.accounting.assert_matched_backbones` (which polices the
shared trunk -- tokenizers / encoder / action_encoder / predictor) and adds the
objective-specific pieces that are *not* part of the trunk: the raw-only decoder
and the JEPA's (training-only) EMA target copy.

The two models are built from the shared constructors
``test_raw_world_model.build_raw_world_model`` and ``test_jepa.build_jepa_model``.
Both use the same tiny widths (the raw builder's predictor ``_D_PRED == _D`` and
the JEPA builder's predictor ``d_model == _D``, all ``== 16``), so calling them
directly -- as SEPARATE instances, with the same modalities / channels / actuators
and different seeds (counts, not values) -- yields a genuinely matched trunk
without forking any magic numbers.

Four behaviours are locked:

* the report's per-component trunk counts equal each raw component's
  :func:`count_parameters`;
* the decoder (raw-only) and the EMA copy (JEPA training-only) are reported
  separately from the trunk, the EMA copy is identity-aware (nonzero under EMA,
  zero under a shared-trunk policy), and the deployed JEPA total excludes it;
* a widened encoder trips the actionable, component-named ``ValueError``;
* ``to_dict()`` round-trips through ``json.dumps`` with only plain scalars.
"""

from __future__ import annotations

import json

import pytest

from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.jepa import TargetUpdatePolicy
from fusion_jepa.utils.accounting import count_parameters
from fusion_jepa.utils.capacity import MatchReport, verify_matched_capacity
from tests.unit.test_jepa import _D, _N_HEADS, _S, build_jepa_model
from tests.unit.test_raw_world_model import build_raw_world_model


def test_raw_and_jepa_shared_components_match_exactly():
    # Different seeds are fine: the report is about parameter *counts*, not
    # values, and the two builders wire identical trunk widths.
    raw = build_raw_world_model(seed=0)
    jepa = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=1)

    report = verify_matched_capacity(raw, jepa)

    assert isinstance(report, MatchReport)
    # Every shared-trunk component count equals the raw component's own count.
    assert report.trunk_components["tokenizers"] == count_parameters(raw.tokenizers)
    assert report.trunk_components["encoder"] == count_parameters(raw.encoder)
    assert report.trunk_components["action_encoder"] == count_parameters(
        raw.action_encoder
    )
    assert report.trunk_components["predictor"] == count_parameters(raw.predictor)
    assert report.trunk_total == sum(report.trunk_components.values())
    # ...and equally the JEPA's own online components (separate instances).
    assert report.trunk_components["encoder"] == count_parameters(jepa.encoder)
    assert report.trunk_components["predictor"] == count_parameters(jepa.predictor)


def test_report_separates_decoder_and_ema_copy():
    raw = build_raw_world_model(seed=0)
    jepa_ema = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=1)

    report = verify_matched_capacity(raw, jepa_ema)

    # The decoder is raw-only and reported apart from the shared trunk.
    assert report.decoder_params > 0
    assert report.decoder_params == count_parameters(raw.decoder)
    assert report.deployed_raw_total == count_parameters(raw)
    assert report.deployed_raw_total == report.trunk_total + report.decoder_params

    # Under EMA the target trunk is a distinct deepcopy: the EMA copy equals the
    # target tokenizers + encoder, i.e. the trunk's tokenizer + encoder counts.
    expected_ema = count_parameters(jepa_ema.target_tokenizers) + count_parameters(
        jepa_ema.target_encoder
    )
    assert report.ema_copy_params == expected_ema
    assert report.ema_copy_params == (
        report.trunk_components["tokenizers"] + report.trunk_components["encoder"]
    )
    assert report.ema_copy_params > 0

    # The deployed JEPA excludes the training-only EMA copy.
    assert report.deployed_jepa_total == report.trunk_total
    assert (
        report.deployed_jepa_total
        == count_parameters(jepa_ema) - report.ema_copy_params
    )
    assert report.deployed_jepa_total < count_parameters(jepa_ema)
    assert report.policy == "ema"

    # Under a shared-trunk policy the target IS the online trunk (same objects),
    # so identity detection must report NO EMA copy -- never double-counted.
    jepa_shared = build_jepa_model(policy=TargetUpdatePolicy.SHARED_STOPGRAD, seed=1)
    shared = verify_matched_capacity(raw, jepa_shared)
    assert shared.ema_copy_params == 0
    assert shared.deployed_jepa_total == shared.trunk_total
    assert shared.deployed_jepa_total == count_parameters(jepa_shared)
    assert shared.policy == "shared_stopgrad"


def test_mismatch_raises_when_encoder_width_differs():
    raw = build_raw_world_model(seed=0)
    jepa = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=1)
    # Widen ONLY the JEPA's online encoder so the first (and only) mismatch is
    # the 'encoder' component. verify never runs a forward, so the resulting
    # width inconsistency is intentional and harmless.
    jepa.encoder = ContextEncoder(
        d_model=_D * 2, n_heads=_N_HEADS, n_blocks=2, n_state_tokens=_S
    )

    with pytest.raises(ValueError, match="encoder"):
        verify_matched_capacity(raw, jepa)


def test_to_dict_is_json_serializable_and_plain():
    raw = build_raw_world_model(seed=0)
    jepa = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=1)
    report = verify_matched_capacity(raw, jepa)

    payload = report.to_dict()
    dumped = json.dumps(payload)  # must not raise
    assert json.loads(dumped) == payload

    # Every value is a plain JSON scalar (or a plain dict of them).
    assert isinstance(payload["policy"], str)
    assert payload["policy"] == "ema"
    for key in (
        "trunk_total",
        "decoder_params",
        "ema_copy_params",
        "deployed_raw_total",
        "deployed_jepa_total",
    ):
        assert isinstance(payload[key], int)
    assert isinstance(payload["trunk_components"], dict)
    assert all(
        isinstance(name, str) and isinstance(count, int)
        for name, count in payload["trunk_components"].items()
    )
