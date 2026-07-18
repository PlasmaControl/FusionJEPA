"""Unit tests for the YAML -> RawWorldModel builder (Task 2.13).

These lock the config-driven constructor `fusion_jepa.models.build`:

* the committed reference config `configs/model/raw_predictor_small.yaml` builds a
  runnable :class:`RawWorldModel` whose forward matches the synthetic target
  shapes, and whose parameter budget sits in the documented broad ranges
  (~12M total; encoder ~7.4M, predictor ~2.4M, decoder ~1-2.5M);
* the builder DERIVES the coupled trunk widths from the encoder (tokenizer /
  decoder width, predictor ``d_latent_in`` / ``n_state_tokens``, decoder
  ``d_latent``) and takes ``d_action`` from the action encoder (action d != trunk
  d is supported);
* the landed 2.6 constraint -- decoder ``d_model`` MUST equal the trunk width --
  is enforced with an actionable error;
* only ``scalar_series`` is buildable: ``profile`` is rejected with an actionable
  error (RawWorldModel cannot drive a ProfileTokenizer -- 3-arg tokenizer call +
  ``channel_embed`` reuse vs. the profile's 4-arg ``coords`` / ``patch_embed``),
  and an unknown type is likewise an actionable error.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.build import build_raw_world_model
from fusion_jepa.models.decoders import QueryConditionedDecoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.raw_world_model import RawWorldModel
from fusion_jepa.models.tokenizers import ScalarSeriesTokenizer
from fusion_jepa.objectives.raw_prediction import RawPredictionObjective
from fusion_jepa.utils.accounting import parameter_report
from tests.fixtures.synthetic import make_synthetic_fusion_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "raw_predictor_small.yaml"

# Synthetic-batch shapes that match configs/model/raw_predictor_small.yaml
# (n_channels=8 per modality, n_actuators=4, device_context width 3).
_MODALITIES = ("slow_ts", "profile")
_N_CHANNELS = 8
_N_ACTUATORS = 4


def _config_dict() -> dict:
    """A small, self-contained valid model config (independent of the YAML)."""
    return {
        "encoder": {
            "d_model": 32,
            "n_heads": 4,
            "n_blocks": 2,
            "n_state_tokens": 4,
        },
        "modalities": {
            "slow_ts": {
                "type": "scalar_series",
                "n_channels": 3,
                "patch_len": 2,
                "n_time_freqs": 4,
            },
        },
        "action_encoder": {
            "n_actuators": 2,
            "d_model": 24,
            "n_time_freqs": 4,
        },
        "predictor": {
            "d_model": 24,
            "n_heads": 4,
            "n_blocks": 2,
            "device_vocab": ["MAST"],
            "d_device_context": 3,
        },
        "decoder": {"n_heads": 4, "n_blocks": 2},
    }


def _synthetic_batch(B: int = 2):
    return make_synthetic_fusion_batch(
        B=B,
        modalities=_MODALITIES,
        n_channels=_N_CHANNELS,
        T=8,
        H=4,
        A=_N_ACTUATORS,
        seed=0,
        missing_fraction=0.1,
    )


def test_committed_yaml_builds_and_forward_matches_target_shapes():
    model = build_raw_world_model(MODEL_CONFIG)
    assert isinstance(model, RawWorldModel)
    assert set(model.tokenizers) == set(_MODALITIES)

    batch = _synthetic_batch()
    preds = model(batch)

    assert set(preds) == set(_MODALITIES)
    for modality in _MODALITIES:
        assert preds[modality].shape == batch.target[modality].shape
        assert preds[modality].dtype == torch.float32
        assert torch.isfinite(preds[modality]).all()


def test_committed_yaml_forward_backward_is_differentiable():
    model = build_raw_world_model(MODEL_CONFIG)
    batch = _synthetic_batch()
    objective = RawPredictionObjective(distance="smooth_l1", smooth_l1_beta=1.0)

    out = objective(model(batch), batch.target, batch.target_mask)
    out.total.backward()

    # A representative parameter on the prediction path received a gradient.
    grad = model.predictor.out_proj.weight.grad
    assert grad is not None and bool(torch.any(grad != 0))


def test_parameter_report_sits_in_documented_broad_ranges():
    model = build_raw_world_model(MODEL_CONFIG)
    report = parameter_report(model)

    # Broad ranges (NOT exact counts), per the config's documented budget.
    assert 5_000_000 < report["encoder"] < 10_000_000
    assert 1_500_000 < report["predictor"] < 4_000_000
    assert 800_000 < report["decoder"] < 4_000_000
    assert report["tokenizers"] > 0
    assert 9_000_000 < report["total"] < 16_000_000


def test_builder_derives_coupled_trunk_widths():
    model = build_raw_world_model(MODEL_CONFIG)
    trunk = model.encoder.d_model

    # Tokenizers and the decoder share the trunk width; the decoder query tokens
    # reuse each tokenizer's channel_embed, which is [C, trunk].
    for tokenizer in model.tokenizers.values():
        assert tokenizer.d_model == trunk
        assert tokenizer.channel_embed.shape[1] == trunk
    assert model.decoder.d_model == trunk
    assert model.decoder.d_latent == trunk

    # The predictor's latent I/O + bottleneck come from the encoder; its action
    # width comes from the action encoder (action d != trunk d is supported).
    assert model.predictor.d_latent_in == trunk
    assert model.predictor.n_state_tokens == model.encoder.n_state_tokens
    assert model.action_encoder.d_model != trunk  # 256 != 320 in the reference
    assert model.predictor.action_proj.in_features == model.action_encoder.d_model


def test_builder_accepts_dict_dictconfig_and_path_equivalently():
    from_dict = build_raw_world_model(_config_dict())
    from_omega = build_raw_world_model(OmegaConf.create(_config_dict()))
    from_path = build_raw_world_model(MODEL_CONFIG)

    n_dict = parameter_report(from_dict)["total"]
    n_omega = parameter_report(from_omega)["total"]
    assert n_dict == n_omega  # same config -> same parameter count
    assert isinstance(from_path, RawWorldModel)


def test_builder_rejects_decoder_d_model_mismatch():
    cfg = _config_dict()
    cfg["decoder"]["d_model"] = cfg["encoder"]["d_model"] + 8  # conflicts w/ trunk
    with pytest.raises(ValueError, match="d_model"):
        build_raw_world_model(cfg)


def test_builder_rejects_profile_tokenizer_type():
    """``type: profile`` is rejected up front, not silently constructed.

    A ProfileTokenizer is constructible but NOT runnable inside RawWorldModel:
    ``RawWorldModel.forward`` calls every tokenizer with three arguments
    (``values, value_mask, times``) and reuses each tokenizer's ``channel_embed``
    to build target queries, while ``ProfileTokenizer.forward`` requires a fourth
    ``coords`` argument and exposes ``patch_embed``. The builder must therefore
    reject profile configs with an actionable error rather than hand back a model
    that crashes in the forward pass. (RED: the f835209 builder CONSTRUCTED this
    config -- see the fix-round RED evidence in the Task 2.13 report.)
    """
    cfg = _config_dict()
    cfg["modalities"] = {
        "profile": {
            "type": "profile",
            "radial_patch": 2,
            "n_radial_points": 6,
        },
    }
    with pytest.raises(ValueError) as excinfo:
        build_raw_world_model(cfg)
    msg = str(excinfo.value)
    # The error names the offending type, states WHY (the two contract
    # mismatches), and names the allowed type(s).
    assert "profile" in msg
    assert "channel_embed" in msg
    assert "coords" in msg
    assert "scalar_series" in msg


def test_builder_scalar_series_is_the_reference_tokenizer_type():
    model = build_raw_world_model(MODEL_CONFIG)
    for tokenizer in model.tokenizers.values():
        assert isinstance(tokenizer, ScalarSeriesTokenizer)


def test_builder_rejects_unknown_tokenizer_type():
    cfg = _config_dict()
    cfg["modalities"]["slow_ts"]["type"] = "quantized_nonsense"
    with pytest.raises(ValueError, match="quantized_nonsense"):
        build_raw_world_model(cfg)


def test_builder_reports_missing_required_section():
    cfg = _config_dict()
    del cfg["encoder"]
    with pytest.raises((ValueError, KeyError), match="encoder"):
        build_raw_world_model(cfg)


def test_builder_component_types_are_the_landed_classes():
    model = build_raw_world_model(MODEL_CONFIG)
    assert isinstance(model.encoder, ContextEncoder)
    assert isinstance(model.action_encoder, ActionEncoder)
    assert isinstance(model.predictor, LatentPredictor)
    assert isinstance(model.decoder, QueryConditionedDecoder)
