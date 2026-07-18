"""Config -> :class:`RawWorldModel` builder (Task 2.13).

Until now the raw world model was only ever composed by hand (the tiny
``build_raw_world_model`` helper in ``tests/unit/test_raw_world_model.py``). This
module is the production builder that turns a model config -- a YAML file, a plain
mapping, or an OmegaConf ``DictConfig`` -- into a runnable
:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`, pinned to the exact
landed constructor signatures of the five composed sub-modules.

The config is a mapping with five sections::

    encoder:        {d_model, n_heads, n_blocks, n_state_tokens, [mlp_ratio, dropout]}
    modalities:     {<name>: {type: scalar_series|profile, ...}, ...}
    action_encoder: {n_actuators, d_model, n_time_freqs}
    predictor:      {d_model, n_heads, n_blocks, device_vocab, d_device_context,
                     [n_horizon_freqs, mlp_ratio, dropout]}
    decoder:        {n_heads, n_blocks, [n_coord_freqs, n_horizon_freqs,
                     mlp_ratio, dropout]}

Width coupling (derived, not repeated in the config)
---------------------------------------------------
The *trunk width* is ``encoder.d_model``. The tokenizers and the decoder are
built at that width because the decoder's query tokens reuse each tokenizer's
``channel_embed`` (the landed 2.6 constraint). The predictor's latent I/O
(``d_latent_in``), its bottleneck (``n_state_tokens``), and the decoder's
``d_latent`` are likewise taken from the encoder, and the predictor's
``d_action`` is taken from ``action_encoder.d_model`` -- the action width MAY
differ from the trunk width. A config that *does* pin a conflicting
``decoder.d_model`` is an actionable build error rather than a silent shape crash
deep in the forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.decoders import QueryConditionedDecoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.raw_world_model import RawWorldModel
from fusion_jepa.models.tokenizers import ProfileTokenizer, ScalarSeriesTokenizer

__all__ = ["build_raw_world_model", "load_model_config"]

_TOKENIZER_TYPES = ("scalar_series", "profile")


def load_model_config(config: str | Path | Mapping[str, Any] | DictConfig) -> dict:
    """Normalise ``config`` to a plain, fully-resolved ``dict``.

    Accepts a path to a YAML file, an OmegaConf ``DictConfig``, or a plain
    mapping. Interpolations are resolved eagerly so the returned dict is a
    self-contained snapshot of the config.
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(Path(config))
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(
        "model config must be a path, an OmegaConf config, or a mapping, got "
        f"{type(config).__name__}"
    )


def _section(cfg: Mapping[str, Any], key: str) -> dict:
    """Return ``cfg[key]`` as a dict, or raise an actionable error."""
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"model config is missing required section {key!r}")
    value = cfg[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"model config section {key!r} must be a mapping")
    return dict(value)


def _require(section: Mapping[str, Any], key: str, ctx: str) -> Any:
    """Return ``section[key]``, or raise naming the missing ``ctx.key``."""
    if key not in section or section[key] is None:
        raise ValueError(f"model config {ctx!r} is missing required field {key!r}")
    return section[key]


def _build_tokenizer(name: str, spec: Mapping[str, Any], d_model: int) -> nn.Module:
    """Build one modality tokenizer at the shared trunk ``d_model``."""
    ttype = spec.get("type", "scalar_series")
    ctx = f"modalities.{name}"
    if ttype == "scalar_series":
        return ScalarSeriesTokenizer(
            n_channels=int(_require(spec, "n_channels", ctx)),
            d_model=d_model,
            patch_len=int(_require(spec, "patch_len", ctx)),
            n_time_freqs=int(_require(spec, "n_time_freqs", ctx)),
            modality=name,
        )
    if ttype == "profile":
        return ProfileTokenizer(
            d_model=d_model,
            radial_patch=int(_require(spec, "radial_patch", ctx)),
            n_radial_points=int(_require(spec, "n_radial_points", ctx)),
            modality=name,
        )
    raise ValueError(
        f"modality {name!r} has unknown tokenizer type {ttype!r}; expected one "
        f"of {_TOKENIZER_TYPES}"
    )


def build_raw_world_model(
    config: str | Path | Mapping[str, Any] | DictConfig,
) -> RawWorldModel:
    """Build a :class:`RawWorldModel` from a model config.

    Args:
        config: a YAML path, an OmegaConf ``DictConfig``, or a plain mapping with
            the ``encoder`` / ``modalities`` / ``action_encoder`` / ``predictor``
            / ``decoder`` sections documented in the module docstring.

    Returns:
        A runnable :class:`RawWorldModel` with the coupled trunk widths derived
        from the encoder and the action width from the action encoder.

    Raises:
        ValueError: on a missing section/field, an unknown tokenizer ``type``, or
            a ``decoder.d_model`` that conflicts with the trunk width.
    """
    cfg = load_model_config(config)

    encoder_cfg = _section(cfg, "encoder")
    trunk_d = int(_require(encoder_cfg, "d_model", "encoder"))
    n_state_tokens = int(_require(encoder_cfg, "n_state_tokens", "encoder"))

    modalities = _section(cfg, "modalities")
    if not modalities:
        raise ValueError("model config 'modalities' must define at least one modality")
    tokenizers = {
        name: _build_tokenizer(name, spec, trunk_d)
        for name, spec in modalities.items()
    }

    encoder = ContextEncoder(
        d_model=trunk_d,
        n_heads=int(_require(encoder_cfg, "n_heads", "encoder")),
        n_blocks=int(_require(encoder_cfg, "n_blocks", "encoder")),
        n_state_tokens=n_state_tokens,
        mlp_ratio=float(encoder_cfg.get("mlp_ratio", 4.0)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
    )

    action_cfg = _section(cfg, "action_encoder")
    d_action = int(_require(action_cfg, "d_model", "action_encoder"))
    action_encoder = ActionEncoder(
        n_actuators=int(_require(action_cfg, "n_actuators", "action_encoder")),
        d_model=d_action,
        n_time_freqs=int(_require(action_cfg, "n_time_freqs", "action_encoder")),
    )

    predictor_cfg = _section(cfg, "predictor")
    predictor = LatentPredictor(
        d_model=int(_require(predictor_cfg, "d_model", "predictor")),
        n_heads=int(_require(predictor_cfg, "n_heads", "predictor")),
        n_blocks=int(_require(predictor_cfg, "n_blocks", "predictor")),
        d_latent_in=trunk_d,  # derived: predicted latent == encoder state width
        n_state_tokens=n_state_tokens,  # derived: == encoder bottleneck width
        device_vocab=list(_require(predictor_cfg, "device_vocab", "predictor")),
        d_device_context=int(
            _require(predictor_cfg, "d_device_context", "predictor")
        ),
        d_action=d_action,  # derived: predictor consumes the action-encoder width
        n_horizon_freqs=int(predictor_cfg.get("n_horizon_freqs", 6)),
        mlp_ratio=float(predictor_cfg.get("mlp_ratio", 4.0)),
        dropout=float(predictor_cfg.get("dropout", 0.0)),
    )

    decoder_cfg = _section(cfg, "decoder")
    decoder_d_model = int(decoder_cfg.get("d_model", trunk_d))
    if decoder_d_model != trunk_d:
        raise ValueError(
            "decoder d_model must equal the trunk width (encoder d_model="
            f"{trunk_d}) because the decoder's query tokens reuse each "
            f"tokenizer's channel embedding; got decoder d_model="
            f"{decoder_d_model}. Omit decoder.d_model to derive it, or set it to "
            "the encoder width."
        )
    decoder = QueryConditionedDecoder(
        d_latent=trunk_d,  # derived: == predictor output latent width
        d_model=trunk_d,
        n_heads=int(_require(decoder_cfg, "n_heads", "decoder")),
        n_blocks=int(_require(decoder_cfg, "n_blocks", "decoder")),
        n_coord_freqs=int(decoder_cfg.get("n_coord_freqs", 6)),
        n_horizon_freqs=int(decoder_cfg.get("n_horizon_freqs", 6)),
        mlp_ratio=float(decoder_cfg.get("mlp_ratio", 4.0)),
        dropout=float(decoder_cfg.get("dropout", 0.0)),
    )

    return RawWorldModel(
        tokenizers=tokenizers,
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
    )
