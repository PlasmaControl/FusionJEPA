"""Config -> :class:`RawWorldModel` builder (Task 2.13).

Until now the raw world model was only ever composed by hand (the tiny
``build_raw_world_model`` helper in ``tests/unit/test_raw_world_model.py``). This
module is the production builder that turns a model config -- a YAML file, a plain
mapping, or an OmegaConf ``DictConfig`` -- into a runnable
:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`, pinned to the exact
landed constructor signatures of the five composed sub-modules.

The config is a mapping with five sections::

    encoder:        {d_model, n_heads, n_blocks, n_state_tokens, [mlp_ratio, dropout]}
    modalities:     {<name>: {type: scalar_series, ...}, ...}
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
from fusion_jepa.models.jepa import JEPAModel, TargetUpdatePolicy
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.raw_world_model import RawWorldModel
from fusion_jepa.models.tokenizers import ScalarSeriesTokenizer
from fusion_jepa.objectives.collapse_regularizers import (
    VarianceCovarianceRegularizer,
)

__all__ = ["build_jepa_model", "build_raw_world_model", "load_model_config"]

# Only ``scalar_series`` is buildable. ``RawWorldModel`` cannot drive any other
# tokenizer today (see ``_build_tokenizer`` for the exact contract mismatch that
# makes ``profile`` constructible-but-not-runnable).
_TOKENIZER_TYPES = ("scalar_series",)


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
    """Build one modality tokenizer at the shared trunk ``d_model``.

    Only ``scalar_series`` is supported. ``RawWorldModel`` cannot drive any other
    tokenizer as it is landed today: ``RawWorldModel.forward`` calls every
    tokenizer with exactly three arguments -- ``tokenizer(values, value_mask,
    times)`` -- and builds its target queries by reusing each tokenizer's
    ``channel_embed`` parameter. A :class:`ProfileTokenizer` satisfies neither
    contract: its ``forward`` requires a fourth ``coords`` argument and it exposes
    ``patch_embed`` (not ``channel_embed``). A profile model is therefore
    *constructible but not runnable* -- it crashes in the forward pass -- so the
    builder rejects it up front rather than handing back a broken model. Wiring
    radial coordinates through ``RawWorldModel`` is out-of-scope M3
    data-integration work and needs a ``RawWorldModel`` change.
    """
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
        raise ValueError(
            f"modality {name!r} requests tokenizer type 'profile', which the "
            "builder rejects: RawWorldModel.forward calls every tokenizer with "
            "three arguments (values, value_mask, times) and reuses each "
            "tokenizer's 'channel_embed' to build its target queries, but "
            "ProfileTokenizer.forward requires a fourth 'coords' argument and "
            "exposes 'patch_embed' instead of 'channel_embed'. A profile model "
            "would be constructible but crash in the forward pass. Profile "
            "support requires a RawWorldModel change (radial-coordinate wiring, "
            "out-of-scope M3 data-integration work). Allowed tokenizer type(s): "
            f"{_TOKENIZER_TYPES}."
        )
    raise ValueError(
        f"modality {name!r} has unknown tokenizer type {ttype!r}; allowed "
        f"tokenizer type(s): {_TOKENIZER_TYPES}"
    )


def _build_trunk(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build the four matched-capacity trunk components + derived widths.

    Shared VERBATIM by :func:`build_raw_world_model` and :func:`build_jepa_model`,
    which is exactly what makes the raw baseline and the JEPA a matched comparison
    *by construction*: identical tokenizers, :class:`ContextEncoder`,
    :class:`ActionEncoder`, and :class:`LatentPredictor`, built from the same
    config sections with the same derived widths. Returns the four components plus
    the derived ``trunk_d`` / ``n_state_tokens`` / ``d_action`` widths.
    """
    encoder_cfg = _section(cfg, "encoder")
    trunk_d = int(_require(encoder_cfg, "d_model", "encoder"))
    n_state_tokens = int(_require(encoder_cfg, "n_state_tokens", "encoder"))

    modalities = _section(cfg, "modalities")
    if not modalities:
        raise ValueError("model config 'modalities' must define at least one modality")
    tokenizers = {
        name: _build_tokenizer(name, spec, trunk_d) for name, spec in modalities.items()
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
        d_device_context=int(_require(predictor_cfg, "d_device_context", "predictor")),
        d_action=d_action,  # derived: predictor consumes the action-encoder width
        n_horizon_freqs=int(predictor_cfg.get("n_horizon_freqs", 6)),
        mlp_ratio=float(predictor_cfg.get("mlp_ratio", 4.0)),
        dropout=float(predictor_cfg.get("dropout", 0.0)),
    )

    return {
        "tokenizers": tokenizers,
        "encoder": encoder,
        "action_encoder": action_encoder,
        "predictor": predictor,
        "trunk_d": trunk_d,
        "n_state_tokens": n_state_tokens,
        "d_action": d_action,
    }


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
        ValueError: on a missing section/field, an unsupported tokenizer ``type``
            (only ``scalar_series`` is runnable; ``profile`` is rejected -- see
            :func:`_build_tokenizer`), or a ``decoder.d_model`` that conflicts
            with the trunk width.
    """
    cfg = load_model_config(config)

    trunk = _build_trunk(cfg)
    trunk_d = trunk["trunk_d"]
    tokenizers = trunk["tokenizers"]

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
        encoder=trunk["encoder"],
        action_encoder=trunk["action_encoder"],
        predictor=trunk["predictor"],
        decoder=decoder,
    )


def _build_collapse_regularizer(cfg: Mapping[str, Any]) -> object | None:
    """Build the optional VICReg-style collapse regularizer from ``cfg``.

    Returns ``None`` when the config has no ``collapse_regularizer`` section. The
    section's fields (all optional) are ``std_target`` / ``var_weight`` /
    ``cov_weight`` -- see
    :class:`~fusion_jepa.objectives.collapse_regularizers.VarianceCovarianceRegularizer`.
    """
    section = cfg.get("collapse_regularizer")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("model config 'collapse_regularizer' must be a mapping")
    return VarianceCovarianceRegularizer(
        std_target=float(section.get("std_target", 1.0)),
        var_weight=float(section.get("var_weight", 1.0)),
        cov_weight=float(section.get("cov_weight", 1.0)),
    )


def build_jepa_model(
    config: str | Path | Mapping[str, Any] | DictConfig,
) -> JEPAModel:
    """Build a :class:`~fusion_jepa.models.jepa.JEPAModel` from a model config.

    The JEPA reuses the SAME per-component trunk builders as
    :func:`build_raw_world_model` (via :func:`_build_trunk`), so a JEPA built from
    the trunk sections of ``config`` is capacity-matched by construction to a raw
    baseline built from the same sections -- checkable with
    :func:`~fusion_jepa.utils.capacity.verify_matched_capacity`.

    Args:
        config: a YAML path, an OmegaConf ``DictConfig``, or a plain mapping with
            the ``encoder`` / ``modalities`` / ``action_encoder`` / ``predictor``
            trunk sections, plus the JEPA-only ``policy`` (default ``"ema"``) and
            ``ema_decay`` (default ``0.996``), and an optional
            ``collapse_regularizer`` section (required by the
            ``end_to_end_regularized`` policy).

    Returns:
        A runnable :class:`JEPAModel`.

    Raises:
        ValueError: on a missing trunk section/field, an unsupported tokenizer
            ``type``, or a ``decoder`` section (the JEPA has no decoder -- where
            the raw baseline decodes latents to raw values, the JEPA compares
            predicted latents against a target-encoded latent, so a ``decoder``
            here is a config mistake and is rejected up front).
    """
    cfg = load_model_config(config)

    # Reject a decoder section: the JEPA has no decoder. This is a config mistake
    # (very likely a raw-baseline config handed to the JEPA builder), so fail
    # loudly rather than silently ignore it.
    if cfg.get("decoder") is not None:
        raise ValueError(
            "JEPA model config must NOT contain a 'decoder' section: the JEPA has "
            "no decoder (it compares predicted latents against a target-encoded "
            "latent rather than decoding to raw values). Remove the 'decoder' "
            "section, or use build_raw_world_model for a raw baseline."
        )

    trunk = _build_trunk(cfg)

    policy = cfg.get("policy", "ema")
    ema_decay = float(cfg.get("ema_decay", 0.996))
    collapse_regularizer = _build_collapse_regularizer(cfg)

    return JEPAModel(
        tokenizers=trunk["tokenizers"],
        encoder=trunk["encoder"],
        action_encoder=trunk["action_encoder"],
        predictor=trunk["predictor"],
        policy=TargetUpdatePolicy.coerce(policy),
        ema_decay=ema_decay,
        collapse_regularizer=collapse_regularizer,
    )
