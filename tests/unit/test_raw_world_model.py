"""Unit tests for the raw world model composition (Task 2.6).

The raw baseline composes the IDENTICAL classes later shared with the JEPA:
tokenizers -> merge -> :class:`ContextEncoder` -> (:class:`ActionEncoder`,
:class:`LatentPredictor`) -> :class:`QueryConditionedDecoder`. Two behaviours are
locked:

* :func:`test_forward_on_synthetic_batch_matches_target_shapes` -- a forward on a
  deterministic synthetic :class:`FusionBatch` returns one prediction tensor per
  target modality, each shaped exactly like that modality's target values;
* :func:`test_gradients_reach_tokenizers_encoder_action_encoder_predictor_decoder`
  -- a scalar built from the predictions backpropagates a gradient into every one
  of the five composed sub-modules, proving the whole pipeline is differentiable
  end-to-end (no path is accidentally detached).

``build_raw_world_model`` is the tiny shared constructor both this module and
``test_decoders.py`` use; it wires consistent (tiny) widths across the five
components.
"""

from collections.abc import Sequence

import torch

from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.decoders import QueryConditionedDecoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.raw_world_model import RawWorldModel
from fusion_jepa.models.tokenizers import ScalarSeriesTokenizer
from tests.fixtures.synthetic import make_synthetic_fusion_batch

# Tiny, mutually consistent widths. ``D`` is shared by the tokenizers, the
# context encoder, the predicted-latent dim, and the decoder (the decoder's
# query channel embeddings reuse the tokenizer channel embeddings, so their
# widths must agree).
_D = 16
_S = 4
_N_HEADS = 4
_D_ACTION = 12
_D_PRED = 16
_PATCH_LEN = 2
_N_TIME_FREQS = 4
_DEVICE_VOCAB = ("MAST",)
_D_DEVICE_CONTEXT = 3


def build_raw_world_model(
    modalities: Sequence[str] = ("slow_ts", "profile"),
    *,
    n_channels: int = 3,
    n_actuators: int = 2,
    device_vocab: Sequence[str] = _DEVICE_VOCAB,
    seed: int = 0,
) -> RawWorldModel:
    """Compose a tiny raw world model over ``modalities`` with consistent dims."""
    torch.manual_seed(seed)
    tokenizers = {
        modality: ScalarSeriesTokenizer(
            n_channels=n_channels,
            d_model=_D,
            patch_len=_PATCH_LEN,
            n_time_freqs=_N_TIME_FREQS,
            modality=modality,
        )
        for modality in modalities
    }
    encoder = ContextEncoder(
        d_model=_D,
        n_heads=_N_HEADS,
        n_blocks=2,
        n_state_tokens=_S,
    )
    action_encoder = ActionEncoder(
        n_actuators=n_actuators,
        d_model=_D_ACTION,
        n_time_freqs=_N_TIME_FREQS,
    )
    predictor = LatentPredictor(
        d_model=_D_PRED,
        n_heads=_N_HEADS,
        n_blocks=2,
        d_latent_in=_D,
        n_state_tokens=_S,
        device_vocab=list(device_vocab),
        d_device_context=_D_DEVICE_CONTEXT,
        d_action=_D_ACTION,
    )
    decoder = QueryConditionedDecoder(
        d_latent=_D,
        d_model=_D,
        n_heads=_N_HEADS,
        n_blocks=2,
    )
    return RawWorldModel(
        tokenizers=tokenizers,
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
    )


def test_forward_on_synthetic_batch_matches_target_shapes():
    modalities = ("slow_ts", "profile")
    batch = make_synthetic_fusion_batch(
        B=2, modalities=modalities, n_channels=3, T=4, H=3, A=2
    )
    model = build_raw_world_model(modalities=modalities, n_channels=3, n_actuators=2)

    preds = model(batch)

    assert set(preds) == set(modalities)
    for modality in modalities:
        assert preds[modality].shape == batch.target[modality].shape
        assert preds[modality].dtype == torch.float32
        assert torch.isfinite(preds[modality]).all()


def test_gradients_reach_tokenizers_encoder_action_encoder_predictor_decoder():
    modalities = ("slow_ts", "profile")
    batch = make_synthetic_fusion_batch(
        B=2, modalities=modalities, n_channels=3, T=4, H=3, A=2
    )
    model = build_raw_world_model(modalities=modalities, n_channels=3, n_actuators=2)

    preds = model(batch)
    # A simple reconstruction loss against the (fully observed) targets exercises
    # every predicted element and thus every upstream parameter.
    loss = sum(
        ((preds[m] - batch.target[m]) ** 2).mean() for m in modalities
    )
    loss.backward()

    def _has_grad(param: torch.Tensor) -> bool:
        return param.grad is not None and bool(torch.any(param.grad != 0))

    # One representative *tokenization-path* parameter per tokenizer (the input
    # projection, reached only through encoder -> predictor -> decoder).
    for modality in modalities:
        assert _has_grad(model.tokenizers[modality].proj.weight)

    # Encoder: reached via the context bottleneck feeding the predictor.
    assert any(_has_grad(p) for p in model.encoder.parameters())
    # Action encoder: reached because every synthetic action is causally
    # admissible at the requested horizon and therefore attended.
    assert _has_grad(model.action_encoder.proj.weight)
    # Predictor: the output projection sits directly on the prediction path.
    assert _has_grad(model.predictor.out_proj.weight)
    # Decoder: the scalar readout projection.
    assert _has_grad(model.decoder.out_proj.weight)
