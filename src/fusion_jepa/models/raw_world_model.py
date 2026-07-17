"""Raw (reconstruction) world model for Fusion-JEPA (M2, Task 2.6).

This composes the shared building blocks of Milestone 2 into the *raw baseline*::

    per-modality tokenizers -> merge_token_sets
        -> ContextEncoder (state bottleneck z)
        -> LatentPredictor(z, encoded actions, device) -> z_hat
        -> QueryConditionedDecoder(z_hat, target queries) -> raw predictions

The baseline predicts future *raw* signal values (the decoder reconstructs the
target window from the predicted latents). Its whole point is to be a *matched*
comparison to the JEPA that Milestone 3 builds: M3 reuses the IDENTICAL
tokenizer/encoder/action-encoder/predictor classes and configuration and only
swaps the raw reconstruction objective for a latent-prediction one. Because the
predictor, encoder, and tokenizers are the same objects and the readout is the
single shared :class:`~fusion_jepa.models.decoders.QueryConditionedDecoder`
(no modality-keyed decoder modules), the capacity match is structural rather
than something a training script has to police.

Time-base wiring (the load-bearing detail)
------------------------------------------
The :class:`~fusion_jepa.data.batch.FusionBatch` carries **absolute** float64
times (``context_times``, ``target_times``, ``action_times``). The predictor,
however, reasons in seconds **relative to the context end** -- that is the base
its causality contract and horizon embeddings are defined on. This model owns the
conversion:

* ``context_end`` = the latest context time per sample (times are monotone, so
  this is the final context frame);
* the predictor's ``action_times`` = ``batch.action_times - context_end``;
* the predictor's ``horizons`` = ``batch.horizon_seconds`` -- already a duration
  measured from the context end (``batch`` guarantees it equals
  ``target_end - context_end``), reshaped to the ``[B, 1]`` the predictor wants
  (``K == 1``: one predicted latent set for the whole target window, matching the
  single target-window latent the JEPA target encoder would produce).

The context tokenizers and the action encoder keep receiving **absolute** times
for their Fourier position features (the same base they use everywhere), so token
positions stay on one consistent physical clock; only the predictor's causality /
horizon reasoning is re-based, exactly as the 2.5 wiring contract specifies. The
decoder's query horizons are likewise ``target_time - context_end`` (built inside
:meth:`QueryConditionedDecoder.build_target_queries`), so queries and predictor
share the relative base.

Device conditioning
-------------------
The device vocabulary is fixed at predictor construction; this model simply
forwards ``batch.device_id`` / ``batch.device_context`` through. An id outside
the predictor's vocabulary is an actionable error raised by the predictor.

Scope
-----
Context modalities are tokenized by scalar-series tokenizers (the batch carries
no per-signal radial coordinate axis, which a profile tokenizer would need), and
the decoder's query channel embeddings reuse each tokenizer's channel embedding,
so a modality's tokenizer must expose a ``channel_embed`` of shape
``[C, d_model]`` (as :class:`ScalarSeriesTokenizer` does). Both requirements are
checked with actionable errors.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.decoders import QueryConditionedDecoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.types import merge_token_sets


class RawWorldModel(nn.Module):
    """Compose tokenizers, encoder, action encoder, predictor, and decoder.

    Args:
        tokenizers: ``{modality: tokenizer}`` mapping. Each tokenizer is called
            as ``tokenizer(values, value_mask, times)`` on that context
            modality and must expose a ``channel_embed`` parameter of shape
            ``[C, d_model]`` (reused as the decoder's query channel embedding).
        encoder: the shared :class:`ContextEncoder`.
        action_encoder: the shared :class:`ActionEncoder`.
        predictor: the shared :class:`LatentPredictor` (its ``device_vocab`` was
            fixed at construction and governs valid ``batch.device_id`` values).
        decoder: the single shared :class:`QueryConditionedDecoder`.

    ``forward(batch)`` returns ``{modality: preds}`` with each prediction shaped
    exactly like that modality's target values ``[B, C, T_tgt]``.
    """

    def __init__(
        self,
        tokenizers: Mapping[str, nn.Module],
        encoder: ContextEncoder,
        action_encoder: ActionEncoder,
        predictor: LatentPredictor,
        decoder: QueryConditionedDecoder,
    ) -> None:
        super().__init__()
        if not tokenizers:
            raise ValueError("RawWorldModel requires at least one tokenizer")
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.decoder = decoder

    def _channel_registry(self, modalities: list[str]) -> dict[str, Tensor]:
        """Gather each modality's tokenizer channel embedding for query building.

        Reusing the tokenizer's channel embedding keeps channel identity shared
        between the input (tokenization) and output (query) sides and adds no new
        modality-keyed parameter.
        """
        registry: dict[str, Tensor] = {}
        for modality in modalities:
            if modality not in self.tokenizers:
                raise ValueError(
                    f"no tokenizer registered for target modality {modality!r}"
                )
            tokenizer = self.tokenizers[modality]
            channel_embed = getattr(tokenizer, "channel_embed", None)
            if channel_embed is None:
                raise ValueError(
                    f"tokenizer for {modality!r} must expose a 'channel_embed' "
                    "parameter to build target queries"
                )
            registry[modality] = channel_embed
        return registry

    def forward(self, batch: FusionBatch) -> dict[str, Tensor]:
        """Predict raw target values for every target modality in ``batch``."""
        context_times = batch.context_times  # [B, T] float64 (absolute)

        # 1. Tokenize each context modality (absolute times) and merge into one
        #    flat token sequence for the shared encoder.
        token_sets = []
        for modality, values in batch.context.items():
            if modality not in self.tokenizers:
                raise ValueError(
                    f"no tokenizer registered for context modality {modality!r}"
                )
            tokenizer = self.tokenizers[modality]
            token_sets.append(
                tokenizer(values, batch.context_mask[modality], context_times)
            )
        tokens, token_mask, _ = merge_token_sets(token_sets)

        # 2. Encode context into the fixed-size state bottleneck.
        z_ctx, z_ctx_mask = self.encoder(tokens, token_mask)

        # 3. Encode the actuator waveforms (absolute times for the token
        #    position features, matching the context tokenizers).
        action_tokens, action_valid = self.action_encoder(
            batch.actions, batch.action_mask, batch.action_times
        )

        # 4. Predict future latents. Re-base action times / horizons to the
        #    context end -- the predictor's causality + horizon time base.
        context_end = context_times.to(torch.float64).max(dim=1).values  # [B]
        action_times_rel = batch.action_times.to(torch.float64) - context_end.unsqueeze(
            1
        )
        horizons = batch.horizon_seconds.to(torch.float64).reshape(-1, 1)  # [B, 1]
        z_hat = self.predictor(
            context_latents=z_ctx,
            context_mask=z_ctx_mask,
            action_tokens=action_tokens,
            action_mask=action_valid,
            action_times=action_times_rel,
            horizons=horizons,
            device_id=batch.device_id,
            device_context=batch.device_context,
            device_context_mask=batch.device_context_mask,
        )  # [B, 1, S, d_latent]

        # 5. Build per-modality target queries and read them out of z_hat with
        #    the single shared decoder. The predictor's K == 1 latent slice is
        #    the only horizon slice the decoder attends onto.
        registry = self._channel_registry(list(batch.target))
        queries_by_modality = self.decoder.build_target_queries(batch, registry)

        preds: dict[str, Tensor] = {}
        for modality, (queries, query_mask, shape) in queries_by_modality.items():
            out = self.decoder(
                z_hat,
                queries.unsqueeze(1),  # [B, 1, Q, d_model]
                query_mask.unsqueeze(1),  # [B, 1, Q]
            )  # [B, 1, Q]
            n_channels, n_frames = shape
            preds[modality] = out.reshape(out.shape[0], n_channels, n_frames)
        return preds
