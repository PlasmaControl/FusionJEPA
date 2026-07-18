"""JEPA world model with explicit target-update policies (M3, Task 3.2).

This is the Milestone 3 counterpart of
:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`. It reuses the
*identical* building blocks -- per-modality tokenizers, the shared
:class:`~fusion_jepa.models.encoder.ContextEncoder`, the
:class:`~fusion_jepa.models.action_encoder.ActionEncoder`, and the
:class:`~fusion_jepa.models.predictor.LatentPredictor` -- so the JEPA and the raw
baseline are a *matched* comparison by construction (the capacity match is
structural, not something a training script has to police). Where the raw baseline
decodes predicted latents back to raw signal values, the JEPA instead compares the
predicted latents ``z_hat`` against a *target-encoded* latent ``z_target`` of the
future window.

Online vs. target trunk
-----------------------
Per the M3 plan, only the **tokenizers + context encoder** get a target twin; the
action encoder and predictor are online-only (there is nothing to encode a target
*with* them). Exactly how the target trunk relates to the online trunk, and
whether gradient flows through it, is chosen *explicitly* at construction via
:class:`TargetUpdatePolicy` -- never inferred implicitly:

* :attr:`TargetUpdatePolicy.EMA` -- the target tokenizers+encoder are a
  ``copy.deepcopy`` of the online ones with every parameter frozen
  (``requires_grad_(False)``). The target forward runs under ``torch.no_grad``.
  A separate :class:`~fusion_jepa.training.ema.EmaUpdater` (Task 3.1) nudges the
  target toward the online weights; :meth:`JEPAModel.target_encoder_pairs`
  exposes exactly the ``(online, target)`` module pairs it consumes. deepcopy
  preserves module registration and parameter iteration order, so zipping the
  online/target ``.parameters()`` is a sound one-to-one pairing.
* :attr:`TargetUpdatePolicy.SHARED_STOPGRAD` -- the target trunk *is* the online
  trunk (same module objects, shared weights); the target latent is
  ``detach()``ed so no gradient flows back through the target branch.
* :attr:`TargetUpdatePolicy.END_TO_END_REGULARIZED` -- the target trunk is again
  the online trunk, but the target latent is *not* detached (gradient flows
  through both branches). Without a stop-gradient this objective is only
  well-posed alongside a collapse regularizer, so construction *requires* a
  non-``None`` ``collapse_regularizer`` (the regularizer itself is Task 3.4's
  scope: it is stored verbatim and never invoked here).

Time-base wiring
----------------
The context path mirrors :class:`RawWorldModel` exactly: context tokenizers and
the action encoder receive **absolute** float64 times for their Fourier position
features, while the predictor's causality / horizon reasoning is re-based to the
context end (``action_times - context_end``; ``horizons = horizon_seconds``,
reshaped to the ``[B, 1]`` the predictor wants -- ``K == 1``, one predicted latent
set for the single target window). The target path tokenizes the future window at
its own absolute ``target_times`` -- the same absolute-time convention the context
tokenizers use -- and encodes it through the target trunk into ``z_target``.

Multi-horizon note
------------------
The landed :class:`~fusion_jepa.data.batch.FusionBatch` carries exactly one target
window, so ``z_target`` is ``[B, K=1, S, D]`` here. If batches ever carry ``H``
target windows, target encoding will fold ``[B, H, ...] -> [B*H, ...]`` before the
trunk and unfold back, matching the predictor's ``K`` axis; that is deliberately
not built until such batches exist (YAGNI).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from fusion_jepa.data.batch import FusionBatch
from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.types import merge_token_sets


class TargetUpdatePolicy(Enum):
    """How the JEPA target trunk relates to the online trunk.

    The value is the lowercase canonical name, so both the enum member and its
    lowercase string name are accepted at construction (see :meth:`coerce`).
    """

    EMA = "ema"
    SHARED_STOPGRAD = "shared_stopgrad"
    END_TO_END_REGULARIZED = "end_to_end_regularized"

    @classmethod
    def coerce(cls, value: "TargetUpdatePolicy | str") -> "TargetUpdatePolicy":
        """Return ``value`` as a member, accepting a lowercase string name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                valid = ", ".join(policy.value for policy in cls)
                raise ValueError(
                    f"unknown target-update policy {value!r}; expected a "
                    f"TargetUpdatePolicy or one of: {valid}"
                ) from None
        raise TypeError(
            "policy must be a TargetUpdatePolicy or its lowercase string name, "
            f"got {type(value).__name__}"
        )


@dataclass
class JEPAOutput:
    """The JEPA forward result.

    Attributes:
        z_hat: ``[B, K, S, d_latent]`` predicted future state latents (``K == 1``
            for the landed single-window batch).
        z_target: ``[B, K, S, D]`` target-encoded latents of the future window.
            Gradient semantics follow the model's :class:`TargetUpdatePolicy`.
        target_valid: ``[B, K]`` bool, ``True`` iff at least one target sample is
            observed for that example (any ``target_mask`` entry ``True`` across
            all target modalities).
    """

    z_hat: Tensor
    z_target: Tensor
    target_valid: Tensor


class JEPAModel(nn.Module):
    """Compose tokenizers, encoder, action encoder, and predictor into a JEPA.

    Args:
        tokenizers: ``{modality: tokenizer}`` mapping wrapped in an
            :class:`~torch.nn.ModuleDict`. Each tokenizer is called as
            ``tokenizer(values, value_mask, times)`` and must cover every context
            and target modality of the batch. Requires ``channel_embed`` (the
            model builder rejects profile-type tokenizers for exactly this reason;
            here the requirement is inherited from the shared components).
        encoder: the shared :class:`ContextEncoder`.
        action_encoder: the shared :class:`ActionEncoder` (online-only).
        predictor: the shared :class:`LatentPredictor` (online-only).
        policy: a :class:`TargetUpdatePolicy` member or its lowercase string name.
        ema_decay: the EMA decay carried for whoever builds the external
            :class:`~fusion_jepa.training.ema.EmaUpdater`; stored, not applied
            here (the updater owns the nudge).
        collapse_regularizer: required (non-``None``) for
            :attr:`TargetUpdatePolicy.END_TO_END_REGULARIZED`; stored verbatim and
            never invoked (Task 3.4 owns its use). Deliberately *not* registered
            as a submodule so an ``nn.Module`` regularizer never injects
            parameters into this matched-capacity trunk.

    ``forward(batch)`` returns a :class:`JEPAOutput`.
    """

    def __init__(
        self,
        tokenizers: Mapping[str, nn.Module],
        encoder: ContextEncoder,
        action_encoder: ActionEncoder,
        predictor: LatentPredictor,
        policy: TargetUpdatePolicy | str = TargetUpdatePolicy.EMA,
        ema_decay: float = 0.996,
        collapse_regularizer: object | None = None,
    ) -> None:
        super().__init__()
        if not tokenizers:
            raise ValueError("JEPAModel requires at least one tokenizer")

        self.policy = TargetUpdatePolicy.coerce(policy)
        self.ema_decay = float(ema_decay)

        self.tokenizers = nn.ModuleDict(tokenizers)
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor

        if (
            self.policy is TargetUpdatePolicy.END_TO_END_REGULARIZED
            and collapse_regularizer is None
        ):
            raise ValueError(
                "policy=END_TO_END_REGULARIZED requires a non-None "
                "collapse_regularizer: without a stop-gradient the target branch "
                "would collapse, so a collapse_regularizer must be supplied to "
                "keep the objective well-posed"
            )
        # Stored verbatim and never invoked here (Task 3.4 owns its use). Bypass
        # nn.Module.__setattr__ so an nn.Module regularizer is NOT registered as
        # a submodule -- that would silently add parameters to this trunk and
        # break the matched-capacity invariant with the raw baseline.
        object.__setattr__(self, "collapse_regularizer", collapse_regularizer)

        if self.policy is TargetUpdatePolicy.EMA:
            # A frozen deepcopy: target twins evolve only via the external
            # EmaUpdater, never via backprop.
            self.target_tokenizers = copy.deepcopy(self.tokenizers)
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_tokenizers.parameters():
                param.requires_grad_(False)
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)
        else:
            # SHARED_STOPGRAD and END_TO_END_REGULARIZED both share the online
            # trunk (same objects); they differ only in whether the target
            # latent is detached in forward.
            self.target_tokenizers = self.tokenizers
            self.target_encoder = self.encoder

    def forward(self, batch: FusionBatch) -> JEPAOutput:
        """Predict future latents and encode the target window."""
        z_hat = self._encode_context(batch)  # [B, 1, S, d_latent]
        z_target = self._encode_target(batch)  # [B, 1, S, D]
        target_valid = self._target_valid(batch)  # [B, 1]
        return JEPAOutput(z_hat=z_hat, z_target=z_target, target_valid=target_valid)

    def _encode_context(self, batch: FusionBatch) -> Tensor:
        """Mirror :class:`RawWorldModel` context wiring; return ``z_hat``."""
        context_times = batch.context_times  # [B, T] float64 (absolute)

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

        z_ctx, z_ctx_mask = self.encoder(tokens, token_mask)

        action_tokens, action_valid = self.action_encoder(
            batch.actions, batch.action_mask, batch.action_times
        )

        # Re-base action times / horizons to the context end -- the predictor's
        # causality + horizon time base -- exactly as the raw baseline does.
        context_end = context_times.to(torch.float64).max(dim=1).values  # [B]
        action_times_rel = batch.action_times.to(torch.float64) - context_end.unsqueeze(
            1
        )
        horizons = batch.horizon_seconds.to(torch.float64).reshape(-1, 1)  # [B, 1]
        return self.predictor(
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

    def _encode_target(self, batch: FusionBatch) -> Tensor:
        """Encode the future window through the target trunk; apply grad policy.

        The target latent is structurally finite even for a fully-masked target
        window: the tokenizer substitutes a learned missing-fill for unobserved
        samples (never NaN), and the encoder's always-valid state tokens keep
        every softmax row off the all-``-inf`` path.
        """
        if self.policy is TargetUpdatePolicy.EMA:
            # Frozen twins AND a no_grad guard -- belt and braces.
            with torch.no_grad():
                z = self._run_target_trunk(batch)
        else:
            z = self._run_target_trunk(batch)
            if self.policy is TargetUpdatePolicy.SHARED_STOPGRAD:
                z = z.detach()
        return z.unsqueeze(1)  # [B, S, D] -> [B, K=1, S, D]

    def _run_target_trunk(self, batch: FusionBatch) -> Tensor:
        """Tokenize + encode the target window (absolute times); return z [B,S,D]."""
        target_times = batch.target_times  # [B, T_tgt] float64 (absolute)
        token_sets = []
        for modality, values in batch.target.items():
            if modality not in self.target_tokenizers:
                raise ValueError(
                    f"no tokenizer registered for target modality {modality!r}"
                )
            tokenizer = self.target_tokenizers[modality]
            token_sets.append(
                tokenizer(values, batch.target_mask[modality], target_times)
            )
        tokens, token_mask, _ = merge_token_sets(token_sets)
        z, _ = self.target_encoder(tokens, token_mask)
        return z

    @staticmethod
    def _target_valid(batch: FusionBatch) -> Tensor:
        """``[B, 1]`` bool: True iff any target sample is observed per example.

        A window is valid when at least one ``target_mask`` entry is ``True``
        across all target modalities -- the OR over every modality's per-example
        "any observed sample" reduction.
        """
        valid: Tensor | None = None
        for mask in batch.target_mask.values():
            observed = mask.reshape(mask.shape[0], -1).any(dim=1)  # [B]
            valid = observed if valid is None else valid | observed
        assert valid is not None  # batch always carries >= 1 target modality
        return valid.unsqueeze(1)  # [B, 1]

    def target_encoder_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        """Return ``(online, target)`` module pairs for the EMA updater.

        Directly consumable by :class:`~fusion_jepa.training.ema.EmaUpdater`,
        which zips each pair's ``.parameters()`` in definition order. Only the
        :attr:`TargetUpdatePolicy.EMA` policy has a separate target trunk to
        update; the other policies share the online trunk, so there is nothing to
        EMA-update and this raises an actionable error.
        """
        if self.policy is not TargetUpdatePolicy.EMA:
            raise ValueError(
                "target_encoder_pairs() is only defined for the EMA policy; "
                f"policy {self.policy.value!r} shares the online trunk, so there "
                "is no separate target to EMA-update"
            )
        return [
            (self.tokenizers, self.target_tokenizers),
            (self.encoder, self.target_encoder),
        ]
