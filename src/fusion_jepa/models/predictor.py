"""Action-conditioned latent predictor for Fusion-JEPA (M2, Task 2.5).

This is the core world model, shared verbatim by the raw baseline and the JEPA --
the matched-comparison invariant requires both to predict future latents with the
*same* module, so it lives here once. Given a fixed-size context bottleneck
(``S`` state latents from :class:`~fusion_jepa.models.encoder.ContextEncoder`), a
sequence of encoded action tokens, and a device descriptor, it predicts the
future state latents at one or more requested horizons.

Sequence layout
---------------
Per sample, a single conditioning block is assembled once::

    [ proj(context_latents) ; proj(action_tokens) ; device_token ]
      \\_____ S tokens _____/   \\____ H tokens ____/   \\_ 1 token _/

and ``S`` learned *query* tokens are appended, one query per predicted state
token. For ``K`` requested horizons the query block is materialized ``K`` times --
each copy adds ``Fourier(horizon_k)`` so a query knows how far ahead it predicts.
The ``K`` horizons are laid out along the batch axis (effective batch ``B*K``), so
horizons never attend across one another and each carries its own causal mask.
After ``n_blocks`` self-attention layers the query positions are read out,
layer-normed, and projected back to ``d_latent_in`` -> ``z_hat [B, K, S,
d_latent_in]``.

Causality contract (spec invariant 3)
-------------------------------------
A horizon-``h`` query may attend **only** action tokens whose time is ``<= h``;
actions strictly later than ``h`` must never influence its prediction. Both
``horizons`` and ``action_times`` are measured in seconds **relative to the
context end** (the same time base), so the admissibility test is directly
``action_time <= horizon``.

Because each horizon occupies its own batch element, causality is a *per-column*
property there: an inadmissible action column is marked ignored for **every** row
of that horizon's sequence (a ``key_padding_mask`` in the
:class:`~fusion_jepa.models.encoder.MaskedBlock` convention, ``True == ignore``).
Masking future-action columns from the whole sequence -- not only the query
rows -- is what makes the guarantee airtight *at every depth*: no token (context,
action, device, or query) ever absorbs a future action, so nothing can carry one
into a query indirectly across blocks. A full row-dependent additive attention
mask is therefore unnecessary, and :class:`MaskedBlock` is reused unchanged (no
edit to ``encoder.py``). :func:`test_actions_beyond_horizon_do_not_leak` locks
the invariant: a finite masked key contributes an exact ``0 * value == 0``, so a
horizon's prediction is bit-identical under any change to its future actions.

All-masked safety
-----------------
The device token and the ``S`` query columns are never masked, so every softmax
row always has ``>= 1 + S`` valid keys -- a sample with no admissible actions (or
no actions at all) still yields finite latents, never an all-``-inf`` row.
Mask-invalid context/action positions are additionally zeroed *before* their
input projection: the batch contract only guarantees finite-where-masked, and an
attention-ignored key still enters the matmul as ``0 * value``, which is NaN for
a non-finite value. (Causality-masked actions need no such treatment -- they are
mask-valid and therefore finite by contract.)

Rollout
-------
:meth:`LatentPredictor.rollout` is *defined* as repeated single-horizon direct
:meth:`forward` calls: step ``t`` predicts ``step_seconds`` ahead of step
``t-1``'s prediction, feeding the previous ``z_hat`` back in as the next context
(``context_mask`` is all-valid since predicted state tokens are always present),
and shifting ``action_times`` by ``-t * step_seconds`` so the causal window
advances with the context end. Step 0 is byte-for-byte a single-horizon
``forward`` call, so a 1-step rollout equals the direct call exactly
(:func:`test_single_step_rollout_equals_direct_call`).

Device conditioning
-------------------
``device_id`` strings are mapped through a registered vocabulary to a learned
embedding (an unknown id is an actionable error, never a silent fallback). The
``device_context`` vector is masked-filled with a learned per-dimension value
*before* projection -- missing is not zero, as everywhere in this codebase -- and
added to the embedding to form the single device token.

Determinism
-----------
``dropout`` defaults to ``0.0``, so forward carries no randomness in either mode;
all randomness lives in parameter init. ``float64`` times are cast to ``float32``
only inside the Fourier computation, while the causality comparison stays in
``float64`` for exactness.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from fusion_jepa.models.encoder import MaskedBlock
from fusion_jepa.models.tokenizers import _FourierFeatures

# Std for learned query tokens / missing-fill parameters, matching the tokenizer
# and encoder convention so the projected raw signals dominate at init.
_INIT_STD = 0.02
# Fourier band count for the per-horizon time embedding.
_HORIZON_N_FREQS = 6


def _dtype_satisfies_contract(tensor: Tensor, expected: torch.dtype) -> bool:
    """Whether ``tensor`` meets a strict ``expected`` dtype contract, autocast-aware.

    Outside an autocast region the contract is strict -- only ``expected`` passes.
    Inside an *active* autocast region for the tensor's device, ``nn.Linear`` (and
    the rest of the trunk) legitimately emit the autocast compute dtype (bf16 on
    CUDA), so that dtype is accepted too. This is applied ONLY to the trunk
    activations flowing INTO the predictor (context latents, action tokens); the
    float64 time contracts and the raw-batch ``device_context`` never pass through
    an autocast op, so their checks stay strict and are left untouched.
    """
    if tensor.dtype == expected:
        return True
    device_type = tensor.device.type
    if torch.is_autocast_enabled(device_type):
        return tensor.dtype == torch.get_autocast_dtype(device_type)
    return False


class LatentPredictor(nn.Module):
    """Predict future state latents from context, actions, and a device token.

    Args:
        d_model: internal transformer width.
        n_heads: attention heads per block (must divide ``d_model``).
        n_blocks: number of stacked :class:`MaskedBlock` layers.
        d_latent_in: dimension of the input context latents and the predicted
            output latents (the shared encoder's state width).
        n_state_tokens: number of state tokens ``S`` -- both the context
            bottleneck width and the number of query/output tokens per horizon.
        device_vocab: ordered, unique device-id strings. ``n_devices`` embeddings
            are allocated; a ``device_id`` outside this vocabulary raises.
        d_device_context: width ``Dc`` of the device-context vector (may be ``0``).
        d_action: last-dim ``D_act`` of the encoded action tokens; defaults to
            ``d_model`` (the action encoder's width may differ, so action tokens
            are always projected to ``d_model``).
        n_horizon_freqs: Fourier bands for the per-horizon time embedding.
        mlp_ratio: MLP hidden-dim multiplier passed to each block.
        dropout: dropout passed to each block; ``0.0`` (default) keeps forward
            deterministic.

    ``forward`` returns ``z_hat [B, K, S, d_latent_in]`` (float32 outside
    autocast, the active autocast dtype -- bf16 -- inside it); see the module
    docstring for the sequence layout, causality contract, and time base.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 3,
        d_latent_in: int = 320,
        n_state_tokens: int = 16,
        *,
        device_vocab: Sequence[str],
        d_device_context: int,
        d_action: int | None = None,
        n_horizon_freqs: int = _HORIZON_N_FREQS,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            d_model < 1
            or n_heads < 1
            or n_blocks < 1
            or d_latent_in < 1
            or n_state_tokens < 1
        ):
            raise ValueError(
                "d_model, n_heads, n_blocks, d_latent_in, n_state_tokens must "
                "all be >= 1"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if d_device_context < 0:
            raise ValueError("d_device_context must be >= 0")

        device_list = list(device_vocab)
        if not device_list:
            raise ValueError("device_vocab must contain at least one device id")
        if len(set(device_list)) != len(device_list):
            raise ValueError("device_vocab entries must be unique")

        d_action = d_model if d_action is None else d_action
        if d_action < 1:
            raise ValueError("d_action must be >= 1")

        self.d_model = d_model
        self.d_latent_in = d_latent_in
        self.n_state_tokens = n_state_tokens
        self.d_device_context = d_device_context
        self._device_index = {name: idx for idx, name in enumerate(device_list)}

        # Project the two variable-width inputs to the internal width; both are
        # projected (context latents are d_latent_in, action tokens d_action).
        self.context_proj = nn.Linear(d_latent_in, d_model)
        self.action_proj = nn.Linear(d_action, d_model)

        # Device conditioning: a shared embedding table (one row per known
        # device) plus a masked-filled context projection. When Dc == 0 (real
        # tokamark windows carry no device context) the device token is the
        # embedding alone -- no empty Linear, which also keeps init warning-free.
        self.device_embedding = nn.Embedding(len(device_list), d_model)
        if d_device_context > 0:
            self.device_context_proj: nn.Linear | None = nn.Linear(
                d_device_context, d_model
            )
            self.device_context_fill: nn.Parameter | None = nn.Parameter(
                torch.empty(d_device_context)
            )
        else:
            self.device_context_proj = None
            self.device_context_fill = None

        # Learned query tokens (one per output state token) + horizon embedding.
        self.query_tokens = nn.Parameter(torch.empty(n_state_tokens, d_model))
        self.horizon_features = _FourierFeatures(n_horizon_freqs, d_model)

        self.blocks = nn.ModuleList(
            [
                MaskedBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_latent_in)

        nn.init.normal_(self.query_tokens, std=_INIT_STD)
        if self.device_context_fill is not None:
            nn.init.normal_(self.device_context_fill, std=_INIT_STD)

    @property
    def n_devices(self) -> int:
        """Number of registered device embeddings."""
        return self.device_embedding.num_embeddings

    def _device_token(
        self,
        device_id: list[str],
        device_context: Tensor,
        device_context_mask: Tensor,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        """Build the ``[B, 1, d_model]`` device conditioning token."""
        if len(device_id) != batch:
            raise ValueError(
                f"device_id must have length B={batch}, got {len(device_id)}"
            )
        try:
            indices = [self._device_index[name] for name in device_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown device_id {exc.args[0]!r}; registered devices: "
                f"{sorted(self._device_index)}"
            ) from None
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        embed = self.device_embedding(idx)  # [B, d_model]

        if self.device_context_proj is not None:
            # Missing context entries -> learned per-dim fill (never zero).
            # nan_to_num first so a NaN's arbitrary bits never survive even
            # transiently.
            safe = torch.nan_to_num(device_context)
            filled = torch.where(
                device_context_mask, safe, self.device_context_fill.view(1, -1)
            )
            embed = embed + self.device_context_proj(filled)
        return embed.unsqueeze(1)

    def forward(
        self,
        context_latents: Tensor,
        context_mask: Tensor,
        action_tokens: Tensor,
        action_mask: Tensor,
        action_times: Tensor,
        horizons: Tensor,
        device_id: list[str],
        device_context: Tensor,
        device_context_mask: Tensor,
    ) -> Tensor:
        """Predict future state latents at each requested horizon.

        Args:
            context_latents: ``[B, S, d_latent_in]`` context bottleneck; float32
                outside autocast, the active autocast dtype (bf16) inside it.
            context_mask: ``[B, S]`` bool, ``True`` where a state token is valid.
            action_tokens: ``[B, H, d_action]`` encoded action tokens; float32
                outside autocast, the active autocast dtype (bf16) inside it.
            action_mask: ``[B, H]`` bool, ``True`` where a timestep is valid.
            action_times: ``[B, H]`` float64 action times in seconds, relative to
                the context end (same base as ``horizons``).
            horizons: ``[B, K]`` float64 prediction horizons in seconds, relative
                to the context end.
            device_id: length-``B`` list of device-id strings.
            device_context: ``[B, Dc]`` float32 device-context vector.
            device_context_mask: ``[B, Dc]`` bool, ``True`` where observed.

        Returns:
            ``z_hat [B, K, S, d_latent_in]`` predicted state latents; float32
                outside autocast, the active autocast dtype (bf16) inside it.
        """
        B, S, H, K = self._validate(
            context_latents,
            context_mask,
            action_tokens,
            action_mask,
            action_times,
            horizons,
            device_context,
            device_context_mask,
        )
        device = context_latents.device

        # Masked positions may legally carry non-finite values (the batch
        # contract only guarantees finite-where-masked). Their columns are
        # attention-ignored, but an ignored key still enters the matmul as
        # 0 * value -- and 0 * NaN = NaN -- so neutralize them BEFORE the
        # projection. Zeroing is numerical hygiene, not imputation: unlike a
        # tokenizer's partially-observed patch, these tokens never contribute.
        context_safe = torch.where(
            context_mask.unsqueeze(-1),
            context_latents,
            torch.zeros_like(context_latents),
        )
        actions_safe = torch.where(
            action_mask.unsqueeze(-1),
            action_tokens,
            torch.zeros_like(action_tokens),
        )
        context = self.context_proj(context_safe)  # [B, S, d_model]
        actions = self.action_proj(actions_safe)  # [B, H, d_model]
        device_token = self._device_token(
            device_id, device_context, device_context_mask, B, device
        )  # [B, 1, d_model]
        cond = torch.cat([context, actions, device_token], dim=1)  # [B, Lc, d]
        n_cond = cond.shape[1]  # S + H + 1

        # Per-horizon query block: S shared learned queries + Fourier(horizon).
        horizon_feat = self.horizon_features(horizons.to(torch.float32))  # [B,K,d]
        queries = self.query_tokens.view(1, 1, S, self.d_model) + horizon_feat.view(
            B, K, 1, self.d_model
        )  # [B, K, S, d_model]

        # Lay the K horizons along the batch axis so each is an independent
        # sequence with its own causal mask (no cross-horizon attention).
        cond_bk = cond.unsqueeze(1).expand(B, K, n_cond, self.d_model)
        cond_bk = cond_bk.reshape(B * K, n_cond, self.d_model)
        query_bk = queries.reshape(B * K, S, self.d_model)
        seq = torch.cat([cond_bk, query_bk], dim=1)  # [B*K, Lc+S, d_model]

        key_padding_mask = self._causal_key_padding_mask(
            context_mask, action_mask, action_times, horizons, B, S, H, K
        )

        for block in self.blocks:
            seq = block(seq, key_padding_mask=key_padding_mask)

        query_out = self.query_norm(seq[:, n_cond:, :])  # [B*K, S, d_model]
        z_hat = self.out_proj(query_out)  # [B*K, S, d_latent_in]
        return z_hat.reshape(B, K, S, self.d_latent_in)

    def _causal_key_padding_mask(
        self,
        context_mask: Tensor,
        action_mask: Tensor,
        action_times: Tensor,
        horizons: Tensor,
        B: int,
        S: int,
        H: int,
        K: int,
    ) -> Tensor:
        """Build the ``[B*K, S+H+1+S]`` ignore mask (``True == ignore``).

        Context columns follow ``context_mask``; action columns are ignored when
        masked *or* when the action time exceeds the horizon (the causality
        contract); the device column and the ``S`` query columns are always
        attendable so no softmax row is ever fully ``-inf``.
        """
        ctx_ignore = (~context_mask).unsqueeze(1).expand(B, K, S).reshape(B * K, S)

        # Admissible iff valid AND time <= horizon; comparison stays float64.
        admissible = action_mask.unsqueeze(1) & (
            action_times.unsqueeze(1) <= horizons.unsqueeze(2)
        )  # [B, K, H]
        act_ignore = (~admissible).reshape(B * K, H)

        always = torch.zeros(
            B * K, 1 + S, dtype=torch.bool, device=context_mask.device
        )
        return torch.cat([ctx_ignore, act_ignore, always], dim=1)

    def rollout(
        self,
        context_latents: Tensor,
        context_mask: Tensor,
        action_tokens: Tensor,
        action_mask: Tensor,
        action_times: Tensor,
        device_id: list[str],
        device_context: Tensor,
        device_context_mask: Tensor,
        *,
        step_seconds: float,
        n_steps: int,
    ) -> Tensor:
        """Autoregress ``n_steps`` fixed-size steps of ``step_seconds`` each.

        Each step is a single-horizon :meth:`forward` call re-conditioned on the
        previous step's prediction; ``action_times`` shift by ``-t*step_seconds``
        so the causal window advances with the context end. Step 0 is exactly a
        direct single-horizon call, so a 1-step rollout equals the direct call.

        Returns ``[B, n_steps, S, d_latent_in]`` predicted latents; float32
        outside autocast, the active autocast dtype (bf16) inside it.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        B = context_latents.shape[0]
        device = context_latents.device

        current = context_latents
        current_mask = context_mask
        preds: list[Tensor] = []
        for step in range(n_steps):
            horizons = torch.full(
                (B, 1), step_seconds, dtype=torch.float64, device=device
            )
            shifted_times = action_times - step * step_seconds
            z = self.forward(
                current,
                current_mask,
                action_tokens,
                action_mask,
                shifted_times,
                horizons,
                device_id,
                device_context,
                device_context_mask,
            )  # [B, 1, S, d_latent_in]
            z_step = z[:, 0]  # [B, S, d_latent_in]
            preds.append(z_step)
            current = z_step
            # Predicted state tokens are always present -> all-valid next context.
            current_mask = torch.ones(
                B, self.n_state_tokens, dtype=torch.bool, device=device
            )
        return torch.stack(preds, dim=1)

    def _validate(
        self,
        context_latents: Tensor,
        context_mask: Tensor,
        action_tokens: Tensor,
        action_mask: Tensor,
        action_times: Tensor,
        horizons: Tensor,
        device_context: Tensor,
        device_context_mask: Tensor,
    ) -> tuple[int, int, int, int]:
        """Validate ranks/dtypes/shapes and return ``(B, S, H, K)``."""
        if context_latents.ndim != 3:
            raise ValueError(
                f"context_latents must be [B, S, d_latent_in], got "
                f"{tuple(context_latents.shape)}"
            )
        B, S, D = context_latents.shape
        if S != self.n_state_tokens:
            raise ValueError(
                f"context_latents has S={S} state tokens, predictor expects "
                f"{self.n_state_tokens}"
            )
        if D != self.d_latent_in:
            raise ValueError(
                f"context_latents last dim {D} must equal d_latent_in="
                f"{self.d_latent_in}"
            )
        # Trunk activation: the shared encoder emits this, so under an active
        # autocast region it is legitimately the autocast dtype (bf16). Outside
        # autocast the strict float32 contract stands.
        if not _dtype_satisfies_contract(context_latents, torch.float32):
            raise ValueError(
                f"context_latents must be float32 (or the active autocast "
                f"dtype), got {context_latents.dtype}"
            )
        if tuple(context_mask.shape) != (B, S) or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be a bool tensor of shape [B, S]")

        if action_tokens.ndim != 3 or action_tokens.shape[0] != B:
            raise ValueError(
                f"action_tokens must be [B, H, d_action], got "
                f"{tuple(action_tokens.shape)}"
            )
        H = action_tokens.shape[1]
        # Trunk activation: the action encoder emits this through nn.Linear, so
        # under an active autocast region it is legitimately the autocast dtype
        # (bf16). Outside autocast the strict float32 contract stands.
        if not _dtype_satisfies_contract(action_tokens, torch.float32):
            raise ValueError(
                f"action_tokens must be float32 (or the active autocast dtype), "
                f"got {action_tokens.dtype}"
            )
        if tuple(action_mask.shape) != (B, H) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be a bool tensor of shape [B, H]")
        if (
            tuple(action_times.shape) != (B, H)
            or action_times.dtype != torch.float64
        ):
            raise ValueError(f"action_times must be float64 [B, H]=({B}, {H})")

        if horizons.ndim != 2 or horizons.shape[0] != B:
            raise ValueError(
                f"horizons must be [B, K], got {tuple(horizons.shape)}"
            )
        K = horizons.shape[1]
        if K < 1:
            raise ValueError(
                "horizons must request at least one horizon (K >= 1); an "
                "empty horizon axis would silently produce an empty batch"
            )
        if horizons.dtype != torch.float64:
            raise ValueError(f"horizons must be float64, got {horizons.dtype}")

        Dc = self.d_device_context
        if (
            tuple(device_context.shape) != (B, Dc)
            or device_context.dtype != torch.float32
        ):
            raise ValueError(f"device_context must be float32 [B, Dc]=({B}, {Dc})")
        if (
            tuple(device_context_mask.shape) != (B, Dc)
            or device_context_mask.dtype != torch.bool
        ):
            raise ValueError(
                f"device_context_mask must be a bool tensor of shape [B, {Dc}]"
            )
        return B, S, H, K
