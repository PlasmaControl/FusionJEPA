"""Action encoder for Fusion-JEPA (M2, Task 2.4).

The action encoder turns a shot's actuator waveforms into a per-timestep token
sequence -- the conditioning signal the action-conditioned world model attends
to. Actuator identity and timing must be structurally preserved, so the encoding
is permutation-sensitive across actuator channels and shift-sensitive in time.

One token per timestep: token ``h`` summarizes all ``A`` actuator values at
timestep ``h`` as ``linear(actuator values) + Fourier(time)``. The ``H`` output
tokens line up one-to-one with the ``H`` input timesteps -- no patching, since
each timestep is an independent action to condition on.

Mask / missing semantics
------------------------
The ``FusionBatch`` action mask is *per-timestep* (``action_mask [B, H]``), not
per-channel. Two distinct notions of "absent" are handled separately:

* **Missing channel (per-channel).** A non-finite entry (``NaN``) in ``actions``
  marks that one actuator as unobserved at that timestep. Each such entry is
  replaced by a learned *per-actuator* fill value BEFORE the linear projection
  (never by ``0`` -- missing is not zero), so the network can distinguish an
  unobserved actuator from a genuine zero command. The arbitrary bits behind a
  ``NaN`` never reach the projection.
* **Masked timestep (per-timestep).** ``action_mask[b, h] == False`` marks the
  whole timestep invalid; the returned ``u_mask`` is exactly ``action_mask``.
  The token is still computed and kept finite, but downstream stages ignore it.

Because every token is computed independently along ``H`` (no cross-timestep
mixing), changing the ``actions`` values at one timestep can only move that
timestep's own token: a masked timestep's values never affect any other token,
and a valid token is bit-identical regardless of what sits at masked positions.

All-invalid / all-``NaN`` inputs stay finite -- ``NaN`` is neutralized before the
projection and nothing downstream can reintroduce it. Forward is fully
deterministic (all randomness lives in parameter init); ``float64`` times are
cast to ``float32`` only inside the Fourier computation.
"""

import torch
from torch import Tensor, nn

from fusion_jepa.models.tokenizers import _FourierFeatures

# Std for the learned per-actuator missing-fill parameters, matching the
# tokenizer / encoder convention so the raw-signal projection dominates the token
# at initialization.
_EMBED_INIT_STD = 0.02


class ActionEncoder(nn.Module):
    """Encode actuator waveforms into one conditioning token per timestep.

    Args:
        n_actuators: number of actuator channels ``A`` in the input.
        d_model: token embedding dimension ``D``.
        n_time_freqs: number of Fourier bands for the token time feature.

    ``forward(actions, action_mask, action_times)`` takes:

    * ``actions`` -- ``[B, H, A]`` float32 actuator values (a ``NaN`` marks a
      per-channel missing/unobserved actuator at that timestep).
    * ``action_mask`` -- ``[B, H]`` bool, ``True`` where the timestep is valid.
    * ``action_times`` -- ``[B, H]`` float64 physical timestep times in seconds.

    and returns ``(u_tokens [B, H, D] float32, u_mask [B, H] bool)`` with one
    token per timestep. ``u_mask`` is exactly ``action_mask`` -- a masked
    timestep yields an invalid (but finite) token. The per-actuator linear makes
    the encoding permutation-sensitive across channels (actuator identity is
    positional), and the added ``Fourier(action_times)`` makes it shift-sensitive
    in time.
    """

    def __init__(self, n_actuators: int, d_model: int, n_time_freqs: int) -> None:
        super().__init__()
        if n_actuators < 1 or d_model < 1:
            raise ValueError("n_actuators and d_model must both be >= 1")
        self.n_actuators = n_actuators
        self.d_model = d_model

        # Per-actuator linear: actuator identity is positional, so each input
        # index owns a distinct weight column -> permuting channels changes the
        # output. This maps the A-vector at each timestep to a D-token.
        self.proj = nn.Linear(n_actuators, d_model)
        # One learned fill value per actuator, substituted for missing (NaN)
        # entries in value-space before projection (never zero: missing != zero).
        self.missing_fill = nn.Parameter(torch.empty(n_actuators))
        self.time_features = _FourierFeatures(n_time_freqs, d_model)

        nn.init.normal_(self.missing_fill, std=_EMBED_INIT_STD)

    def forward(
        self, actions: Tensor, action_mask: Tensor, action_times: Tensor
    ) -> tuple[Tensor, Tensor]:
        if actions.ndim != 3:
            raise ValueError(
                f"actions must be [B, H, A], got {tuple(actions.shape)}"
            )
        B, H, A = actions.shape
        if A != self.n_actuators:
            raise ValueError(
                f"actions has {A} actuators, encoder expects {self.n_actuators}"
            )
        if actions.dtype != torch.float32:
            raise ValueError(f"actions must be float32, got {actions.dtype}")
        if tuple(action_mask.shape) != (B, H) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be a bool tensor of shape [B, H]")
        if (
            tuple(action_times.shape) != (B, H)
            or action_times.dtype != torch.float64
        ):
            raise ValueError(f"action_times must be float64 [B, H]=({B}, {H})")

        # Per-channel missing (NaN) -> learned per-actuator fill, before proj.
        # nan_to_num first so a NaN's arbitrary bits never survive even
        # transiently; torch.where then selects the learned fill for those slots.
        finite = torch.isfinite(actions)
        safe = torch.nan_to_num(actions)
        filled = torch.where(finite, safe, self.missing_fill.view(1, 1, A))

        tokens = self.proj(filled)
        # Fourier(time): one token per timestep, so the time enters directly.
        time_feat = self.time_features(action_times.to(torch.float32))
        tokens = tokens + time_feat

        # A masked timestep is invalid; its (finite) token is ignored downstream.
        return tokens, action_mask
