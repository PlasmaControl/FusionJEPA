"""Persistence (copy-forward) baseline for Fusion-JEPA (M2, Task 2.9).

The persistence baseline is experiment ``E00``: the naive, parameter-free
comparison the learned raw baseline (``E02``, :class:`RawWorldModel`) has to
beat. For every target channel it predicts *the same value at every horizon* --
that channel's **last observed context value** -- which is the classic "future
equals present" null model for a forecasting task.

Shared output contract
----------------------
:meth:`PersistenceBaseline.forward` returns ``{modality: preds}`` with each
prediction shaped exactly like that modality's target values ``[B, C, T_tgt]``
and dtype ``float32`` -- byte-for-byte the contract
:class:`~fusion_jepa.models.raw_world_model.RawWorldModel` produces. Because the
two models emit the same structure, the *identical* eval/scoring code path runs
on both (E00 vs E02) with no special-casing.

Mask semantics (the load-bearing detail)
-----------------------------------------
``batch.context_mask[modality]`` is ``True`` where a frame is observed. The
finite-where-masked contract (see :mod:`fusion_jepa.data.batch`) lets a context
value be ``NaN`` wherever its mask is ``False``. Persistence therefore must
select by the mask, never by a blind ``nan_to_num``: for each (example, channel)
it walks back from the end to the last frame whose mask is ``True`` and copies
that (guaranteed finite) value forward. A masked *tail* is skipped; only the
last genuinely observed frame is persisted.

A channel with **no** observed context frame anywhere has nothing to copy. This
implementation predicts a finite ``0.0`` for it (disclosed choice). The raw
prediction objective masks such positions out whenever the corresponding target
is also missing, so the constant is scored-away in the matched comparison -- but
the tensor is guaranteed NaN-free regardless, so it can never poison a metric.

The module holds no parameters, so it never needs ``.eval()`` / ``no_grad``
bookkeeping to behave; it is a pure function of the batch wrapped as an
:class:`~torch.nn.Module` for a uniform call site with the learned models.
"""

import torch
from torch import Tensor, nn

from fusion_jepa.data.batch import FusionBatch


class PersistenceBaseline(nn.Module):
    """Zero-parameter copy-forward baseline over a :class:`FusionBatch`.

    ``forward(batch)`` returns ``{modality: preds}`` for every target modality,
    each ``[B, C, T_tgt]`` float32, repeating each channel's last observed
    context value across all target frames (``0.0`` for a never-observed
    channel or a target modality with no context).
    """

    def forward(self, batch: FusionBatch) -> dict[str, Tensor]:
        """Copy each channel's last observed context value across the horizon."""
        preds: dict[str, Tensor] = {}
        for modality, target in batch.target.items():
            n_examples, n_channels, n_target_frames = target.shape
            values = batch.context.get(modality)
            mask = batch.context_mask.get(modality)
            if values is None or mask is None:
                # Disclosed edge: nothing to persist for a target modality that
                # is absent from the context -> a finite all-zero block.
                preds[modality] = target.new_zeros(
                    (n_examples, n_channels, n_target_frames),
                    dtype=torch.float32,
                )
                continue
            last_valid = self._last_valid_value(values, mask.bool())  # [B, C]
            preds[modality] = (
                last_valid.unsqueeze(-1)
                .expand(-1, -1, n_target_frames)
                .contiguous()
            )
        return preds

    @staticmethod
    def _last_valid_value(values: Tensor, mask: Tensor) -> Tensor:
        """Return the last observed value per (example, channel).

        Args:
            values: ``[B, C, T_ctx]`` context values (float; may be ``NaN``
                wherever ``mask`` is ``False``).
            mask: ``[B, C, T_ctx]`` bool; ``True`` marks an observed frame.

        Returns:
            ``[B, C]`` float32 tensor: the value at each (example, channel)'s
            latest observed frame, or ``0.0`` where no frame is observed. Never
            ``NaN`` -- masked entries are selected out, not numerically patched.
        """
        n_ctx_frames = values.shape[-1]
        frame_index = torch.arange(n_ctx_frames, device=mask.device)
        # -1 at masked frames so the per-(B, C) max lands on the last observed
        # frame, or stays -1 when nothing is observed.
        masked_index = torch.where(
            mask, frame_index, torch.full_like(frame_index, -1)
        )
        last_index = masked_index.max(dim=-1).values  # [B, C]
        has_observation = last_index >= 0
        gather_index = last_index.clamp_min(0).unsqueeze(-1)  # [B, C, 1]
        gathered = values.gather(-1, gather_index).squeeze(-1)  # [B, C]
        # ``torch.where`` discards the NaN branch for never-observed channels.
        zeros = torch.zeros_like(gathered)
        return torch.where(has_observation, gathered, zeros).to(torch.float32)
