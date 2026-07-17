"""Shared return contract for Fusion-JEPA training objectives."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class LossOutput:
    """What every training objective returns for one batch.

    Attributes:
        total: Scalar tensor to call ``.backward()`` on.
        terms: Named scalar tensors summing (up to weighting) into ``total``,
            kept for per-term logging and gradient inspection.
        diagnostics: Named plain-``float`` metrics for logging only; these carry
            no gradient and must never be backpropagated through.
    """

    total: Tensor
    terms: dict[str, Tensor]
    diagnostics: dict[str, float]
