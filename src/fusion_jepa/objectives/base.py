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

    def __post_init__(self) -> None:
        """Enforce the scalar-tensor / plain-float contract at construction."""
        if not isinstance(self.total, Tensor):
            raise ValueError(
                "LossOutput.total must be a torch.Tensor, got "
                f"{type(self.total).__name__}"
            )
        if self.total.ndim != 0:
            raise ValueError(
                "LossOutput.total must be a scalar (0-d) tensor, got ndim "
                f"{self.total.ndim}"
            )
        for key, value in self.terms.items():
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"LossOutput.terms[{key!r}] must be a torch.Tensor, got "
                    f"{type(value).__name__}"
                )
            if value.ndim != 0:
                raise ValueError(
                    f"LossOutput.terms[{key!r}] must be a scalar (0-d) tensor, "
                    f"got ndim {value.ndim}"
                )
        for key, value in self.diagnostics.items():
            if not isinstance(value, float) or isinstance(value, bool):
                raise ValueError(
                    f"LossOutput.diagnostics[{key!r}] must be a plain float, got "
                    f"{type(value).__name__}"
                )
