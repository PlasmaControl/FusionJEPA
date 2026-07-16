"""Canonical data contracts for Fusion-JEPA."""

from fusion_jepa.data.batch import (
    FusionBatch,
    FusionSample,
    collate_fusion,
    validate_batch,
)

__all__ = [
    "FusionBatch",
    "FusionSample",
    "collate_fusion",
    "validate_batch",
]
