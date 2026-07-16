"""Tests for JSONL metrics logging."""

import json

import numpy as np
import torch

from fusion_jepa.utils.logging import MetricsLogger, read_metrics


def test_append_and_readback_round_trip(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)

    logger.log(1, loss=0.5, phase="train")
    logger.log(2, loss=0.25, phase="train")

    assert read_metrics(path) == [
        {"step": 1, "loss": 0.5, "phase": "train"},
        {"step": 2, "loss": 0.25, "phase": "train"},
    ]


def test_nonfinite_values_recorded_explicitly(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)

    logger.log(3, loss=float("nan"), upper=float("inf"), lower=-float("inf"))

    raw_record = path.read_text().strip()
    record = json.loads(raw_record)
    assert record == {
        "step": 3,
        "loss": None,
        "upper": None,
        "lower": None,
        "nonfinite_keys": ["loss", "upper", "lower"],
    }
    assert "NaN" not in raw_record
    assert "Infinity" not in raw_record


def test_tensor_values_coerced_to_python_scalars(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)

    logger.log(
        torch.tensor(4),
        loss=torch.tensor(0.5),
        count=np.int64(7),
        ratio=np.float32(0.25),
    )

    assert read_metrics(path) == [
        {"step": 4, "loss": 0.5, "count": 7, "ratio": 0.25}
    ]
