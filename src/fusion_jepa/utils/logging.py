"""JSONL metrics logging utilities."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


class MetricsLogger:
    """Append metrics records to a JSONL file."""

    def __init__(self, path: str | Path, flush_every: int = 1) -> None:
        self._file = Path(path).open("a", encoding="utf-8")
        self._flush_every = flush_every
        self._records_since_flush = 0

    def log(self, step: Any, **metrics: Any) -> None:
        """Append one metrics record."""
        record = {"step": _to_python_scalar(step)}
        nonfinite_keys = []

        for key, value in metrics.items():
            value = _to_python_scalar(value)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
                nonfinite_keys.append(key)
            record[key] = value

        if nonfinite_keys:
            record["nonfinite_keys"] = nonfinite_keys

        self._file.write(json.dumps(record, allow_nan=False) + "\n")
        self._records_since_flush += 1
        if self._records_since_flush >= self._flush_every:
            self._file.flush()
            self._records_since_flush = 0

    def close(self) -> None:
        """Flush and close the metrics file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read metrics records from a JSONL file."""
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]
