"""Tests for signal registry loading, validation, and TokaMark coverage."""

from pathlib import Path

import pytest
import yaml
from tokamark.tasks import get_task_config

from fusion_jepa.data.registry import load_registry, validate_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
MAST_REGISTRY = REPO_ROOT / "signal_registry" / "mast.yaml"
GROUP23_TASKS = (
    "task_2-1",
    "task_2-2",
    "task_2-3",
    "task_3-1",
    "task_3-2",
    "task_3-3",
)


def _valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "canonical_name": "mast.summary.ip",
        "device": "MAST",
        "source_name": "summary-ip",
        "units": "A",
        "kind": "measurement",
        "sharing_label": "internal",
        "review_status": "pending_physics_review",
        "description": "Measured plasma current.",
    }
    entry.update(overrides)
    return entry


def _write_registry(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"signals": entries}), encoding="utf-8")
    return path


def test_mast_registry_validates_against_schema() -> None:
    registry = load_registry(MAST_REGISTRY)

    assert registry
    assert validate_registry(list(registry.values())) == []


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    entry = _valid_entry()
    del entry["description"]
    path = _write_registry(tmp_path, [entry])

    with pytest.raises(ValueError, match="missing required field.*description"):
        load_registry(path)


def test_duplicate_canonical_name_rejected(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_entry(), _valid_entry()])

    with pytest.raises(ValueError, match="duplicate canonical_name.*mast.summary.ip"):
        load_registry(path)


def test_shared_core_requires_approved_review_status(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        [_valid_entry(sharing_label="shared_core")],
    )

    with pytest.raises(ValueError, match="shared_core.*approved"):
        load_registry(path)


def test_null_actuator_extras_rejected(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        [
            _valid_entry(
                kind="actuator",
                command_or_measured=None,
                bounds=None,
                rate_limit=None,
                safe_to_perturb=None,
            )
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        load_registry(path)

    message = str(exc_info.value)
    assert "command_or_measured" in message
    assert "safe_to_perturb" in message


def test_unknown_field_rejected_with_value_error(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_entry(typo_field="unexpected")])

    with pytest.raises(ValueError, match="typo_field"):
        load_registry(path)


def test_every_group23_task_signal_has_registry_entry() -> None:
    registry = load_registry(MAST_REGISTRY)
    registered_sources = {entry.source_name for entry in registry.values()}
    task_sources = set()

    for task_name in GROUP23_TASKS:
        segmenter = get_task_config(task_name)["task_window_segmenter"]
        for role in ("input_keys", "actuator_keys", "output_keys"):
            task_sources.update(
                f"{source}-{signal}" for source, signal in segmenter[role] or []
            )

    assert task_sources <= registered_sources
