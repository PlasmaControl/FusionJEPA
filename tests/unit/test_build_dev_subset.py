"""Tests for the deterministic TokaMark development-subset builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from fusion_jepa.data.splits import SplitManifest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA git repository")


def _load_script():
    path = _repo_root() / "scripts" / "build_dev_subset.py"
    spec = importlib.util.spec_from_file_location("build_dev_subset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_dev_subset = _load_script()

SIGNALS = {
    ("magnetics", "flux_loop_flux"),
    ("summary", "ip"),
    ("pf_active", "coil_current"),
}


def _write_v3_array(store: Path, name: str, values: np.ndarray) -> None:
    array = store / name
    array.mkdir(parents=True)
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(values.shape),
        "data_type": str(values.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(values.shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
        "dimension_names": None,
    }
    (array / "zarr.json").write_text(json.dumps(metadata))
    chunk = array / "c" / "0"
    chunk.parent.mkdir()
    chunk.write_bytes(values.tobytes())


def _write_v3_store(path: Path, signals: set[tuple[str, str]]) -> None:
    path.mkdir(parents=True)
    group_metadata = json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {}}
    )
    (path / "zarr.json").write_text(group_metadata)
    for source_name in sorted({source for source, _ in signals} | {"irrelevant"}):
        source = path / source_name
        source.mkdir()
        (source / "zarr.json").write_text(group_metadata)
    for source_name, signal_name in sorted(signals):
        _write_v3_array(
            path,
            f"{source_name}/{signal_name}",
            np.arange(4, dtype=np.float32),
        )
    _write_v3_array(
        path, "irrelevant/not_for_tasks", np.arange(2, dtype=np.int8)
    )


@pytest.fixture
def tiny_store(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    layouts = {
        "train": {
            "101": SIGNALS,
            "102": SIGNALS - {("summary", "ip")},
            "103": SIGNALS - {("pf_active", "coil_current")},
        },
        "val": {
            "201": SIGNALS,
            "202": SIGNALS - {("summary", "ip")},
            "203": SIGNALS - {("pf_active", "coil_current")},
        },
        "test": {
            "301": SIGNALS,
            "302": SIGNALS - {("summary", "ip")},
            "303": SIGNALS - {("pf_active", "coil_current")},
        },
    }
    for shots in layouts.values():
        for shot, signals in shots.items():
            _write_v3_store(source / f"{shot}.zarr", signals)

    split_path = tmp_path / "splits.yaml"
    SplitManifest(
        name="tiny",
        source="test fixture",
        source_hash="fixture",
        splits={split: sorted(shots) for split, shots in layouts.items()},
    ).save(split_path)
    monkeypatch.setattr(build_dev_subset, "_split_manifest_path", lambda: split_path)
    monkeypatch.setattr(build_dev_subset, "_task_signals", lambda tasks: SIGNALS)
    return source, layouts


def _build(source: Path, dest: Path, shots_per_split: int = 2):
    return build_dev_subset.build_subset(
        source=str(source),
        dest=dest,
        tasks=["group_2", "group_3"],
        shots_per_split=shots_per_split,
        seed=17,
    )


def test_selection_deterministic_across_runs(tiny_store, tmp_path: Path) -> None:
    source, _ = tiny_store

    first = _build(source, tmp_path / "dest-one")
    second = _build(source, tmp_path / "dest-two")

    assert first["selection"] == second["selection"]


def test_split_membership_preserved(tiny_store, tmp_path: Path) -> None:
    source, layouts = tiny_store
    dest = tmp_path / "dest"

    manifest = _build(source, dest)

    for split, selected in manifest["selection"].items():
        assert set(selected) <= set(layouts[split])
        assert {path.stem for path in (dest / split).glob("*.zarr")} == set(
            selected
        )


def test_modality_and_missingness_coverage_enforced(
    tiny_store, tmp_path: Path
) -> None:
    source, _ = tiny_store
    dest = tmp_path / "dest"

    manifest = _build(source, dest, shots_per_split=1)

    for split, selected in manifest["selection"].items():
        assert len(selected) == 3
        assert manifest["expanded"][split] == {"requested": 1, "selected": 3}
        observed_patterns = {
            frozenset(SIGNALS - build_dev_subset._signals_in_shot(source, shot))
            for shot in ("101", "102", "103")
        }
        selected_patterns = {
            frozenset(SIGNALS - build_dev_subset._signals_in_shot(source, shot))
            for shot in selected
        }
        if split != "train":
            offset = {"val": 100, "test": 200}[split]
            observed_patterns = {
                frozenset(
                    SIGNALS
                    - build_dev_subset._signals_in_shot(
                        source, str(int(shot) + offset)
                    )
                )
                for shot in ("101", "102", "103")
            }
        assert selected_patterns == observed_patterns
        copied_modalities = {
            source_name
            for shot in selected
            for source_name, _ in build_dev_subset._signals_in_shot(
                dest / split, shot
            )
        }
        assert copied_modalities == {"magnetics", "summary", "pf_active"}
        assert not any(
            path.name == "not_for_tasks"
            for path in (dest / split).rglob("not_for_tasks")
        )


def test_manifest_written_with_hashes(tiny_store, tmp_path: Path) -> None:
    source, _ = tiny_store
    dest = tmp_path / "dest"

    returned = _build(source, dest)
    manifest_path = dest / "_manifest" / "dev_subset.yaml"
    written = yaml.safe_load(manifest_path.read_text())

    assert written == returned
    assert written["source"] == str(source)
    assert written["seed"] == 17
    assert written["tasks"] == ["group_2", "group_3"]
    assert written["shots_per_split"] == 2
    assert written["file_manifest"]["checksum"] == "sha256"
    assert written["file_manifest"]["files"]
    for relpath, entry in written["file_manifest"]["files"].items():
        assert entry["sha256"] == hashlib.sha256(
            (dest / relpath).read_bytes()
        ).hexdigest()
