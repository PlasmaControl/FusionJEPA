"""Tests for dataset and file manifest utilities."""

from fusion_jepa.utils.manifests import (
    file_manifest,
    manifest_hash,
    read_manifest,
    verify_manifest,
    write_manifest,
)


def test_round_trip_preserves_payload(tmp_path) -> None:
    payload = {
        "patterns": ["**/*.py"],
        "files": {"src/example.py": {"size": 12, "sha256": "abc"}},
        "checksum": "sha256",
    }
    path = tmp_path / "manifest.yaml"

    write_manifest(payload, path)

    assert read_manifest(path) == payload
    assert path.read_text().splitlines()[0] == "checksum: sha256"


def test_manifest_hash_invariant_to_key_order() -> None:
    first = {"checksum": "sha256", "files": {"b": 2, "a": 1}}
    second = {"files": {"a": 1, "b": 2}, "checksum": "sha256"}

    assert manifest_hash(first) == manifest_hash(second)


def test_verify_detects_modified_and_missing_files(tmp_path) -> None:
    (tmp_path / "kept.txt").write_text("before")
    (tmp_path / "missing.txt").write_text("remove me")
    manifest = file_manifest(tmp_path, ["**/*.txt"])

    (tmp_path / "kept.txt").write_text("after!")
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "extra.txt").write_text("new")
    report = verify_manifest(tmp_path, manifest)

    assert report.changed == ["kept.txt"]
    assert report.missing == ["missing.txt"]
    assert report.extra == ["extra.txt"]
    assert not report.ok


def test_size_only_mode_for_large_trees(tmp_path) -> None:
    (tmp_path / "changed.bin").write_bytes(b"abcd")
    (tmp_path / "same_size.bin").write_bytes(b"wxyz")
    manifest = file_manifest(tmp_path, ["**/*.bin"], checksum="size_only")

    assert all("sha256" not in entry for entry in manifest["files"].values())

    (tmp_path / "changed.bin").write_bytes(b"longer")
    (tmp_path / "same_size.bin").write_bytes(b"1234")
    report = verify_manifest(tmp_path, manifest)

    assert report.changed == ["changed.bin"]
    assert report.ok is False
