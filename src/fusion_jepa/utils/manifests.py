"""Dataset and file manifest utilities."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VerificationReport:
    """Differences between a file manifest and files on disk."""

    missing: list[str]
    changed: list[str]
    extra: list[str]

    @property
    def ok(self) -> bool:
        """Return whether the manifest matches the files on disk."""
        return not (self.missing or self.changed or self.extra)


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a manifest as sorted-key YAML."""
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=True)


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a YAML manifest."""
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_files(root: Path, patterns: list[str]) -> dict[str, Path]:
    files = {
        path.relative_to(root).as_posix(): path
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    return dict(sorted(files.items()))


def file_manifest(
    root: str | Path,
    patterns: list[str],
    checksum: str = "sha256",
) -> dict[str, Any]:
    """Build a manifest for files under a root matching glob patterns."""
    if checksum not in {"sha256", "size_only"}:
        raise ValueError("checksum must be 'sha256' or 'size_only'")

    root = Path(root)
    files: dict[str, dict[str, int | str]] = {}
    for relative_path, path in _matching_files(root, patterns).items():
        entry: dict[str, int | str] = {"size": path.stat().st_size}
        if checksum == "sha256":
            entry["sha256"] = _sha256(path)
        files[relative_path] = entry

    return {"checksum": checksum, "patterns": patterns, "files": files}


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash a manifest using canonical key-sorted serialization."""
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_manifest(
    root: str | Path,
    manifest: dict[str, Any],
) -> VerificationReport:
    """Compare files on disk with a stored manifest."""
    root = Path(root)
    expected = manifest["files"]
    actual = _matching_files(root, manifest["patterns"])
    expected_paths = set(expected)
    actual_paths = set(actual)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    changed = []

    for relative_path in sorted(expected_paths & actual_paths):
        path = actual[relative_path]
        entry = expected[relative_path]
        differs = path.stat().st_size != entry["size"]
        if manifest["checksum"] == "sha256" and not differs:
            differs = _sha256(path) != entry["sha256"]
        if differs:
            changed.append(relative_path)

    return VerificationReport(missing=missing, changed=changed, extra=extra)
