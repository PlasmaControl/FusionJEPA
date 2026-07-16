"""Creation and completion of reproducible run artifact directories."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fusion_jepa.config import save_resolved, scientific_subset
from fusion_jepa.utils.manifests import manifest_hash, read_manifest

_NON_SCIENTIFIC_KEYS = {"cluster", "output_path", "runs_root", "timestamp"}


@dataclass
class RunContext:
    """Identifiers and filesystem location for a run."""

    run_dir: Path
    run_hash: str
    run_name: str


def _git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Return revision and working-tree state for a Git repository."""
    root = Path(repo_root)
    diff = _git(root, "diff", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def compute_run_hash(
    scientific_cfg: dict[str, Any],
    code_revision: Any,
    upstream_manifest_hash: Any,
) -> str:
    """Return the stable short hash identifying a scientific run."""
    filtered = {
        key: value
        for key, value in scientific_cfg.items()
        if key not in _NON_SCIENTIFIC_KEYS
    }
    payload = json.dumps(
        [filtered, code_revision, upstream_manifest_hash],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _config_value(cfg: Any, key: str) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key)
    return None


def make_run_name(cfg: Any, run_hash: str) -> str:
    """Return the human-readable run directory name."""
    experiment = _config_value(cfg, "experiment_name") or _config_value(
        cfg, "experiment"
    )
    if not experiment:
        raise ValueError("cfg must contain experiment_name or experiment")
    name = str(experiment).replace("_", "-")
    return f"{name}_seed{_config_value(cfg, 'seed')}__{run_hash}"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA Git repository")


def _environment_text() -> str:
    lines = [f"python=={sys.version.split()[0]}"]
    for package in ("numpy", "omegaconf", "torch"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        lines.append(f"{package}=={version}")
    return "\n".join(lines) + "\n"


def create_run_dir(
    cfg: Any,
    argv: Sequence[str],
    base: str | Path,
) -> RunContext:
    """Create a run directory and its initial provenance artifacts."""
    repo_root = _repo_root()
    revision = git_state(repo_root)
    upstream_path = repo_root / "manifests" / "upstream.yaml"
    upstream_hash = ""
    if upstream_path.exists():
        upstream_hash = manifest_hash(read_manifest(upstream_path))

    run_hash = compute_run_hash(scientific_subset(cfg), revision, upstream_hash)
    run_name = make_run_name(cfg, run_hash)
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    save_resolved(cfg, run_dir / "config.resolved.yaml")
    (run_dir / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text(
        _environment_text(), encoding="utf-8"
    )
    with (run_dir / "git.json").open("w", encoding="utf-8") as file:
        json.dump(revision, file, sort_keys=True)
    (run_dir / "metrics.jsonl").touch()
    if upstream_path.exists():
        shutil.copy2(upstream_path, run_dir / "upstream.yaml")

    return RunContext(run_dir=run_dir, run_hash=run_hash, run_name=run_name)


def write_completion(
    run_dir: str | Path,
    *,
    status: str,
    started_at: Any,
    warnings: Any,
    failure_reason: Any,
    **extra: Any,
) -> None:
    """Write the terminal run status as strict JSON."""
    completion = {
        "status": status,
        "started_at": started_at,
        "warnings": warnings,
        "failure_reason": failure_reason,
        **extra,
    }
    with (Path(run_dir) / "completion.json").open("w", encoding="utf-8") as file:
        json.dump(completion, file, allow_nan=False, sort_keys=True)
