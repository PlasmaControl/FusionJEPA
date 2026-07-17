#!/usr/bin/env python3
"""Build a deterministic, coverage-preserving TokaMark development subset.

The documented destination is
``/lustre/orion/fus187/proj-shared/mast/tokamark/dev_subset_v1``.  ``--dest``
is nevertheless required so an operator must explicitly confirm the target.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import fsspec

from fusion_jepa.data.registry import load_registry
from fusion_jepa.data.splits import SplitManifest
from fusion_jepa.utils.manifests import file_manifest, read_manifest, write_manifest
from fusion_jepa.utils.reproducibility import derive_seed

_DEFAULT_DEST = (
    "/lustre/orion/fus187/proj-shared/mast/tokamark/dev_subset_v1"
)
_METADATA_FILES = {".zgroup", ".zattrs", ".zarray", "zarr.json"}


class BuildSubsetError(Exception):
    """Raised for actionable development-subset build errors."""


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA git repository")


def _split_manifest_path() -> Path:
    return _repo_root() / "manifests" / "splits" / "tokamark_official.yaml"


def _validate_dest(raw: Path) -> Path:
    """Refuse repository-internal and home-directory destinations."""
    dest = raw.expanduser().resolve()
    repo_root = _repo_root()
    if dest == repo_root or dest.is_relative_to(repo_root):
        raise BuildSubsetError(
            f"--dest {dest} is inside the Fusion-JEPA repository working "
            f"tree ({repo_root}); choose a scratch or proj-shared location"
        )
    home = Path.home().resolve()
    if dest == home or dest.is_relative_to(home):
        raise BuildSubsetError(
            f"--dest {dest} is under $HOME ({home}), which is quota-limited; "
            "choose a scratch or proj-shared location instead"
        )
    return dest


def _task_signals(tasks: list[str]) -> set[tuple[str, str]]:
    """Resolve requested task groups to registry-backed upstream signals."""
    from tokamark.tasks import GROUP_TASKS, get_task_config

    registry = load_registry(_repo_root() / "signal_registry" / "mast.yaml")
    registered = {spec.source_name for spec in registry.values()}
    signals: set[tuple[str, str]] = set()
    for task_group in tasks:
        try:
            group_number = int(task_group.removeprefix("group_"))
            task_ids = GROUP_TASKS[group_number]
        except (KeyError, ValueError) as exc:
            raise BuildSubsetError(f"unknown task group: {task_group!r}") from exc
        for task_id in task_ids:
            config = get_task_config(task_id)
            groups = config["sources_and_signals"]
            for pairs in groups.values():
                signals.update((str(source), str(signal)) for source, signal in pairs)

    unregistered = sorted(
        f"{source}-{signal}"
        for source, signal in signals
        if f"{source}-{signal}" not in registered
    )
    if unregistered:
        raise BuildSubsetError(
            "task signals missing from signal_registry/mast.yaml: "
            + ", ".join(unregistered)
        )
    return signals


def _source_fs(source: str):
    options: dict[str, Any] = {}
    if source.startswith("s3://"):
        upstream = read_manifest(_repo_root() / "manifests" / "upstream.yaml")
        options = {
            "anon": not bool(
                os.environ.get("AWS_ACCESS_KEY_ID")
                or os.environ.get("AWS_SECRET_ACCESS_KEY")
            ),
            "client_kwargs": {
                "endpoint_url": upstream["tokamark_dataset"]["s3_endpoint"]
            },
        }
    return fsspec.core.url_to_fs(source, **options)


def _candidate_shots(fs, root: str) -> list[str]:
    shots = []
    for entry in fs.ls(root, detail=True):
        name = str(entry["name"]).rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            shots.append(name.removesuffix(".zarr"))
    return sorted(shots)


def _shot_files(fs, root: str, shot: str) -> list[str]:
    shot_root = posixpath.join(root.rstrip("/"), f"{shot}.zarr")
    return sorted(fs.find(shot_root))


def _present_signals(
    fs, root: str, shot: str, relevant: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    shot_root = posixpath.join(root.rstrip("/"), f"{shot}.zarr") + "/"
    relpaths = [path[len(shot_root) :] for path in _shot_files(fs, root, shot)]
    return {
        (source, signal)
        for source, signal in relevant
        if any(
            relpath == f"{source}/{signal}"
            or relpath.startswith(f"{source}/{signal}/")
            for relpath in relpaths
        )
    }


def _signals_in_shot(root: Path, shot: str) -> set[tuple[str, str]]:
    """Return source/signal pairs in a local shot store (test/introspection aid)."""
    shot_root = Path(root) / f"{shot}.zarr"
    if not shot_root.exists():
        return set()
    found = set()
    for source in shot_root.iterdir():
        if source.is_dir():
            found.update(
                (source.name, signal.name)
                for signal in source.iterdir()
                if signal.is_dir()
            )
    return found


def _select_split(
    candidates: list[str],
    present: dict[str, set[tuple[str, str]]],
    relevant: set[tuple[str, str]],
    requested: int,
    seed: int,
    split: str,
) -> list[str]:
    patterns: dict[frozenset[tuple[str, str]], list[str]] = {}
    for shot in candidates:
        pattern = frozenset(relevant - present[shot])
        patterns.setdefault(pattern, []).append(shot)

    rng = random.Random(derive_seed(seed, "dev_subset", split, *candidates))
    selected = []
    for pattern in sorted(patterns, key=lambda item: sorted(item)):
        representatives = patterns[pattern]
        selected.append(representatives[rng.randrange(len(representatives))])

    remaining = [shot for shot in candidates if shot not in selected]
    rng.shuffle(remaining)
    target = min(len(candidates), max(requested, len(selected)))
    selected.extend(remaining[: target - len(selected)])
    return sorted(selected)


def _is_needed(relpath: str, signals: set[tuple[str, str]]) -> bool:
    parts = relpath.split("/")
    sources = {source for source, _ in signals}
    if parts[-1] in _METADATA_FILES and len(parts) == 1:
        return True
    if (
        parts[-1] in _METADATA_FILES
        and len(parts) == 2
        and parts[0] in sources
    ):
        return True
    return any(
        relpath == f"{source}/{signal}"
        or relpath.startswith(f"{source}/{signal}/")
        for source, signal in signals
    )


def _copy_shot(fs, root: str, dest: Path, split: str, shot: str, signals) -> None:
    source_root = posixpath.join(root.rstrip("/"), f"{shot}.zarr")
    target_root = dest / split / f"{shot}.zarr"
    for source_path in _shot_files(fs, root, shot):
        relpath = source_path[len(source_root) :].lstrip("/")
        if not _is_needed(relpath, signals):
            continue
        target = target_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(source_path, "rb") as source_file, target.open("wb") as out:
            shutil.copyfileobj(source_file, out)


def build_subset(
    *,
    source: str,
    dest: Path,
    tasks: list[str],
    shots_per_split: int,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select and optionally copy a deterministic subset from ``source``."""
    if shots_per_split < 1:
        raise BuildSubsetError("--shots-per-split must be at least 1")
    dest_path = Path(dest)
    if not dry_run and dest_path.exists() and any(dest_path.iterdir()):
        raise BuildSubsetError(
            f"--dest {dest_path} already exists and is not empty; this builder "
            "only overwrites the files it selects, so a rerun with different "
            "--tasks/--seed/--source would leave stale shot stores and "
            "obsolete subtrees behind -- unreported by the new manifest, "
            "silently mixing unselected data into the subset. Point --dest at "
            "a fresh directory (an empty or nonexistent path)."
        )
    relevant = _task_signals(tasks)
    split_manifest = SplitManifest.load(_split_manifest_path())
    fs, root = _source_fs(source)
    available = set(_candidate_shots(fs, root))
    selection: dict[str, list[str]] = {}
    expanded: dict[str, dict[str, int]] = {}

    for split, assigned in split_manifest.splits.items():
        candidates = sorted(available.intersection(assigned))
        if not candidates:
            raise BuildSubsetError(f"no candidate shots found for split {split!r}")
        present = {
            shot: _present_signals(fs, root, shot, relevant) for shot in candidates
        }
        selected = _select_split(
            candidates, present, relevant, shots_per_split, seed, split
        )
        selection[split] = selected
        if len(selected) > shots_per_split:
            expanded[split] = {
                "requested": shots_per_split,
                "selected": len(selected),
            }

    manifest: dict[str, Any] = {
        "source": source,
        "seed": seed,
        "tasks": tasks,
        "shots_per_split": shots_per_split,
        "selection": selection,
        "expanded": expanded,
    }
    if dry_run:
        return manifest

    for split, shots in selection.items():
        for shot in shots:
            _copy_shot(fs, root, dest_path, split, shot, relevant)
    patterns = [
        f"{split}/{shot}.zarr/**/*"
        for split, shots in selection.items()
        for shot in shots
    ]
    manifest["file_manifest"] = file_manifest(dest_path, patterns, checksum="sha256")
    manifest_path = dest_path / "_manifest" / "dev_subset.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local root or S3 URL.")
    parser.add_argument(
        "--dest",
        required=True,
        help=f"Destination outside the repo and $HOME (documented: {_DEFAULT_DEST}).",
    )
    parser.add_argument("--tasks", default="group_2,group_3")
    parser.add_argument("--shots-per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dest = _validate_dest(Path(args.dest))
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        manifest = build_subset(
            source=args.source,
            dest=dest,
            tasks=tasks,
            shots_per_split=args.shots_per_split,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (BuildSubsetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for split, shots in manifest["selection"].items():
        suffix = ""
        if split in manifest["expanded"]:
            suffix = " (expanded for coverage)"
        print(f"{split}: {len(shots)} shots{suffix}: {', '.join(shots)}")
    if args.dry_run:
        print("dry run: no files written")
    else:
        print(f"manifest: {dest / '_manifest' / 'dev_subset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
