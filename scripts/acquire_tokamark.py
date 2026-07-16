#!/usr/bin/env python3
"""Resumable TokaMark dataset downloader (Task 1.4).

Pulls the ~2 TB TokaMark dataset (see ``manifests/upstream.yaml`` ->
``tokamark_dataset``) from its upstream S3-compatible object store (or, as a
fallback, the gated Hugging Face mirror) into a local destination directory,
via `fsspec <https://filesystem-spec.readthedocs.io/>`_ so any backend --
including a plain ``file://`` path -- can be used as the source. That last
point is what makes the six unit tests in
``tests/unit/test_acquire_tokamark.py`` fully offline: they point
``--source-url`` at a ``tmp_path`` fixture tree instead of the real bucket.

Design (see the Task 1.4 brief):

* ``--dest`` is required and refuses destinations inside this repository's
  working tree or under ``$HOME`` (both are quota- or version-control-
  inappropriate for a multi-terabyte dataset).
* Resume semantics never delete anything: an existing local file whose size
  matches the remote size is left alone (skip); a size mismatch is left
  untouched and reported unless ``--overwrite`` is given, in which case it is
  refetched.
* ``<dest>/_manifest/files.json`` accumulates ``relpath -> {size, [sha256]}``
  for every file this or a prior invocation fetched or verified -- runs with
  different ``--include`` globs (e.g. staging modality groups one at a time)
  merge into the same manifest rather than clobbering it.
* ``manifests/datasets/tokamark_v1.yaml`` (committed, at the repository root
  discovered from this file's location) records a one-shot summary of the
  most recent sync: source URL, file count, byte total, retrieval
  timestamp, and checksum mode. Written via
  ``fusion_jepa.utils.manifests.write_manifest``.

Usage::

    # Preview what would be pulled, no writes.
    python scripts/acquire_tokamark.py --dest /path/to/dest --dry-run

    # Real pull (resumable; safe to re-run/interrupt).
    python scripts/acquire_tokamark.py --dest /lustre/orion/fus187/proj-shared/mast/tokamark/v1

    # Fall back to the gated Hugging Face mirror if S3 anon access fails.
    HF_TOKEN=... python scripts/acquire_tokamark.py --dest ... --source hf
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec

from fusion_jepa.utils.manifests import read_manifest, write_manifest

_SOURCE_CHOICES = ("s3", "hf")
_CHECKSUM_CHOICES = ("none", "sha256")


class AcquireError(Exception):
    """Raised for actionable, user-facing acquisition errors."""


@dataclass(frozen=True)
class RemoteFile:
    """One file in the remote listing."""

    relpath: str
    size: int


@dataclass
class SyncResult:
    """Outcome of a single ``_sync_files`` pass."""

    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


def _repo_root() -> Path:
    """Walk up from this file to find the repo root (holds ``.git``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA git repository")


def _dataset_summary_path() -> Path:
    """Where the committed per-dataset summary manifest is written."""
    return _repo_root() / "manifests" / "datasets" / "tokamark_v1.yaml"


def _validate_dest(raw: Path) -> Path:
    """Resolve ``--dest`` and refuse repo-internal or ``$HOME`` locations."""
    dest = raw.expanduser().resolve()

    repo_root = _repo_root()
    if dest == repo_root or dest.is_relative_to(repo_root):
        raise AcquireError(
            f"--dest {dest} is inside the Fusion-JEPA repository working "
            f"tree ({repo_root}); a multi-terabyte dataset does not belong "
            "in version control -- choose a scratch or proj-shared "
            "location, e.g. /lustre/orion/fus187/proj-shared/mast/tokamark/v1"
        )

    home = Path.home().resolve()
    if dest == home or dest.is_relative_to(home):
        raise AcquireError(
            f"--dest {dest} is under $HOME ({home}), which is quota-"
            "limited; choose a scratch or proj-shared location instead"
        )

    return dest


def _resolve_source_url(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Build the fsspec source URL and its storage options."""
    if args.source_url:
        return args.source_url, {}

    upstream = read_manifest(_repo_root() / "manifests" / "upstream.yaml")
    dataset = upstream["tokamark_dataset"]

    if args.source == "s3":
        anon = not (
            os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        storage_options = {
            "anon": anon,
            "client_kwargs": {"endpoint_url": dataset["s3_endpoint"]},
        }
        return dataset["s3_path"], storage_options

    # args.source == "hf"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise AcquireError(
            "source=hf requires an HF_TOKEN environment variable (the "
            f"{dataset['hf_repo_id']} dataset is gated)"
        )
    url = f"hf://datasets/{dataset['hf_repo_id']}@{dataset['revision']}"
    return url, {"token": token}


def _list_remote_files(fs: fsspec.AbstractFileSystem, root: str) -> list[RemoteFile]:
    """Recursively list files under ``root`` on filesystem ``fs``."""
    root_norm = root.rstrip("/")
    files = []
    for path, info in fs.find(root_norm, detail=True).items():
        if info.get("type") == "directory":
            continue
        relpath = path[len(root_norm) :].lstrip("/")
        if not relpath:
            continue
        files.append(RemoteFile(relpath=relpath, size=int(info["size"])))
    files.sort(key=lambda file: file.relpath)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_files(
    fs: fsspec.AbstractFileSystem,
    root: str,
    remote_files: list[RemoteFile],
    dest: Path,
    *,
    resume: bool,
    overwrite: bool,
    checksum: str,
    workers: int,
) -> SyncResult:
    """Fetch/verify remote files into ``dest``. Never deletes anything."""
    result = SyncResult()
    root_norm = root.rstrip("/")

    def handle(remote: RemoteFile) -> None:
        local_path = dest / remote.relpath
        exists = local_path.exists()
        needs_fetch = not exists

        if exists:
            local_size = local_path.stat().st_size
            if local_size != remote.size:
                if overwrite:
                    needs_fetch = True
                else:
                    result.mismatched.append(remote.relpath)
                    return
            elif not resume:
                needs_fetch = True

        if needs_fetch:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(f"{root_norm}/{remote.relpath}", str(local_path))
            result.fetched.append(remote.relpath)
        else:
            result.skipped.append(remote.relpath)

        entry: dict[str, Any] = {"size": local_path.stat().st_size}
        if checksum == "sha256":
            entry["sha256"] = _sha256(local_path)
        result.entries[remote.relpath] = entry

    if workers <= 1:
        for remote in remote_files:
            handle(remote)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(handle, remote) for remote in remote_files]:
                future.result()

    return result


def _write_files_manifest(dest: Path, entries: dict[str, dict[str, Any]]) -> Path:
    """Merge ``entries`` into ``<dest>/_manifest/files.json``."""
    manifest_dir = dest / "_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "files.json"

    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.update(entries)

    manifest_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _write_dataset_summary(source_url: str, result: SyncResult, checksum: str) -> Path:
    """Write the committed ``manifests/datasets/tokamark_v1.yaml`` summary."""
    summary = {
        "source_url": source_url,
        "file_count": len(result.entries),
        "byte_total": sum(entry["size"] for entry in result.entries.values()),
        "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checksum": checksum,
    }
    path = _dataset_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(summary, path)
    return path


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _print_dry_run(remote_files: list[RemoteFile], dest: Path, byte_total: int) -> None:
    print("Fusion-JEPA TokaMark acquisition -- dry run (writes nothing)")
    print(f"remote files: {len(remote_files)}")
    for remote in remote_files:
        print(f"  {remote.relpath}  ({remote.size} bytes)")
    print(f"total remote bytes: {byte_total}")

    usage = shutil.disk_usage(_existing_ancestor(dest))
    print(
        f"disk usage at {dest}: {usage.free} bytes free of {usage.total} "
        f"total (need {byte_total})"
    )
    if usage.free < byte_total:
        print(
            "warning: free space is less than the remote dataset size",
            file=sys.stderr,
        )


def _print_summary(result: SyncResult) -> None:
    print(f"fetched: {len(result.fetched)}")
    print(f"skipped (already present): {len(result.skipped)}")
    print(f"mismatched, left untouched: {len(result.mismatched)}")
    if result.mismatched:
        print(
            "concern: the following files differ in size from the remote "
            "source and were left untouched (rerun with --overwrite to "
            "replace them):",
            file=sys.stderr,
        )
        for relpath in sorted(result.mismatched):
            print(f"  {relpath}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acquire_tokamark.py",
        description="Resumable downloader for the TokaMark dataset.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Local destination directory (must be outside the repo and $HOME).",
    )
    parser.add_argument(
        "--source",
        choices=_SOURCE_CHOICES,
        default="s3",
        help="Remote backend to use when --source-url is not given (default: s3).",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Explicit fsspec URL to pull from, e.g. file:///path for offline "
        "testing. Overrides --source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the remote dataset and report sizes/disk usage without "
        "writing anything.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files whose local size already matches the remote size "
        "(default: on). --no-resume refetches everything.",
    )
    parser.add_argument(
        "--checksum",
        choices=_CHECKSUM_CHOICES,
        default="none",
        help="Verification level recorded in the manifest (default: none).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch files whose local size mismatches the remote size "
        "(required to replace mismatches; without it they are left "
        "untouched and reported).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="GLOB",
        help="Only fetch remote relpaths matching this glob (repeatable; "
        "useful for staging modality groups).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent file fetches (default: 4).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the TokaMark acquisition CLI."""
    args = _build_parser().parse_args(argv)

    try:
        dest = _validate_dest(Path(args.dest))
    except AcquireError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        source_url, storage_options = _resolve_source_url(args)
    except AcquireError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        fs, root = fsspec.core.url_to_fs(source_url, **storage_options)
        remote_files = _list_remote_files(fs, root)
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable CLI error
        print(f"error: failed to list remote source {source_url}: {exc}", file=sys.stderr)
        return 1

    if args.include:
        remote_files = [
            remote
            for remote in remote_files
            if any(fnmatch.fnmatch(remote.relpath, pattern) for pattern in args.include)
        ]

    if args.dry_run:
        _print_dry_run(remote_files, dest, sum(f.size for f in remote_files))
        return 0

    result = _sync_files(
        fs,
        root,
        remote_files,
        dest,
        resume=args.resume,
        overwrite=args.overwrite,
        checksum=args.checksum,
        workers=max(1, args.workers),
    )

    _write_files_manifest(dest, result.entries)
    _write_dataset_summary(source_url, result, args.checksum)
    _print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
