"""Unit tests for scripts/acquire_tokamark.py, the resumable TokaMark
dataset downloader (Task 1.4).

``scripts/`` is not an installed package, so the module under test is
loaded directly from its file path via ``importlib``. Every test drives the
sync through ``--source-url file://...`` fixtures under ``tmp_path`` --
fully offline, no network. The committed ``manifests/datasets/`` summary
write target is redirected to a per-test ``tmp_path`` location (via
monkeypatching ``_dataset_summary_path``) so running the suite never
touches the real repo-tracked ``manifests/datasets/tokamark_v1.yaml``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA git repository")


def _load_acquire_tokamark():
    script_path = _repo_root() / "scripts" / "acquire_tokamark.py"
    spec = importlib.util.spec_from_file_location("acquire_tokamark", script_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the dataclass decorator looks itself up via
    # sys.modules[cls.__module__], which only exists once registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acquire_tokamark = _load_acquire_tokamark()


@pytest.fixture(autouse=True)
def _redirect_dataset_summary(monkeypatch, tmp_path) -> Path:
    """Never let a test write the real committed dataset summary manifest."""
    summary_path = tmp_path / "_repo_manifests" / "datasets" / "tokamark_v1.yaml"
    monkeypatch.setattr(
        acquire_tokamark, "_dataset_summary_path", lambda: summary_path
    )
    return summary_path


def _make_source(tmp_path: Path, files: dict[str, bytes]) -> Path:
    source = tmp_path / "source"
    for relpath, content in files.items():
        path = source / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source


def _source_url(source: Path) -> str:
    return f"file://{source}"


def test_destination_required(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        acquire_tokamark.main(["--source-url", "file:///nonexistent"])

    assert exc_info.value.code == 2
    assert "--dest" in capsys.readouterr().err


def test_refuses_repo_internal_or_home_destination(monkeypatch, tmp_path, capsys) -> None:
    source = _make_source(tmp_path, {"a.txt": b"hello"})

    repo_root = _repo_root()
    repo_internal_dest = repo_root / "tmp-acquire-tokamark-test-should-not-exist"
    exit_code = acquire_tokamark.main(
        [
            "--dest",
            str(repo_internal_dest),
            "--source-url",
            _source_url(source),
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "repository" in err.lower()
    assert not repo_internal_dest.exists()

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    home_dest = fake_home / "tokamark-data"
    exit_code = acquire_tokamark.main(
        ["--dest", str(home_dest), "--source-url", _source_url(source)]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "home" in err.lower()
    assert not home_dest.exists()


def test_dry_run_prints_sizes_and_writes_nothing(tmp_path, capsys) -> None:
    source = _make_source(
        tmp_path, {"a.txt": b"x" * 100, "group2/b.txt": b"y" * 250}
    )
    dest = tmp_path / "dest"

    exit_code = acquire_tokamark.main(
        ["--dest", str(dest), "--source-url", _source_url(source), "--dry-run"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "a.txt" in out
    assert "group2/b.txt" in out
    assert "350" in out  # total remote bytes (100 + 250)
    assert not dest.exists()


def test_resume_skips_complete_files_and_fetches_partial(tmp_path) -> None:
    source = _make_source(
        tmp_path, {"complete.txt": b"same-content", "missing.txt": b"new-content"}
    )
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "complete.txt").write_bytes(b"same-content")
    before_mtime = (dest / "complete.txt").stat().st_mtime_ns

    exit_code = acquire_tokamark.main(
        ["--dest", str(dest), "--source-url", _source_url(source)]
    )

    assert exit_code == 0
    # Untouched: size already matched, so it must not have been rewritten.
    assert (dest / "complete.txt").stat().st_mtime_ns == before_mtime
    assert (dest / "complete.txt").read_bytes() == b"same-content"
    # Fetched: did not exist locally before this run.
    assert (dest / "missing.txt").read_bytes() == b"new-content"

    manifest = json.loads((dest / "_manifest" / "files.json").read_text())
    assert manifest["complete.txt"]["size"] == len(b"same-content")
    assert manifest["missing.txt"]["size"] == len(b"new-content")


def test_mismatched_file_untouched_without_overwrite(tmp_path) -> None:
    source = _make_source(tmp_path, {"a.txt": b"remote-content-longer"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.txt").write_bytes(b"stale")  # different size -> mismatch

    exit_code = acquire_tokamark.main(
        ["--dest", str(dest), "--source-url", _source_url(source)]
    )

    assert exit_code == 0
    assert (dest / "a.txt").read_bytes() == b"stale"  # left untouched
    manifest_path = dest / "_manifest" / "files.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert "a.txt" not in manifest

    # Rerunning with --overwrite does replace the mismatched file.
    exit_code = acquire_tokamark.main(
        [
            "--dest",
            str(dest),
            "--source-url",
            _source_url(source),
            "--overwrite",
        ]
    )
    assert exit_code == 0
    assert (dest / "a.txt").read_bytes() == b"remote-content-longer"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["a.txt"]["size"] == len(b"remote-content-longer")


def test_manifest_and_summary_written(tmp_path, _redirect_dataset_summary) -> None:
    files = {"a.txt": b"aaaa", "group2/b.txt": b"bbbbbbbb"}
    source = _make_source(tmp_path, files)
    dest = tmp_path / "dest"

    exit_code = acquire_tokamark.main(
        [
            "--dest",
            str(dest),
            "--source-url",
            _source_url(source),
            "--checksum",
            "sha256",
        ]
    )

    assert exit_code == 0

    manifest = json.loads((dest / "_manifest" / "files.json").read_text())
    assert manifest["a.txt"]["size"] == 4
    assert manifest["a.txt"]["sha256"] == hashlib.sha256(b"aaaa").hexdigest()
    assert manifest["group2/b.txt"]["size"] == 8
    assert manifest["group2/b.txt"]["sha256"] == hashlib.sha256(b"bbbbbbbb").hexdigest()

    summary_path = _redirect_dataset_summary
    assert summary_path.exists()
    summary = acquire_tokamark.read_manifest(summary_path)
    assert summary["source_url"] == _source_url(source)
    assert summary["file_count"] == 2
    assert summary["byte_total"] == 12
    assert summary["checksum"] == "sha256"
    assert "retrieved" in summary
