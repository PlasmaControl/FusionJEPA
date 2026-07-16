"""Unit tests for the run artifact contract."""

import json
import subprocess

from omegaconf import OmegaConf

from fusion_jepa.utils.run_artifacts import (
    compute_run_hash,
    create_run_dir,
    git_state,
    write_completion,
)


def test_run_hash_deterministic_across_calls():
    config = {"seed": 0, "model": {"width": 64}}

    first = compute_run_hash(config, "abc123", "manifest123")
    second = compute_run_hash(config, "abc123", "manifest123")

    assert first == second
    assert len(first) == 8
    assert all(character in "0123456789abcdef" for character in first)


def test_run_hash_changes_on_scientific_config_change():
    first = compute_run_hash({"seed": 0}, "abc123", "manifest123")
    second = compute_run_hash({"seed": 1}, "abc123", "manifest123")

    assert first != second


def test_run_hash_changes_on_code_revision_change():
    config = {"seed": 0}

    first = compute_run_hash(config, "abc123", "manifest123")
    second = compute_run_hash(config, "def456", "manifest123")

    assert first != second


def test_run_hash_ignores_output_path_and_cluster_block():
    first = {
        "seed": 0,
        "model": {"width": 64},
        "output_path": "/first",
        "runs_root": "/runs/first",
        "timestamp": "2026-01-01",
        "cluster": {"account": "first"},
    }
    second = {"seed": 0, "model": {"width": 64}}

    assert compute_run_hash(first, "abc", "manifest") == compute_run_hash(
        second, "abc", "manifest"
    )


def test_run_dir_contains_required_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo, check=True
    )
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True
    )
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean_state = git_state(repo)
    assert clean_state["dirty"] is False
    assert len(clean_state["commit"]) == 40

    cfg = OmegaConf.create(
        {
            "experiment": "mast_smoke",
            "seed": 0,
            "data": {"batch_size": 2},
            "cluster": {"runs_root": "/ignored"},
        }
    )
    context = create_run_dir(
        cfg, ["train", "seed=0"], repo / "runs", repo_root=repo
    )

    assert context.run_name.startswith("mast-smoke_seed0__")
    required = {
        "command.txt",
        "config.resolved.yaml",
        "environment.txt",
        "git.json",
        "metrics.jsonl",
    }
    assert required <= {path.name for path in context.run_dir.iterdir()}
    assert (context.run_dir / "command.txt").read_text() == "train seed=0\n"
    assert (context.run_dir / "metrics.jsonl").read_text() == ""
    git_json = json.loads((context.run_dir / "git.json").read_text())
    assert git_json["commit"] == head_sha
    assert git_json["dirty"] is False
    assert {
        "branch",
        "commit",
        "diff_sha256",
        "dirty",
    } <= set(git_json)


def test_completion_written_with_failure_reason(tmp_path):
    write_completion(
        tmp_path,
        status="failed",
        started_at="2026-07-16T12:00:00Z",
        warnings=["partial output"],
        failure_reason="out of memory",
        ended_at="2026-07-16T12:01:00Z",
    )

    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion == {
        "status": "failed",
        "started_at": "2026-07-16T12:00:00Z",
        "warnings": ["partial output"],
        "failure_reason": "out of memory",
        "ended_at": "2026-07-16T12:01:00Z",
    }
