"""Unit tests for the M0 command-line shells."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from fusion_jepa.cli import evaluate, inspect_data, train


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USER", "cli-test-user")
    monkeypatch.setenv("SBATCH_ACCOUNT", "cli-test-account")


def _arguments(tmp_path, *extra: str) -> list[str]:
    return [
        "experiment=mast_smoke",
        f"cluster.runs_root={tmp_path / 'runs'}",
        *extra,
    ]


def test_inspect_data_dry_run_exit_zero(tmp_path) -> None:
    assert inspect_data.main(_arguments(tmp_path, "--dry-run")) == 0


def test_train_dry_run_creates_no_run_dir(tmp_path) -> None:
    runs_root = tmp_path / "runs"

    assert train.main(_arguments(tmp_path, "--dry-run")) == 0
    assert not runs_root.exists()


def test_train_without_impl_writes_failed_completion_and_exits_nonzero(
    tmp_path,
) -> None:
    runs_root = tmp_path / "runs"

    assert train.main(_arguments(tmp_path)) == 2
    completion_paths = list(runs_root.glob("*/completion.json"))
    assert len(completion_paths) == 1
    completion = json.loads(completion_paths[0].read_text(encoding="utf-8"))
    assert completion["status"] == "failed"
    assert completion["failure_reason"] == "not_implemented"
    assert completion["warnings"] == []


def test_evaluate_refuses_test_split_without_flag(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"

    assert evaluate.main(_arguments(tmp_path, "split=test")) == 2
    assert "--allow-test" in capsys.readouterr().err
    assert not runs_root.exists()


def test_evaluate_allows_test_split_with_flag(tmp_path) -> None:
    runs_root = tmp_path / "runs"

    assert evaluate.main(_arguments(tmp_path, "split=test", "--allow-test")) == 2
    completion_paths = list(runs_root.glob("*/completion.json"))
    assert len(completion_paths) == 1
    completion = json.loads(completion_paths[0].read_text(encoding="utf-8"))
    assert completion["status"] == "failed"
    assert completion["failure_reason"] == "not_implemented"


def test_unknown_experiment_errors_actionably(tmp_path, capsys) -> None:
    args = [
        "experiment=does_not_exist",
        f"cluster.runs_root={tmp_path / 'runs'}",
        "--dry-run",
    ]

    assert train.main(args) == 2
    error = capsys.readouterr().err
    assert "Unknown experiment" in error
    assert "does_not_exist" in error


def test_module_invocation_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fusion_jepa.cli.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
