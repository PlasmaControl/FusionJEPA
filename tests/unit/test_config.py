"""Tests for the Task 0.2 config system (`fusion_jepa.config`).

These exercise real committed YAML under `configs/` and real OmegaConf
merges end to end -- no mocking of OmegaConf itself. `resolve_config` is the
foundation every later Fusion-JEPA task builds on, so these tests pin down
its merge order, error messages, and the scientific-subset / snapshot
round-trip behavior that a future run-hash will depend on.
"""

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fusion_jepa.config import (
    ClusterConfig,
    ConfigError,
    ExperimentConfig,
    RunSettings,
    TokamarkDataConfig,
    resolve_config,
    save_resolved,
    scientific_subset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


def test_experiment_config_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed mast_smoke experiment (cluster=local, data=tokamark)
    resolves into a fully-populated ExperimentConfig with values pulled from
    all three yaml layers plus the structured defaults."""
    cfg = resolve_config(["experiment=mast_smoke"], config_root=CONFIGS_ROOT)

    # Structured RunSettings defaults, flattened onto ExperimentConfig.
    assert cfg.seed == 0
    assert cfg.dry_run is False
    assert cfg.allow_test_split is False

    # configs/data/tokamark.yaml
    assert cfg.data.task_id == "task_2-3"
    assert cfg.data.batch_size == 32
    assert cfg.data.remote is False
    assert cfg.data.limit_shots is None

    # configs/cluster/local.yaml
    assert cfg.cluster.tokamark_root == "s3://mast/tokamark/v1"
    assert cfg.cluster.tokamark_storage_options.anon is True

    # account is never a committed literal -- it comes from the structured
    # default's env interpolation, which falls back to '' when unset.
    monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
    assert cfg.cluster.account == ""

    # Sanity-check the *other* committed cluster profile too (frontier),
    # since nothing else exercises its actual field values.
    monkeypatch.setenv("USER", "smoketest")
    frontier_cfg = OmegaConf.merge(
        OmegaConf.structured(ClusterConfig),
        OmegaConf.load(CONFIGS_ROOT / "cluster" / "frontier.yaml"),
    )
    resolved_frontier = OmegaConf.to_container(frontier_cfg, resolve=True)
    assert (
        resolved_frontier["tokamark_root"]
        == "/lustre/orion/fus187/proj-shared/mast/tokamark/v1"
    )
    assert (
        resolved_frontier["runs_root"]
        == "/lustre/orion/fus187/proj-shared/smoketest/fusion_jepa_runs"
    )


def test_dotlist_override_applies() -> None:
    """CLI dotlist overrides apply last, on top of every yaml layer, and
    reach both the flattened RunSettings fields and nested data fields."""
    cfg = resolve_config(
        ["experiment=mast_smoke", "seed=7", "data.batch_size=4"],
        config_root=CONFIGS_ROOT,
    )
    assert cfg.seed == 7
    assert cfg.data.batch_size == 4
    # Untouched keys keep their yaml-layer values.
    assert cfg.data.task_id == "task_2-3"


def test_missing_required_key_raises_actionable_error() -> None:
    """An unknown experiment name, and a CLI dotlist key absent from the
    schema, both raise ConfigError naming the offending key/file."""
    with pytest.raises(ConfigError, match="does_not_exist"):
        resolve_config(["experiment=does_not_exist"], config_root=CONFIGS_ROOT)

    with pytest.raises(ConfigError, match="not_a_real_key"):
        resolve_config(
            ["experiment=mast_smoke", "not_a_real_key=1"], config_root=CONFIGS_ROOT
        )


def test_scientific_subset_excludes_cluster_paths() -> None:
    """scientific_subset drops the whole cluster block (every path + the
    account string) but keeps data/run settings that affect results."""
    cfg = resolve_config(["experiment=mast_smoke"], config_root=CONFIGS_ROOT)
    subset = scientific_subset(cfg)

    assert "cluster" not in subset
    assert subset["data"]["task_id"] == "task_2-3"
    assert subset["seed"] == 0

    serialized = repr(subset)
    for leaked in ("tokamark_root", "runs_root", "data_root", "s3://", "SBATCH_ACCOUNT"):
        assert leaked not in serialized

    # Deterministic: same cfg always produces an equal (sorted) subset.
    assert scientific_subset(cfg) == subset


def test_committed_yaml_contains_no_account_string() -> None:
    """No committed YAML under configs/ may hard-code a real SLURM account.

    `account:` lines must either resolve from the environment (contain
    `${oc.env:SBATCH_ACCOUNT`) or be empty; no `#SBATCH -A <value>` or
    `--account=<value>` literal may appear. Data paths that happen to
    contain the project id (e.g. `/lustre/orion/fus187/...`) are allowed --
    only the *account* setting itself is restricted.
    """
    account_line_re = re.compile(r"^\s*account\s*:\s*(.*)$")
    sbatch_account_re = re.compile(r"#\s*SBATCH\s+-A\s+(\S+)")
    cli_account_re = re.compile(r"--account=(\S+)")

    yaml_files = sorted(CONFIGS_ROOT.rglob("*.yaml"))
    assert yaml_files, "expected committed config yaml files under configs/"

    for yaml_file in yaml_files:
        for lineno, line in enumerate(yaml_file.read_text().splitlines(), start=1):
            account_match = account_line_re.match(line)
            if account_match:
                value = account_match.group(1).strip().strip("'\"")
                assert value == "" or "${oc.env:SBATCH_ACCOUNT" in value, (
                    f"{yaml_file}:{lineno}: 'account:' must resolve from env "
                    f"or be empty, got literal {value!r}"
                )

            sbatch_match = sbatch_account_re.search(line)
            assert not sbatch_match, (
                f"{yaml_file}:{lineno}: committed literal '#SBATCH -A' account value"
            )

            cli_match = cli_account_re.search(line)
            assert not cli_match, (
                f"{yaml_file}:{lineno}: committed literal '--account=' value"
            )


def test_resolved_snapshot_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_resolved writes fully-resolved YAML (interpolations baked in);
    reloading it gives back an equal config. Env vars are monkeypatched so
    the test never depends on the real environment's SBATCH_ACCOUNT/USER."""
    monkeypatch.setenv("SBATCH_ACCOUNT", "FUS999")
    monkeypatch.setenv("USER", "testuser")

    cfg = resolve_config(["experiment=mast_smoke"], config_root=CONFIGS_ROOT)
    out_path = tmp_path / "resolved.yaml"
    save_resolved(cfg, out_path)

    raw_text = out_path.read_text()
    assert "${" not in raw_text, "saved snapshot must have interpolations baked in"
    assert "FUS999" in raw_text

    reloaded = OmegaConf.load(out_path)
    assert OmegaConf.to_container(reloaded, resolve=True) == OmegaConf.to_container(
        cfg, resolve=True
    )


def test_dataclasses_are_directly_constructible() -> None:
    """The four dataclasses are a public contract on their own, independent
    of resolve_config -- they must be plain, directly instantiable
    dataclasses with the documented defaults."""
    assert ClusterConfig().tokamark_storage_options == {}
    assert TokamarkDataConfig().task_id == "task_2-3"
    assert RunSettings().seed == 0
    experiment = ExperimentConfig()
    assert isinstance(experiment.cluster, ClusterConfig)
    assert isinstance(experiment.data, TokamarkDataConfig)
