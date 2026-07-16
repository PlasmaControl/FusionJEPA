"""Tests for the pinned upstream ``tokamark`` install (Task 1.1).

``tokamark`` (the UKAEA-IBM-STFC-Fusion-FMs benchmark suite pinned in
``manifests/upstream.yaml``) is installed as a pixi git pypi-dependency, at
the exact commit that manifest records. These tests guard three things: the
package actually imports from the installed environment, the pin committed
in ``pyproject.toml`` has not drifted from the manifest (the single source
of truth Task 0.8 established), and the benchmark's 14 task configs -- the
contract Task 1.5's adapter will build on -- load from the installed
package without any data/network access.
"""

import tomllib
from pathlib import Path

from fusion_jepa.utils.manifests import read_manifest


def _repo_root() -> Path:
    """Walk up from this test file to find the repo root (holds ``.git``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA Git repository")


def test_tokamark_importable() -> None:
    import tokamark  # noqa: F401
    from tokamark.tasks import TASKS_CONFIGS_MAP  # noqa: F401


def test_installed_pin_matches_upstream_manifest() -> None:
    """The ``rev`` pinned in pyproject.toml must equal the commit
    manifests/upstream.yaml records for ``tokamark`` (Task 0.8's pin,
    independently ls-remote-verified).

    tokamark is scoped to the ``data`` pixi feature rather than the shared
    ``[tool.pixi.pypi-dependencies]`` table (Risk R1: its own pyproject.toml
    pins ``jupyterlab-widgets==3.0.15`` exactly, which conflicts with the
    ``fdp`` environment's ga-fdp/toksearch conda stack -- see the comment
    above ``[tool.pixi.feature.data]`` in pyproject.toml and
    docs/decisions/0001-tokamark-pin.md).
    """
    repo_root = _repo_root()
    manifest = read_manifest(repo_root / "manifests" / "upstream.yaml")

    with (repo_root / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    data_feature = pyproject["tool"]["pixi"]["feature"]["data"]
    pinned_rev = data_feature["pypi-dependencies"]["tokamark"]["rev"]

    assert pinned_rev == manifest["tokamark"]["commit"]


def test_all_14_task_configs_loadable() -> None:
    from tokamark.tasks import GROUP_TASKS, TASKS_CONFIGS_MAP, get_task_config

    assert len(TASKS_CONFIGS_MAP) == 14
    assert sum(len(tasks) for tasks in GROUP_TASKS.values()) == 14

    required_keys = {
        "task_name",
        "task_type",
        "sources_and_signals",
        "task_window_segmenter",
        "stride_window",
    }

    for task_name in TASKS_CONFIGS_MAP:
        config = get_task_config(task_name)
        assert required_keys <= set(config)
        assert config["task_name"] == task_name
