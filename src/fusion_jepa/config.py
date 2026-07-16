"""Config system for Fusion-JEPA.

Deliberately NOT a Hydra app: dataclasses + OmegaConf give us structured
defaults, yaml layering, and CLI dotlist overrides, while keeping
``python -m fusion_jepa.cli.train experiment=mast_smoke seed=0`` trivial.

``resolve_config`` merges, in order:

1. ``ExperimentConfig`` structured defaults
2. the named experiment's yaml (``configs/experiment/<name>.yaml``)
3. the cluster yaml it points at (``configs/cluster/<cluster>.yaml``)
4. the data yaml it points at (``configs/data/<data>.yaml``)
5. remaining CLI dotlist overrides

An experiment yaml selects its cluster/data profiles with plain string
pointers (``cluster: local``, ``data: tokamark``) rather than inline blocks,
so profiles are shared and swappable across experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


class ConfigError(ValueError):
    """Raised when config resolution fails for an actionable, user-facing
    reason: an unknown experiment/cluster/data profile, or an override key
    that does not exist in the schema. The message always names the
    offending key or file."""


@dataclass
class ClusterConfig:
    """Cluster-specific topology: filesystem roots, remote storage
    credentials, and the SLURM account. Values come entirely from
    ``configs/cluster/*.yaml`` -- never hardcode a real account string here
    or in a committed yaml; ``account`` must always resolve from the
    environment or be empty.
    """

    data_root: str = MISSING
    tokamark_root: str = MISSING
    tokamark_storage_options: Dict[str, Any] = field(default_factory=dict)
    runs_root: str = MISSING
    account: str = "${oc.env:SBATCH_ACCOUNT,''}"


@dataclass
class TokamarkDataConfig:
    """How to load the upstream `tokamark` MAST dataset."""

    task_id: str = "task_2-3"
    batch_size: int = 32
    num_workers: int = 4
    validate_batches: bool = True
    remote: bool = False
    limit_shots: Optional[int] = None


@dataclass
class RunSettings:
    """Settings that identify *this run* of an experiment. Flattened onto
    ``ExperimentConfig`` (via inheritance) rather than nested under a
    ``run:`` key, so `seed=0` works as a bare CLI dotlist override."""

    seed: int = 0
    split: str = "validation"
    dry_run: bool = False
    allow_test_split: bool = False


@dataclass
class ExperimentConfig(RunSettings):
    """Top-level config resolved by `resolve_config`."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    data: TokamarkDataConfig = field(default_factory=TokamarkDataConfig)


def _default_config_root() -> Path:
    """Walk up from this file to find the repo's `configs/` directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    raise ConfigError(
        f"Could not locate a 'configs' directory by walking up from {here}; "
        "pass config_root= explicitly."
    )


def resolve_config(
    argv: Sequence[str], config_root: Optional[Union[str, Path]] = None
) -> DictConfig:
    """Resolve a full `ExperimentConfig` from CLI-style dotlist args.

    `argv` must contain an `experiment=<name>` entry naming a yaml file
    under `<config_root>/experiment/`; every other entry is treated as a
    dotlist override applied last (e.g. `seed=7`, `data.batch_size=4`).
    `config_root` defaults to the repo's `configs/` directory but can be
    overridden (e.g. by tests, or a config tree living elsewhere).
    """
    root = Path(config_root) if config_root is not None else _default_config_root()

    experiment_name: Optional[str] = None
    remaining: List[str] = []
    for item in argv:
        key, sep, value = item.partition("=")
        if sep and key == "experiment":
            experiment_name = value
        else:
            remaining.append(item)

    if not experiment_name:
        raise ConfigError(
            "resolve_config requires an 'experiment=<name>' entry in argv, "
            f"got {list(argv)!r}"
        )

    experiment_path = root / "experiment" / f"{experiment_name}.yaml"
    if not experiment_path.exists():
        raise ConfigError(
            f"Unknown experiment '{experiment_name}': no such file {experiment_path}"
        )
    experiment_raw = OmegaConf.load(experiment_path)

    cluster_name = experiment_raw.get("cluster")
    data_name = experiment_raw.get("data")
    if not cluster_name:
        raise ConfigError(f"{experiment_path} is missing required key 'cluster'")
    if not data_name:
        raise ConfigError(f"{experiment_path} is missing required key 'data'")

    cluster_path = root / "cluster" / f"{cluster_name}.yaml"
    data_path = root / "data" / f"{data_name}.yaml"
    if not cluster_path.exists():
        raise ConfigError(
            f"Unknown cluster '{cluster_name}' referenced by {experiment_path}: "
            f"no such file {cluster_path}"
        )
    if not data_path.exists():
        raise ConfigError(
            f"Unknown data profile '{data_name}' referenced by {experiment_path}: "
            f"no such file {data_path}"
        )

    cluster_raw = OmegaConf.load(cluster_path)
    data_raw = OmegaConf.load(data_path)
    # cluster/data are pure profile-name pointers, not part of the merged
    # schema themselves -- everything else in the experiment yaml (if
    # anything) is merged in directly.
    experiment_overrides = OmegaConf.create(
        {k: v for k, v in experiment_raw.items() if k not in ("cluster", "data")}
    )

    cfg = OmegaConf.structured(ExperimentConfig)
    try:
        cfg = OmegaConf.merge(cfg, experiment_overrides)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"cluster": cluster_raw}))
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": data_raw}))
    except OmegaConfBaseException as exc:
        raise ConfigError(
            f"While loading experiment '{experiment_name}' "
            f"(cluster={cluster_name!r}, data={data_name!r}): {exc}"
        ) from exc

    if remaining:
        try:
            cli_cfg = OmegaConf.from_dotlist(remaining)
        except Exception as exc:
            raise ConfigError(
                f"Invalid CLI override syntax {remaining!r}: {exc}"
            ) from exc
        try:
            cfg = OmegaConf.merge(cfg, cli_cfg)
        except OmegaConfBaseException as exc:
            raise ConfigError(
                f"Unknown or invalid CLI override in {remaining!r}: {exc}"
            ) from exc

    assert isinstance(cfg, DictConfig)
    return cfg


# Top-level config keys that describe *where*/*how* a run executed rather
# than the science it ran -- excluded from the run-hash input. The cluster
# block currently carries every path and the SLURM account; extend this set
# if a future field adds path/logging settings outside of it.
_NON_SCIENTIFIC_TOP_LEVEL_KEYS = ("cluster", "experiment_name", "experiment")


def scientific_subset(cfg: DictConfig) -> Dict[str, Any]:
    """Return a deterministic, plain-dict view of `cfg` with the cluster
    block removed -- the input to a future run-hash. Two configs that differ
    only in *where* they ran (cluster, paths, account) but agree on every
    scientific setting (data, seed, ...) must produce an equal subset.

    The dropped keys are removed *before* resolving interpolations, not
    after: cluster-only values (e.g. `${oc.env:HOME}`, `${oc.env:USER}`)
    must never need to resolve just to be thrown away, since an unset env
    var there is irrelevant to the (cluster-free) output.
    """
    # resolve=False keeps interpolations as literal strings, so dropping a
    # key here can never trigger resolution of a value we're about to
    # discard.
    unresolved: Dict[str, Any] = OmegaConf.to_container(  # type: ignore[assignment]
        cfg, resolve=False
    )
    for key in _NON_SCIENTIFIC_TOP_LEVEL_KEYS:
        unresolved.pop(key, None)

    # Only the remaining (scientific) keys get resolved.
    remainder = OmegaConf.create(unresolved)
    resolved: Dict[str, Any] = OmegaConf.to_container(  # type: ignore[assignment]
        remainder, resolve=True
    )
    return _sort_recursively(resolved)


def _sort_recursively(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_recursively(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_sort_recursively(item) for item in value]
    return value


def save_resolved(cfg: DictConfig, path: Union[str, Path]) -> None:
    """Write `cfg` to `path` as fully-resolved YAML: every `${oc.env:...}`
    (and any other) interpolation is replaced by its concrete value, so the
    snapshot is self-contained and reproducible independent of the
    environment that produced it.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
