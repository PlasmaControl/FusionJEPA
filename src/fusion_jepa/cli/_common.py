"""Shared helpers for the M0 command-line shells."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, open_dict

from fusion_jepa.config import ConfigError, resolve_config

USAGE = "usage: python -m fusion_jepa.cli.{train,evaluate,inspect_data} " \
    "experiment=NAME [key=value ...] [--dry-run] [--allow-test]"


@dataclass(frozen=True)
class ParsedArgv:
    """Dotlist configuration arguments and CLI-only switches."""

    dotlist: list[str]
    dry_run: bool = False
    allow_test: bool = False


def parse_argv(argv: Sequence[str]) -> ParsedArgv:
    """Separate config dotlist entries from CLI-only flags."""
    dotlist: list[str] = []
    dry_run = False
    allow_test = False
    for item in argv:
        if item in {"--help", "-h"}:
            print(USAGE)
            raise SystemExit(0)
        if item == "--dry-run":
            dry_run = True
        elif item == "--allow-test":
            allow_test = True
        elif item.startswith("--"):
            raise ConfigError(f"Unknown option {item!r}; run with --help for usage")
        else:
            dotlist.append(item)
    return ParsedArgv(dotlist, dry_run, allow_test)


def resolve_cli_config(parsed: ParsedArgv) -> DictConfig:
    """Resolve dotlist arguments and apply flat CLI switch settings."""
    cfg = resolve_config(parsed.dotlist)
    cfg.dry_run = parsed.dry_run
    cfg.allow_test_split = parsed.allow_test
    return cfg


def add_experiment_name(cfg: DictConfig, dotlist: Sequence[str]) -> None:
    """Supply the experiment identifier required by run artifact naming."""
    experiment = next(
        item.partition("=")[2]
        for item in dotlist
        if item.partition("=")[:2] == ("experiment", "=")
    )
    with open_dict(cfg):
        cfg.experiment_name = experiment


def dry_run_report(cfg: DictConfig) -> int:
    """Print local reachability and run-root writability checks."""
    tokamark_root = str(cfg.cluster.tokamark_root)
    if tokamark_root.startswith("s3://"):
        tokamark_status = "remote, not checked"
    else:
        tokamark_status = "reachable" if Path(tokamark_root).exists() else "missing"

    runs_root = Path(str(cfg.cluster.runs_root))
    candidate = runs_root.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    writable = candidate.is_dir() and os.access(candidate, os.W_OK)

    print("Fusion-JEPA dry run")
    print(f"tokamark_root: {tokamark_root} ({tokamark_status})")
    print(f"runs_root parent: {runs_root.parent} (writable: {writable})")
    return 0


def report_config_error(exc: ConfigError) -> int:
    """Render an actionable configuration error."""
    print(f"error: {exc}", file=sys.stderr)
    return 2
