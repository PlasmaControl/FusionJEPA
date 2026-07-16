"""M0 evaluation command shell."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Sequence

from fusion_jepa.config import ConfigError
from fusion_jepa.utils.run_artifacts import create_run_dir, write_completion

from ._common import (
    add_experiment_name,
    dry_run_report,
    parse_argv,
    report_config_error,
    resolve_cli_config,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation CLI shell."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_argv(arguments)
    except ConfigError as exc:
        return report_config_error(exc)

    if "split=test" in parsed.dotlist and not parsed.allow_test:
        print(
            "error: split=test requires --allow-test; pass the flag only for "
            "an intentional final test evaluation",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = resolve_cli_config(parsed)
    except ConfigError as exc:
        return report_config_error(exc)

    if parsed.dry_run:
        return dry_run_report(cfg)

    add_experiment_name(cfg, parsed.dotlist)
    started_at = datetime.now(timezone.utc).isoformat()
    context = create_run_dir(cfg, arguments, base=cfg.cluster.runs_root)
    write_completion(
        context.run_dir,
        status="failed",
        started_at=started_at,
        warnings=[],
        failure_reason="not_implemented",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
