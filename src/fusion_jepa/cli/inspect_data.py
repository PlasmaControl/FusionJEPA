"""M0 data inspection command shell."""

from __future__ import annotations

import sys
from typing import Sequence

from fusion_jepa.config import ConfigError

from ._common import (
    dry_run_report,
    parse_argv,
    report_config_error,
    resolve_cli_config,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data-inspection CLI shell."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_argv(arguments)
        cfg = resolve_cli_config(parsed)
    except ConfigError as exc:
        return report_config_error(exc)

    if parsed.dry_run:
        return dry_run_report(cfg)

    print("not implemented (M1)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
