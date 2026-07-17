"""Signal registry loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_FIELDS = (
    "canonical_name",
    "device",
    "source_name",
    "units",
    "kind",
    "sharing_label",
    "review_status",
    "description",
)
ACTUATOR_FIELDS = (
    "command_or_measured",
    "bounds",
    "rate_limit",
    "safe_to_perturb",
)
ENUMS = {
    "kind": {"measurement", "actuator", "derived"},
    "sharing_label": {"private", "internal", "shared_core"},
    "review_status": {"pending_physics_review", "approved", "rejected"},
    "command_or_measured": {"command", "measured"},
}


@dataclass
class SignalSpec:
    """A registered fusion signal."""

    canonical_name: str
    device: str
    source_name: str
    units: str
    kind: str
    sharing_label: str
    review_status: str
    description: str
    command_or_measured: str | None = None
    bounds: list[float] | None = None
    rate_limit: float | None = None
    safe_to_perturb: bool | None = None


def _entry_mapping(entry: object) -> Mapping[str, Any]:
    if isinstance(entry, SignalSpec):
        return vars(entry)
    if isinstance(entry, Mapping):
        return entry
    return {}


def validate_registry(entries: Any) -> list[str]:
    """Return all structural violations found in registry entries."""
    violations: list[str] = []
    seen_names: set[str] = set()

    if not isinstance(entries, list):
        return ["registry field 'signals' must be a list"]

    for index, raw_entry in enumerate(entries):
        entry = _entry_mapping(raw_entry)
        label = f"entry {index}"
        if not entry:
            violations.append(f"{label} must be a mapping")
            continue

        name = entry.get("canonical_name")
        if isinstance(name, str):
            label = f"entry {name!r}"
            if name in seen_names:
                violations.append(f"duplicate canonical_name {name!r}")
            seen_names.add(name)

        for field in REQUIRED_FIELDS:
            if field not in entry or entry[field] is None:
                violations.append(f"{label}: missing required field {field!r}")

        for field in ("kind", "sharing_label", "review_status"):
            value = entry.get(field)
            if value is not None and value not in ENUMS[field]:
                violations.append(f"{label}: invalid {field} value {value!r}")

        if entry.get("kind") == "actuator":
            for field in ACTUATOR_FIELDS:
                if field not in entry:
                    violations.append(
                        f"{label}: missing required actuator field {field!r}"
                    )
            command_type = entry.get("command_or_measured")
            if command_type is not None and command_type not in ENUMS[
                "command_or_measured"
            ]:
                violations.append(
                    f"{label}: invalid command_or_measured value {command_type!r}"
                )

        if (
            entry.get("sharing_label") == "shared_core"
            and entry.get("review_status") != "approved"
        ):
            violations.append(
                f"{label}: shared_core signals require review_status 'approved'"
            )

    return violations


def load_registry(path: str | Path) -> dict[str, SignalSpec]:
    """Load a YAML registry, raising with every structural violation."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    entries = document.get("signals") if isinstance(document, Mapping) else None
    violations = validate_registry(entries)
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise ValueError(f"invalid signal registry:\n{details}")

    specs = [SignalSpec(**entry) for entry in entries]
    return {spec.canonical_name: spec for spec in specs}
