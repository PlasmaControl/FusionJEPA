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
KNOWN_FIELDS = set(REQUIRED_FIELDS) | set(ACTUATOR_FIELDS)
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

        for field in entry.keys() - KNOWN_FIELDS:
            violations.append(f"{label}: unrecognized field {field!r}")

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
            if (
                "command_or_measured" in entry
                and command_type not in ENUMS["command_or_measured"]
            ):
                violations.append(
                    f"{label}: invalid command_or_measured value {command_type!r}"
                )

            safe_to_perturb = entry.get("safe_to_perturb")
            if "safe_to_perturb" in entry and not isinstance(
                safe_to_perturb, bool
            ):
                violations.append(
                    f"{label}: safe_to_perturb must be a bool, got "
                    f"{safe_to_perturb!r}"
                )

            bounds = entry.get("bounds")
            if bounds is not None and (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(value) not in (int, float) for value in bounds)
            ):
                violations.append(
                    f"{label}: bounds must be null or a 2-item list of numbers"
                )

            rate_limit = entry.get("rate_limit")
            if rate_limit is not None and type(rate_limit) not in (int, float):
                violations.append(
                    f"{label}: rate_limit must be null or a number, got "
                    f"{rate_limit!r}"
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
