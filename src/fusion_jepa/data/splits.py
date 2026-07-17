"""Leakage-safe shot-level dataset split manifests."""

from dataclasses import asdict, dataclass
from pathlib import Path

from fusion_jepa.utils.manifests import read_manifest, write_manifest


@dataclass
class SplitManifest:
    """Assignment of whole shots to mutually exclusive dataset splits."""

    name: str
    source: str
    source_hash: str
    splits: dict[str, list[str]]

    def __post_init__(self) -> None:
        """Enforce shot disjointness at construction time."""
        self.assert_disjoint()

    def split_of(self, shot_id: str) -> str:
        """Return the split containing ``shot_id``."""
        for split, shot_ids in self.splits.items():
            if shot_id in shot_ids:
                return split
        raise KeyError(
            f"shot id {shot_id!r} is not present in any split; "
            "check the split manifest and source dataset"
        )

    def assert_disjoint(self) -> None:
        """Raise if any shot is assigned to more than one split."""
        assignments: dict[str, set[str]] = {}
        for split, shot_ids in self.splits.items():
            for shot_id in shot_ids:
                assignments.setdefault(shot_id, set()).add(split)
        overlaps = {
            shot_id: sorted(splits)
            for shot_id, splits in assignments.items()
            if len(splits) > 1
        }
        if overlaps:
            details = ", ".join(
                f"{shot_id!r} ({', '.join(splits)})"
                for shot_id, splits in sorted(overlaps.items())
            )
            raise ValueError(f"shots appear in more than one split: {details}")

    def save(self, path: str | Path) -> None:
        """Persist this manifest as YAML."""
        write_manifest(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        """Load and validate a YAML split manifest."""
        # __post_init__ enforces disjointness at construction; no extra call needed.
        return cls(**read_manifest(path))
