"""Tests for the committed upstream dependency manifest.

``manifests/upstream.yaml`` pins the science's upstream dependencies: the
TokaMark benchmark code, its CNN baseline repo, and the TokaMark dataset
itself. These tests load the real committed file (not a fixture) via
``fusion_jepa.utils.manifests.read_manifest`` -- the same Task 0.5 utility
``create_run_dir`` uses to fold the manifest into every run's provenance --
and pin down its schema so a future accidental edit (a branch name creeping
in where a commit SHA belongs, a required field dropped) fails loudly.
"""

import re
from pathlib import Path

from fusion_jepa.utils.manifests import manifest_hash, read_manifest

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")

_REPO_ENTRIES = ("tokamark", "tokamark_baseline")
_REPO_REQUIRED_KEYS = {"url", "tag", "commit", "license"}
_DATASET_REQUIRED_KEYS = {
    "hf_repo_id",
    "gated",
    "s3_endpoint",
    "s3_path",
    "size_compressed",
    "size_unpacked",
    "license",
    "revision",
    "retrieved",
}


def _repo_root() -> Path:
    """Walk up from this test file to find the repo root (holds ``.git``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA Git repository")


def _load_manifest() -> dict:
    return read_manifest(_repo_root() / "manifests" / "upstream.yaml")


def test_schema_required_keys_present() -> None:
    manifest = _load_manifest()

    assert set(manifest) == {"tokamark", "tokamark_baseline", "tokamark_dataset"}
    for name in _REPO_ENTRIES:
        assert _REPO_REQUIRED_KEYS <= set(manifest[name]), name
    assert _DATASET_REQUIRED_KEYS <= set(manifest["tokamark_dataset"])


def test_commits_are_full_shas_not_branch_names() -> None:
    manifest = _load_manifest()

    for name in _REPO_ENTRIES:
        commit = manifest[name]["commit"]
        assert _FULL_SHA.match(commit), f"{name}.commit is not a full sha: {commit!r}"


def test_manifest_hash_computable() -> None:
    manifest = _load_manifest()

    digest = manifest_hash(manifest)

    assert isinstance(digest, str)
    assert _HEX_DIGEST.match(digest)
