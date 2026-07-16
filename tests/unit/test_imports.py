"""Packaging smoke tests for Task 0.1 (faith -> fusion_jepa rename).

These guard the dual-package transition: the new ``fusion_jepa`` package
must be installed and versioned correctly, the legacy
``tokamak_foundation_model`` package (which all training scripts still
import) must keep working untouched, and the legacy ``faith`` package must
remain importable as a deprecated shim rather than disappearing outright.
"""

import importlib
from importlib.metadata import version

import pytest

import fusion_jepa


def test_import_fusion_jepa_version() -> None:
    """fusion_jepa imports cleanly and its __version__ matches the installed
    distribution metadata, proving pyproject.toml's [project.name] and
    [tool.hatch.version] path were repointed at the new package."""
    assert fusion_jepa.__version__ == version("fusion_jepa")


def test_import_tokamak_foundation_model_still_works() -> None:
    """The legacy tokamak_foundation_model package -- still imported by every
    training script -- must keep importing unmodified during the transition."""
    module = importlib.import_module("tokamak_foundation_model")
    assert module is not None


def test_import_faith_emits_deprecation_warning() -> None:
    """The legacy faith package is now a deprecated shim: it must still
    import (old callers keep working) but warn and point at fusion_jepa,
    and it must re-export the same __version__ as fusion_jepa."""
    with pytest.warns(DeprecationWarning, match="fusion_jepa"):
        faith = importlib.import_module("faith")
    assert faith.__version__ == fusion_jepa.__version__
