"""Deprecated: import :mod:`fusion_jepa` instead.

``faith`` was FusionAIHub's original installable package name. The project
is transitioning to ``fusion_jepa`` (see ``src/fusion_jepa/__init__.py``);
this module remains only so that code still importing ``faith`` keeps
working during the transition, and it will be removed once all callers
have migrated.
"""

import warnings

from fusion_jepa import __version__ as __version__

warnings.warn(
    "The 'faith' package is deprecated and will be removed in a future "
    "release; import 'fusion_jepa' instead.",
    DeprecationWarning,
    stacklevel=2,
)
