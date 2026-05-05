"""Export module for fastnn.

This module is deprecated. Use `fastnn.io` instead.

Provides backward-compatible imports for existing code.
"""

import warnings

warnings.warn(
    "fastnn.export is deprecated, use fastnn.io instead",
    DeprecationWarning,
    stacklevel=2
)

from fastnn.io.export import *  # noqa: F401, F403
