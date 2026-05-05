"""Serialization module for fastnn.

This module is deprecated. Use `fastnn.io` instead.

Provides backward-compatible imports for existing code.
"""

import warnings

warnings.warn(
    "fastnn.serialization is deprecated, use fastnn.io instead",
    DeprecationWarning,
    stacklevel=2
)

from fastnn.io.serialization import *  # noqa: F401, F403
