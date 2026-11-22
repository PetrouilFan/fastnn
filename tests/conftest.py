"""Pytest configuration for the v3 inference runtime.

The 2.x Python facade this file used to isolate (a process-global storage pool)
is gone: sessions own their arena and threads, so a test that completes leaves
nothing behind for the next one. The file is kept because the test directory is
shared with the fn3 Python bindings, and a future fixture belongs here.
"""

import fastnn  # noqa: F401  (the extension must import before any test collects)
