"""Pytest configuration with memory pool isolation fixture."""

import pytest
import fastnn


@pytest.fixture(autouse=True)
def isolate_memory_pool():
    """Clear the Rust storage pool after each test to prevent cross-test pollution.

    This prevents memory corruption issues that can occur when the static Rust
    storage pool holds onto recycled memory blocks during test teardown, especially
    when other libraries (like PyTorch) are also loaded in the same pytest process.
    """
    yield
    fastnn._core.clear_storage_pool()
