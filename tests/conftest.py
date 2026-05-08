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


@pytest.fixture
def small_mlp():
    """Create a small MLP for testing."""
    return fastnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)


@pytest.fixture
def adam_optimizer(small_mlp):
    """Create Adam optimizer for the small MLP."""
    return fastnn.Adam(small_mlp.parameters(), lr=0.01)


@pytest.fixture
def sample_data():
    """Create XOR sample data for training tests."""
    X = fastnn.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
    y = fastnn.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])
    return X, y
