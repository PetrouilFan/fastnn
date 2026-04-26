"""Pytest configuration with memory pool isolation and shared test fixtures."""

import pytest
import numpy as np
import fastnn


# ============================================================================
# Autouse Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def isolate_memory_pool():
    """Clear the Rust storage pool after each test to prevent cross-test pollution.

    This prevents memory corruption issues that can occur when the static Rust
    storage pool holds onto recycled memory blocks during test teardown, especially
    when other libraries (like PyTorch) are also loaded in the same pytest process.
    """
    yield
    fastnn._core.clear_storage_pool()


# ============================================================================
# Random Seed Fixtures
# ============================================================================


@pytest.fixture(autouse=False)
def random_seed():
    """Set a deterministic random seed for reproducible tests."""
    fastnn.set_seed(42)
    np.random.seed(42)


@pytest.fixture(autouse=False)
def random_seed_123():
    """Alternative deterministic random seed."""
    fastnn.set_seed(123)
    np.random.seed(123)


# ============================================================================
# Tensor Creation Fixtures
# ============================================================================


@pytest.fixture
def small_tensor():
    """A small 1D tensor for quick tests."""
    return fastnn.tensor([1.0, 2.0, 3.0], [3])


@pytest.fixture
def small_tensor_2d():
    """A small 2D tensor for quick tests."""
    return fastnn.tensor([[1.0, 2.0], [3.0, 4.0]], [2, 2])


@pytest.fixture
def small_tensor_3d():
    """A small 3D tensor for quick tests."""
    return fastnn.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [2, 2, 2])


@pytest.fixture
def random_tensor_fixture():
    """A random tensor for gradient tests."""
    from tests.test_utils import random_tensor
    return random_tensor([3, 4], requires_grad=True)


@pytest.fixture
def random_tensor_pair_fixture():
    """A pair of random tensors for binary operation tests."""
    from tests.test_utils import random_tensor
    a = random_tensor([3, 4], requires_grad=True)
    b = random_tensor([3, 4], requires_grad=True)
    return a, b


@pytest.fixture
def random_tensor_3d():
    """A random 3D tensor for conv/pool tests."""
    from tests.test_utils import random_tensor
    return random_tensor([2, 3, 8, 8], requires_grad=True)


@pytest.fixture
def random_tensor_4d():
    """A random 4D tensor for conv/pool tests."""
    from tests.test_utils import random_tensor
    return random_tensor([2, 3, 16, 16], requires_grad=True)


@pytest.fixture
def batch_tensor():
    """A batch of tensors for training tests."""
    from tests.test_utils import random_tensor
    return random_tensor([8, 10], requires_grad=True)


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def simple_linear():
    """A simple linear layer for tests."""
    from tests.test_utils import make_linear
    return make_linear(10, 5, requires_grad=True)


@pytest.fixture
def simple_mlp():
    """A simple MLP for tests."""
    from tests.test_utils import make_mlp
    return make_mlp(2, [8], 1, requires_grad=True)


@pytest.fixture
def simple_conv2d():
    """A simple Conv2d layer for tests."""
    from tests.test_utils import make_conv2d
    return make_conv2d(3, 16, kernel_size=3, stride=1, padding=1, requires_grad=True)


@pytest.fixture
def simple_transformer():
    """A simple Transformer model for tests."""
    from tests.test_utils import make_transformer
    return make_transformer(
        vocab_size=50,
        max_seq_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        num_classes=2,
        dropout_p=0.1,
        requires_grad=True,
    )


# ============================================================================
# Optimizer Fixtures
# ============================================================================


@pytest.fixture
def simple_optimizer(simple_mlp):
    """A simple optimizer for tests."""
    from tests.test_utils import make_optimizer
    return make_optimizer(simple_mlp.parameters(), name="adam", lr=0.01)


@pytest.fixture
def adam_optimizer(simple_linear):
    """Adam optimizer for linear layer tests."""
    from tests.test_utils import make_optimizer
    return make_optimizer(simple_linear.parameters(), name="adam", lr=0.001)


@pytest.fixture
def sgd_optimizer(simple_linear):
    """SGD optimizer for linear layer tests."""
    from tests.test_utils import make_optimizer
    return make_optimizer(simple_linear.parameters(), name="sgd", lr=0.01, momentum=0.9)


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def train_data():
    """Small training dataset for quick tests."""
    X = fastnn.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
    y = fastnn.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])
    return X, y


@pytest.fixture
def train_loader(train_data):
    """DataLoader for quick tests."""
    X, y = train_data
    ds = fastnn.TensorDataset(X, y)
    return fastnn.DataLoader(ds, batch_size=2, shuffle=False)


@pytest.fixture
def random_batch():
    """Random batch for training tests."""
    from tests.test_utils import random_tensor
    X = random_tensor([8, 10], requires_grad=False)
    y = random_tensor([8, 1], requires_grad=False)
    return X, y


# ============================================================================
# Dtype Fixtures
# ============================================================================


@pytest.fixture(params=["f32", "f64"])
def dtype_fixture(request):
    """Parameterized dtype fixture."""
    return request.param


@pytest.fixture
def f32_tensor():
    """A float32 tensor."""
    return fastnn.tensor([1.0, 2.0, 3.0], [3], dtype="f32")


@pytest.fixture
def f64_tensor():
    """A float64 tensor."""
    return fastnn.tensor([1.0, 2.0, 3.0], [3], dtype="f64")


# ============================================================================
# Device Fixtures
# ============================================================================


@pytest.fixture
def cpu_device():
    """CPU device string."""
    return "cpu"


# ============================================================================
# Loss Function Fixtures
# ============================================================================


@pytest.fixture
def mse_loss():
    """MSE loss function."""
    return fastnn.mse_loss


@pytest.fixture
def cross_entropy_loss():
    """Cross entropy loss function."""
    return fastnn.cross_entropy_loss


# ============================================================================
# Gradient Checking Fixtures
# ============================================================================


@pytest.fixture
def gradient_tolerance():
    """Default tolerance for gradient checks."""
    return {"atol": 1e-3, "rtol": 1e-3}


@pytest.fixture
def high_precision_gradient_tolerance():
    """High precision tolerance for gradient checks."""
    return {"atol": 1e-4, "rtol": 1e-4}
