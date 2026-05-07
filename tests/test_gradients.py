"""Gradient checking tests - uses utilities from test_utils."""

import numpy as np
import fastnn as fnn

from tests.test_utils import (
    numerical_gradient_elementwise,
    check_gradient,
    check_unary_gradient,
    check_binary_gradient,
    make_tensor,
    requires_grad,
    check_loss_gradient,
)


def test_relu_grad():
    """Test ReLU gradient."""
    x_data = np.array([1.0, -1.0, 0.0, 2.0, -3.0], dtype=np.float32)
    check_unary_gradient("relu", x_data)


def test_sigmoid_grad():
    """Test sigmoid gradient."""
    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    check_unary_gradient("sigmoid", x_data)


def test_tanh_grad():
    """Test tanh gradient."""
    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    check_unary_gradient("tanh", x_data)


def test_gelu_grad():
    """Test GELU gradient."""
    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    check_unary_gradient("gelu", x_data)


def test_silu_grad():
    """Test SiLU gradient."""
    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    check_unary_gradient("silu", x_data)


def test_add_grad():
    """Test addition gradient."""
    a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    check_binary_gradient("add", a_data, b_data)


def test_mul_grad():
    """Test multiplication gradient."""
    a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    check_binary_gradient("mul", a_data, b_data)


def test_matmul_grad():
    """Test matrix multiplication gradient."""
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    check_binary_gradient("matmul", a_data, b_data)


def test_sum_grad():
    """Test sum gradient with numerical checking."""
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = x.sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), [1.0, 1.0, 1.0])


def test_mean_grad():
    """Test mean gradient with numerical checking."""
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = x.mean()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_cross_entropy_grad():
    """Test cross entropy gradient with numerical checking."""
    logits = fnn.tensor([[2.0, 1.0, 0.1]], [1, 3])
    targets = fnn.tensor([0], [1])
    check_loss_gradient(fnn.cross_entropy_loss, logits, targets)


def test_no_grad_context():
    """Test no_grad context manager."""
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    with fnn.no_grad():
        y = 2 * x
    assert y.grad_fn is None


def test_detach():
    """Test tensor detach method."""
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = 2 * x
    z = y.detach()
    assert z.grad_fn is None
