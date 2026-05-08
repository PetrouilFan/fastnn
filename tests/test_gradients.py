"""Gradient checking tests - uses utilities from test_utils."""

import numpy as np
import fastnn as fnn
import pytest

from tests.test_utils import (
    numerical_gradient_elementwise,
    check_gradient,
    check_unary_gradient,
    check_binary_gradient,
    make_tensor,
    requires_grad,
    check_loss_gradient,
)


@pytest.mark.parametrize("op_name", ["relu", "sigmoid", "tanh", "gelu", "silu"])
def test_unary_gradient_ops(op_name):
    """Test gradient for unary operations."""
    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    check_unary_gradient(op_name, x_data)


@pytest.mark.parametrize("op_name", ["add", "mul", "matmul"])
def test_binary_gradient_ops(op_name):
    """Test gradient for binary operations."""
    if op_name == "matmul":
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    else:
        a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    check_binary_gradient(op_name, a_data, b_data)


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
