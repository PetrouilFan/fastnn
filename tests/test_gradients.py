"""Gradient checking tests - uses utilities from test_utils."""

import numpy as np
import fastnn as fnn

from tests.test_utils import (
    numerical_gradient,
    numerical_gradient_elementwise,
    check_gradient,
    check_unary_gradient,
    check_binary_gradient,
    make_tensor,
    requires_grad,
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


if __name__ == "__main__":
    test_relu_grad()
    print("  PASSED: test_relu_grad")
    test_sigmoid_grad()
    print("  PASSED: test_sigmoid_grad")
    test_tanh_grad()
    print("  PASSED: test_tanh_grad")
    test_gelu_grad()
    print("  PASSED: test_gelu_grad")
    test_silu_grad()
    print("  PASSED: test_silu_grad")
    test_add_grad()
    print("  PASSED: test_add_grad")
    test_mul_grad()
    print("  PASSED: test_mul_grad")
    test_matmul_grad()
    print("  PASSED: test_matmul_grad")
    print("\n=== Gradient Tests PASSED ===")
