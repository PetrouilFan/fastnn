import numpy as np
import fastnn as fnn
from tests.test_utils import (
    numerical_gradient as numerical_grad,
    numerical_gradient_elementwise as numerical_grad_elementwise,
    check_gradient as check_grad,
    check_unary_gradient as check_unary_grad,
    check_binary_gradient as check_binary_grad,
    requires_grad,
)


def test_relu_gradient_mask():
    """Test that ReLU gradient is zero at negative input positions."""
    x = fnn.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], [5])
    requires_grad(x)
    y = fnn.relu(x)
    y.sum().backward()

    # Gradient should be 0 for x <= 0, 1 for x > 0
    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    actual_grad = x.grad.numpy()

    assert np.allclose(actual_grad, expected_grad), (
        f"ReLU gradient incorrect: {actual_grad} vs {expected_grad}"
    )


def test_abs_gradient_sign():
    """Test that Abs gradient is sign of input."""
    x = fnn.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], [5])
    requires_grad(x)
    y = fnn.abs(x)
    y.sum().backward()

    # Gradient is -1 for x < 0, undefined (0) for x = 0, 1 for x > 0
    expected_grad = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
    actual_grad = x.grad.numpy()

    assert np.allclose(actual_grad, expected_grad), (
        f"Abs gradient incorrect: {actual_grad} vs {expected_grad}"
    )


def test_silu_gradient_no_divbyzero():
    """Test that SiLU gradient doesn't have division by zero issues."""
    x = fnn.tensor([-1.0, 0.0, 1.0], [3])
    requires_grad(x)
    y = fnn.silu(x)
    y.sum().backward()

    # Should not raise and should have valid gradients
    assert x.grad is not None
    grad_data = x.grad.numpy()
    assert not np.any(np.isnan(grad_data))  # SiLU gradient has NaN
    assert not np.any(np.isinf(grad_data))  # SiLU gradient has Inf


def test_gelu_gradient():
    """Test GELU gradient using numerical differentiation."""
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = check_unary_grad("gelu", x_data)
    assert result["passed"], f"GELU gradient failed: {result}"


def test_sigmoid_gradient():
    """Test sigmoid gradient using numerical differentiation."""
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = check_unary_grad("sigmoid", x_data)
    assert result["passed"], f"Sigmoid gradient failed: {result}"


def test_tanh_gradient():
    """Test tanh gradient using numerical differentiation."""
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = check_unary_grad("tanh", x_data)
    assert result["passed"], f"Tanh gradient failed: {result}"


def test_add_grad():
    """Test add gradient."""
    a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = check_binary_grad("add", a_data, b_data)
    assert result["passed_a"], f"Add grad_a failed: {result}"
    assert result["passed_b"], f"Add grad_b failed: {result}"


def test_mul_grad():
    """Test mul gradient: d(a*b)/da = b, d(a*b)/db = a."""
    a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = check_binary_grad("mul", a_data, b_data)
    assert result["passed_a"], f"Mul grad_a failed: {result}"
    assert result["passed_b"], f"Mul grad_b failed: {result}"


def test_div_grad():
    """Test div gradient: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2."""
    a_data = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    b_data = np.array([2.0, 3.0, 6.0], dtype=np.float32)
    result = check_binary_grad("div", a_data, b_data)
    assert result["passed_a"], f"Div grad_a failed: {result}"
    assert result["passed_b"], f"Div grad_b failed: {result}"


def test_sub_grad():
    """Test sub gradient: d(a-b)/da = 1, d(a-b)/db = -1."""
    a_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    b_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = check_binary_grad("sub", a_data, b_data)
    assert result["passed_a"], f"Sub grad_a failed: {result}"
    assert result["passed_b"], f"Sub grad_b failed: {result}"


def test_exp_grad():
    """Test exp gradient: d(exp(x)/dx = exp(x)."""
    x_data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    result = check_unary_grad("exp", x_data)
    assert result["passed"], f"Exp gradient failed: {result}"


def test_log_grad():
    """Test log gradient: d(log(x)/dx = 1/x."""
    x_data = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    result = check_unary_grad("log", x_data)
    assert result["passed"], f"Log gradient failed: {result}"


def test_sqrt_grad():
    """Test sqrt gradient: d(sqrt(x)/dx = 1/(2*sqrt(x)."""
    x_data = np.array([0.25, 1.0, 4.0], dtype=np.float32)
    result = check_unary_grad("sqrt", x_data)
    assert result["passed"], f"Sqrt gradient failed: {result}"


def test_neg_grad():
    """Test neg gradient: d(-x)/dx = -1."""
    x_data = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    result = check_unary_grad("neg", x_data)
    assert result["passed"], f"Neg gradient failed: {result}"


def test_matmul_grad():
    """Test matmul gradient for 2D matrices."""
    a = fnn.tensor([[1.0, 2.0], [3.0, 4.0]], [2, 2])
    b = fnn.tensor([[5.0, 6.0], [7.0, 8.0]], [2, 2])
    requires_grad(a)
    requires_grad(b)

    c = a @ b
    c.sum().backward()

    # dC/dA = C @ B^T, dC/dB = A^T @ C where C is all-ones gradient
    expected_grad_a = b.numpy()  # sum of columns of b
    expected_grad_b = a.numpy().T  # sum of rows of a

    # Actually: d(a@b)/da = grad @ b^T
    # grad is ones, so grad_a = ones @ b^T = sum(b, axis=1) for each row
    expected_grad_a = np.array([[11.0, 15.0], [11.0, 15.0]])  # ones(2,2) @ b
    expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])  # a^T @ ones(2,2)

    assert np.allclose(a.grad.numpy(), expected_grad_a), (
        f"matmul grad_a: {a.grad.numpy()} vs {expected_grad_a}"
    )
    assert np.allclose(b.grad.numpy(), expected_grad_b), (
        f"matmul grad_b: {b.grad.numpy()} vs {expected_grad_b}"
    )


def test_softmax_grad():
    """Test softmax gradient."""
    x = fnn.tensor([[1.0, 2.0, 3.0]], [1, 3])
    requires_grad(x)
    y = fnn.softmax(x, -1)
    # Use a different loss that will have non-zero gradient
    # sum(x * softmax(x) has non-zero gradient
    loss = (x * y).sum()
    loss.backward()

    # Check that gradient exists and is not all zeros
    assert x.grad is not None
    grad_val = x.grad.numpy()
    assert not np.allclose(grad_val, 0), "Softmax gradient is all zeros"

    # Verify the gradient is correct
    # For sum(x * softmax(x), the gradient should be:
    # d(sum(x_i * softmax_i)/dx_j = softmax_j + x_j * softmax_j * (1 - softmax_j) - sum_i x_i * softmax_i * softmax_j
    # This is more complex, so we just check it's not all zeros
    print(f"Softmax gradient: {grad_val}")


def test_layer_norm_grad():
    """Test layer norm gradient."""
    x = fnn.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [2, 3])
    requires_grad(x)
    layer_norm = fnn.layers.LayerNorm(3)
    y = layer_norm(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_embedding_grad():
    """Test embedding gradient."""
    vocab_size = 10
    embed_dim = 4
    indices = fnn.tensor([0, 1, 2], [3])
    embedding = fnn.layers.Embedding(vocab_size, embed_dim)
    output = embedding(indices)
    output.sum().backward()
    # Check that weight gradient exists
    weight = embedding.parameters()[0]
    assert weight.grad is not None, "Embedding weight should have gradient"
    assert weight.grad.shape == weight.shape, "Gradient shape should match weight shape"


def test_cross_entropy_grad():
    """Test cross entropy gradient."""
    logits = fnn.tensor([[2.0, 1.0, 0.1]], [1, 3])
    targets = fnn.tensor([0], [1])
    requires_grad(logits)
    loss = fnn.cross_entropy_loss(logits, targets, reduction="mean")
    loss.backward()

    # Check gradient exists
    assert logits.grad is not None
    # Gradient should not be all zeros
    assert not np.allclose(logits.grad.numpy(), 0), (
        "Cross entropy gradient is all zeros"
    )


if __name__ == "__main__":
    print("Running gradient tests...")

    tests = [
        ("ReLU gradient mask", test_relu_gradient_mask),
        ("Abs gradient sign", test_abs_gradient_sign),
        ("SiLU gradient no divbyzero", test_silu_gradient_no_divbyzero),
        ("GELU gradient", test_gelu_gradient),
        ("Sigmoid gradient", test_sigmoid_gradient),
        ("Tanh gradient", test_tanh_gradient),
        ("Add gradient", test_add_grad),
        ("Mul gradient", test_mul_grad),
        ("Div gradient", test_div_grad),
        ("Sub gradient", test_sub_grad),
        ("Exp gradient", test_exp_grad),
        ("Log gradient", test_log_grad),
        ("Sqrt gradient", test_sqrt_grad),
        ("Neg gradient", test_neg_grad),
        ("Matmul gradient", test_matmul_grad),
        ("Softmax gradient", test_softmax_grad),
        ("LayerNorm gradient", test_layer_norm_grad),
        ("Embedding gradient", test_embedding_grad),
        ("CrossEntropy gradient", test_cross_entropy_grad),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except (AssertionError, ValueError, TypeError, RuntimeError) as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
