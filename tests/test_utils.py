"""Shared test utilities and fixtures for fastnn tests.

This module provides common test helpers to eliminate duplicate test code
across the test suite. It includes fixtures for tensor creation, model
setup, gradient checking, and numerical differentiation.
"""

import numpy as np
import fastnn


# ============================================================================
# Tensor Creation Utilities
# ============================================================================


def make_tensor(data, shape=None, requires_grad=False):
    """Create a fastnn tensor with optional gradient tracking.

    Args:
        data: Data to populate the tensor (list, np.ndarray, or scalar).
        shape: Shape of the tensor. If None, inferred from data.
        requires_grad: Whether to enable gradient tracking.

    Returns:
        fastnn tensor with optional gradient tracking enabled.
    """
    if isinstance(data, np.ndarray):
        flat = data.flatten().tolist()
        shape = list(data.shape) if shape is None else shape
    else:
        flat = data
    t = fastnn.tensor(flat, shape if shape is not None else [])
    if requires_grad:
        t.requires_grad_(True)
    return t


def tensor_like(other, requires_grad=False):
    """Create a tensor with the same shape/dtype as another tensor.

    Args:
        other: Reference tensor.
        requires_grad: Whether to enable gradient tracking.

    Returns:
        New tensor with same shape/dtype.
    """
    t = fastnn.zeros(other.shape, dtype=other.dtype)
    if requires_grad:
        t.requires_grad_(True)
    return t


def random_tensor(shape, requires_grad=False, kind="normal",
                  low=-1.0, high=1.0, mean=0.0, std=1.0):
    """Create a random tensor for testing.

    Args:
        shape: Shape of the tensor.
        requires_grad: Whether to enable gradient tracking.
        kind: "normal" for Gaussian, "uniform" for uniform distribution.
        low: Lower bound for uniform distribution.
        high: Upper bound for uniform distribution.
        mean: Mean for normal distribution.
        std: Standard deviation for normal distribution.

    Returns:
        Random tensor with optional gradient tracking.
    """
    if kind == "normal":
        np_data = np.random.normal(mean, std, shape).astype(np.float32)
    else:
        np_data = np.random.uniform(low, high, shape).astype(np.float32)

    t = fastnn.tensor(np_data.flatten().tolist(), list(shape))
    if requires_grad:
        t.requires_grad_(True)
    return t


def random_tensor_pair(shape, requires_grad=False):
    """Create a pair of random tensors for binary operation tests.

    Args:
        shape: Shape for both tensors.
        requires_grad: Whether to enable gradient tracking.

    Returns:
        Tuple of two random tensors.
    """
    a = random_tensor(shape, requires_grad, kind="normal")
    b = random_tensor(shape, requires_grad, kind="normal")
    return a, b


def requires_grad(*tensors):
    """Enable gradient tracking on one or more tensors.

    Convenience function to reduce boilerplate in tests that call
    requires_grad_(True) repeatedly.

    Args:
        *tensors: One or more fastnn tensors to enable gradient tracking for.

    Returns:
        The input tensors (modified in-place), or a single tensor if only
        one was provided.
    """
    for t in tensors:
        t.requires_grad_(True)
    if len(tensors) == 1:
        return tensors[0]
    return tensors


# ============================================================================
# Model Creation
# ============================================================================


def make_linear(in_features, out_features, bias=True, requires_grad=True):
    """Create a Linear layer with optional gradient tracking.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include bias.
        requires_grad: Whether to enable gradient tracking on parameters.

    Returns:
        Linear layer with optional gradient tracking.
    """
    linear = fastnn.layers.Linear(in_features, out_features, bias=bias)
    if requires_grad:
        for p in linear.parameters():
            p.requires_grad_(True)
    return linear


def make_conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                padding=1, bias=True, requires_grad=True):
    """Create a Conv2d layer with optional gradient tracking.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Stride of convolution.
        padding: Padding of convolution.
        bias: Whether to include bias.
        requires_grad: Whether to enable gradient tracking on parameters.

    Returns:
        Conv2d layer with optional gradient tracking.
    """
    conv = fastnn.layers.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
    if requires_grad:
        for p in conv.parameters():
            p.requires_grad_(True)
    return conv


def make_mlp(input_dim, hidden_dims, output_dim, requires_grad=True):
    """Create an MLP model with optional gradient tracking.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension.
        requires_grad: Whether to enable gradient tracking on parameters.

    Returns:
        MLP model with optional gradient tracking.
    """
    model = fastnn.models.MLP(input_dim, hidden_dims, output_dim)
    if requires_grad:
        for p in model.parameters():
            p.requires_grad_(True)
    return model


def make_transformer(vocab_size=100, max_seq_len=16, d_model=64,
                     num_heads=4, num_layers=2, ff_dim=128,
                     num_classes=2, dropout_p=0.1, requires_grad=True):
    """Create a Transformer model with optional gradient tracking.

    Args:
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        ff_dim: Feed-forward dimension.
        num_classes: Number of output classes.
        dropout_p: Dropout probability.
        requires_grad: Whether to enable gradient tracking on parameters.

    Returns:
        Transformer model with optional gradient tracking.
    """
    model = fastnn.models.Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout_p=dropout_p,
    )
    if requires_grad:
        for p in model.parameters():
            p.requires_grad_(True)
    return model


# ============================================================================
# Optimizer Creation
# ============================================================================


def make_optimizer(parameters, name="adam", lr=0.001, **kwargs):
    """Create an optimizer by name.

    Args:
        parameters: Model parameters to optimize.
        name: Optimizer name (adam, adamw, sgd, muon, lion).
        lr: Learning rate.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer.
    """
    name = name.lower()
    if name == "adam":
        return fastnn.Adam(parameters, lr=lr, **kwargs)
    elif name == "adamw":
        return fastnn.AdamW(parameters, lr=lr, **kwargs)
    elif name == "sgd":
        return fastnn.SGD(parameters, lr=lr, **kwargs)
    elif name == "muon":
        return fastnn.Muon(parameters, lr=lr, **kwargs)
    elif name == "lion":
        return fastnn.Lion(parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================================
# Numerical Gradient Checking
# ============================================================================


def numerical_gradient_elementwise(op, x, eps=1e-4):
    """Compute numerical gradient for elementwise operations.

    Args:
        op: Function to compute gradient of.
        x: Input tensor.
        eps: Finite difference epsilon.

    Returns:
        Numerical gradient as numpy array.
    """
    x_np = x.numpy()
    grad = np.zeros_like(x_np)

    for i in range(x.numel()):
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps

        x_plus_t = fastnn.tensor(x_plus.flatten().tolist(), list(x_np.shape))
        x_minus_t = fastnn.tensor(x_minus.flatten().tolist(), list(x_np.shape))

        f_plus = op(x_plus_t).numpy().flat[0]
        f_minus = op(x_minus_t).numpy().flat[0]
        grad.flat[i] = (f_plus - f_minus) / (2 * eps)

    return grad


def check_gradient(op, *inputs, reduction="sum", atol=1e-3, rtol=1e-3,
                   eps=1e-4):
    """Check analytical gradient against numerical gradient.

    Args:
        op: Function to test.
        *inputs: Input tensors.
        reduction: How to reduce output ("sum", "mean", or None).
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        eps: Finite difference epsilon.

    Returns:
        List of (index, passed, details) tuples for each input.
    """
    input_tensors = []
    for inp in inputs:
        t = fastnn.tensor(inp.numpy().flatten().tolist(),
                          list(inp.numpy().shape))
        t.requires_grad_(True)
        input_tensors.append(t)

    output = op(*input_tensors)

    if reduction == "sum":
        output = output.sum()
    elif reduction == "mean":
        output = output.mean()

    output.backward()

    results = []
    for i, inp in enumerate(input_tensors):
        if inp.grad is None:
            results.append((i, False, "No gradient computed"))
            continue

        def op_wrapper(*args):
            out = op(*args)
            if reduction == "sum":
                return out.sum()
            elif reduction == "mean":
                return out.mean()
            return out

        numerical = numerical_gradient_elementwise(
            lambda x: op_wrapper(
                *[x if j == i else input_tensors[j]
                  for j in range(len(input_tensors))]
            ),
            inp,
            eps,
        )

        analytical = inp.grad.numpy()
        passed = np.allclose(analytical, numerical, atol=atol, rtol=rtol)

        results.append(
            (i, passed,
             {"analytical": analytical, "numerical": numerical,
              "diff": np.abs(analytical - numerical)})
        )

    return results


def check_unary_gradient(op_name, x_data, atol=1e-3, rtol=1e-3):
    """Test gradient for a unary operation.

    Args:
        op_name: Name of operation (relu, sigmoid, tanh, gelu, silu,
                 exp, log, sqrt, abs, neg).
        x_data: Input data as numpy array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dict with op name, passed flag, and gradient arrays.
    """
    ops = {
        "relu": fastnn.relu,
        "sigmoid": fastnn.sigmoid,
        "tanh": fastnn.tanh,
        "gelu": fastnn.gelu,
        "silu": fastnn.silu,
        "exp": fastnn.exp,
        "log": fastnn.log,
        "sqrt": fastnn.sqrt,
        "abs": fastnn.abs,
        "neg": fastnn.neg,
    }

    if op_name not in ops:
        raise ValueError(f"Unknown op: {op_name}")

    op = ops[op_name]
    x = fastnn.tensor(x_data.flatten().tolist(),
                      list(x_data.shape))
    x.requires_grad_(True)

    y = op(x)
    y.sum().backward()

    numerical = np.zeros_like(x_data)
    for i in range(x_data.size):
        x_plus = x_data.copy()
        x_minus = x_data.copy()
        x_plus.flat[i] += 1e-3
        x_minus.flat[i] -= 1e-3

        x_plus_t = fastnn.tensor(x_plus.flatten().tolist(),
                                 list(x_plus.shape))
        x_minus_t = fastnn.tensor(x_minus.flatten().tolist(),
                                  list(x_minus.shape))

        y_plus = op(x_plus_t)
        y_minus = op(x_minus_t)

        numerical.flat[i] = (y_plus.sum().numpy().flat[0]
                             - y_minus.sum().numpy().flat[0]) / (2 * 1e-3)

    numerical = numerical.flatten()
    analytical = x.grad.numpy().flatten()

    passed = np.allclose(analytical, numerical, atol=atol, rtol=rtol)
    return {
        "op": op_name,
        "passed": passed,
        "analytical": analytical,
        "numerical": numerical,
        "diff": np.abs(analytical - numerical)
    }


def check_binary_gradient(op_name, a_data, b_data, atol=1e-3, rtol=1e-3):
    """Test gradient for a binary operation.

    Args:
        op_name: Name of operation (add, sub, mul, div).
        a_data: First input as numpy array.
        b_data: Second input as numpy array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dict with op name, passed flags, and gradient arrays.
    """
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "matmul": lambda a, b: a @ b,
    }

    if op_name not in ops:
        raise ValueError(f"Unknown op: {op_name}")

    op = ops[op_name]

    a = fastnn.tensor(a_data.flatten().tolist(),
                      list(a_data.shape))
    b = fastnn.tensor(b_data.flatten().tolist(),
                      list(b_data.shape))
    a.requires_grad_(True)
    b.requires_grad_(True)

    y = op(a, b)
    y.sum().backward()

    numerical_a = np.zeros_like(a_data)
    for i in range(a_data.size):
        a_plus = a_data.copy()
        a_minus = a_data.copy()
        a_plus.flat[i] += 1e-3
        a_minus.flat[i] -= 1e-3

        a_plus_t = fastnn.tensor(a_plus.flatten().tolist(),
                                 list(a_plus.shape))
        a_minus_t = fastnn.tensor(a_minus.flatten().tolist(),
                                  list(a_minus.shape))

        y_plus = op(a_plus_t, b)
        y_minus = op(a_minus_t, b)

        numerical_a.flat[i] = (y_plus.sum().numpy().flat[0]
                               - y_minus.sum().numpy().flat[0]) / (2 * 1e-3)

    numerical_b = np.zeros_like(b_data)
    for i in range(b_data.size):
        b_plus = b_data.copy()
        b_minus = b_data.copy()
        b_plus.flat[i] += 1e-3
        b_minus.flat[i] -= 1e-3

        b_plus_t = fastnn.tensor(b_plus.flatten().tolist(),
                                 list(b_plus.shape))
        b_minus_t = fastnn.tensor(b_minus.flatten().tolist(),
                                  list(b_minus.shape))

        y_plus = op(a, b_plus_t)
        y_minus = op(a, b_minus_t)

        numerical_b.flat[i] = (y_plus.sum().numpy().flat[0]
                               - y_minus.sum().numpy().flat[0]) / (2 * 1e-3)

    numerical_a = numerical_a.flatten()
    numerical_b = numerical_b.flatten()
    analytical_a = a.grad.numpy().flatten()
    analytical_b = b.grad.numpy().flatten()

    passed_a = np.allclose(analytical_a, numerical_a, atol=atol, rtol=rtol)
    passed_b = np.allclose(analytical_b, numerical_b, atol=atol, rtol=rtol)

    return {
        "op": op_name,
        "passed_a": passed_a,
        "passed_b": passed_b,
        "analytical_a": analytical_a,
        "numerical_a": numerical_a,
        "analytical_b": analytical_b,
        "numerical_b": numerical_b,
    }


# ============================================================================
# Layer Gradient Checking
# ============================================================================


def check_linear_gradient(input_dim, output_dim, batch_size=4,
                          atol=1e-3, rtol=1e-3, eps=1e-4):
    """Check gradient for Linear layer.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        batch_size: Batch size for test.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        eps: Epsilon for numerical gradient.

    Returns:
        Dict with gradient check results for weight and bias.
    """
    # Create layer
    layer = fastnn.layers.Linear(input_dim, output_dim, bias=True)
    for p in layer.parameters():
        p.requires_grad_(True)

    # Create random input
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    x = fastnn.tensor(x_np.flatten().tolist(), [batch_size, input_dim])
    x.requires_grad_(True)

    # Forward pass
    y = layer(x)
    y.sum().backward()

    # Get analytical gradients
    params = list(layer.parameters())
    weight_grad_analytical = params[0].grad.numpy().copy()
    bias_grad_analytical = params[1].grad.numpy().copy() if len(params) > 1 else None

    # Numerical gradient for weight
    weight_grad_numerical = np.zeros_like(params[0].numpy())
    w_np = params[0].numpy()
    b_np = params[1].numpy() if len(params) > 1 else None

    for i in range(w_np.size):
        w_plus = w_np.copy()
        w_minus = w_np.copy()
        w_plus.flat[i] += eps
        w_minus.flat[i] -= eps

        # Create layers with modified weights using from_weights
        w_plus_t = fastnn.tensor(w_plus.flatten().tolist(), list(w_plus.shape))
        w_minus_t = fastnn.tensor(w_minus.flatten().tolist(), list(w_minus.shape))

        if b_np is not None:
            b_t = fastnn.tensor(b_np.flatten().tolist(), list(b_np.shape))
            layer_plus = fastnn.layers.Linear.from_weights(w_plus_t, b_t)
            layer_minus = fastnn.layers.Linear.from_weights(w_minus_t, b_t)
        else:
            layer_plus = fastnn.layers.Linear.from_weights(w_plus_t, None)
            layer_minus = fastnn.layers.Linear.from_weights(w_minus_t, None)

        # Forward passes
        x_t = fastnn.tensor(x_np.flatten().tolist(), [batch_size, input_dim])
        y_plus = layer_plus(x_t).sum().numpy().flat[0]
        y_minus = layer_minus(x_t).sum().numpy().flat[0]

        weight_grad_numerical.flat[i] = (y_plus - y_minus) / (2 * eps)

    # Numerical gradient for bias
    bias_grad_numerical = None
    if b_np is not None:
        bias_grad_numerical = np.zeros_like(b_np)
        w_t = fastnn.tensor(w_np.flatten().tolist(), list(w_np.shape))

        for i in range(b_np.size):
            b_plus = b_np.copy()
            b_minus = b_np.copy()
            b_plus.flat[i] += eps
            b_minus.flat[i] -= eps

            b_plus_t = fastnn.tensor(b_plus.flatten().tolist(), list(b_plus.shape))
            b_minus_t = fastnn.tensor(b_minus.flatten().tolist(), list(b_minus.shape))

            layer_plus = fastnn.layers.Linear.from_weights(w_t, b_plus_t)
            layer_minus = fastnn.layers.Linear.from_weights(w_t, b_minus_t)

            x_t = fastnn.tensor(x_np.flatten().tolist(), [batch_size, input_dim])
            y_plus = layer_plus(x_t).sum().numpy().flat[0]
            y_minus = layer_minus(x_t).sum().numpy().flat[0]

            bias_grad_numerical.flat[i] = (y_plus - y_minus) / (2 * eps)

    # Compare
    weight_passed = np.allclose(weight_grad_analytical, weight_grad_numerical,
                                atol=atol, rtol=rtol)
    bias_passed = (np.allclose(bias_grad_analytical, bias_grad_numerical,
                               atol=atol, rtol=rtol)
                   if bias_grad_analytical is not None else True)

    return {
        "weight_passed": weight_passed,
        "bias_passed": bias_passed,
        "weight_analytical": weight_grad_analytical,
        "weight_numerical": weight_grad_numerical,
        "weight_diff": np.abs(weight_grad_analytical - weight_grad_numerical),
        "bias_analytical": bias_grad_analytical,
        "bias_numerical": bias_grad_numerical,
        "bias_diff": (np.abs(bias_grad_analytical - bias_grad_numerical)
                      if bias_grad_analytical is not None else None),
    }


def check_conv2d_gradient(in_channels, out_channels, kernel_size,
                          batch_size=2, input_h=8, input_w=8,
                          stride=1, padding=0, atol=1e-3, rtol=1e-3,
                          eps=1e-4):
    """Check gradient for Conv2d layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        batch_size: Batch size for test.
        input_h: Input height.
        input_w: Input width.
        stride: Stride of convolution.
        padding: Padding of convolution.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        eps: Epsilon for numerical gradient.

    Returns:
        Dict with gradient check results for weight and bias.
    """
    # Create layer
    layer = fastnn.layers.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=True)
    for p in layer.parameters():
        p.requires_grad_(True)

    # Create random input
    x_np = np.random.randn(batch_size, in_channels, input_h, input_w).astype(np.float32)
    x = fastnn.tensor(x_np.flatten().tolist(),
                      [batch_size, in_channels, input_h, input_w])
    x.requires_grad_(True)

    # Forward pass
    y = layer(x)
    y.sum().backward()

    # Get analytical gradients
    params = list(layer.parameters())
    weight_grad_analytical = params[0].grad.numpy().copy()
    bias_grad_analytical = params[1].grad.numpy().copy() if len(params) > 1 else None

    # Numerical gradient for weight (sample a few elements for speed)
    weight_grad_numerical = np.zeros_like(params[0].numpy())
    w_np = params[0].numpy()
    b_np = params[1].numpy() if len(params) > 1 else None

    # Sample a subset of weight elements for efficiency
    max_samples = min(10, w_np.size)
    indices = np.random.choice(w_np.size, max_samples, replace=False)

    for idx in indices:
        w_plus = w_np.copy()
        w_minus = w_np.copy()
        w_plus.flat[idx] += eps
        w_minus.flat[idx] -= eps

        # Create layers with modified weights using from_weights
        w_plus_t = fastnn.tensor(w_plus.flatten().tolist(), list(w_plus.shape))
        w_minus_t = fastnn.tensor(w_minus.flatten().tolist(), list(w_minus.shape))

        if b_np is not None:
            b_t = fastnn.tensor(b_np.flatten().tolist(), list(b_np.shape))
            layer_plus = fastnn.layers.Conv2d.from_weights(w_plus_t, b_t)
            layer_minus = fastnn.layers.Conv2d.from_weights(w_minus_t, b_t)
        else:
            layer_plus = fastnn.layers.Conv2d.from_weights(w_plus_t, None)
            layer_minus = fastnn.layers.Conv2d.from_weights(w_minus_t, None)

        # Forward passes
        x_t = fastnn.tensor(x_np.flatten().tolist(),
                            [batch_size, in_channels, input_h, input_w])
        y_plus = layer_plus(x_t).sum().numpy().flat[0]
        y_minus = layer_minus(x_t).sum().numpy().flat[0]

        weight_grad_numerical.flat[idx] = (y_plus - y_minus) / (2 * eps)

    # Fill in the rest with analytical values for comparison
    for idx in range(w_np.size):
        if idx not in indices:
            weight_grad_numerical.flat[idx] = weight_grad_analytical.flat[idx]

    # Numerical gradient for bias
    bias_grad_numerical = None
    if b_np is not None:
        bias_grad_numerical = np.zeros_like(b_np)

        for i in range(b_np.size):
            b_plus = b_np.copy()
            b_minus = b_np.copy()
            b_plus.flat[i] += eps
            b_minus.flat[i] -= eps

            w_t = fastnn.tensor(w_np.flatten().tolist(), list(w_np.shape))
            b_plus_t = fastnn.tensor(b_plus.flatten().tolist(), list(b_plus.shape))
            b_minus_t = fastnn.tensor(b_minus.flatten().tolist(), list(b_minus.shape))

            layer_plus = fastnn.layers.Conv2d.from_weights(w_t, b_plus_t)
            layer_minus = fastnn.layers.Conv2d.from_weights(w_t, b_minus_t)

            x_t = fastnn.tensor(x_np.flatten().tolist(),
                                [batch_size, in_channels, input_h, input_w])
            y_plus = layer_plus(x_t).sum().numpy().flat[0]
            y_minus = layer_minus(x_t).sum().numpy().flat[0]

            bias_grad_numerical.flat[i] = (y_plus - y_minus) / (2 * eps)

    # Compare
    weight_passed = np.allclose(weight_grad_analytical, weight_grad_numerical,
                                atol=atol, rtol=rtol)
    bias_passed = (np.allclose(bias_grad_analytical, bias_grad_numerical,
                               atol=atol, rtol=rtol)
                   if bias_grad_analytical is not None else True)

    return {
        "weight_passed": weight_passed,
        "bias_passed": bias_passed,
        "weight_analytical": weight_grad_analytical,
        "weight_numerical": weight_grad_numerical,
        "weight_diff": np.abs(weight_grad_analytical - weight_grad_numerical),
        "bias_analytical": bias_grad_analytical,
        "bias_numerical": bias_grad_numerical,
        "bias_diff": (np.abs(bias_grad_analytical - bias_grad_numerical)
                      if bias_grad_analytical is not None else None),
    }


def check_layer_norm_gradient(normalized_shape, batch_size=2,
                              atol=1e-3, rtol=1e-3, eps=1e-4):
    """Check gradient for LayerNorm.

    Args:
        normalized_shape: Shape of input to normalize.
        batch_size: Batch size for test.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        eps: Epsilon for numerical gradient.

    Returns:
        Dict with gradient check results.
    """
    if isinstance(normalized_shape, int):
        input_shape = [batch_size, normalized_shape]
    else:
        input_shape = [batch_size] + list(normalized_shape)

    # Create layer
    layer = fastnn.layers.LayerNorm(normalized_shape)
    for p in layer.parameters():
        p.requires_grad_(True)

    # Create random input
    x_np = np.random.randn(*input_shape).astype(np.float32)
    x = fastnn.tensor(x_np.flatten().tolist(), input_shape)
    x.requires_grad_(True)

    # Forward pass
    y = layer(x)
    y.sum().backward()

    # Get analytical gradients
    params = list(layer.parameters())
    weight_grad_analytical = params[0].grad.numpy().copy()
    bias_grad_analytical = params[1].grad.numpy().copy()

    # Numerical gradient for weight
    weight_grad_numerical = np.zeros_like(params[0].numpy())
    w_np = params[0].numpy()
    b_np = params[1].numpy()

    for i in range(w_np.size):
        w_plus = w_np.copy()
        w_minus = w_np.copy()
        w_plus.flat[i] += eps
        w_minus.flat[i] -= eps

        w_plus_t = fastnn.tensor(w_plus.flatten().tolist(), list(w_plus.shape))
        b_t = fastnn.tensor(b_np.flatten().tolist(), list(b_np.shape))
        layer_plus = fastnn.layers.LayerNorm.from_weights(w_plus_t, b_t)

        w_minus_t = fastnn.tensor(w_minus.flatten().tolist(), list(w_minus.shape))
        layer_minus = fastnn.layers.LayerNorm.from_weights(w_minus_t, b_t)

        x_t = fastnn.tensor(x_np.flatten().tolist(), input_shape)
        y_plus = layer_plus(x_t).sum().numpy().flat[0]
        y_minus = layer_minus(x_t).sum().numpy().flat[0]

        weight_grad_numerical.flat[i] = (y_plus - y_minus) / (2 * eps)

    # Numerical gradient for bias
    bias_grad_numerical = np.zeros_like(b_np)
    w_t = fastnn.tensor(w_np.flatten().tolist(), list(w_np.shape))

    for i in range(b_np.size):
        b_plus = b_np.copy()
        b_minus = b_np.copy()
        b_plus.flat[i] += eps
        b_minus.flat[i] -= eps

        b_plus_t = fastnn.tensor(b_plus.flatten().tolist(), list(b_plus.shape))
        b_minus_t = fastnn.tensor(b_minus.flatten().tolist(), list(b_minus.shape))

        layer_plus = fastnn.layers.LayerNorm.from_weights(w_t, b_plus_t)
        layer_minus = fastnn.layers.LayerNorm.from_weights(w_t, b_minus_t)

        x_t = fastnn.tensor(x_np.flatten().tolist(), input_shape)
        y_plus = layer_plus(x_t).sum().numpy().flat[0]
        y_minus = layer_minus(x_t).sum().numpy().flat[0]

        bias_grad_numerical.flat[i] = (y_plus - y_minus) / (2 * eps)

    # Compare
    weight_passed = np.allclose(weight_grad_analytical, weight_grad_numerical,
                                atol=atol, rtol=rtol)
    bias_passed = np.allclose(bias_grad_analytical, bias_grad_numerical,
                              atol=atol, rtol=rtol)

    return {
        "weight_passed": weight_passed,
        "bias_passed": bias_passed,
        "weight_analytical": weight_grad_analytical,
        "weight_numerical": weight_grad_numerical,
        "weight_diff": np.abs(weight_grad_analytical - weight_grad_numerical),
        "bias_analytical": bias_grad_analytical,
        "bias_numerical": bias_grad_numerical,
        "bias_diff": np.abs(bias_grad_analytical - bias_grad_numerical),
    }


# ============================================================================
# Loss Function Gradient Checking
# ============================================================================


def check_loss_gradient(loss_fn, logits, targets, atol=1e-3, rtol=1e-3,
                        eps=1e-4):
    """Check gradient for a loss function.

    Args:
        loss_fn: Loss function (e.g., fastnn.cross_entropy_loss).
        logits: Logits tensor.
        targets: Target tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        eps: Epsilon for numerical gradient.

    Returns:
        Dict with gradient check results.
    """
    logits_np = logits.numpy().copy()
    targets_np = targets.numpy().copy()

    # Analytical gradient
    logits_t = fastnn.tensor(logits_np.flatten().tolist(),
                             list(logits_np.shape))
    targets_t = fastnn.tensor(targets_np.flatten().tolist(),
                              list(targets_np.shape))
    logits_t.requires_grad_(True)

    loss = loss_fn(logits_t, targets_t)
    loss.backward()

    grad_analytical = logits_t.grad.numpy().copy()

    # Numerical gradient
    grad_numerical = np.zeros_like(logits_np)
    for i in range(logits_np.size):
        logits_plus = logits_np.copy()
        logits_minus = logits_np.copy()
        logits_plus.flat[i] += eps
        logits_minus.flat[i] -= eps

        logits_plus_t = fastnn.tensor(logits_plus.flatten().tolist(),
                                      list(logits_plus.shape))
        logits_minus_t = fastnn.tensor(logits_minus.flatten().tolist(),
                                       list(logits_minus.shape))
        targets_t = fastnn.tensor(targets_np.flatten().tolist(),
                                  list(targets_np.shape))

        loss_plus = loss_fn(logits_plus_t, targets_t).numpy().flat[0]
        loss_minus = loss_fn(logits_minus_t, targets_t).numpy().flat[0]

        grad_numerical.flat[i] = (loss_plus - loss_minus) / (2 * eps)

    # Compare
    passed = np.allclose(grad_analytical, grad_numerical, atol=atol, rtol=rtol)

    return {
        "passed": passed,
        "analytical": grad_analytical,
        "numerical": grad_numerical,
        "diff": np.abs(grad_analytical - grad_numerical),
    }


# ============================================================================
# Training Loop Utilities
# ============================================================================


def training_step(model, x, y, optimizer, loss_fn=fastnn.mse_loss):
    """Execute a single training step.

    Args:
        model: Model to train.
        x: Input tensor.
        y: Target tensor.
        optimizer: Optimizer.
        loss_fn: Loss function (default: mse_loss).

    Returns:
        Loss value.
    """
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_model(model, loader, optimizer, loss_fn=fastnn.mse_loss,
               epochs=1, max_batches=None, return_losses=False):
    """Train a model for multiple epochs.

    Args:
        model: Model to train.
        loader: DataLoader providing (x, y) batches.
        optimizer: Optimizer.
        loss_fn: Loss function (default: mse_loss).
        epochs: Number of epochs to train.
        max_batches: Maximum batches per epoch (None for all).
        return_losses: If True, return list of (epoch, avg_loss) tuples.

    Returns:
        Dict with 'initial_loss' and 'final_loss' keys, or dict with 'losses' if return_losses=True.
    """
    if hasattr(model, 'train_mode'):
        model.train_mode()
    elif hasattr(model, 'train'):
        model.train()
    initial_loss = None
    final_loss = None
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for x_batch, y_batch in loader:
            if max_batches is not None and batch_count >= max_batches:
                break
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == epochs - 1:
            final_loss = avg_loss
        if return_losses:
            losses.append((epoch, avg_loss))

    if return_losses:
        return {"losses": losses}
    return {"initial_loss": initial_loss, "final_loss": final_loss}


def evaluate(model, loader, loss_fn=fastnn.mse_loss):
    """Evaluate model on a data loader.

    Args:
        model: Model to evaluate.
        loader: DataLoader.
        loss_fn: Loss function (default: mse_loss).

    Returns:
        Dict with loss value.
    """
    if hasattr(model, 'eval_mode'):
        model.eval_mode()
    elif hasattr(model, 'eval'):
        model.eval()
    total_loss = 0
    with fastnn.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()
    if hasattr(model, 'train_mode'):
        model.train_mode()
    elif hasattr(model, 'train'):
        model.train()
    return {"loss": total_loss / len(loader)}


# ============================================================================
# Tensor Comparison Utilities
# ============================================================================


def assert_tensor_equal(actual, expected, atol=1e-5, rtol=1e-3):
    """Assert that two tensors are equal within tolerance.

    Args:
        actual: Actual tensor.
        expected: Expected tensor or numpy array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    if not isinstance(expected, np.ndarray):
        expected = expected.numpy()
    actual_np = actual.numpy()
    assert np.allclose(actual_np, expected, atol=atol, rtol=rtol),         f"Tensors not equal:\n  actual: {actual_np}\n  expected: {expected}"


def assert_gradient_correct(op, *inputs, atol=1e-3, rtol=1e-3):
    """Assert that gradients are correct for an operation.

    Args:
        op: Function to test.
        *inputs: Input tensors.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    results = check_gradient(op, *inputs, atol=atol, rtol=rtol)
    for i, passed, details in results:
        assert passed, f"Gradient check failed for input {i}: {details}"


def assert_allclose(actual, expected, atol=1e-5, rtol=1e-3, msg=None):
    """Assert that two arrays/tensors are close within tolerance.

    Args:
        actual: Actual array/tensor.
        expected: Expected array/tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Optional error message.
    """
    if hasattr(actual, 'numpy'):
        actual = actual.numpy()
    if hasattr(expected, 'numpy'):
        expected = expected.numpy()
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        if msg is None:
            msg = f"Arrays not close:\n  actual: {actual}\n  expected: {expected}"
        raise AssertionError(msg)


def assert_shape_equal(tensor, expected_shape, msg=None):
    """Assert that a tensor has the expected shape.

    Args:
        tensor: Tensor to check.
        expected_shape: Expected shape tuple.
        msg: Optional error message.
    """
    actual_shape = tensor.shape
    if actual_shape != expected_shape:
        if msg is None:
            msg = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        raise AssertionError(msg)


def assert_dtype_equal(tensor, expected_dtype, msg=None):
    """Assert that a tensor has the expected dtype.

    Args:
        tensor: Tensor to check.
        expected_dtype: Expected dtype string.
        msg: Optional error message.
    """
    actual_dtype = tensor.dtype
    if actual_dtype != expected_dtype:
        if msg is None:
            msg = f"Dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
        raise AssertionError(msg)


def assert_not_none(value, msg=None):
    """Assert that a value is not None.

    Args:
        value: Value to check.
        msg: Optional error message.
    """
    if value is None:
        if msg is None:
            msg = "Value is None"
        raise AssertionError(msg)


def assert_is_none(value, msg=None):
    """Assert that a value is None.

    Args:
        value: Value to check.
        msg: Optional error message.
    """
    if value is not None:
        if msg is None:
            msg = f"Value is not None: {value}"
        raise AssertionError(msg)


def assert_has_grad(tensor, msg=None):
    """Assert that a tensor has a gradient.

    Args:
        tensor: Tensor to check.
        msg: Optional error message.
    """
    if tensor.grad is None:
        if msg is None:
            msg = "Tensor has no gradient"
        raise AssertionError(msg)


def assert_no_nan(tensor, msg=None):
    """Assert that a tensor contains no NaN values.

    Args:
        tensor: Tensor to check.
        msg: Optional error message.
    """
    if np.any(np.isnan(tensor.numpy())):
        if msg is None:
            msg = "Tensor contains NaN values"
        raise AssertionError(msg)


def assert_no_inf(tensor, msg=None):
    """Assert that a tensor contains no Inf values.

    Args:
        tensor: Tensor to check.
        msg: Optional error message.
    """
    if np.any(np.isinf(tensor.numpy())):
        if msg is None:
            msg = "Tensor contains Inf values"
        raise AssertionError(msg)


def assert_finite(tensor, msg=None):
    """Assert that a tensor contains only finite values.

    Args:
        tensor: Tensor to check.
        msg: Optional error message.
    """
    if not np.all(np.isfinite(tensor.numpy())):
        if msg is None:
            msg = "Tensor contains non-finite values"
        raise AssertionError(msg)


# ============================================================================
# Device/Memory Utilities
# ============================================================================


def get_device(tensor):
    """Get the device of a tensor.

    Args:
        tensor: Tensor to check.

    Returns:
        Device string.
    """
    return tensor.device


def to_device(tensor, device):
    """Move tensor to device (CPU only for now).

    Args:
        tensor: Tensor to move.
        device: Target device.

    Returns:
        Tensor on target device.
    """
    # fastnn tensors are always on CPU
    return tensor


def num_parameters(model):
    """Get total number of parameters in a model.

    Args:
        model: Model to check.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Get number of trainable parameters in a model.

    Args:
        model: Model to check.

    Returns:
        Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.grad is not None)


# ============================================================================
# DataLoader Utilities
# ============================================================================


def create_dataloader(X, y, batch_size=32, shuffle=False):
    """Create a DataLoader from tensors.

    Args:
        X: Input tensor.
        y: Target tensor.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.

    Returns:
        DataLoader instance.
    """
    ds = fastnn.TensorDataset(X, y)
    return fastnn.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def iterate_batches(loader):
    """Iterate through DataLoader batches.

    Args:
        loader: DataLoader to iterate.

    Yields:
        Tuple of (x_batch, y_batch).
    """
    for x_batch, y_batch in loader:
        yield x_batch, y_batch


# ============================================================================
# Optimizer Utilities
# ============================================================================


def get_learning_rate(optimizer):
    """Get the learning rate from an optimizer.

    Args:
        optimizer: Optimizer instance.

    Returns:
        Learning rate or None.
    """
    # Most optimizers store lr in param_groups
    if hasattr(optimizer, 'lr'):
        return optimizer.lr
    elif hasattr(optimizer, 'learning_rate'):
        return optimizer.learning_rate
    return None


def set_learning_rate(optimizer, lr):
    """Set the learning rate for an optimizer.

    Args:
        optimizer: Optimizer instance.
        lr: New learning rate.

    Note:
        This may not work for all optimizers depending on implementation.
    """
    # This may not work for all optimizers depending on implementation
    if hasattr(optimizer, 'lr'):
        optimizer.lr = lr
    elif hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate = lr