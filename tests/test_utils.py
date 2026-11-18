"""Shared inference-test helpers.

These helpers create tensors and models for forward-path tests. Training,
gradient and optimizer helpers were deleted with the training stack (#65).
"""

import numpy as np
import fastnn


def make_tensor(data, shape=None):
    """Create a fastnn tensor from a list or NumPy array."""
    if isinstance(data, np.ndarray):
        flat = data.flatten().tolist()
        shape = list(data.shape) if shape is None else shape
    else:
        flat = data
    return fastnn.tensor(flat, shape if shape is not None else [])


def make_transformer(vocab_size=100, max_seq_len=16, d_model=64,
                     num_heads=4, num_layers=2, ff_dim=128,
                     num_classes=2, dropout_p=0.1):
    """Create a small Transformer encoder for forward-path tests."""
    return fastnn.models.Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout_p=dropout_p,
    )


def assert_shape_equal(tensor, expected_shape, msg=None):
    """Assert a tensor's shape equals the expected list/tuple."""
    actual = list(tensor.shape)
    expected = list(expected_shape)
    assert actual == expected, msg or f"shape {actual} != expected {expected}"


def assert_not_none(value, msg=None):
    assert value is not None, msg or "value is None"


def assert_allclose(actual, expected, atol=1e-5, rtol=1e-3, msg=None):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol, err_msg=msg or ""
    )


def assert_no_nan(tensor, msg=None):
    data = np.asarray(tensor.numpy())
    assert not np.isnan(data).any(), msg or "tensor contains NaN"


def assert_finite(tensor, msg=None):
    data = np.asarray(tensor.numpy())
    assert np.isfinite(data).all(), msg or "tensor contains a non-finite value"


def num_parameters(model):
    """Count parameter elements in a model."""
    return sum(int(np.prod(list(p.shape))) for p in model.parameters())
