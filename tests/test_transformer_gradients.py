"""Numerical regression gates for transformer training paths."""
import pytest
import numpy as np
import fastnn as fnn


def tensor(value, train=False):
    value = np.asarray(value, dtype=np.float32)
    result = fnn.tensor(value, list(value.shape))
    return result.requires_grad_(True) if train else result


def weighted_loss(output, weights):
    return (output * tensor(weights)).reshape([-1]).sum(0, False)


def numerical_gradient(value, loss):
    value = np.asarray(value, dtype=np.float64).copy()
    result = np.zeros_like(value)
    eps = 1e-4
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original + eps
        plus = loss(value)
        value[index] = original - eps
        minus = loss(value)
        value[index] = original
        result[index] = (plus - minus) / (2 * eps)
    return result


def test_layer_norm_gradients_match_numpy():
    values = np.linspace(-1.3, 2.1, 24).reshape(2, 3, 4)
    values[..., 1] += 0.7
    weights = np.sin(np.arange(24)).reshape(values.shape)
    model = fnn.LayerNorm(4, eps=1e-5)
    x = tensor(values, True)
    output = model(x)
    assert output.requires_grad
    weighted_loss(output, weights).backward()
    def normalized(v):
        centered = v - v.mean(axis=-1, keepdims=True)
        return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
    expected = numerical_gradient(values, lambda v: (normalized(v) * weights).sum())
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=2e-4, rtol=2e-3)
    params = dict(model.named_parameters())
    np.testing.assert_allclose(params['weight'].grad.numpy(), (normalized(values) * weights).sum(axis=(0, 1)), atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(params['bias'].grad.numpy(), weights.sum(axis=(0, 1)), atol=2e-4, rtol=2e-3)


def test_views_preserve_exact_gradient_routing():
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    weights = np.arange(8, dtype=np.float32).reshape(2, 4) + 1
    for start in (0, 1):
        x = tensor(values, True)
        output = x.permute([1, 0, 2])[start:start + 1].contiguous().reshape([2, 4])
        weighted_loss(output, weights).backward()
        expected = np.zeros_like(values)
        expected[:, start, :] = weights
        np.testing.assert_array_equal(x.grad.numpy(), expected)


def test_batched_matmul_shared_weight_gradients():
    a = np.linspace(-0.7, 1.1, 12).reshape(2, 2, 3)
    b = np.linspace(-0.4, 0.8, 6).reshape(3, 2)
    weights = np.cos(np.arange(8)).reshape(2, 2, 2)
    x, w = tensor(a, True), tensor(b, True)
    weighted_loss(x @ w, weights).backward()
    da = numerical_gradient(a, lambda v: ((v @ b) * weights).sum())
    db = numerical_gradient(b, lambda v: ((a @ v) * weights).sum())
    np.testing.assert_allclose(x.grad.numpy(), da, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(w.grad.numpy(), db, atol=2e-5, rtol=2e-4)


def test_scaled_attention_gradients_match_numpy():
    values = np.linspace(-0.8, 0.9, 12).reshape(1, 2, 2, 3)
    weights = np.cos(np.arange(12)).reshape(values.shape)
    x = tensor(values, True)
    scores = (x @ x.permute([0, 1, 3, 2])).mul_scalar(0.5)
    output = fnn.softmax(scores, 3) @ x
    weighted_loss(output, weights).backward()
    def loss(v):
        scores = (v @ v.swapaxes(-1, -2)) * 0.5
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        return ((probs @ v) * weights).sum()
    expected = numerical_gradient(values, loss)
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("width", [8, 128])
def test_transformer_all_parameter_groups_receive_finite_gradients(width):
    fnn.set_seed(731)
    model = fnn.models.Transformer(vocab_size=12, max_seq_len=4, d_model=width,
        num_heads=2, num_layers=1, ff_dim=16, num_classes=2, dropout_p=0.0)
    tokens = tensor([[1, 2, 3], [3, 1, 2]])
    logits = model(tokens)
    weighted_loss(logits, [[0.3, -0.7], [-0.2, 0.9]]).backward()
    parameters = list(model.parameters())
    assert len(parameters) > 10
    for index, parameter in enumerate(parameters):
        assert parameter.grad is not None, f"parameter {index} has no gradient"
        gradient = parameter.grad.numpy()
        assert list(gradient.shape) == parameter.shape
        assert np.isfinite(gradient).all(), f"parameter {index} has a nonfinite gradient"
    # Embeddings, the first attention projection, and the classifier all participate.
    for index in (0, 1, 2, len(parameters) - 2):
        assert np.max(np.abs(parameters[index].grad.numpy())) > 1e-8


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [False, True])
def test_mean_backward_preserves_axis_and_singletons(axis, keepdim):
    values = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    x = tensor(values, True)
    output = x.mean(axis, keepdim)
    weights = np.arange(np.prod(output.shape), dtype=np.float32).reshape(output.shape) + 0.3
    weighted_loss(output, weights).backward()
    kept = weights if keepdim else np.expand_dims(weights, axis)
    expected = np.broadcast_to(kept, values.shape) / values.shape[axis]
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-6, rtol=1e-6)
