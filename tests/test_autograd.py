import numpy as np
import fastnn as fnn


def test_relu_grad():
    x = fnn.tensor([1.0, -2.0, 3.0], [3])
    x.requires_grad_(True)
    y = fnn.relu(x)
    y.backward()
    assert x.grad is not None


def test_sigmoid_grad():
    x = fnn.tensor([0.5, -0.5], [2])
    x.requires_grad_(True)
    y = fnn.sigmoid(x)
    y.backward()
    assert x.grad is not None


def test_matmul_grad():
    a = fnn.tensor([[1.0, 2.0], [3.0, 4.0]], [2, 2])
    b = fnn.tensor([[5.0, 6.0], [7.0, 8.0]], [2, 2])
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a @ b
    c.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_no_grad_context():
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    with fnn.no_grad():
        y = x * 2
    assert y.grad_fn is None


def test_detach():
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = x * 2
    z = y.detach()
    assert z.grad_fn is None


def test_sum_grad():
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = x.sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), [1.0, 1.0, 1.0])


def test_mean_grad():
    x = fnn.tensor([1.0, 2.0, 3.0], [3])
    x.requires_grad_(True)
    y = x.mean()
    y.backward()
    assert x.grad is not None


def test_cross_entropy_grad():
    logits = fnn.tensor([[2.0, 1.0, 0.1]], [1, 3])
    targets = fnn.tensor([0], [1])
    logits.requires_grad_(True)
    loss = fnn.cross_entropy_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
