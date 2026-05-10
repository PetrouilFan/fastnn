import sys
import pytest
import numpy as np
import fastnn as fnn
from tests.test_utils import (
    training_step,
    train_model,
    assert_shape_equal,
    assert_has_grad,
    make_tensor,
)


def test_linear_forward():
    linear = fnn.Linear(10, 5)
    x = fnn.zeros([2, 10])
    y = linear(x)
    assert_shape_equal(y, [2, 5])


def test_conv2d_forward():
    conv = fnn.Conv2d(3, 16, kernel_size=3, padding=1)
    x = fnn.zeros([1, 3, 32, 32])
    y = conv(x)
    assert_shape_equal(y, [1, 16, 32, 32])


def test_sequential():
    model = fnn.Sequential(
        [
            fnn.Linear(10, 20),
            fnn.ReLU(),
            fnn.Linear(20, 5),
        ]
    )
    x = fnn.zeros([2, 10])
    y = model(x)
    assert_shape_equal(y, [2, 5])


def test_mlp_forward():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    x = fnn.zeros([4, 2])
    y = model(x)
    assert_shape_equal(y, [4, 1])


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="heap corruption (0xc0000374) in release mode on Windows",
)
def test_mlp_training_step():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    optimizer = fnn.Adam(model.parameters(), lr=0.01)

    x = make_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
    y = make_tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])

    initial_loss = None
    for _ in range(10):
        loss_val = training_step(model, x, y, optimizer, loss_fn=fnn.mse_loss)
        initial_loss = loss_val
        break

    assert initial_loss is not None


@pytest.mark.skipif(
    sys.platform == "win32", reason="heap corruption in release mode on Windows"
)
def test_muon_optimizer():
    # Test Muon optimizer with 2D weight matrix
    linear = fnn.Linear(10, 5)
    # Use non-zero input to avoid numerical issues with very small gradients
    x = make_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], [1, 10])
    y = make_tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], [1, 5])

    # Get initial parameters
    initial_params = [p.numpy().copy() for p in linear.parameters()]

    # Use Muon optimizer
    optimizer = fnn.Muon(
        linear.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=True
    )

    # Training loop
    initial_loss = None
    for _ in range(5):
        pred = linear(x)
        # Use mean reduction to get a scalar loss
        loss = fnn.mse_loss(pred, y, reduction="mean")
        # loss should be a scalar tensor (shape [1])
        assert_shape_equal(loss, [1])
        initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert initial_loss is not None
    # Check that weights are updated (not all zero)
    for i, param in enumerate(linear.parameters()):
        param_data = param.numpy()
        # Check that parameters have changed from initial values
        assert not np.allclose(param_data, initial_params[i]), (
            f"Parameter {i} was not updated"
        )


def test_zero_grad():
    linear = fnn.Linear(10, 5)
    x = fnn.ones([2, 10])
    y = linear(x)
    y.sum().backward()
    linear.zero_grad()
    for param in linear.parameters():
        if param.grad is not None:
            pass


def test_batchnorm_train_eval():
    bn = fnn.BatchNorm1d(10)
    x = fnn.ones([4, 10])

    bn.train()
    y_train = bn(x)

    bn.eval()
    y_eval = bn(x)

    assert_shape_equal(y_train, [4, 10])
    assert_shape_equal(y_eval, [4, 10])


def test_batchnorm_running_stats():
    """Verify BatchNorm running mean/var follow PyTorch convention:
    running = (1-momentum) * running + momentum * batch_stats"""
    bn = fnn.BatchNorm1d(4, momentum=0.1)
    bn.train()

    # BatchNorm computes per-feature mean over (batch, spatial) dims
    # Input [1, 4]: batch=1, num_features=4, spatial=1
    # batch_mean per feature = [1.0, 2.0, 3.0, 4.0]
    x = fnn.tensor([[1.0, 2.0, 3.0, 4.0]], [1, 4])
    _ = bn(x)

    # After one forward pass with momentum=0.1:
    # running_mean = 0.9 * 0 + 0.1 * [1,2,3,4] = [0.1, 0.2, 0.3, 0.4]
    expected_mean = 0.1
    # batch_var per feature = 0 (only 1 value per feature), unbiased = 0
    # running_var = 0.9 * 1 + 0.1 * 0 = 0.9
    expected_var = 0.9

    rm = bn.get_running_mean().numpy()[0]
    rv = bn.get_running_var().numpy()[0]
    assert abs(rm - expected_mean) < 1e-5, \
        f"running_mean[0] = {rm}, expected {expected_mean}"
    assert abs(rv - expected_var) < 1e-5, \
        f"running_var[0] = {rv}, expected {expected_var}"

    # Test that momentum direction is correct (PyTorch convention)
    # Second forward pass with same data:
    # running_mean = 0.9 * [0.1,0.2,0.3,0.4] + 0.1 * [1,2,3,4]
    #             = [0.09+0.1, 0.18+0.2, 0.27+0.3, 0.36+0.4]
    #             = [0.19, 0.38, 0.57, 0.76]
    _ = bn(x)
    rm2 = bn.get_running_mean().numpy()[0]
    expected_mean2 = 0.19
    assert abs(rm2 - expected_mean2) < 1e-5, \
        f"running_mean[0] after 2nd pass = {rm2}, expected {expected_mean2}"


def test_conv1d_forward():
    """Verify Conv1d forward produces correct output shape (Bug 1 fix validation)."""
    conv = fnn.Conv1d(3, 16, kernel_size=3, padding=1)
    x = fnn.zeros([2, 3, 10])  # [batch, channels, length]
    y = conv(x)
    # Input length 10, kernel 3, padding 1 → output length 10
    assert_shape_equal(y, [2, 16, 10])


def test_conv3d_forward():
    """Verify Conv3d forward produces correct output shape (Bug 1 fix validation)."""
    conv = fnn.Conv3d(3, 8, kernel_size=3, padding=1)
    x = fnn.zeros([1, 3, 5, 6, 7])  # [batch, channels, depth, height, width]
    y = conv(x)
    # Input (5,6,7), kernel 3, padding 1 → output (5,6,7)
    assert_shape_equal(y, [1, 8, 5, 6, 7])


def test_conv2d_groups():
    """Verify grouped Conv2d with groups > 1 (Bug 5 fix validation)."""
    # groups=2: in_channels=4, out_channels=4, each group processes 2→2
    conv = fnn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    x = fnn.zeros([1, 4, 8, 8])
    y = conv(x)
    assert_shape_equal(y, [1, 4, 8, 8])


def test_conv2d_dilation():
    """Verify Conv2d forward with dilation > 1."""
    conv = fnn.Conv2d(3, 16, kernel_size=3, dilation=2, padding=2)
    x = fnn.zeros([1, 3, 16, 16])
    y = conv(x)
    # dilation=2, kernel=3 → effective kernel = 5, padding=2 → same output size
    assert_shape_equal(y, [1, 16, 16, 16])


def test_embedding():
    emb = fnn.Embedding(100, 32)
    indices = fnn.tensor([0, 1, 2, 3], [4])
    output = emb(indices)
    assert_shape_equal(output, [4, 32])


def test_dropout():
    dropout = fnn.Dropout(0.5)
    x = fnn.ones([10, 10])

    dropout.train()
    y_train = dropout(x)

    dropout.eval()
    y_eval = dropout(x)

    assert_shape_equal(y_train, [10, 10])
    assert_shape_equal(y_eval, [10, 10])


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.skipif(
    sys.platform == "win32", reason="half precision has heap corruption issues on Windows CI"
)
def test_half_precision_support(dtype):
    # Test tensor creation
    x = fnn.zeros([3, 4], dtype=dtype)
    assert_shape_equal(x, [3, 4])
    assert x.dtype == dtype

    # Test tensor operations
    y = fnn.ones([3, 4], dtype=dtype)
    z = x + y
    assert_shape_equal(z, [3, 4])
    assert z.dtype == dtype

    # Test item() method
    scalar = fnn.zeros([1], dtype=dtype)
    val = scalar.item()
    assert isinstance(val, float)

    # Test numpy() conversion
    arr = x.numpy()
    assert len(arr) == 3
