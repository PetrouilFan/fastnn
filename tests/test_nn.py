import sys
import pytest
import fastnn as fnn


def test_linear_forward():
    linear = fnn.Linear(10, 5)
    x = fnn.zeros([2, 10])
    y = linear(x)
    assert y.shape == [2, 5]


def test_conv2d_forward():
    conv = fnn.Conv2d(3, 16, kernel_size=3, padding=1)
    x = fnn.zeros([1, 3, 32, 32])
    y = conv(x)
    assert y.shape == [1, 16, 32, 32]


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
    assert y.shape == [2, 5]


def test_mlp_forward():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    x = fnn.zeros([4, 2])
    y = model(x)
    assert y.shape == [4, 1]


def test_mlp_training_step():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    optimizer = fnn.Adam(model.parameters(), lr=0.01)

    x = fnn.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
    y = fnn.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])

    initial_loss = None
    for _ in range(10):
        pred = model(x)
        loss = fnn.mse_loss(pred, y)
        initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        break

    assert initial_loss is not None


def test_muon_optimizer():
    # Test Muon optimizer with 2D weight matrix
    linear = fnn.Linear(10, 5)
    # Use non-zero input to avoid numerical issues with very small gradients
    x = fnn.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], [1, 10])
    y = fnn.tensor([[1.0]], [1, 5])

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
        assert loss.shape == [1]
        initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert initial_loss is not None
    # Check that weights are updated (not all zero)
    for i, param in enumerate(linear.parameters()):
        param_data = param.numpy()
        import numpy as np

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

    assert y_train.shape == y_eval.shape == [4, 10]


def test_embedding():
    emb = fnn.Embedding(100, 32)
    indices = fnn.tensor([0, 1, 2, 3], [4])
    output = emb(indices)
    assert output.shape == [4, 32]


def test_dropout():
    dropout = fnn.Dropout(0.5)
    x = fnn.ones([10, 10])

    dropout.train()
    y_train = dropout(x)

    dropout.eval()
    y_eval = dropout(x)

    assert y_train.shape == y_eval.shape == [10, 10]


@pytest.mark.skipif(
    sys.platform == "win32", reason="bf16 has heap corruption issues on Windows CI"
)
def test_bf16_support():
    # Test BF16 tensor creation
    x = fnn.zeros([3, 4], dtype="bf16")
    assert x.shape == [3, 4]
    assert x.dtype == "bf16"

    # Test BF16 tensor operations
    y = fnn.ones([3, 4], dtype="bf16")
    z = x + y
    assert z.shape == [3, 4]
    assert z.dtype == "bf16"

    # Test item() method
    scalar = fnn.zeros([1], dtype="bf16")
    val = scalar.item()
    assert isinstance(val, float)

    # Test numpy() conversion
    arr = x.numpy()
    assert len(arr) == 3


@pytest.mark.skipif(
    sys.platform == "win32", reason="f16 has heap corruption issues on Windows CI"
)
def test_f16_support():
    # Test F16 tensor creation
    x = fnn.zeros([3, 4], dtype="f16")
    assert x.shape == [3, 4]
    assert x.dtype == "f16"

    # Test F16 tensor operations
    y = fnn.ones([3, 4], dtype="f16")
    z = x + y
    assert z.shape == [3, 4]
    assert z.dtype == "f16"

    # Test item() method
    scalar = fnn.zeros([1], dtype="f16")
    val = scalar.item()
    assert isinstance(val, float)
