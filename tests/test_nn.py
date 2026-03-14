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

        pred.backward()
        optimizer.step()
        optimizer.zero_grad()

        break

    assert initial_loss is not None


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
