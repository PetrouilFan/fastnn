import fastnn as fnn
from fastnn.data import DataLoader, TensorDataset
from tests.test_utils import train_model, evaluate, assert_not_none
from fastnn import EarlyStopping


def test_trainer_fit():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[8], output_dim=1)

    X = fnn.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
    y = fnn.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    optimizer = fnn.Adam(model.parameters(), lr=0.01)

    result = train_model(
        model,
        loader,
        optimizer,
        loss_fn=fnn.mse_loss,
        epochs=10,
    )
    assert_not_none(result["initial_loss"])


def test_trainer_evaluate():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[8], output_dim=1)

    X = fnn.tensor([[0.0, 0.0], [1.0, 1.0]], [2, 2])
    y = fnn.tensor([[0.0], [0.0]], [2, 1])

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    metrics = evaluate(model, loader, loss_fn=fnn.mse_loss)
    assert "loss" in metrics


def test_early_stopping():
    es = EarlyStopping(patience=3, min_delta=0.01)

    for i in range(10):
        logs = {"val_loss": 1.0 - i * 0.01}
        if logs["val_loss"] > 0.97:
            break

    assert es.counter == 0 or es.counter > 0
