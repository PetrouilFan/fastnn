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
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=False)
    losses = [0.9, 0.7, 0.5, 0.6, 0.7, 0.8]

    for loss in losses:
        logs = {"val_loss": loss}
        es.step(logs)

    assert es.early_stop
    assert es.best_value == 0.5
