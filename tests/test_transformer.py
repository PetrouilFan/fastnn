"""
Pytest test suite for Transformer encoder.
"""

import numpy as np
import fastnn as fnn
from tests.test_utils import make_transformer


def test_transformer_forward():
    """Test basic transformer forward pass."""
    model = make_transformer()

    # Test forward pass with batch
    x = fnn.randint(low=0, high=100, shape=[32, 16])
    logits = model(x)
    assert logits.shape == [32, 2], (
        f"Expected shape [32, 2], got {logits.shape}"
    )


def test_transformer_training():
    """Test transformer training step."""
    model = make_transformer()
    optimizer = fnn.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Load dataset
    X_train = np.load("tests/X_train.npy")
    y_train = np.load("tests/y_train.npy")

    def np_to_fnn(arr):
        return fnn.tensor(arr.flatten().tolist(), list(arr.shape))

    X_train_t = np_to_fnn(X_train)
    y_train_t = np_to_fnn(y_train.astype(np.float32).reshape(-1))

    train_ds = fnn.TensorDataset(X_train_t, y_train_t)
    train_loader = fnn.DataLoader(train_ds, batch_size=32, shuffle=False)

    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(1):
        epoch_loss = 0.0
        batch_count = 0
        for x_batch, y_batch in train_loader:
            if batch_count >= 10:
                break
            logits = model(x_batch)
            y_int = fnn.tensor(
                [int(v) for v in y_batch.numpy().flatten()],
                [len(y_batch.numpy().flatten())],
            )
            loss = fnn.cross_entropy_loss(logits, y_int)

            loss_val = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss_val
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 0:  # Only 1 epoch
            final_loss = avg_loss

    # Basic assertions
    assert initial_loss is not None, "Initial loss should be recorded"
    assert final_loss is not None, "Final loss should be recorded"
    assert initial_loss > 0, "Initial loss should be positive"
    assert final_loss > 0, f"Final loss should be positive, got {final_loss}"


def test_transformer_parameters():
    """Test that transformer parameters are accessible."""
    model = make_transformer()

    params = list(model.parameters())
    assert len(params) > 0, "Transformer should have parameters"

    # Check that parameters have gradients using a more standard loss
    x = fnn.randint(low=0, high=100, shape=[4, 16])
    logits = model(x)

    # Create targets for cross-entropy loss
    targets = fnn.randint(low=0, high=2, shape=[4])
    loss = fnn.cross_entropy_loss(logits, targets)
    loss.backward()

    # Check that at least some parameters have gradients
    params_with_grad = [p for p in params if p.grad is not None]
    assert len(params_with_grad) > 0, (
        "At least some parameters should have gradients after backward pass"
    )
