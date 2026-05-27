"""
Pytest test suite for Transformer encoder.
"""

import numpy as np
import fastnn as fnn
from tests.test_utils import (
    make_transformer,
    make_tensor,
    train_model,
    assert_shape_equal,
    assert_has_grad,
)


def test_transformer_forward():
    """Test basic transformer forward pass."""
    model = make_transformer()

    # Test forward pass with batch
    x = fnn.randint(low=0, high=100, shape=[32, 16])
    logits = model(x)
    assert_shape_equal(logits, [32, 2])


def test_transformer_training():
    """Test transformer training step."""
    model = make_transformer()
    optimizer = fnn.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Create self-contained sample data (no external files)
    X_np = np.random.randint(0, 100, (32, 16), dtype=np.int64)
    y_np = np.random.randint(0, 2, (32,), dtype=np.int64)

    X_train_t = make_tensor(X_np)
    y_train_t = make_tensor(y_np)

    train_ds = fnn.TensorDataset(X_train_t, y_train_t)
    train_loader = fnn.DataLoader(train_ds, batch_size=32, shuffle=False)

    # Use train_model helper instead of manual loop
    result = train_model(
        model,
        train_loader,
        optimizer,
        loss_fn=fnn.cross_entropy_loss,
        epochs=1,
        max_batches=10,
    )

    # Basic assertions
    assert result["initial_loss"] is not None, "Initial loss should be recorded"
    assert result["final_loss"] is not None, "Final loss should be recorded"
    assert result["initial_loss"] > 0, "Initial loss should be positive"
    assert result["final_loss"] > 0, f"Final loss should be positive, got {result['final_loss']}"


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
    assert_has_grad(params[0])
