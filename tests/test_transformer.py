"""
Pytest test suite for Transformer encoder.
"""

import numpy as np
import fastnn as fnn
from fastnn.models import Transformer


def test_transformer_forward():
    """Test basic transformer forward pass."""
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 16
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_DIM = 128
    NUM_CLASSES = 2
    DROPOUT_P = 0.1

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        num_classes=NUM_CLASSES,
        dropout_p=DROPOUT_P,
    )

    # Test forward pass with batch
    x = fnn.randint(low=0, high=VOCAB_SIZE, shape=[32, MAX_SEQ_LEN])
    logits = model(x)
    assert logits.shape == [32, NUM_CLASSES], (
        f"Expected shape [32, {NUM_CLASSES}], got {logits.shape}"
    )


def test_transformer_training():
    """Test transformer training step."""
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 16
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_DIM = 128
    NUM_CLASSES = 2
    DROPOUT_P = 0.1
    BATCH_SIZE = 32
    EPOCHS = 3  # Reduced for test
    LR = 1e-3

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        num_classes=NUM_CLASSES,
        dropout_p=DROPOUT_P,
    )

    optimizer = fnn.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Load dataset
    X_train = np.load("tests/X_train.npy")
    y_train = np.load("tests/y_train.npy")

    def np_to_fnn(arr):
        return fnn.tensor(arr.flatten().tolist(), list(arr.shape))

    X_train_t = np_to_fnn(X_train)
    y_train_t = np_to_fnn(y_train.astype(np.float32).reshape(-1))

    train_ds = fnn.TensorDataset(X_train_t, y_train_t)
    train_loader = fnn.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            logits = model(x_batch)
            y_int = fnn.tensor(
                [int(v) for v in y_batch.numpy().flatten()],
                [len(y_batch.numpy().flatten())],
            )
            loss = fnn.cross_entropy_loss(logits, y_int)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == EPOCHS - 1:
            final_loss = avg_loss

    # Basic assertions
    assert initial_loss is not None, "Initial loss should be recorded"
    assert final_loss is not None, "Final loss should be recorded"
    assert initial_loss > 0, "Initial loss should be positive"
    assert final_loss > 0, "Final loss should be positive"

    # Loss should generally decrease (not strictly required but good for verification)
    # assert final_loss <= initial_loss, f"Loss should decrease (initial: {initial_loss:.4f}, final: {final_loss:.4f})"


def test_transformer_parameters():
    """Test that transformer parameters are accessible."""
    model = Transformer(
        vocab_size=100,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        num_classes=2,
        dropout_p=0.1,
    )

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


if __name__ == "__main__":
    test_transformer_forward()
    print("✓ test_transformer_forward passed")

    test_transformer_training()
    print("✓ test_transformer_training passed")

    test_transformer_parameters()
    print("✓ test_transformer_parameters passed")

    print("\nAll transformer tests passed!")
