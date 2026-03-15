"""
Quick test for Transformer encoder - single epoch.
"""

import numpy as np
import fastnn as fnn
from fastnn.models import Transformer
import time

# Config - reduced for quick testing
VOCAB_SIZE = 100
MAX_SEQ_LEN = 16
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
NUM_CLASSES = 2
DROPOUT_P = 0.1
BATCH_SIZE = 32
EPOCHS = 3  # 3 epochs to match test_transformer_training
LR = 1e-3
SEED = 42

fnn.set_seed(SEED)

# Load dataset
X_train = np.load("tests/X_train.npy")
y_train = np.load("tests/y_train.npy")
X_test = np.load("tests/X_test.npy")
y_test = np.load("tests/y_test.npy")


def np_to_fnn(arr):
    return fnn.tensor(arr.flatten().tolist(), list(arr.shape))


X_train_t = np_to_fnn(X_train)
y_train_t = np_to_fnn(y_train.astype(np.float32).reshape(-1))
X_test_t = np_to_fnn(X_test)
y_test_t = np_to_fnn(y_test.astype(np.float32).reshape(-1))

train_ds = fnn.TensorDataset(X_train_t, y_train_t)
test_ds = fnn.TensorDataset(X_test_t, y_test_t)
train_loader = fnn.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

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


def compute_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with fnn.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            logits_np = np.array(logits.numpy())
            preds_np = np.argmax(logits_np, axis=1)
            labels_np = np.array(y_batch.numpy()).flatten().astype(int)
            correct += (preds_np == labels_np).sum()
            total += len(labels_np)
    model.train()
    return correct / total if total > 0 else 0.0


print("=" * 60)
print(f"Quick Transformer Test ({EPOCHS} epochs)")
print(f"  d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
print("=" * 60)

model.train()
epoch_start = time.time()
epoch_losses = []

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
    epoch_losses.append(avg_loss)
    train_acc = compute_accuracy(train_loader, model)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | loss={avg_loss:.4f} | train_acc={train_acc:.4f}"
    )

epoch_time = time.time() - epoch_start
avg_loss = epoch_losses[-1]  # Use loss from last epoch
train_acc = compute_accuracy(train_loader, model)

print("=" * 60)
print("QUICK TEST COMPLETE")
print(f"  Loss: {avg_loss:.4f}")
print(f"  Train accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
print("=" * 60)

# Basic sanity check
assert avg_loss > 0, "Loss should be positive"
assert 0 <= train_acc <= 1, "Accuracy should be between 0 and 1"
print("\n✓ PASS: Basic transformer forward/backward pass successful")
