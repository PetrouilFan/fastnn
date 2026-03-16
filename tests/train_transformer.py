"""
Train TransformerEncoder on synthetic sentiment classification dataset.
Uses fastnn only (no PyTorch/TensorFlow).
"""

import numpy as np
import fastnn as fnn
from fastnn.models import Transformer
import time

# ── Config ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 100
MAX_SEQ_LEN = 16
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
NUM_CLASSES = 2
DROPOUT_P = 0.1
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
SEED = 42
# ────────────────────────────────────────────────────────────────────────────

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
test_loader = fnn.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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
            # argmax is broken, compute manually
            logits_np = np.array(logits.numpy())
            preds_np = np.argmax(logits_np, axis=1)
            labels_np = np.array(y_batch.numpy()).flatten().astype(int)
            correct += (preds_np == labels_np).sum()
            total += len(labels_np)
    model.train()
    return correct / total if total > 0 else 0.0


print("=" * 60)
print("FastNN Transformer Training")
print(f"  d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
print(f"  ff_dim={FF_DIM}, dropout={DROPOUT_P}, lr={LR}")
print(f"  train_size={len(X_train)}, test_size={len(X_test)}")
print("=" * 60)

model.train()
best_acc = 0.0
train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    epoch_start = time.time()

    for x_batch, y_batch in train_loader:
        logits = model(x_batch)

        # Convert integer labels to long for cross entropy
        y_int = fnn.tensor(
            [int(v) for v in y_batch.numpy().flatten()],
            [len(y_batch.numpy().flatten())],
        )

        loss = fnn.cross_entropy_loss(logits, y_int)
        # DEBUG: Print loss value
        # print(f"Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start

    if epoch % 5 == 0 or epoch == 1:
        train_acc = compute_accuracy(train_loader, model)
        test_acc = compute_accuracy(test_loader, model)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
            f"time={epoch_time:.2f}s"
        )

total_time = time.time() - train_start

# ── Final evaluation ────────────────────────────────────────────────────────
model.eval()
final_train_acc = compute_accuracy(train_loader, model)
final_test_acc = compute_accuracy(test_loader, model)

print("=" * 60)
print("TRAINING COMPLETE")
print(f"  Total training time : {total_time:.1f}s")
print(f"  Final train accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
print(f"  Final test accuracy : {final_test_acc:.4f}  ({final_test_acc * 100:.2f}%)")
print(f"  Best test accuracy  : {best_acc:.4f}  ({best_acc * 100:.2f}%)")
print("=" * 60)

# Save model
fnn.save_model(model._model, "tests/transformer_checkpoint.safetensors")
print("Model saved to tests/transformer_checkpoint.safetensors")

# ── Sanity check: single sample inference ──────────────────────────────────
model.eval()
sample_input = fnn.tensor(X_test.tolist(), [1, MAX_SEQ_LEN])
with fnn.no_grad():
    sample_logits = model(sample_input)
sample_pred = int(np.argmax(sample_logits.numpy()))
sample_true = int(y_test[0])
print(
    f"\nSample inference: pred={sample_pred}, true={sample_true}, "
    f"{'CORRECT' if sample_pred == sample_true else 'WRONG'}"
)

assert final_test_acc > 0.70, (
    f"FAIL: Test accuracy {final_test_acc:.4f} is below 70% threshold. "
    f"Check attention implementation and learning rate."
)
print("\n✓ PASS: Test accuracy exceeds 70% threshold.")
