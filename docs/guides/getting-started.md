# Getting Started

This guide walks through installing fastnn and running your first model.

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) for Python environment management
- Rust stable (for source builds)

### Build from Source

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

This builds the Rust extension via maturin and installs the `fastnn` Python package.

## Your First Program

A tensor operation with autograd:

```python
import fastnn as fnn

x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
w = fnn.tensor([0.5, 0.0, 0.0, 0.5], [2, 2])

y = fnn.matmul(x, w)
print(y.numpy())

# Autograd
z = y.sum()
z.backward()
print(x.grad)
```

## Training an MLP on XOR

```python
import fastnn as fnn

# XOR data
X = fnn.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
y = fnn.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

# Model
model = fnn.Sequential(
    fnn.Linear(2, 16),
    fnn.ReLU(),
    fnn.Linear(16, 1),
)
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

# Training loop
model.train()
for epoch in range(100):
    pred = model(X)
    loss = fnn.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4}")

# Inference
model.eval()
with fnn.no_grad():
    preds = model(X)
    print("Predictions:", preds.numpy().round(2))
```

## Next Steps

- [Training Basics](training/training-basics.md) — deeper dive into training loops
- [Tensor Reference](../reference/tensors.md) — tensor creation and operations
- [NN Module Catalog](../reference/nn-modules.md) — available layer types
- [API Reference](../reference/api-reference.md) — full function reference
