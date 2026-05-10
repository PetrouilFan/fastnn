# Getting Started

## Installation

### Prerequisites

- Python 3.12 or higher
- numpy >= 1.24

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd fastnn

# Install/build the library
uv pip install -e .
```

This will build the Rust extension and install the Python package.

## Quick Start

Here's a complete example of training a MLP on XOR data:

```python
import fastnn as fnn

# Create training data (XOR problem)
X = fnn.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
y = fnn.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

# Build model
model = fnn.models.MLP(
    input_dim=2, 
    hidden_dims=[16, 16], 
    output_dim=1, 
    activation="relu"
)

# Create optimizer
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

# Prepare data
ds = fnn.TensorDataset(X, y)
loader = fnn.DataLoader(ds, batch_size=4, shuffle=True)

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4}")

# Inference
with fnn.no_grad():
    preds = model(X)
    print("Predictions:", preds.numpy().round(2))
```

## Basic Concepts

### Tensors

Tensors are the core data structure in FastNN. Create them from Python lists (flattened data + shape):

```python
# 2D tensor (matrix)
x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])

# 1D tensor (vector)
v = fnn.tensor([1.0, 2.0, 3.0], [3])

# Scalar
s = fnn.tensor([5.0], [1])
```

### Autograd

FastNN automatically tracks operations for gradient computation:

```python
x = fnn.tensor([1.0, 2.0])
y = x * 2
z = y.sum()
z.backward()  # Computes gradients
print(x.grad)  # gradient w.r.t. x
```

### Modules

Neural network layers inherit from `Module` and can be combined using `Sequential`:

```python
model = fnn.Sequential([
    fnn.Linear(10, 64),
    fnn.ReLU(),
    fnn.Linear(64, 32),
    fnn.ReLU(),
    fnn.Linear(32, 1),
])
```

### Training

The typical training loop in FastNN:

1. Forward pass: compute predictions
2. Compute loss
3. Backward pass: compute gradients
4. Optimizer step: update parameters
5. Zero gradients

## Development Commands

```bash
# Build the Rust extension
uv pip install -e .

# Run Rust tests
cargo test

# Run Python tests
uv run pytest tests/ -v

# Run packed precision benchmarks
cargo bench --bench packed_bench
```
