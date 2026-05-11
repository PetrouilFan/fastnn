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

### Packed Precision (Quantized Inference)

FastNN stores weights in packed u32 words for 4×–8× memory savings and up to 7.4× GEMV speedup over PyTorch f32:

```python
import fastnn as fnn

# Create packed linear layers (4-bit, 8-bit, 16-bit, or 32-bit)
layer4 = fnn.Linear4(128, 64)   # 4-bit packed weights
layer8 = fnn.Linear8(128, 64)   # 8-bit packed weights

# Create packed tensors from f32 data
pt = fnn.packed_tensor_from_f32([0.5, -1.2, 3.7, ...], shape=[64, 128], dtype="u4")
```

### ONNX Model Import

Load and run models from PyTorch or other frameworks:

```python
import fastnn as fnn

# Import ONNX model
info = fnn.convert_from_onnx("model.onnx", "model.fnn")

# Build and run
model = fnn.build_model_from_fnn("model.fnn")
output = model(input_tensor)

# Or use the YOLO wrapper directly
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")
```

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
