# Getting Started

## Installation

### Prerequisites

- Python 3.11 or higher
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
model = fnn.Sequential(
    fnn.Linear(2, 16),
    fnn.ReLU(),
    fnn.Linear(16, 1),
)

# Create optimizer
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

# Training loop
model.train()
for epoch in range(100):
    pred = model(X)
    loss = fnn.mse_loss(pred, y)
    
    # Backward pass
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

### Quantized Inference (AOT Compiler Pass)

Weight quantization is handled by the AOT compiler as a compile-time pass — not through separate layer types. Enable 4-bit or 8-bit quantization when compiling an ONNX model:

```python
from fastnn.io import AotExecutor

executor = AotExecutor(nodes, params, input_names, output_names, quantize=4)
output = executor.forward({"input": tensor})
```

This replaces eligible `MatMul` and `Conv2d` weight nodes with packed U4/U8 variants carrying per-channel scale and zero-point metadata. Dequantization is fused into the GEMM/conv kernels.

### ONNX Model Import

Load and run models from PyTorch or other frameworks:

```python
import fastnn as fnn

# Import ONNX model
info = fnn.convert_from_onnx("model.onnx", "model.fnn")

# Build and run (auto-detects model provenance)
model = fnn.build_model_from_fnn("model.fnn")
output = model(input_tensor)

# Or use the YOLO wrapper directly for object detection
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
