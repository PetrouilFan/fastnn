# IO & Serialization

FastNN uses a custom binary format (.fnn for models, .fno for optimizers).

## Unified API (fastnn.io)

The `fastnn.io` module provides a unified interface for saving/loading models and converting from other formats.

```python
import fastnn as fnn

# Save/Load models (custom binary format)
fnn.io.save(model, "model.fnn")        # Save model
loaded_model = fnn.io.load("model.fnn")       # Load model

# Convert from other formats
fnn.io.convert_from_pytorch(torch_model, "model.fnn")
info = fnn.io.convert_from_onnx("model.onnx", "model.fnn")
```

### Save Model

```python
import fastnn as fnn

# Build and train model
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
# ... training code ...

# Save model (custom binary format)
fnn.io.save(model, "model.fnn")
```

### Load Model

```python
import fastnn as fnn

# Create model architecture (must match saved architecture)
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Load weights
loaded_model = fnn.io.load("model.fnn")

# Use for inference
loaded_model.eval()
with fnn.no_grad():
    prediction = loaded_model(test_input)
```

### Convert from PyTorch

```python
import torch
import torch.nn as nn
from fastnn.io import convert_from_pytorch

# Create PyTorch model
torch_model = nn.Linear(784, 10)

# Convert to fastnn format
convert_from_pytorch(torch_model, "model.fnn")
```

### Convert from ONNX

```python
from fastnn.io import convert_from_onnx

# Import ONNX model and save as fastnn format
info = convert_from_onnx("model.onnx", "model.fnn")
print(f"Imported {info['parameters']} parameters")
print(f"Input shape: {info['input_shape']}")
print(f"Output shape: {info['output_shape']}")
```


## Serialization Format

FastNN uses a custom binary format with:
- Magic bytes header (`b"FNN\x00"` for models, `b"FNO\x00"` for optimizers)
- Version number (currently version 2 for models)
- Tensor data stored as float32 with shape information

## DLPack Interop

`to_dlpack()` / `from_dlpack()` exist in the Rust backend but are **NOT exported** to Python.
The documentation previously referenced these functions, but they are not currently available from Python.

## Allocator Statistics

Monitor memory usage:

```python
# Get allocator statistics
stats = fnn.allocator_stats()
print(stats)
```

This returns information about:
- Total allocated memory
- Number of allocations
- Cache statistics

## Registered Operations

List all registered operations:

```python
ops = fnn.list_registered_ops()
print(ops)
```

This shows all tensor operations available in the library.
