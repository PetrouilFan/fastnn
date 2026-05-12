# IO & Serialization

FastNN uses a custom binary format (.fnn for models, .fno for optimizers).

## Unified API (fastnn.io)

The `fastnn.io` module provides a unified interface for saving/loading models and converting from other formats.

```python
import fastnn as fnn

# Save/Load models (custom binary format)
fnn.io.save(model, "model.fnn")             # Save model (default: fnn-v2)
fnn.io.save(model, "model.fnn", format="fnn-v3")  # Save as v3 (dtype-tagged)
loaded_model = fnn.io.load("model.fnn")       # Load model (auto-detects version)

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

# Save model (fnn-v2 — all f32 parameters)
fnn.io.save(model, "model.fnn")

# Save model (fnn-v3 — dtype-tagged, supports packed tensors)
fnn.io.save(model, "model.fnn", format="fnn-v3")
```

### Load Model

```python
import fastnn as fnn

# Create model architecture (must match saved architecture)
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Load weights (auto-detects v2 or v3 format)
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
- Version number (currently **v2** for f32-only, **v3** for dtype-tagged packed tensors)

### Version 2 (fnn-v2)
- All parameters stored as float32
- Optional gradient storage per parameter
- Backward compatible with v1 loaders

### Version 3 (fnn-v3)
- Dtype-tagged tensors: each parameter stores its precision tag (F32=0, F16=1, U8=2, U4=3)
- Per-channel scale and zero-point arrays for quantized parameters
- Compact binary: packed U4/U8/F16 data stored as raw bytes, not expanded f32
- Enables direct loading of quantized models without re-quantization

### Format support

| Version | Save | Load | Features |
|---------|------|------|----------|
| v1      | —    | ✓    | Basic f32 parameters |
| v2      | ✓    | ✓    | f32 + optional gradients |
| v3      | ✓    | ✓    | Dtype-tagged packed tensors |

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
