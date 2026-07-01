# IO & Serialization

Model persistence, format conversion, and serialization utilities in `fastnn.io`.

## Save / Load

```python
import fastnn as fnn

# Save model (default fnn-v2, f32 parameters)
fnn.io.save(model, "model.fnn")

# Save as v3 (dtype-tagged packed tensors)
fnn.io.save(model, "model.fnn", format="fnn-v3")

# Load model (auto-detects v2 or v3)
loaded_model = fnn.io.load("model.fnn")
```

**`fnn.io.save(model, path, format="fnn-v2")`** serializes a model to the custom `.fnn` binary format. Use `format="fnn-v3"` for dtype-tagged packed tensor support.

**`fnn.io.load(path)`** deserializes a `.fnn` model file. The format version is auto-detected (v1, v2, or v3). The model architecture on disk must match the architecture in code.

Use the loaded model for inference:

```python
loaded_model.eval()
with fnn.no_grad():
    prediction = loaded_model(test_input)
```

## Format Conversion

### From PyTorch

```python
from fastnn.io import convert_from_pytorch

convert_from_pytorch(torch_model, "model.fnn")
```

Converts PyTorch models to `.fnn` by mapping standard `nn.Module` types.

### From ONNX

```python
from fastnn.io import convert_from_onnx

info = convert_from_onnx("model.onnx", "model.fnn")
# Returns: parameters, input_shape, output_shape
```

Imports an ONNX model and saves it as `.fnn`. Returns metadata.

## Serialization Format

Custom binary with magic bytes (`b"FNN\x00"` for models, `b"FNO\x00"` for optimizers).

| Version | Save | Load | Features |
|---------|------|------|----------|
| v1 | -- | Yes | Basic f32 parameters |
| v2 | Yes | Yes | f32 + optional gradient storage |
| v3 | Yes | Yes | Dtype-tagged packed tensors (F32, F16, U8, U4, I4Codebook) |

### Version 2 (fnn-v2)

All parameters stored as float32 with optional per-parameter gradient storage. Backward compatible with v1. Suitable for full-precision training checkpoints.

### Version 3 (fnn-v3)

Dtype-tagged tensors: each parameter carries a precision tag (F32=0, F16=1, U8=2, U4=3, I4Codebook=4) with per-channel scale and zero-point arrays. I4Codebook additionally stores the 64-byte codebook and per-block scales. Packed U4/U8/F16 data stored as raw bytes, reducing file size by up to 8x. Enables direct loading of quantized models without re-quantization.

## ONNX Import and Export

The ONNX importer supports standard operators including Conv, BatchNorm, ReLU, GELU, Sigmoid, Tanh, Add, and MatMul. Custom operators can be added via the fastnn op registration API.

For detailed operator coverage, troubleshooting tips, and the training export contract:

- [ONNX Support](../models/onnx.md) -- Operator coverage and import guide
- [ONNX Training Export](../models/onnx-training-export.md) -- Training export workflow

## Utilities

```python
stats = fnn.allocator_stats()       # Memory statistics
ops = fnn.list_registered_ops()     # All registered tensor ops
fnn.clear_storage_pool()            # Clear storage pool cache
```

## DLPack

`to_dlpack()` and `from_dlpack()` exist in the Rust backend but are not currently exported to Python.

## See also

- [Optimizers](optimizers.md) -- Saving optimizer state with model checkpoints
- [API Reference](api-reference.md) -- Complete IO function signatures
- [Architecture](../internals/architecture.md) -- AOT compiler pipeline
- [Performance Roadmap](../internals/performance-roadmap.md) -- Packed precision types
- [ONNX Support](../models/onnx.md) -- ONNX operator coverage
- [Index](../index.md) -- Full documentation index
