# IO & Serialization

Legacy 2.x model persistence, format conversion, and serialization in
`fastnn.io`. The installed `fastnn-convert` command writes `.fnn` files. For
3.0 decoder artifacts (`.fnn3`), use [`fastnn.inference`](python-api.md) or
the [inference CLI](../quickstart-fn3.md).

## Save / Load

```python
import fastnn as fnn

model = fnn.Linear(4, 2)

# Save the parameters as v2 (the default, all f32)
fnn.io.save(model, "model.fnn")

# load() returns {name: tensor}, not a model
params = fnn.io.load("model.fnn")
print(sorted(params))  # ['bias', 'weight']

restored = fnn.Linear(4, 2)
restored.set_weight(params["weight"])
restored.set_bias(params["bias"])

test_input = fnn.tensor([1.0, 2.0, 3.0, 4.0], [1, 4])
prediction = restored(test_input)
```

**`fnn.io.save(model, path, format="fnn-v2")`** serializes a model's named parameters to the custom `.fnn` binary format. The default v2 stores f32 tensors. `format="fnn-v3"` writes dtype-tagged tensors, but that path currently writes the magic/version prefix twice, so `fnn.io.load` cannot read the result; use `write_fnn_file_v3` and `read_fnn_header` / `read_fnn_parameters` for v3 payloads (the pattern the v3 tests use).

**`fnn.io.load(path)`** returns a `{name: tensor}` dict of parameters; it does not rebuild a model. v1 and v2 files are auto-detected, and a correctly written v3 file loads too. Because the file stores no architecture, construct the model in code and assign the parameters, as the `set_weight` / `set_bias` calls above do. Layer setters are the reliable path on this tree; the top-level `fnn.load_state_dict(model, params)` helper is not usable because it calls `PyTensor.copy_`, which this extension no longer provides.

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

Custom binary with magic bytes (`b"FNN\x00"`).

| Version | Save | Load | Features |
|---------|------|------|----------|
| v1 | -- | Yes | Basic f32 parameters |
| v2 | Yes | Yes | f32 parameters (gradient flag is read for archived files) |
| v3 | Yes | Yes | Dtype-tagged packed tensors (F32, F16, U8, U4, I4Codebook) |

"Save" means a writer exists for the layout: `io.save` writes v1 and v2, and
`write_fnn_file_v3` writes v3. The `format="fnn-v3"` option of `io.save` is the
broken wrapper noted in [Save / Load](#save-load).

### Version 2 (fnn-v2)

All parameters stored as float32, with a per-parameter flag byte kept for compatibility with archived files. Suitable for full-precision inference checkpoints.

### Version 3 (fnn-v3)

Written with `write_fnn_file_v3`. Dtype-tagged tensors: each parameter carries a precision tag (F32=0, F16=1, U8=2, U4=3, I4Codebook=4) with per-channel scale and zero-point arrays. I4Codebook additionally stores the 64-byte codebook and per-block scales. Packed U4/U8/F16 data stored as raw bytes, reducing file size by up to 8x. Enables direct loading of quantized models without re-quantization.

## ONNX Import and Export

The ONNX importer supports standard operators including Conv, BatchNorm, ReLU, GELU, Sigmoid, Tanh, Add, and MatMul. Custom operators can be added via the fastnn op registration API.

For detailed operator coverage and troubleshooting tips:

- [ONNX Support](../models/onnx.md) -- Operator coverage and import guide

## Utilities

```python
stats = fnn.allocator_stats()       # Memory statistics
ops = fnn.list_registered_ops()     # All registered tensor ops
fnn.clear_storage_pool()            # Clear storage pool cache
```

## DLPack

`to_dlpack()` and `from_dlpack()` exist in the Rust backend but are not currently exported to Python.

## See also

- [API Reference](api-reference.md) -- Complete IO function signatures
- [Architecture](../internals/architecture.md) -- AOT compiler pipeline
- [Performance Roadmap](../internals/performance-roadmap.md) -- Packed precision types
- [ONNX Support](../models/onnx.md) -- ONNX operator coverage
- [Index](../index.md) -- Full documentation index
