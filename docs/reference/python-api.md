# Python API Reference

The 3.0 inference API is `fastnn.fn3`: one versioned artifact, an explicit
device, and sessions that own their KV cache. The 2.x eager tensor/nn forward
facade remains importable as a secondary surface; it has no training names.

## `fastnn.fn3`

```python
import fastnn.fn3 as fn3

fn3.devices() -> list[Device]

fn3.prepare(onnx, artifact, *, max_total=None, max_seq=None, max_past=None,
            overwrite=False) -> PrepareReport
fn3.inspect(artifact) -> ArtifactInfo
fn3.load(artifact, source=None, *, device=None, capacity=None) -> Model

Model.session() -> Session
Session.prefill(tokens) -> Logits          # [batch, 1, vocab]
Session.decode(tokens) -> Logits           # [batch, 1, vocab]
Session.reset() -> None
```

`Device.id / kind / index / name`; `Geometry`, `Envelope`, `Metadata`,
`PrepareReport`, `ArtifactInfo`, `SessionStats` and `Logits` are documented in
`fastnn/fn3/__init__.py` and expose the same fields as the Rust types.

### Example

```python
import fastnn.fn3 as fn3

report = fn3.prepare("model.onnx", "model.fn3prep")
print(report.status, report.size_bytes, report.sha256)

model = fn3.load("model.fn3prep", source="model.onnx", device="cpu")
session = model.session()
logits = session.prefill([464, 2068, 7586, 21831])
print(int(logits.argmax()), session.cache_len)
```

### Devices

`device="cpu"` is always available. `device="vulkan"` requires a build with the
`vulkan` feature and a physical device; otherwise `Fn3DeviceError` names the
request and the available devices. The API never falls back to CPU silently.

### Errors

Every core failure maps to a typed exception (`Fn3DeviceError`,
`Fn3ArtifactError`, `Fn3ImportError`, `Fn3SessionError`, `Fn3UnsupportedError`,
...). No panic crosses the FFI boundary.

## Eager forward surface

`fastnn.tensor`, `fastnn.nn`, `fastnn.io` and `fastnn.AotExecutor` remain for
the 2.x compiled path. They expose forward operations only; `requires_grad`,
`backward`, optimizers, schedulers, losses, data loaders and callbacks were
deleted with the training stack (FN3-19 / #65).

## See also

- [API Reference](api-reference.md) -- forward Python API table
- [Tensors](tensors.md) -- tensor creation and operations
- [ONNX Support](../models/onnx.md) -- ONNX import and quantized inference
- [Index](../index.md) -- full documentation index
