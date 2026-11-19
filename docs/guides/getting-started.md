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

## Prepare and run an inference session

Start with a supported ONNX decoder export (see the
[pinned example in the CLI quickstart](../quickstart-fn3.md#1-the-model)).

```python
import fastnn.inference as fn3

report = fn3.prepare("model.onnx", "model.fnn3")
print(report.status, report.geometry)

model = fn3.load("model.fnn3", source="model.onnx", device="cpu")
session = model.session()

logits = session.prefill([464, 2068, 7586, 21831])
print(int(logits.argmax()), session.cache_len)

for token in [640, 11, 612]:
    logits = session.decode([token])
    print(int(logits.argmax()), session.cache_len)
```

`prepare` writes a versioned artifact; `load` re-checks it; `session` owns the
KV cache so each decode step only processes the new token. A device request the
build or the machine cannot satisfy raises `Fn3DeviceError` instead of falling
back to CPU.

## Next Steps

- [Python API](../reference/python-api.md) — prepare, load and run sessions
- [CLI Quickstart](../quickstart-fn3.md) — run on CPU or Vulkan with `fastnn-infer`
- [Migration from 2.x](migrate-from-2x.md) — transition from the old command or find the retained eager and AOT APIs
