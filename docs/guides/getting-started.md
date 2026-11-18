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

A forward tensor operation:

```python
import fastnn as fnn

x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
w = fnn.tensor([0.5, 0.0, 0.0, 0.5], [2, 2])

y = fnn.matmul(x, w)
print(y.numpy())
```

## Prepare and run an inference session

```python
import fastnn.fn3 as fn3

report = fn3.prepare("model.onnx", "model.fn3prep")
print(report.status, report.geometry)

model = fn3.load("model.fn3prep", source="model.onnx", device="cpu")
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

- [Tensor Reference](../reference/tensors.md) — tensor creation and operations
- [NN Module Catalog](../reference/nn-modules.md) — available layer types
- [API Reference](../reference/api-reference.md) — full function reference
