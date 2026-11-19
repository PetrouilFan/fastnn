# fastnn

[![CI](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml/badge.svg)](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PetrouilFan/fastnn/branch/main/graph/badge.svg)](https://codecov.io/gh/PetrouilFan/fastnn)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)
[![Downloads](https://img.shields.io/github/downloads/PetrouilFan/fastnn/total)](https://github.com/PetrouilFan/fastnn)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Inference-only neural networks in Rust, with a Python binding and a Vulkan
compute backend.**

fastnn 3.0 imports an ONNX decoder, compiles it into a versioned prepared-model
artifact, plans it for a bounded shape envelope, and runs prefill plus cached
decode on a CPU reference backend or on a physical Vulkan device. It is a
deletion-first rewrite: there is no autograd, no optimizer, no training loop and
no wgpu backend in this tree.

> ⚠️ Alpha. APIs, the prepared-model format and performance may change before
> the stable release.

---

## Highlights

- **One inference path** — `PreparedModel` is immutable and shareable; each
  `Session` owns its KV cache, step counter and plan cache. CPU and Vulkan run
  the same plan through the same backend trait.
- **No silent fallback** — an explicit device request either resolves to that
  device or fails naming the request and the available devices. A backend that
  cannot run an op returns `UnsupportedOp`; it never returns an approximation.
- **Strict ONNX import** — a bounds-checked protobuf reader maps the pinned
  GPT-2 decoder to a bounded graph, folds integer shape arithmetic at plan
  time, and rejects unsupported ops by name.
- **Versioned artifact** — the `.fnn3` payload carries a format version,
  dtype codes, the value/node tables and weight bytes with CRC and SHA-256
  verification.
- **CPU + Vulkan** — a portable FP32 reference backend, plus Vulkan transform,
  embedding, pointwise, matmul, norm, reduce and fused-attention kernels with
  validation-layer-clean dispatch.
- **Rust, CLI and Python share one engine** — `fastnn-infer` and `fastnn.inference`
  call the same `Session::prefill` / `Session::decode`.

## Installation

**From source** (currently recommended):

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

**Prerequisites:** Rust stable, Python ≥ 3.11, [uv](https://github.com/astral-sh/uv).

The Python package is built with `maturin` and exposes the Rust extension as
`fastnn._core`.

## Quick Start

### Python: prepare once, then decode

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

Pass `device="vulkan"` on a build with `--features vulkan` to run the same
session on a physical GPU. A request the build or the machine cannot satisfy
raises `Fn3DeviceError`; it never falls back to CPU.

### CLI

```bash
cargo build --release --no-default-features --features cli --bin fastnn-infer
./target/release/fastnn-infer prepare --onnx model.onnx --out model.fnn3
./target/release/fastnn-infer inspect model.fnn3
./target/release/fastnn-infer run --model model.fnn3 --onnx model.onnx --tokens 464,2068,7586
./target/release/fastnn-infer generate --model model.fnn3 --onnx model.onnx --tokens 464,2068 --steps 4
```

### Rust

```rust
use fastnn::inference::model::PreparedModel;
use fastnn::inference::DeviceRequest;

let request = DeviceRequest::parse("cpu")?;
let model = PreparedModel::load("model.fnn3", None, request)?;
let mut session = model.session()?;
let logits = session.prefill(&[464, 2068, 7586, 21831])?;
```

See `src/fn3/AGENTS.md` and `docs/roadmap/fn3-architecture.md` for the contract.

## Project Components

| Component | Description |
|-----------|-------------|
| 3.0 inference core | `src/fn3/`: contracts, storage, CPU reference, Vulkan platform and kernels, ONNX import, prepared artifact, planner, session/KV |
| CLI | `src/bin/fastnn-infer.rs`: prepare, inspect, run, generate, diagnose |
| Python | `fastnn/inference/` over `fastnn._core.inference`; CPU and honest Vulkan refusal |
| Secondary 2.x AOT tree | `src/ir/`, `src/compiler/`, `src/backend/`, eager `src/tensor/`/`src/nn/` forward API, ONNX conversion, `.fnn` I/O. Inference only; kept for the legacy compiled path and its quantized CPU pipeline |

## Documentation

- [Getting Started](docs/guides/getting-started.md) — build, prepare a model, run a session
- [Architecture](docs/internals/architecture.md) — the 3.0 inference core and the secondary AOT pipeline
- [API Reference](docs/reference/api-reference.md) — Rust and Python surface
- [CHANGELOG](CHANGELOG.md) — release history

## Testing

```bash
cargo test --lib
cargo test --test fn3_contracts
uv run pytest tests/ -m "not slow"       # Python binding and tooling
cargo test --features vulkan --test fn3_vulkan_smoke   # physical GPU
```

The pinned model and parity fixtures live under `fixtures/` and
`tools/parity/`; they are never committed as weights.
