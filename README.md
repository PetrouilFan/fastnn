# fastnn

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

fastnn is a Rust neural-network runtime and training library with Python
bindings. It provides an eager tensor/autograd API and an ahead-of-time graph
compilation path for experiments with compiler passes, arena memory planning,
quantization, and backend dispatch.

fastnn is under active development. APIs, supported graph paths, and performance
characteristics may change between minor versions.

## Installation

Prerequisites:

- Rust stable
- Python 3.11 or newer
- `uv` for Python environment management

Build and install from source:

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

The Python package is built with `maturin` and exposes the Rust extension as
`fastnn._core`.

## Quick start

```python
import fastnn as fnn

x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
w = fnn.tensor([0.5, 0.0, 0.0, 0.5], [2, 2])

y = fnn.matmul(x, w)
print(y.numpy())
```

For a longer introduction to tensors, modules, optimizers, and training loops,
see [Getting Started](docs/getting-started.md).

## Project components

| Component | Description |
| --- | --- |
| Rust core | Tensor operations, autograd, graph IR, compiler passes, and backend implementations. |
| Python package | Python-facing tensor, module, optimizer, data, callback, and model I/O APIs. |
| AOT compiler | Graph construction, shape/type inference, fusion passes, quantization passes, and memory planning. |
| Eager mode | Tensor/autograd execution with neural-network modules and optimizers. |
| Backends | CPU execution paths and optional WGPU code paths where supported. |
| Import/export | `.fnn`, PyTorch conversion, and ONNX import paths for supported models/operators. |

## Supported capabilities

fastnn currently includes:

- Tensor operations and eager-mode autograd.
- Neural-network modules, losses, optimizers, schedulers, data loaders, and callbacks.
- Ahead-of-time graph compilation for selected workloads.
- CPU kernels with runtime dispatch for supported SIMD targets.
- Optional WGPU backend paths behind the `gpu` feature.
- U4/U8 weight quantization paths in selected compiled inference flows.
- ONNX import and model conversion utilities for supported operator sets.
- Benchmark and regression tooling for maintained CPU execution paths.

Some APIs and backend paths are experimental. Check the relevant documentation
and tests before relying on a path for production workloads.

## Testing

Common local checks:

```bash
cargo test
uv run pytest tests/ -v
```

Additional focused checks are documented in [Development](docs/development.md).

## Benchmarking

Performance results should be reported with the exact command, hardware,
commit, and comparison baseline. The maintained CPU benchmark suite is:

```bash
cargo +stable bench --bench cpu_baselines
```

See [BENCHMARKS.md](BENCHMARKS.md) for benchmark commands, baseline capture,
JSON export, and the performance-claim policy.

## Documentation

| Document | Purpose |
| --- | --- |
| [Getting Started](docs/getting-started.md) | Installation and first examples. |
| [Tensors](docs/tensors.md) | Tensor creation, operations, and autograd behavior. |
| [Training](docs/training.md) | Training loops, data loading, callbacks, and compiled training notes. |
| [Neural Network Modules](docs/nn-modules.md) | Layer and module reference. |
| [Optimizers](docs/optimizers.md) | Optimizer and scheduler APIs. |
| [Python API](docs/python-api.md) | Python-facing APIs for compiled execution and runtime features. |
| [ONNX Support](docs/onnx.md) | ONNX import and conversion paths. |
| [Architecture](docs/architecture.md) | IR, compiler passes, memory planning, and backend internals. |
| [Development](docs/development.md) | Repository layout, contribution workflow, and validation commands. |
| [Models](docs/models.md) | Pre-built model architectures (MLP, Transformer, YOLO). |
| [IO & Serialization](docs/io.md) | Save/load models, ONNX import, and serialization API. |
| [API Reference](docs/api-reference.md) | Complete Python API reference. |
| [ARM NEON](docs/arm-neon.md) | ARM NEON SIMD kernel documentation. |
| [v2.3 Roadmap](docs/v2.3-roadmap.md) | Current development roadmap. |
| [Performance Roadmap](docs/performance-roadmap.md) | GPU, fusion, optimizer, and packed precision roadmap. |
| [ONNX Training Export](docs/onnx-training-export.md) | ONNX training export contract. |
| [Release Process](docs/release-process.md) | Release workflow and checklist. |
| [Benchmarks](BENCHMARKS.md) | Maintained benchmark suites and reporting policy. |

## Development

Useful commands while working on the repository:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test
uv run pytest tests/ -v
```

Behavior changes should include tests. Performance changes should include a
reproducible benchmark command or telemetry guardrail.

## Project status

fastnn is currently focused on:

- CPU execution performance and copy reduction.
- Arena memory planning and compiled graph execution.
- Correctness coverage for compiler/runtime paths.
- Maintained benchmark baselines for performance work.
- Validation of optional WGPU backend paths.

For current planning documents, see [v2.3 Roadmap](docs/v2.3-roadmap.md) and
[Performance Roadmap](docs/performance-roadmap.md).

## Contributing

Contributions should keep the README concise and move detailed API, architecture,
and performance material into the documentation. Include tests for behavior
changes and reproducible benchmark commands for performance claims.

See [Development](docs/development.md) for repository layout and local validation.

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
