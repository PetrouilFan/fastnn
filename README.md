# fastnn

[![CI](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml/badge.svg)](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PetrouilFan/fastnn/branch/main/graph/badge.svg)](https://codecov.io/gh/PetrouilFan/fastnn)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)
[![Downloads](https://img.shields.io/github/downloads/PetrouilFan/fastnn/total)](https://github.com/PetrouilFan/fastnn)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/PetrouilFan/fastnn)](https://github.com/PetrouilFan/fastnn)
[![Last Commit](https://img.shields.io/github/last-commit/PetrouilFan/fastnn)](https://github.com/PetrouilFan/fastnn)

**Fast, compiled neural networks in Rust with Python bindings.**

fastnn is an ahead-of-time compiled neural network runtime with eager-mode training. Write models in Python or Rust, compile through a 90+ op IR with 6 optimization passes, and deploy on CPU (ARM NEON, x86 SIMD) with optional WGPU GPU acceleration.

> ⚠️ Under active development. APIs, supported graph paths, and performance characteristics may change between minor versions.

---

## Highlights

- **AOT-compiled graphs** — Write with an eager-like API, deploy with compiled graph performance. Shape inference, fusion, quantization, and memory planning happen at compile time.
- **90+ IR ops, 6 compiler passes** — Operator fusion (ReLU, GELU, SiLU, MatMul+Add), weight quantization (I4/I8/FP4/FP8, I4Codebook per-block), constant folding, dead code elimination, and arena memory planning.
- **CPU + GPU backends** — ARM NEON and x86 SIMD CPU kernels with runtime dispatch, plus an experimental WGPU backend (Vulkan, Metal, DX12).
- **ONNX import/export** — Import 90+ ONNX ops, export trained models. Full YOLOv5/v8/v10/v11 pipeline with NMS post-processing.
- **Python bindings (PyO3) + Rust native API** — Use from Python with numpy interop, or embed directly in Rust projects.

## Installation

**From source** (currently recommended):

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

**Prerequisites:** Rust stable, Python ≥ 3.11, [uv](https://github.com/astral-sh/uv).

The Python package is built with `maturin` and exposes the Rust extension as `fastnn._core`.

## Quick Start

### Python: Eager-mode training

```python
import fastnn as fnn

model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    pred = model(batch_x)
    loss = fnn.mse_loss(pred, batch_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Python: Compiled AOT inference

```python
from fastnn.io import AotExecutor

executor = AotExecutor(
    nodes=model_nodes,
    params=model_params,
    input_names=["input"],
    output_names=["output"],
    quantize=4,                    # 4-bit weight quantization (or "i4cb" for codebook)
)
outputs = executor.forward({"input": input_tensor})
```

### Rust: GraphBuilder API

```rust
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType};
use fastnn::backend::cpu::CpuBackend;

let gb = GraphBuilder::new();
let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
let weight = gb.parameter(&[784, 10], IrDType::F32);
let mm = gb.matmul(&input, &weight);

let result = gb.compile_and_execute_with_quantize(
    &[&mm], CpuBackend, Some(8),
    &[input_data, weight_data],
);
```

## Project Components

| Component | Description |
|-----------|-------------|
| Rust core | Tensor ops, autograd, graph IR, compiler passes, CPU + WGPU backends |
| Python bindings | Python-facing tensor, module, optimizer, data, callback, and I/O APIs |
| AOT compiler | Graph IR → shape/type inference → fusion → quantization → memory planning |
| Eager mode | Imperative tensor/autograd with nn modules, losses, and optimizers |
| Backends | CPU (ARM NEON/x86 SIMD, OpenBLAS) + experimental WGPU (Vulkan/Metal/DX12) |
| Import/export | `.fnn` format, ONNX import (90+ ops), PyTorch model conversion |

## Documentation

Full documentation is in the [`docs/index.md`](docs/index.md) directory:

- [Getting Started](docs/guides/getting-started.md) — First steps and examples
- [API Reference](docs/reference/api-reference.md) — Complete API documentation
- [Architecture](docs/internals/architecture.md) — Compiler pipeline internals
- [Development](docs/internals/development.md) — Contributing guide and codebase walkthrough
- [CHANGELOG](CHANGELOG.md) — Release history

## Testing

```bash
cargo test                           # Rust tests
uv run pytest tests/ -v              # Python tests
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repository layout, coding standards, and the PR workflow.

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
