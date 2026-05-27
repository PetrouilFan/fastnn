# fastnn

**fastnn** is a neural network library built in Rust with Python bindings. It combines an AOT compiler pipeline (90+ IR opcodes, operator fusion, per-channel weight quantization, arena-based memory planning) with an eager-mode autograd engine for research and training.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

> **Version:** 2.2.0 — AOT compiler, compiled training (6 optimizers), FlashAttention SIMD, WGPU quantized inference, residual+add+norm fusion

---

## Installation

### Prerequisites

| Tool   | Version  | Purpose                   |
|--------|----------|---------------------------|
| Rust   | stable   | Build the core library    |
| Python | ≥ 3.12   | Python bindings           |
| uv     | latest   | Python dependency manager |

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build & Install

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

This builds the Rust extension via [maturin](https://www.maturin.rs) and installs the Python package.

### Platform Support

fastnn automatically selects the best available instruction set at runtime:

| Platform             | ISA Support                          |
|----------------------|--------------------------------------|
| x86-64 (desktop)     | AVX-512 → AVX2 → scalar fallback     |
| ARM64 (Raspberry Pi 4/5, Apple Silicon) | NEON intrinsics         |
| Other                | Scalar fallback                      |

---

## Features

### IR-Based AOT Compiler Pipeline

The core of fastnn is an ahead-of-time compiler that transforms computation graphs through a series of passes:

```
ComputeGraph → Shape Inference → Operator Fusion → Quantization (opt.)
→ Memory Planning → Backend Compile → Execute
```

- **ComputeGraph IR** — 90 opcodes with symbolic dimension expressions (`DimExpr::Known`, `Symbol`, `Bounded`), tensor types carrying dtype + shape, and `GraphBuilder` fluent API
- **Shape Inference** — Resolves symbolic dimensions at compile time; falls back to bounded estimates with runtime tightening
- **Operator Fusion** — Modular `FusionPass` trait with independent forward fusions (Op+ReLU, MatMul+BiasAdd+ReLU, Conv2d+BiasAdd+ReLU, residual+add+norm) and backward fusions
- **Memory Planning** — Greedy first-fit arena allocator with live-range analysis and runtime shape tightening
- **Arena-based execution** — Zero runtime allocation after compilation

### Compiled Training

The forward+backward+optimizer pipeline compiles ahead of time into a single `ExecutablePlan` with persistent memory arena reuse:

| Optimizer | Status |
|-----------|--------|
| SGD       | Compiled |
| Adam      | Compiled |
| AdamW     | Compiled |
| Muon      | Compiled |
| Lion      | Compiled |
| RMSprop   | Compiled |

```python
model = fnn.compile_train_model(
    graph_bytes=graph_bytes,
    loss_node_id=...,
    param_ids=[...],
    param_data=[...],
    batch_input_ids=[...],
    optimizer="adamw",
    lr=0.001,
)
loss = model.train_step([batch_bytes])
```

### Native Weight Quantization

Weights are quantized at **compile time** — no runtime overhead:

- **4-bit (U4x8)** — 8 values per `u32` word
- **8-bit (U8x4)** — 4 values per `u32` word
- **Per-channel** — One (scale, zero_point) pair per output channel
- **Fused dequantization** inside GEMM/conv kernels — no separate dequant step
- **CPU + WGPU GPU quantized inference**

```python
executor = fnn.AotExecutor(nodes, params, input_names, output_names, quantize=4)
output = executor.forward({"input": tensor})
```

### Eager-Mode Training

Full eager-mode autograd engine with neural network modules and optimizers:

```python
import fastnn as fnn

X = fnn.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
y = fnn.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

model = fnn.Sequential(
    fnn.Linear(2, 16),
    fnn.ReLU(),
    fnn.Linear(16, 1),
)

optimizer = fnn.Adam(model.parameters(), lr=1e-2)
for epoch in range(200):
    pred = model(X)
    loss = fnn.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Neural Network Modules

Layers with full autograd support: `Linear`, `Conv1d/2d/3d`, `ConvTranspose2d`, `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Dropout/Dropout2d`, `Embedding`, `Upsample`, `MaxPool1d/2d`, `AvgPool1d/2d`, `AdaptiveAvgPool2d`, `MultiHeadAttention`, `TransformerBlock/Encoder`, `ResidualBlock`, `Flatten`.

Activations (layer + tensor method): `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `SiLU`, `LeakyReLU`, `Softplus`, `Hardswish`, `ELU`, `Mish`, `PReLU`, `Softmax`, `LogSoftmax`.

### Optimizers

All with fused update steps: `SGD`, `Adam`, `AdamW`, `Muon`, `Lion`, `RMSprop`. Plus learning rate schedulers (`StepLR`, `CosineAnnealingLR`, `ExponentialLR`, `ReduceLROnPlateau`), gradient clipping, and callbacks.

### ONNX Model Import

90+ ONNX operator types supported via `OnnxConverter`:

```python
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
model = fnn.build_model_from_fnn("model.fnn")
executor = fnn.AotExecutor(nodes, params, input_names, output_names)
output = executor.forward({"input": input_tensor})
```

### YOLO Object Detection

```python
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")
```

### FlashAttention

Memory-efficient attention with tiled online-softmax and SIMD tile matmul:

```python
output = fnn.flash_attention(q, k, v, causal=True)
```

### GPU Acceleration (WGPU)

Cross-platform GPU compute via wgpu (Vulkan, Metal, DX12). Quantized U4/U8 inference runs entirely on GPU via WGSL compute shaders.

---

## Quick Start

### Train an MLP (Python)

```python
import fastnn as fnn

X = fnn.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
y = fnn.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

model = fnn.Sequential(
    fnn.Linear(2, 16),
    fnn.ReLU(),
    fnn.Linear(16, 1),
)

optimizer = fnn.Adam(model.parameters(), lr=1e-2)
for epoch in range(200):
    pred = model(X)
    loss = fnn.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
with fnn.no_grad():
    print(model(X).numpy().round(2))
```

### Compile and Execute a Graph (Rust)

```rust
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};
use fastnn::backend::cpu::CpuBackend;

let gb = GraphBuilder::new();
let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
let weight_tt = TensorType::new(vec![DimExpr::Known(784), DimExpr::Known(10)], IrDType::F32);
let weight = gb.constant(&weight_bytes, weight_tt);
let output = gb.matmul(&input, &weight);

let (plan, mem, graph) = gb.compile_with_quantize(&[&output], CpuBackend, Some(4))?;
```

### Complete ONNX Pipeline

```python
import fastnn as fnn
from fastnn.io import AotExecutor

info = fnn.convert_from_onnx("model.onnx", "model.fnn")
model = fnn.build_model_from_fnn("model.fnn")
executor = fnn.AotExecutor(nodes, params, input_names, output_names)
output = executor.forward({"input": input_tensor})
```

---

## Performance

Performance numbers are hardware-dependent and must be backed by a runnable benchmark command.

- Run the maintained CPU suite with `cargo +stable bench --bench cpu_baselines`
- Save a regression baseline with `cargo +stable bench --bench cpu_baselines -- --save-baseline <name>`
- Export a normalized JSON summary with `python scripts/criterion_to_json.py --criterion-dir target/criterion --output benchmark-results/<name>.json`

Public speed claims should point to a checked-in benchmark command, name the comparison baseline, and avoid hard-coded tables that cannot be reproduced locally.

> See `BENCHMARKS.md` for commands, baseline capture format, and the performance-claim policy.

---

## Architecture

```
┌──────────────┐     ┌────────────────────────────────────────────────┐
│  Python API  │────▶│                Rust Core                       │
│  (PyO3)      │     │                                                │
│  AotExecutor │     │  ┌──────────────────────────────────────────┐  │
│  Tensor API  │     │  │        ComputeGraph (IR)                 │  │
│              │     │  │  90 opcodes · DimExpr · IrDType · F32    │  │
│              │     │  │  F16 · BF16 · I64 · U4/U8 quantized      │  │
│              │     │  └─────────────┬────────────────────────────┘  │
│              │     │                │                                │
│              │     │  ┌─────────────▼────────────────────────────┐  │
│              │     │  │      Compiler Passes                    │  │
│              │     │  │  Shape Inference → AutoCast → Type Inf   │  │
│              │     │  │  → Operator Fusion → Quantization (opt.) │  │
│              │     │  │  → Const Folding → DCE → Memory Planning │  │
│              │     │  └─────────────┬────────────────────────────┘  │
│              │     │                │                                │
│              │     │  ┌─────────────▼────────────────────────────┐  │
│              │     │  │        Backend Dispatch                   │  │
│              │     │  │  CpuBackend (AVX-512/AVX2/NEON/scalar)    │  │
│              │     │  │  WgpuBackend (Vulkan/Metal/DX12)          │  │
│              │     │  │  Arena-based execution                     │  │
│              │     │  └────────────────────────────────────────────┘  │
│              │     │                                                  │
│              │     │  ┌────────────────────────────────────────────┐  │
│              │     │  │     Eager Mode (Training)                  │  │
│              │     │  │  Tensor · Autograd · Optimizers · Modules   │  │
│              │     │  └────────────────────────────────────────────┘  │
│              │     │  ┌────────────────────────────────────────────┐  │
│              │     │  │     Compiled Training                      │  │
│              │     │  │  build_backward_graph → TrainingPass       │  │
│              │     │  │  → CompiledTrainingModel (6 optimizers)    │  │
│              │     │  └────────────────────────────────────────────┘  │
└──────────────┘     └──────────────────────────────────────────────────┘
```

---

## Build Flags

| Feature       | Description                                       | Default |
|---------------|---------------------------------------------------|---------|
| `simd`        | SIMD kernels (AVX2, AVX512, NEON, F16C)           | on      |
| `neon`        | ARM NEON SIMD kernels for packed GEMV (aarch64)   | on      |
| `parallel`    | Rayon multi-threaded parallelism                  | on      |
| `simd-avx512` | AVX-512 kernels (requires AVX-512 CPU)            | on      |
| `fusion-forward` | Enable forward operator fusion passes          | on      |
| `fusion-op-relu` | Op+Relu fusion pass                            | on      |
| `fusion-matmul-add-relu` | MatMul+BiasAdd+Relu fusion pass       | on      |
| `fusion-backward` | Enable backward operator fusion passes         | off     |
| `fusion-residual-add-norm` | Enable residual+add+norm fusion pass    | off     |
| `cli`         | Standalone runtime binary (`fastnn-runtime`)       | off     |
| `openblas`    | Link against OpenBLAS for large matmul            | off     |
| `blas`        | BLAS-accelerated matmul (requires system cblas)   | off     |

---

## Testing

```bash
# Rust unit and integration tests
cargo test

# Quantized pipeline integration tests
cargo test --test quantized_pipeline

# Python tests
uv run pytest tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and quick start |
| [Tensors](docs/tensors.md) | Tensor creation, operations, autograd |
| [NN Modules](docs/nn-modules.md) | Neural network layer reference |
| [Optimizers](docs/optimizers.md) | Training optimization algorithms |
| [Training](docs/training.md) | Data loaders, callbacks, training loops |
| [Models](docs/models.md) | Pre-built model architectures (MLP, Transformer, YOLO) |
| [IO & Serialization](docs/io.md) | Model save/load, ONNX import |
| [Python API](docs/python-api.md) | Compiled training, FlashAttention, WGPU |
| [Architecture](docs/architecture.md) | AOT compiler pipeline internals |
| [Development](docs/development.md) | Module layout and contribution guide |
| [ONNX Support](docs/onnx.md) | ONNX import, YOLO, quantized export |

---

## Roadmap

- [x] **v1.0** — Core tensor ops, autograd, Conv2d, optimizers, FlashAttention
- [x] **v1.1** — Packed precision, SWAR ops, GPU backend, modular architecture
- [x] **v1.2** — Fused kernels (Conv+BN+Activation), ONNX import, fused optimizer updates
- [x] **v1.3** — Batch GEMM, ARM NEON, WGPU packed conv, calibration/profiling
- [x] **v2.0** — AOT compiler pipeline, IR-based execution, native U4/U8 quantization
- [x] **v2.1** — Modular fusion (forward + backward), CLI binary, error handling (`try_*` API)
- [x] **v2.2** — Compiled training (6 optimizers), FlashAttention SIMD, WGPU quantized inference, residual+add+norm fusion, ARM NEON validation suite
- [ ] **v2.3** (planned) — Multi-GPU training, 2-bit quantization, sparse computation

---

## Contributing

fastnn is developed in two layers:

1. **Rust core** (`src/`) — IR nodes, compiler passes, backends, tensor ops, autograd, neural network modules, optimizers
2. **Python facade** (`fastnn/`) — Public API, data loading, model I/O, ONNX pipeline

See [`docs/development.md`](docs/development.md) for architecture, module layout, and step-by-step guides for adding ops, passes, and bindings.

---

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
