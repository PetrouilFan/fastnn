# fastnn

**fastnn** is a high-performance neural network inference library built from scratch in Rust, with seamless Python bindings. The v2.0.0 release introduces a complete **ahead-of-time (AOT) compiler pipeline** built on a first-class IR — replacing the legacy DAG/layer dispatch with a unified graph-based compilation path that supports **operator fusion, shape inference, memory planning, and native 4-bit/8-bit weight quantization** as first-class compiler passes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

> **Version:** v2.0.0 — AOT compiler pipeline, IR-based execution, native U4/U8 quantization

---

## What's New in v2.0.0

### AOT Compiler Pipeline

The v2.0.0 release replaces the old DAG/layer dispatch with a **fully compiled execution path**:

```
ComputeGraph → Shape Inference → Operator Fusion → Quantization (opt.) → Memory Planning → Backend Compile → Execute
```

- **`ComputeGraph`** — First-class IR representation with 30+ opcodes (MatMul, Conv2d, Softmax, etc.), symbolic dimension support (`DimExpr::Symbol`, `Bounded`, `Known`), and per-node tensor types carrying dtype + shape.
- **`GraphBuilder`** — Ergonomic Rust API for building graphs: `input()`, `constant()`, `matmul()`, `conv2d_with_params()`, `relu()`, `softmax()`, etc.
- **`GraphExecutor`** — Compiles a `ComputeGraph` through the full pipeline and dispatches execution on a `CpuBackend` (or `WgpuBackend` for supported ops).
- **Shape inference** — Resolves symbolic dimensions at compile time; falls back to bounded estimates with runtime tightening via `ShapeEnv`.
- **Operator fusion** — Merges `MatMul + Add`, `MatMul + Add + ReLU`, `Conv2d + Add`, `Conv2d + Add + ReLU` into single fused kernels.
- **Memory planning** — Greedy first-fit allocator with live-range analysis; reuses arena slots for non-overlapping tensors.

### Native Weight Quantization (U4x8 / U8x4)

Weights are quantized at **compile time** — no runtime overhead:

```python
from fastnn.io import AotExecutor

# Build and compile with 4-bit quantization
executor = AotExecutor(nodes, params, input_names, output_names, quantize=4)
output = executor.forward(*inputs)

# Or 8-bit quantization
executor = AotExecutor(nodes, params, input_names, output_names, quantize=8)
```

From Rust:

```rust
use fastnn::ir::builder::GraphBuilder;
use fastnn::backend::cpu::CpuBackend;

let gb = GraphBuilder::new();
let input = gb.input(&[1, 784], IrDType::F32);
let weight = gb.constant(&weight_bytes, weight_tt);
let output = gb.matmul(&input, &weight);

// Compile with 4-bit quantization
let (plan, mem, graph) = gb.compile_with_quantize(&[&output], CpuBackend, Some(4))?;
```

Key properties:
- **Per-channel quantization** — One (scale, zero_point) pair per output channel for U4/U8 weights.
- **Packed representation** — U4x8 (8 values per u32) and U8x4 (4 values per u32) with SWAR/SIMD dequantization inside GEMM/conv kernels.
- **f32 output** — Dequantization is fused into the GEMM/conv kernel; no separate dequant step.
- **MatMul + Conv2d** — Both support U4 and U8 quantized inference on CPU.
- **WGPU fallback** — Quantized ops fall back to CPU with explicit `UnsupportedOp`; GPU packed shaders planned for v2.1.

### IR-Based Architecture

| Component | Description |
|-----------|-------------|
| `ir::node` | `ComputeGraph`, `Opcode`, `IrDType` (F32/F16/BF16/I32/I64/Bool/U4/U8), `DimExpr`, `TensorType`, `TensorValue` |
| `ir::builder` | `GraphBuilder` + `GraphTensor` — fluent API for graph construction |
| `compiler::passes` | `shape_inference`, `operator_fusion`, `quantization`, `memory_planning` |
| `backend::cpu` | `CpuBackend` — kernel selection, arena-based dispatch, fused kernels |
| `backend::wgpu` | `WgpuBackend` — GPU compute with `UnsupportedOp` fallback for quantized ops |
| `backend::executor` | `GraphExecutor` — ties IR, compiler passes, and backend together |
| `onnx::converter` | `OnnxConverter` — ONNX → `ComputeGraph` with 30+ op mappings |

### Python API Changes

- **`AotExecutor`** now accepts an optional `quantize` parameter (`4`, `8`, or `None`).
- **`DAGModel(quantize=)`** and **`build_dag_model(quantize=)`** pass quantization through to the AOT pipeline.
- **`GraphBuilder.compile_with_quantize()`** and **`compile_and_execute_with_quantize()`** exposed in Rust.
- Existing `DAGExecutor` / packed layer APIs remain available for backward compatibility.

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
curl --proto '=https' --tlsv1.2 -sSSf https://sh.rustup.rs | sh
```

### Build & Install

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .
```

This builds the Rust extension via [maturin](https://www.maturin.rs) and installs the Python package in editable mode.

### Platform Support

fastnn automatically selects the best available instruction set at runtime:

| Platform             | ISA Support                          |
|----------------------|--------------------------------------|
| x86-64 (desktop)     | AVX-512 → AVX2 → scalar fallback     |
| ARM64 (Raspberry Pi 4/5, Apple Silicon) | NEON intrinsics         |
| Other                | Scalar fallback                      |

---

## Quick Start

### Inference with the AOT pipeline

```python
import fastnn as fnn
from fastnn.io import AotExecutor

# Build an ONNX model into a compiled executor with 4-bit quantization
executor = AotExecutor(nodes, params, input_names, output_names, quantize=4)
output = executor.forward(*inputs)

# Or without quantization (f32 default)
executor = AotExecutor(nodes, params, input_names, output_names)
output = executor.forward(*inputs)
```

### Training (eager mode)

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
    print(model(X).numpy().round(2))  # ≈ [[0.], [1.], [1.], [0.]]
```

### Rust: Compile and execute a quantized graph

```rust
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};
use fastnn::backend::cpu::CpuBackend;

let gb = GraphBuilder::new();
let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
let weight_tt = TensorType::new(vec![DimExpr::Known(784), DimExpr::Known(10)], IrDType::F32);
let weight = gb.constant(&weight_bytes, weight_tt);
let output = gb.matmul(&input, &weight);

// Compile with 4-bit quantization
let (plan, mem, graph) = gb.compile_with_quantize(&[&output], CpuBackend, Some(4))?;
let executor = GraphExecutor::new(CpuBackend);
let results = executor.execute(&graph, &plan, &mem, &[&input_bytes])?;
```

---

## Performance

### Packed precision GEMV

| Implementation | Time | GFLOP/s | vs PyTorch f32 | Memory |
|---------------|------|---------|----------------|--------|
| PyTorch f32 (MKL) | 4.04 ms | 8.3 | 1.0× | 64 MB |
| fastnn F16x2 | 1.80 ms | 18.6 | 2.2× | 32 MB |
| fastnn U8x4 | 0.76 ms | 44.4 | 5.3× | 16 MB |
| fastnn U4x8 | 0.55 ms | 61.1 | 7.4× | 8 MB |

> Measured on AMD Ryzen 7 3700X, 8 threads. PyTorch uses MKL BLAS. fastnn uses Rayon + AVX2.

### Fused kernels

| Configuration | PyTorch (separate) | fastnn (fused) | Speedup |
|---------------|-------------------|----------------|---------|
| Conv2d(32→64) + BN + SiLU (64×64) | 81.81 ms | 3.27 ms | 25.0× |
| Conv2d(64→128) + BN + SiLU (32×32) | 42.55 ms | 2.01 ms | 21.2× |
| Conv2d(128→256) + BN + SiLU (16×16) | 26.24 ms | 1.82 ms | 14.4× |

### General benchmarks

| Operation | Size | fastnn | GFLOP/s |
|-----------|------|--------|---------|
| MatMul | 1024×1024 | 24.2 ms | 88.6 |
| GEMM (quantized U4) | 1024×1024 | 8.7 ms | — |
| Conv2d | 2×8×32×32 → 16 | 1.64 ms | — |

---

## Features

- **AOT compiler pipeline** — Shape inference → Operator fusion → Quantization → Memory planning → Backend compile. Compile once, run many times.
- **Native packed precision** — 4-bit (U4x8), 8-bit (U8x4), 16-bit (F16x2), 32-bit (F32x1) inference with SWAR/SIMD-accelerated GEMV/GEMM operating directly on packed u32 words
- **Per-channel quantization** — Scale/zero_point per output channel for U4 and U8; fused dequantization inside GEMM/conv kernels
- **IR-based execution** — `ComputeGraph` with 30+ opcodes, symbolic dimensions, and `GraphBuilder` fluent API
- **Operator fusion** — MatMul+Add, MatMul+Add+ReLU, Conv2d+Add, Conv2d+Add+ReLU fused at compile time
- **Vectorized CPU kernels** — Runtime-dispatched SIMD: AVX-512 → AVX2 → NEON → scalar fallback
- **Autograd engine** — Built-in automatic differentiation with operation tracking and `no_grad` context
- **Convolutions** — Conv1d, Conv2d, Conv3d, ConvTranspose2d with im2col and specialized 1×1, 3×3, and depthwise kernels
- **ONNX model import** — Load and run models from PyTorch, YOLO, and other frameworks (30+ ops supported)
- **GPU acceleration** *(experimental)* — Cross-platform compute via wgpu (Vulkan, Metal, DX12)
- **Optimizers** — SGD, Adam, AdamW, Muon, Lion, RMSprop with fused update steps
- **Data loading** — Multi-threaded DataLoader with automatic resource tuning and prefetch
- **Calibration & profiling** — `ActivationCalibrator` for KL-divergence scale refinement; `PrecisionProfiler` for per-layer sensitivity analysis

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│  Python API  │────▶│             Rust Core                    │
│  (PyO3)      │     │                                          │
│  AotExecutor │     │  ┌────────────────────────────────────┐  │
│  DAGModel    │     │  │        ComputeGraph (IR)           │  │
│  Tensor API  │     │  │  Opcode · DimExpr · IrDType       │  │
│              │     │  │  MatMul · Conv2d · Softmax …      │  │
│              │     │  └─────────────┬──────────────────────┘  │
│              │     │                │                          │
│              │     │  ┌─────────────▼──────────────────────┐  │
│              │     │  │      Compiler Passes              │  │
│              │     │  │  Shape Inference → Operator Fusion  │  │
│              │     │  │  → Quantization → Memory Planning  │  │
│              │     │  └─────────────┬──────────────────────┘  │
│              │     │                │                          │
│              │     │  ┌─────────────▼──────────────────────┐  │
│              │     │  │        Backend Dispatch             │  │
│              │     │  │  CpuBackend · WgpuBackend           │  │
│              │     │  │  Arena-based execution              │  │
│              │     │  └────────────────────────────────────┘  │
│              │     │                                          │
│              │     │  ┌────────────────────────────────────┐  │
│              │     │  │     Eager Mode (training)          │  │
│              │     │  │  Tensor · Autograd · Optimizers     │  │
│              │     │  └────────────────────────────────────┘  │
└──────────────┘     └──────────────────────────────────────────┘
```

---

## Build Flags

| Feature       | Description                                       | Default |
|---------------|---------------------------------------------------|---------|
| `simd`        | SIMD kernels (AVX2, AVX512, NEON, F16C)           | on      |
| `neon`        | ARM NEON SIMD kernels for packed GEMV (aarch64)   | on      |
| `parallel`    | Rayon multi-threaded parallelism                  | on      |
| `simd-avx512` | AVX-512 kernels (requires AVX-512 CPU)            | on      |
| `openblas`    | Link against OpenBLAS for large matmul            | off     |
| `blas`        | BLAS-accelerated matmul (requires system cblas)   | off     |

---

## Testing

```bash
# Rust unit and integration tests
cargo test

# Quantized pipeline integration tests
cargo test --test quantized_pipeline

# Packed precision integration tests
cargo test --test packed_integration

# Python tests
uv run pytest tests/ -v
```

---

## Roadmap

- [x] v1.0 — Core tensor ops, autograd, Conv2d, optimizers, PyTorch export
- [x] v1.1 — Packed precision, SWAR ops, GPU backend, modular architecture
- [x] v1.2 — Fused kernels (Conv+BN+Activation, LayerNorm+GELU), ONNX import, YOLO support
- [x] v1.3 — Packed precision expansion, quantized ONNX, fused packed layers, batch GEMM, ARM NEON, WGPU packed conv
- [x] **v2.0** — AOT compiler pipeline, IR-based execution, native U4/U8 quantization, operator fusion, memory planning
- [ ] v2.1 — Training through the IR pipeline, WGPU packed shaders, ONNX quantized op exporters (QLinearMatMul, QLinearConv)
- [ ] FlashAttention SIMD optimization (AVX2/AVX512 block kernels)
- [ ] Raspberry Pi benchmark suite (ARM NEON validation)
- [ ] Multi-GPU training via wgpu

---

## Contributing

fastnn is developed in two layers:

1. **Rust core** (`src/`) — IR nodes, compiler passes, backends, tensor ops, autograd, neural network modules, optimizers
2. **Python facade** (`fastnn/`) — Public API, data loading, model I/O, ONNX pipeline

See [`docs/development.md`](docs/development.md) for the internal architecture, GPU synchronization policy, and step-by-step guides.

---

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis