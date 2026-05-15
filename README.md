# fastnn

**fastnn** is a high-performance neural network inference library built from scratch in Rust, with seamless Python bindings. The v2.2.0 release extends the AOT compiler pipeline with **compiled training** (6 optimizers), **FlashAttention SIMD** (AVX2/AVX512), **WGPU quantized inference**, **ONNX quantized op exporters**, and **residual+add+norm fusion**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

> **Version:** v2.2.0 — Compiled training, FlashAttention SIMD, WGPU packed shaders, ONNX quantized export

---

## What's New in v2.2.0

### Compiled Training (6 Optimizers)

The forward+backward+optimizer pipeline compiles ahead of time into a single `ExecutablePlan` with persistent memory arena reuse. Supports SGD, Adam, AdamW, **Muon**, **Lion**, and **RMSprop**:

```python
model = fnn.compile_train_model(
    graph_bytes=graph_bytes,
    loss_node_id=...,
    param_ids=[...],
    param_data=[...],
    batch_input_ids=[...],
    optimizer="muon",       # "sgd" | "adamw" | "muon" | "lion" | "rmsprop"
    lr=0.001,
)
loss = model.train_step([batch_bytes])
```

### FlashAttention SIMD Optimization

Tiled online-softmax attention with AVX-512/AVX2 tile matmul — **2-4× speedup** over the baseline implementation.

### WGPU Packed Shaders (U4/U8 Quantized GPU Inference)

U4 and U8 quantized inference runs entirely on GPU with per-channel dequantization via WGSL compute shaders — no CPU copy during qunatized matmul/conv dispatch.

### ONNX Quantized Op Exporters

New `QLinearMatMul` and `QLinearConv` ONNX op exporters with per-channel scale/zp for exporting quantized models.

### Residual + Add + Norm Fusion

Single-pass skip connection + add + layer norm fusion for transformer architectures.

### CLI Enhancements

New `fastnn-runtime compile` (ONNX → .fnnc) and `fastnn-runtime quantize` (offline weight quantization) subcommands.

### ARM NEON Validation Suite

CI cross-compilation pipeline, NEON kernel tests, cross-architecture consistency tests, and Raspberry Pi benchmark runner.

### Compiler Test Suite

52 new compiler pass tests covering type_inference, auto_cast, activation_quantization, shape_inference, memory_planning, and edge cases.

---

## What's New in v2.1.0

### AOT Compiler Pipeline

The v2.1.0 release replaces the old DAG/layer dispatch with a **fully compiled execution path**:

```
ComputeGraph → Shape Inference → Operator Fusion (forward + backward) → Quantization (opt.) → Memory Planning → Backend Compile → Execute
```

- **`ComputeGraph`** — First-class IR representation with 30+ opcodes (MatMul, Conv2d, Softmax, etc.), symbolic dimension support (`DimExpr::Symbol`, `Bounded`, `Known`), and per-node tensor types carrying dtype + shape.
- **`GraphBuilder`** — Ergonomic Rust API for building graphs: `input()`, `constant()`, `matmul()`, `conv2d_with_params()`, `relu()`, `softmax()`, etc.
- **`GraphExecutor`** — Compiles a `ComputeGraph` through the full pipeline and dispatches execution on a `CpuBackend` (or `WgpuBackend` for supported ops).
- **Shape inference** — Resolves symbolic dimensions at compile time; falls back to bounded estimates with runtime tightening via `ShapeEnv`.
- **Operator fusion** — Modular `FusionPass` trait with independent forward fusions (OpRelu, MatMulAddRelu) and backward fusions (BackwardReluMatMul, BackwardMatMulAddRelu), each feature-gated via Cargo.toml. Backward fusion eliminates 3 intermediate allocations per fused backward chain by combining dRelu+Transpose+MatMul into single fused kernels.
- **Memory planning** — Greedy first-fit allocator with live-range analysis; reuses arena slots for non-overlapping tensors. Runtime `tighten()` shrinks worst-case symbolic arena (85 GB) to actual input sizes (40 KB).

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
- **WGPU quantized inference** — U4/U8 quantized ops run on GPU via WGSL compute shaders with per-channel dequantization (v2.2).

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
- **Error handling** — All Python ops now return `PyResult<PyTensor>` with `catch_unwind`, preventing Python process crashes on AOT compilation/execution failures. Previously, a failed `.expect()` in a tensor op would abort the Python process.

### Backward Fusion

The v2.1.0 release introduces **backward graph fusion** that eliminates intermediate allocations in the backward pass:

- **BackwardReluMatMul** — Detects `Mul(dRelu) + Transpose(b) + MatMul(da) + Transpose(a) + MatMul(db)` chains in backward graphs and replaces them with two fused backward MatMul nodes (one for da, one for db).
- **BackwardMatMulAddRelu** — Identifies the `ReduceSum(dbias)` step in the MatMulAddRelu backward chain and marks it for fused execution.
- **Effect** — 3 intermediate allocations eliminated per fused backward chain (dRelu tensor, b^T tensor, a^T tensor).

### Error Handling

- **`try_*` API** — Every tensor operation now has a `try_*` variant returning `Result<Tensor, BackendError>`:
  `try_add`, `try_sub`, `try_mul`, `try_div`, `try_matmul`, `try_neg`, `try_relu`, `try_exp`, `try_ln`, `try_sigmoid`, `try_tanh`, `try_silu`, `try_gelu`, `try_leaky_relu`, `try_softplus`, `try_hardswish`, `try_mish`, `try_elu`, `try_softmax`, `try_sqrt`, `try_clamp`, `try_pow`, `try_abs`, `try_log_softmax`, `try_erf`.
- Panicking variants (`.add()`, `.sub()`, etc.) now delegate to their `try_*` counterparts.
- Python bridge uses `catch_unwind` on all ops — AOT failures produce a Python `RuntimeError` instead of aborting the process.

### CLI Binary

A standalone **`fastnn-runtime`** binary for deploying pre-compiled plans without the compiler stack:

```
cargo build --bin fastnn-runtime --features cli
fastnn-runtime info plan.fnnc memory.json
fastnn-runtime run plan.fnnc memory.json input0.bin input1.bin output.bin
fastnn-runtime bench plan.fnnc memory.json input0.bin input1.bin -i 1000
fastnn-runtime compile model.onnx -o model.fnnc --quantize 4
fastnn-runtime quantize model.fnnc memory.json --bits 4
```

- Feature-gated behind `cli` (opt-in, not in default features).
- Subcommands: `info` (inspect plan), `run` (execute), `bench` (latency/throughput), `compile` (ONNX → .fnnc, NEW in v2.2), `quantize` (offline weight quantization, NEW in v2.2).

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
| FlashAttention (AVX2) | 4×8×1024×64 | ~2.3 ms | — |

> FlashAttention uses tiled online-softmax with AVX-512/AVX2 tile matmul for 2-4× speedup over the baseline (v2.2).

---

## Features

- **AOT compiler pipeline** — Shape inference → Operator fusion → Quantization → Memory planning → Backend compile. Compile once, run many times.
- **Compiled training** — Forward+backward+optimizer pipeline compiled into a single `ExecutablePlan` with persistent memory arena reuse. 6 optimizers: SGD, Adam, AdamW, Muon, Lion, RMSprop (v2.2)
- **FlashAttention SIMD** — Tiled online-softmax attention with AVX-512/AVX2 tile matmul (2-4× speedup, v2.2)
- **Native packed precision** — 4-bit (U4x8), 8-bit (U8x4), 16-bit (F16x2), 32-bit (F32x1) inference with SWAR/SIMD-accelerated GEMV/GEMM operating directly on packed u32 words
- **Per-channel quantization** — Scale/zero_point per output channel for U4 and U8; fused dequantization inside GEMM/conv kernels
- **WGPU quantized inference** — U4/U8 quantized ops run on GPU via WGSL compute shaders (v2.2)
- **IR-based execution** — `ComputeGraph` with 30+ opcodes, symbolic dimensions, and `GraphBuilder` fluent API
- **Operator fusion** — MatMul+Add, MatMul+Add+ReLU, Conv2d+Add, Conv2d+Add+ReLU, residual+add+norm fused at compile time
- **Vectorized CPU kernels** — Runtime-dispatched SIMD: AVX-512 → AVX2 → NEON → scalar fallback
- **Autograd engine** — Built-in automatic differentiation with operation tracking and `no_grad` context
- **Convolutions** — Conv1d, Conv2d, Conv3d, ConvTranspose2d with im2col and specialized 1×1, 3×3, and depthwise kernels
- **ONNX model import** — Load and run models from PyTorch, YOLO, and other frameworks (30+ ops supported)
- **ONNX quantized export** — QLinearMatMul, QLinearConv exporters with per-channel scale/zp (v2.2)
- **GPU acceleration** *(experimental)* — Cross-platform compute via wgpu (Vulkan, Metal, DX12)
- **Optimizers** — SGD, Adam, AdamW, Muon, Lion, RMSprop with fused update steps
- **Data loading** — Multi-threaded DataLoader with automatic resource tuning and prefetch
- **Calibration & profiling** — `ActivationCalibrator` for KL-divergence scale refinement; `PrecisionProfiler` for per-layer sensitivity analysis
- **ARM NEON validation** — CI cross-compilation, NEON kernel tests, Raspberry Pi benchmark runner (v2.2)

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
│              │     │  ┌────────────────────────────────────┐  │
│              │     │  │  Compiled Training (v2.2)          │  │
│              │     │  │  build_backward_graph → TrainingPass│  │
│              │     │  │  → CompiledTrainingModel           │  │
│              │     │  │  6 optimizers · persistent arena    │  │
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
| `fusion-forward` | Enable forward operator fusion passes          | on      |
| `fusion-op-relu` | Op+Relu fusion pass                            | on      |
| `fusion-matmul-add-relu` | MatMul+BiasAdd+Relu fusion pass       | on      |
| `fusion-backward` | Enable backward operator fusion passes         | off     |
| `fusion-residual-add-norm` | Enable residual+add+norm fusion pass (v2.2) | off |
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
- [x] **v2.1** — Modular fusion passes (forward + backward), shape specialization (runtime `tighten()`), CLI binary, error handling (`try_*` API + `PyResult` bridge), backward fusion pattern matching
- [x] **v2.2** — Compiled training (Muon/Lion/RMSprop), WGPU packed shaders, ONNX quantized export, residual+add+norm fusion, FlashAttention SIMD, compiler test suite, ARM NEON validation, CLI compile/quantize
- [ ] v2.3 (planned) — Multi-GPU training via wgpu, 2-bit quantization, sparse computation

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