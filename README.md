# fastnn

**fastnn** is a high-performance neural network library built from scratch in Rust, with seamless Python bindings. It is designed to be a drop-in PyTorch replacement for CPU-based deep learning — achieving **14–25× speedup over PyTorch** on common Conv+BN+Activation pipelines, up to **7.4× faster GEMV with 8× memory savings** via native packed precision, with **quantized ONNX runtime support** for Q/DQ, QLinearConv, and MatMulInteger ops, plus **ARM NEON SIMD** and **batch GEMM** acceleration for packed inference.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

> **Version:** v1.3.0 — Packed precision expansion, quantized ONNX, fused packed layers, batch GEMM, ARM NEON & WGPU packed conv

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

### Training a model

```python
import fastnn as fnn

# Generate XOR training data
X = fnn.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
y = fnn.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

# Build model
model = fnn.Sequential(
    fnn.Linear(2, 16),
    fnn.ReLU(),
    fnn.Linear(16, 16),
    fnn.ReLU(),
    fnn.Linear(16, 1),
)

optimizer = fnn.Adam(model.parameters(), lr=1e-2)
ds = fnn.TensorDataset(X, y)
loader = fnn.DataLoader(ds, batch_size=4, shuffle=True)

model.train()
for epoch in range(200):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loader.reset_sampler()

# Inference
model.eval()
with fnn.no_grad():
    preds = model(X)
    print(preds.numpy().round(2))
# ≈ [[0.], [1.], [1.], [0.]]
```

### Creating and manipulating tensors

```python
import fastnn as fnn

a = fnn.randn([1000, 1000])
b = fnn.randn([1000, 1000])
c = fnn.matmul(a, b)   # Optimized GEMM — 88 GFLOP/s

x = fnn.tensor([1.0, 2.0, 3.0], [3])
y = x * 2
z = y.sum()
z.backward()            # Automatic differentiation
print(x.grad)           # tensor([2., 2., 2.])
```

---

## Why fastnn?

fastnn is built for **one job**: running neural networks fast on CPU hardware — from desktop x86 to ARM edge devices — with no dependency on PyTorch, CUDA, or any external framework.

### 1. Fused kernels: 14–25× faster Conv+BN+Activation

Single-pass `fused_conv_bn_silu` replaces three separate PyTorch operations:

| Configuration | PyTorch (separate) | fastnn (fused) | Speedup |
|---------------|-------------------|----------------|---------|
| Conv2d(32→64) + BN + SiLU (64×64) | 81.81 ms | 3.27 ms | **25.0×** |
| Conv2d(64→128) + BN + SiLU (32×32) | 42.55 ms | 2.01 ms | **21.2×** |
| Conv2d(128→256) + BN + SiLU (16×16) | 26.24 ms | 1.82 ms | **14.4×** |

### 2. Native packed precision: 7.4× faster GEMV, 8× memory savings

Weights stored packed in u32 words (4-bit, 8-bit, 16-bit). SIMD/SWAR kernels operate directly on the packed representation — no unpacking needed.

| Implementation | Time | GFLOP/s | vs PyTorch f32 | Memory |
|---------------|------|---------|----------------|--------|
| PyTorch f32 (MKL) | 4.04 ms | 8.3 | 1.0× | 64 MB |
| fastnn F16x2 | 1.80 ms | 18.6 | **2.2×** | 32 MB |
| fastnn U8x4 | 0.76 ms | 44.4 | **5.3×** | 16 MB |
| fastnn U4x8 | 0.55 ms | 61.1 | **7.4×** | 8 MB |

> Measured on AMD Ryzen 7 3700X, 8 threads. PyTorch uses MKL BLAS. fastnn uses Rayon + AVX2.

### 3. Pure Rust, zero external dependencies

The entire core — tensor operations, autograd, convolutions, optimizers — is implemented in Rust with no bindings to C/C++ libraries. Python bindings are generated via [PyO3](https://pyo3.rs), and the only way to use fastnn from Python is through its clean, PyTorch-compatible API.

---

## Features

- **Native packed precision** — 4-bit, 8-bit, 16-bit, 32-bit inference with SWAR and SIMD-accelerated GEMV/ReLU operating directly on packed u32 words
- **Vectorized CPU kernels** — Runtime-dispatched SIMD: AVX-512 → AVX2 → NEON → scalar fallback. Cephes-style fast approximations for `exp`, `log`
- **Fast GEMM** — Pure-Rust matrix multiplication via `matrixmultiply`: **88 GFLOP/s** at 1024×1024
- **Autograd engine** — Built-in automatic differentiation with operation tracking and `no_grad` context
- **Convolutions** — Conv1d, Conv2d, Conv3d, ConvTranspose2d with im2col and specialized 1×1, 3×3, and depthwise kernels; packed dispatch for MatMul, ConvTranspose, Embedding
- **Fused layers** — `FusedConvBn`, `FusedConvBnReLU`, `FusedConvBnGELU`, `PackedConvRelu`, `PackedLinearGelu`, `LayerNorm+GELU`, `RMSNorm+GELU`; BN folding into packed conv
- **Optimizers** — SGD, Adam, AdamW, Muon, Lion, RMSprop with fused update steps
- **Data loading** — Multi-threaded DataLoader with automatic resource tuning and prefetch
- **GPU acceleration** *(experimental)* — Cross-platform compute via [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12); WGPU packed convolution compute shader
- **ONNX model import** — Load and run models from PyTorch, YOLO, and other frameworks (50 ops supported); Q/DQ folding, NonZero, Unique, Tril, Triu, Pad v2
- **Calibration & profiling** — `ActivationCalibrator` for activation statistics with KL-divergence scale refinement; `PrecisionProfiler` for per-layer sensitivity analysis and automatic mixed-precision configuration

---

## New Features (feat/precision-system)

### Quantized Dispatch

- **Packed dispatch for additional ops** — `MatMul`, `ConvTranspose`, and `Embedding` operations now support packed (U4/U8) weight dispatch through the DAG executor, falling back to f32 when weights aren't packed.
- **QuantizeLinear / DequantizeLinear** — ONNX-standard quantization operators (`OpCode::QuantizeLinear`, `OpCode::DequantizeLinear`) for DAG execution, enabling quantized runtime with scale+round+clamp quantization and (v−zp)×scale dequantization.
- **QLinearConv / MatMulInteger** — Quantized integer matrix ops: `QLinearConv` dequantizes quantized input and weight, runs f32 conv, and requantizes the output; `MatMulInteger` dequantizes both quantized inputs and runs f32 matmul.

### ONNX Import

- **Q/DQ folding** — The ONNX importer (`onnx.py`) detects `QuantizeLinear → DequantizeLinear` patterns around `Conv`/`MatMul` nodes and folds them, converting weights to packed storage during import for zero-cost quantized inference.
- **4 new ONNX ops** — `NonZero`, `Unique`, `Tril`, `Triu`.
- **Pad v2** — Tensor-based pad sizes with 4 padding modes (constant, reflect, replicate, edge).

### Calibration & Profiling

- **`ActivationCalibrator`** (`fastnn.io.act_calibrate`) — Runs calibration data through the model, collects per-layer activation distributions, and refines quantization scales using weighted KL-divergence (NVIDIA's method). Integrates with the existing `Calibrator` infrastructure and `fastnn-convert` CLI.
- **`PrecisionProfiler`** (`fastnn.io.profiler`) — Per-layer sensitivity analysis: quantizes each layer in isolation, measures output MSE perturbation, and sorts by sensitivity. `auto_config()` generates a `PrecisionConfig` that assigns U4 to least-sensitive layers, U8 to mid, and F32 to most-sensitive given a target memory budget.

### Fused Layers

- **`PackedConvRelu`** — Standalone fused packed convolution + in-place ReLU in all 4 precisions (U4/U8/U16/F32), exposed via Python.
- **`PackedLinearGelu`** — Fused packed linear + GELU with Abramowitz & Stegun erf approximation, all 4 precisions.
- **BN folding** — `fold_bn_into_packed_conv()` dequantizes packed weights to f32, applies `W_fused = W × γ / √(σ² + ε)`, and requantizes per-channel — enabling BN fusion in quantized conv layers.

### Performance Kernels

- **Batch GEMM for packed MatMul** — K-tiled batch GEMM (`gemm_batch_packed()`) processes all batch rows within each K-tile before moving to the next, improving L2 cache reuse significantly (~5–6× speedup over per-row GEMV).
- **ARM NEON SIMD** — `gemv_u4x8_neon()` and `gemv_u8x4_neon()` kernels using full SIMD widening chain (int8 → int16 → int32 → f32 → FMA) for aarch64 targets. Gated behind a `neon` feature flag in `Cargo.toml`.
- **WGPU packed conv** — Experimental `conv_packed.wgsl` compute shader implementing packed convolution on GPU with per-channel scaling and bias, plus full Rust scaffolding (`conv2d_packed_wgpu()`) for pipeline creation, buffer management, and dispatch.

---

## Model Import

fastnn can load pretrained models from PyTorch and ONNX:

```python
# From PyTorch
from fastnn.io import convert_from_pytorch
convert_from_pytorch(torch_model, "model.fnn")

# From ONNX
from fastnn.io import convert_from_onnx
info = convert_from_onnx("model.onnx", "model.fnn")

# YOLO object detection
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")
```

See [Model I/O](docs/io.md) and [ONNX support](docs/onnx.md) for details.

---

## Benchmarks

All benchmarks measured on AMD Ryzen 7 3700X (Arch Linux). Steady-state medians with warmup.

| Operation | Size | fastnn | GFLOP/s |
|-----------|------|--------|---------|
| MatMul | 1024×1024 | 24.2 ms | **88.6** |
| ReLU | 1024×1024 | 0.30 ms | — |
| GELU | 1024×1024 | 1.06 ms | — |
| Conv2d | 2×8×32×32 → 16 | 1.64 ms | — |

See [Performance Roadmap](docs/performance-roadmap.md) for packed precision benchmarks, GPU results, and the complete benchmark suite.

---

## API Reference

The full API reference is maintained in [`docs/api-reference.md`](docs/api-reference.md). It covers:

- **Tensor creation** — `zeros`, `ones`, `randn`, `arange`, `linspace`, `eye`, `full`, and `tensor_from_numpy`
- **Tensor operations** — `matmul`, `einsum`, `cat`, `stack`, `repeat`, `gather`, `where`, `expand`, `topk`, `flip`, `transpose`, `permute`, `reshape`, `view`, `squeeze`, `unsqueeze`
- **Reductions** — `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `softmax`, `log_softmax`, `cumsum`
- **Activations** — `relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `leaky_relu`, `elu`, `softplus`, `hardswish`, `mish`
- **Math** — `abs`, `exp`, `log`, `sqrt`, `pow`, `clamp`, `neg`, `erf`, `maximum`, `minimum`
- **Neural network modules** — `Linear`, `Conv1d/2d/3d`, `ConvTranspose2d`, `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Dropout`, `Dropout2d`, `Embedding`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `Upsample`, `Flatten`, `ResidualBlock`, `PReLU`, `FusedConvBn`, `FusedConvBnRelu`, `FusedConvBnGelu`, `Sequential`, `ModuleList`
- **Optimizers** — `SGD`, `Adam`, `AdamW`, `Muon`, `Lion`, `RMSprop`
- **Loss functions** — `mse_loss`, `cross_entropy_loss`, `bce_with_logits`, `huber_loss`
- **Learning rate schedulers** — `StepLR`, `CosineAnnealingLR`, `ExponentialLR`, `ReduceLROnPlateau`
- **Data** — `Dataset`, `TensorDataset`, `DataLoader` (with multi-threaded auto-tuning)
- **Model I/O** — `save`, `load`, `convert_from_pytorch`, `convert_from_onnx`, `load_state_dict`
- **Packed precision Python API** — `Linear4/8/16/32`, `PackedConvRelu`, `PackedLinearGelu`, `PackedTensor4/8/16/32`, `MasterWeightOptimizer4/8/16/32`
- **Calibration & profiling** — `ActivationCalibrator` (KL-divergence scale refinement), `PrecisionProfiler` (per-layer sensitivity analysis, automatic mixed-precision config)
- **Utilities** — `no_grad`, `set_seed`, `set_num_threads`, `set_default_device`, `clip_grad_norm_`, `clip_grad_value_`, `flash_attention`

---

## Build Flags

| Feature       | Description                                       | Default |
|---------------|---------------------------------------------------|---------|
| `simd`        | SIMD kernels (AVX2, AVX512, NEON, F16C)           | on      |
| `parallel`    | Rayon multi-threaded parallelism                  | on      |
| `simd-avx512` | AVX-512 kernels (requires AVX-512 CPU)            | on      |
| `openblas`    | Link against OpenBLAS for large matmul            | off     |
| `blas`        | BLAS-accelerated matmul (requires system cblas)   | off     |

> **Note:** The `blas` feature is disabled by default because the system BLAS reference implementation is very slow. The default `matrixmultiply` pure-Rust GEMM is faster on most systems.

---

## Testing

```bash
# Rust unit and integration tests
cargo test

# Python tests
uv run pytest tests/ -v

# Packed precision benchmarks
cargo bench --bench packed_bench
```

---

## Roadmap

- [x] v1.0 — Core tensor ops, autograd, Conv2d, optimizers, PyTorch export
- [x] v1.1 — Packed precision, SWAR ops, GPU backend, modular architecture
- [x] v1.2 — Fused kernels (Conv+BN+Activation, LayerNorm+GELU), ONNX import, YOLO support
- [x] v1.3 — Packed precision expansion, quantized ONNX, fused packed layers, batch GEMM, ARM NEON, WGPU packed conv
- [ ] FlashAttention SIMD optimization (AVX2/AVX512 block kernels)
- [ ] Raspberry Pi benchmark suite (ARM NEON validation)
- [ ] Multi-GPU training via wgpu
- [ ] Process-based multiprocessing for DataLoader
- [ ] Full fused GPU optimizer kernels
- [ ] `residual + add + norm` fusion

---

## Contributing

fastnn is developed in two layers:

1. **Rust core** (`src/`) — Tensors, kernels, autograd, neural network modules, optimizers
2. **Python facade** (`fastnn/`) — Public API, data loading, model I/O, ONNX pipeline

See [`docs/development.md`](docs/development.md) for the internal architecture, GPU synchronization policy, and step-by-step guides for adding tensor operations, fused kernels, and ONNX ops.

---

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
