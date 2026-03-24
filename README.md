# fastnn

**fastnn** is a high-performance neural network inference library built from scratch in Rust, with seamless Python bindings. It is designed for fast, memory-efficient CPU inference — including on edge devices like Raspberry Pi — using sub-byte quantization and hand-written SIMD kernels.

> **Version:** v0.7.0 — 88+ Performance Optimizations

[![CI](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml/badge.svg)](https://github.com/PetrouilFan/fastnn/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

---

## Overview

fastnn is built for researchers and engineers who need efficient inference on CPU hardware — from desktop x86 to ARM-based edge devices. It is implemented entirely in Rust and exposed to Python via [PyO3](https://pyo3.rs), with no dependency on PyTorch, CUDA, or OpenBLAS (optional).

**Core design goals:**
- Sub-byte quantized inference (4-bit, 8-bit, 16-bit) with native SIMD acceleration
- Full CPU utilization via automatic multi-threading across all cores
- Portable across x86-64 (AVX2/AVX512) and ARM64 (NEON) with runtime dispatch
- A clean, PyTorch-compatible Python API with no learning curve
- Optional GPU acceleration via WebGPU/wgpu (experimental)

---

## Features

- **Native Packed Precision** — Inference at 4-bit, 8-bit, 16-bit, and 32-bit with no post-training quantization step. SWAR and SIMD-accelerated GEMV/ReLU operating directly on packed u32 words. U4x8 achieves **61 GFLOP/s** on CPU (7.4× faster than f32, 8× memory savings).
- **Vectorized CPU Kernels** — Hand-optimized SIMD kernels with runtime dispatch: AVX512 → AVX2 → NEON → scalar fallback. Includes Cephes-style fast approximations for transcendental functions (`exp`, `log`).
- **SIMD-Accelerated In-Place Operations** — `add_()`, `mul_()`, `sub_()`, `div_()` with AVX2 vectorization and rayon parallelism for large tensors (>2048 elements).
- **Multi-threading** — Automatic parallelism via [rayon](https://github.com/rayon-rs/rayon) with cache-aware chunking for both memory-bound and compute-bound workloads.
- **Native Autograd** — Built-in automatic differentiation engine with operation tracking, backward passes, and `no_grad` context support.
- **Optimized Convolutions** — im2col-based Conv2d with specialized kernels for 1×1, depthwise, and 3×3 convolutions at various stride/dilation configurations.
- **BLAS Integration** — OpenBLAS backend for large matrix multiplication. Matmul performance within 1.2× of PyTorch MKL at most sizes.
- **PyTorch Model Export** — Load pretrained PyTorch models and run inference through fastnn (see limitations below).
- **PyO3 Python Bindings** — Train and evaluate models from Python with a PyTorch-like API.
- **Training Utilities** — Datasets, DataLoaders, and Keras-style Callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger).
- **GPU Acceleration** *(experimental)* — Cross-platform GPU compute via [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12). Basic elementwise ops and tiled matrix multiplication. Not recommended for production use yet.

---

## Packed Precision Performance

Native low-bit inference — weights stored packed in u32 words, SIMD/SWAR kernels operating directly on the packed representation without unpacking.

### GEMV (Matrix × Vector) at 4096×4096 — 8 threads

| Implementation         | Time     | GFLOP/s | vs f32 baseline | Memory |
|------------------------|----------|---------|-----------------|--------|
| PyTorch f32 (MKL BLAS) | 4.04ms   | 8.3     | 1.0×            | 64 MB  |
| PyTorch int8 (dynamic) | 4.39ms   | 7.6     | 0.9×            | 64 MB  |
| **fastnn F16x2**       | **1.80ms** | **18.6** | **2.2×**      | **32 MB** |
| **fastnn U8x4**        | **0.76ms** | **44.4** | **5.3×**      | **16 MB** |
| **fastnn U4x8**        | **0.55ms** | **61.1** | **7.4×**      | **8 MB**  |

> Measured on AMD Ryzen 7 3700X, 8 cores. PyTorch uses MKL BLAS. All fastnn results use Rayon multi-threading and AVX2 SIMD.

### Packed Precision Types

| Type     | Bits/Value | Values/u32 | Memory vs f32 | Use Case                       |
|----------|-----------|------------|---------------|--------------------------------|
| `F32x1`  | 32        | 1          | 1×            | Baseline / master weights      |
| `F16x2`  | 16        | 2          | 2×            | Inference with moderate precision |
| `U8x4`   | 8         | 4          | 4×            | Balanced speed/memory          |
| `U4x8`   | 4         | 8          | **8×**        | Maximum compression            |

### How It Works

1. **Packed Storage** — N values packed into a single u32 word (e.g., 8 × 4-bit integers per u32)
2. **SWAR Operations** — Element-wise ops (ReLU, add, max) operate on raw u32 words via bit masking — no unpacking needed
3. **SIMD GEMV** — Type-dispatched AVX2/F16C kernels with prefetch, branchless sign-extend, and 2× instruction-level parallelism
4. **Training** — f32 master weights kept for the optimizer; packed representation used for forward/backward passes

### ReLU Performance (SWAR — no unpacking)

| Elements | F32x1  | F16x2  | U8x4   | U4x8       |
|----------|--------|--------|--------|------------|
| 262,144  | 0.17ms | 0.23ms | 0.16ms | **0.09ms** |

SWAR ReLU uses IEEE 754 sign-bit masking — processes 8 u32 words (up to 64 values) at once via AVX2.

---

## CPU Performance vs PyTorch

Benchmarks measured on AMD Ryzen 7 3700X (Arch Linux), comparing fastnn (f32, OpenBLAS, AVX2 SIMD) against PyTorch CPU (f32, MKL BLAS). All times are steady-state medians with warmup.

### Elementwise Operations (AVX2 SIMD)

| Operation | Size | fastnn | PyTorch | Ratio | Winner |
|-----------|------|--------|---------|-------|--------|
| **Add** | 100×100 | **1.8μs** | 2.8μs | **0.62×** | ✅ fastnn |
| **Mul** | 100×100 | **1.6μs** | 3.6μs | **0.44×** | ✅ fastnn |
| **ReLU** | 100×100 | **2.2μs** | 2.9μs | **0.77×** | ✅ fastnn |
| **Mul** | 256×256 | **13.5μs** | 14.9μs | **0.91×** | ✅ fastnn |
| **Add** | 256×256 | 12.9μs | 12.8μs | 1.01× | ≈ equal |
| **ReLU** | 256×256 | 10.5μs | 21.2μs | **0.49×** | ✅ fastnn |
| **Add** | 512×512 | 55.6μs | 20.2μs | 2.75× | torch |
| **Mul** | 512×512 | 58.7μs | 60.1μs | **0.98×** | ✅ fastnn |
| **ReLU** | 512×512 | 42.7μs | 35.1μs | 1.22× | torch |
| **Add** | 1000×1000 | 417μs | 118μs | 3.5× | torch |
| **ReLU** | 1000×1000 | 183μs | 89μs | 2.1× | torch |

> **Key finding:** Rayon parallelism **hurts** memory-bound operations. Single-threaded AVX2 with sequential memory access is faster than rayon for add/mul because memory bandwidth is shared across cores and rayon's task spawning overhead dominates. fastnn matches or beats PyTorch for tensors up to 256×256 using pure SIMD.

### MatMul (OpenBLAS-accelerated)

| Size | fastnn | PyTorch | Ratio |
|------|--------|---------|-------|
| 64×128×64 | 12μs | 17μs | **0.75×** ✅ |
| 128×256×128 | 74μs | 71μs | 1.03× ≈ |
| 256×512×256 | 450μs | 430μs | 1.05× ≈ |
| 512×1024×512 | 1688μs | 2171μs | **0.78×** ✅ |
| 1024×1024×1024 | 8498μs | 7365μs | 1.15× |

> Matmul went from **28× slower** to **within 1.2×** of PyTorch MKL after OpenBLAS integration and dispatch fast path.

### Packed Precision GEMV (the killer feature)

| Type | 4096×4096 GEMV | GFLOP/s | vs f32 | Memory |
|------|---------------|---------|--------|--------|
| PyTorch f32 (MKL) | 4.04ms | 8.3 | 1.0× | 64 MB |
| **fastnn F16x2** | **1.80ms** | **18.6** | **2.2×** | **32 MB** |
| **fastnn U8x4** | **0.76ms** | **44.4** | **5.3×** | **16 MB** |
| **fastnn U4x8** | **0.55ms** | **61.1** | **7.4×** | **8 MB** |

> Packed precision is where fastnn **excels** — 7.4× faster than PyTorch f32 with 8× memory savings. The SIMD GEMV kernels operate directly on packed u32 words without unpacking.

---

## Performance Optimizations (v0.7.0)

This release includes **88+ performance optimizations** across 18 commits, achieving:
- Matmul: **28× slower → within 1.2×** of PyTorch MKL
- Small tensor ops: **33-60× faster** (rayon threshold fix)
- Training step: **50+ fewer allocations** per backward pass

### Critical Fixes
- **Removed debug `eprintln!`** from MatmulBackward hot path (~100× matmul backward speedup)
- **Memory safety fix** in `add_()`/`mul_()` for non-contiguous tensor views
- **OpenBLAS integration** — switched from reference BLAS to optimized OpenBLAS (28× → 1.2× matmul)
- **Rayon threshold fix** — removed rayon from memory-bound ops (add/mul) where parallel overhead (2ms task spawning) dominated compute

### SIMD Vectorization
- Sigmoid/tanh kernels: AVX2/AVX512 `fast_exp` directly on SIMD registers (8-16×)
- `Tensor::add()`/`mul()`: AVX2 fast paths skipping dispatch for contiguous same-shape F32
- `add_()`/`mul_()`/`sub_()`/`div_()`: AVX2 vectorization with contiguous safety checks
- Softmax: fused exp computation with output storage (no double compute)
- `gt_scalar_kernel`: AVX2 comparison + blendv for relu backward
- `sum_last_dim_contiguous`: AVX2 SIMD reduction kernel with parallel row processing

### Dispatch Overhead Reduction
- `Tensor::add()`/`mul()`/`relu()`: fast paths bypassing dispatcher for common cases
- `Tensor::sum()`: fast path for contiguous last-dim sum
- `matmul_fast_2d()`: direct BLAS call for 2D CPU F32 (skip dispatch)
- `dispatch()`: `unwrap_or_else` instead of `expect+format!` (avoid string alloc)
- `dim_scalar()`: cached `OnceLock<[Tensor; 8]>` for dimension arguments
- Attention/Conv2d/MaxPool2d/BatchNorm1d/LayerNorm: pre-allocated scalar tensors

### Allocation Reduction (50+ fewer allocs per training step)
- Adam/AdamW/SGD/Muon: in-place scalar ops (`mul_scalar_`, `add_scalar_`) — 30+ fewer allocs/step
- TensorIterator: `SmallVec<[&[u8]; 4]>` eliminates per-element heap alloc
- `StoragePool` for output tensor reuse via `Tensor::zeros`
- Fused backward computations (SiLU, Sigmoid, Tanh, Gelu, MSELoss, Log, Sqrt)
- Iterator carry-based offset increment (O(numel) instead of O(numel×ndim))

### Parallelization
- Batched BLAS: rayon parallelization across batch dimension (for compute-bound GEMM)
- MaxPool2d: parallelize over (batch × channels)
- `max_kernel`: parallel rayon reduction across blocks
- im2col: parallelize over batch dimension

### Other
- Atomic ordering: `SeqCst` → `Relaxed` for all training flags
- Dropout: `thread_rng()` batch generation instead of `rand::random()` per element
- `as_f32_slice()`: fixed missing `storage_offset` (correctness bug)
- `to_numpy()`: contiguous F32 fast path, O(n) index tracking
- Python `stack()`: `np.stack` instead of element-by-element loop
- Python `numpy()`: pass `dtype=np.float32` upfront
- Removed `println!` from serialize save/load

---

## Project Structure

```
fastnn/
├── Cargo.toml                  # Rust dependencies (PyO3, rayon, wgpu, half, ...)
├── pyproject.toml              # Python package configuration (maturin)
├── Makefile                    # Common dev tasks (install, build, test, bench)
├── src/
│   ├── lib.rs                  # Python module export & PyO3 bindings
│   ├── tensor.rs               # Core Tensor struct, shape, strides, dtype
│   ├── storage/                # Memory backend, device allocation (CPU/GPU)
│   ├── autograd/               # Automatic differentiation tape and backward graph
│   ├── dispatcher/             # Dynamic kernel dispatch (CPU vs GPU)
│   ├── kernels/
│   │   ├── cpu.rs              # SIMD kernels: AVX2, AVX512, NEON, scalar fallbacks
│   │   ├── blas.rs             # BLAS-accelerated matrix multiplication (optional)
│   │   └── gpu/                # WebGPU compute pipelines and WGSL shaders (experimental)
│   ├── nn/                     # Neural network layers, activations, attention
│   ├── optim/                  # SGD, Adam, AdamW, Muon optimizers
│   ├── train/                  # Trainer, callbacks, metrics, loss functions
│   ├── io/                     # Model serialization (safetensors), DLPack
│   ├── dtypes/                 # Packed precision types (U4x8, U8x4, F16x2, F32x1)
│   ├── swar/                   # SWAR operations (add, sub, relu, max on raw u32)
│   ├── packed_tensor.rs        # PackedTensor<T> with scale/zero dequantization
│   ├── packed_layer.rs         # PackedLinear<T> with auto backend selection
│   └── packed_train.rs         # MasterWeightOptimizer for f32 master weights
├── fastnn/                     # Python package
│   ├── __init__.py             # Public API surface
│   ├── nn.py                   # Sequential, ModuleList
│   ├── parallel.py             # DataParallel / DDP (experimental)
│   ├── models/                 # Pre-built models: MLP, Transformer
│   ├── data.py                 # Dataset, TensorDataset, DataLoader
│   └── callbacks.py            # Training callbacks
├── benches/
│   └── packed_bench.rs         # Packed precision GEMV/ReLU benchmarks
└── tests/
    ├── test_autograd.py        # Gradient checking tests
    ├── test_gradients.py       # Numerical gradient verification
    ├── test_nn.py              # Neural network layer tests
    ├── test_tensor.py          # Tensor operation tests
    ├── bench_comparison.py     # Benchmarks vs PyTorch
    └── bench_tensor_ops.py     # Tensor operation benchmarks
```

---

## Installation

### Prerequisites

| Tool   | Version | Purpose                   |
|--------|---------|---------------------------|
| Rust   | stable  | Build the core library    |
| Python | ≥ 3.12  | Python bindings           |
| uv     | latest  | Python dependency management |

Install Rust via [rustup](https://rustup.rs):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build & Install

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
make install
```

### BLAS Backend

For optimal matmul performance, install OpenBLAS:
```bash
# Arch Linux
sudo pacman -S openblas

# Ubuntu/Debian
sudo apt install libopenblas-dev
```

### Platform Support

fastnn automatically selects the best available instruction set at runtime:

| Platform         | ISA Support                     |
|------------------|---------------------------------|
| x86-64 (desktop) | AVX512 → AVX2 → scalar fallback |
| ARM64 (e.g. Raspberry Pi 4/5) | NEON intrinsics   |
| Any other        | Scalar fallback                 |

---

## Quick Start

### Basic Inference

```python
import fastnn as fnn

# Create tensors
a = fnn.randn([1000, 1000])
b = fnn.randn([1000, 1000])
c = a @ b  # SIMD-accelerated matmul

# Define a model
model = fnn.Sequential(
    fnn.Linear(128, 64),
    fnn.ReLU(),
    fnn.Linear(64, 10),
)

# Inference without gradient tracking
inputs = fnn.randn([32, 128])
with fnn.no_grad():
    outputs = model(inputs)
```

### Training

```python
import fastnn as fnn

model     = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, activation="relu")
optimizer = fnn.Adam(model.parameters(), lr=1e-2)
ds        = fnn.TensorDataset(X, y)
loader    = fnn.DataLoader(ds, batch_size=4, shuffle=True)

callbacks = [
    fnn.EarlyStopping(monitor="loss", patience=10),
    fnn.ModelCheckpoint(dirpath="./checkpoints", monitor="loss", save_best_only=True),
    fnn.LearningRateScheduler(schedule="cosine", lr=1e-2, T_max=100),
]

model.train()
for epoch in range(100):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Controlling Parallelism

```python
# Set CPU thread count (defaults to all cores)
fnn.set_num_threads(4)

print(fnn.allocator_stats())
print(fnn.list_registered_ops())
```

### Packed Precision (Rust API)

```rust
use fastnn::{U4x8, U8x4, PackedTensor, PackedLinear, Linear4, MasterWeightOptimizer};

// 4-bit packed linear layer (512 → 2048, 8× memory savings vs f32)
let layer: Linear4 = PackedLinear::new(512, 2048, true);

// Forward pass — weights unpacked + AVX2 FMA automatically
let input = vec![1.0f32; 512];
let output = layer.forward(&input);

// Training with f32 master weights
let mut opt = MasterWeightOptimizer::<U8x4>::new(
    master_weights, 0.001, (0.9, 0.999), 1e-8, 0.01,
);
let packed_weights = opt.step(&gradients);

// SWAR ReLU — operates on packed u32 words, no unpacking needed
fastnn::backends::cpu::relu_cpu(&mut packed_tensor);
```

---

## PyTorch Model Export

Load pretrained PyTorch models and run inference through fastnn.

```python
import torch
import torchvision.models as models
from fastnn.export import export_pytorch_model, load_fnn_model

# Export
model = models.resnet18(pretrained=True)
model.eval()
export_pytorch_model(model, "resnet18.fnn")

# Load and run
fnn_model = load_fnn_model("resnet18.fnn")
output = fnn_model(fnn.tensor(data, shape))
```

### Supported Layers

| Layer                                    | Status | Notes                              |
|------------------------------------------|--------|------------------------------------|
| `Linear`                                 | ✅     | Weight transpose handled automatically |
| `Conv2d`                                 | ✅     | Full support                       |
| `BatchNorm1d` / `BatchNorm2d`            | ✅     | BatchNorm2d mapped to BatchNorm1d  |
| `ReLU`, `GELU`, `SiLU`                  | ✅     |                                    |
| `LayerNorm`, `Embedding`, `Dropout`     | ✅     |                                    |
| `AdaptiveAvgPool2d`                      | ✅     | Output size must be (1,1)          |
| `MaxPool2d`                              | ✅     |                                    |
| Residual/skip connections (e.g. ResNet BasicBlock) | ❌ | Not yet supported — exported ResNets will produce incorrect outputs |

> **Note:** Skip connections are not currently supported. This means architectures like ResNet, EfficientNet, or any model using `x + residual` will produce incorrect results when exported. Plain sequential models (VGG-style, MLP, basic Transformers) work correctly.

---

## API Reference

### Tensor Creation

| Function                        | Description                      |
|---------------------------------|----------------------------------|
| `fnn.tensor(data, shape)`       | Create tensor from a Python list |
| `fnn.zeros(shape)`              | Tensor of zeros                  |
| `fnn.ones(shape)`               | Tensor of ones                   |
| `fnn.full(shape, value)`        | Tensor filled with a value       |
| `fnn.eye(n)`                    | Identity matrix                  |
| `fnn.arange(end)`               | Integer range `[0, end)`         |
| `fnn.linspace(start, end, n)`   | Linearly spaced values           |
| `fnn.randn(shape)`              | Random normal (Gaussian)         |
| `fnn.rand(shape)`               | Random uniform `[0, 1)`          |
| `fnn.randint(low, high, shape)` | Random integers in `[low, high)` |

### Neural Network Modules

| Module                                              | Description                  |
|-----------------------------------------------------|------------------------------|
| `fnn.Linear(in, out, bias=True)`                    | Fully connected layer        |
| `fnn.Conv2d(cin, cout, kernel, stride, padding)`    | 2D convolution               |
| `fnn.LayerNorm(shape)`                              | Layer normalization          |
| `fnn.BatchNorm1d(features)`                         | Batch normalization          |
| `fnn.Dropout(p)`                                    | Dropout regularization       |
| `fnn.Embedding(num, dim)`                           | Learned word embeddings      |
| `fnn.ReLU` / `fnn.GELU` / `fnn.Sigmoid` / `fnn.Tanh` / `fnn.SiLU` | Activations |
| `fnn.Sequential(*layers)`                           | Sequential container         |
| `fnn.ModuleList(modules)`                           | Indexable module list        |

### Optimizers

| Optimizer                                                        | Description              |
|------------------------------------------------------------------|--------------------------|
| `fnn.SGD(params, lr, momentum=0, weight_decay=0)`               | Stochastic Gradient Descent |
| `fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8)`            | Adam                     |
| `fnn.AdamW(params, lr, betas=(0.9, 0.999), weight_decay=0.01)`  | AdamW (decoupled L2)     |
| `fnn.Muon(params, lr, momentum=0.95)`                            | Muon (orthogonalized momentum) |

### Loss Functions

| Function                                 | Description          |
|------------------------------------------|----------------------|
| `fnn.mse_loss(pred, target)`             | Mean squared error   |
| `fnn.cross_entropy_loss(logits, target)` | Cross-entropy loss   |

---

## Build Flags

| Feature       | Description                                       | Default |
|---------------|---------------------------------------------------|---------|
| `simd`        | SIMD kernels (AVX2, AVX512, NEON, F16C)           | on      |
| `parallel`    | Rayon multi-threaded parallelism                  | on      |
| `simd-avx512` | AVX-512 kernels (requires AVX-512 CPU)            | on      |
| `blas`        | BLAS-accelerated matmul                           | on      |
| `openblas`    | Link against OpenBLAS for large matmul            | off     |

---

## Testing & Benchmarking

```bash
# Rust unit + integration tests
cargo test

# Python unit and integration tests
pytest tests/ --ignore=tests/test_transformer.py

# Benchmarks vs PyTorch
pytest tests/bench_comparison.py tests/bench_tensor_ops.py --benchmark-only

# All tests including transformer (may crash due to pre-existing issue)
pytest tests/ -v
```

---

## Roadmap

- [ ] Thread-local memory pool to eliminate `Tensor::zeros` allocation overhead
- [ ] Non-temporal stores (`_mm256_stream_ps`) for large tensor elementwise ops
- [ ] Sigmoid/tanh/gelu SIMD vectorization for large tensors (AVX2 `fast_exp`)
- [ ] Residual/skip connection support in PyTorch export
- [ ] Raspberry Pi benchmark suite (ARM NEON validation)
- [ ] Stable GPU backend (wgpu)
- [ ] Multi-GPU training (experimental → stable)

---

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
