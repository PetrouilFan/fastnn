# fastnn

**fastnn** is a high-performance neural network library built from scratch in Rust, with seamless Python bindings. It supports both CPU and GPU training and inference — with sub-byte quantization, hand-written SIMD kernels, and a clean PyTorch-compatible API.

> **Version:** v1.1.0 — Modular backend architecture & GPU execution improvements

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

---

## Overview

fastnn is built for researchers and engineers who need efficient neural network computation on CPU hardware — from desktop x86 to ARM-based edge devices. It is implemented entirely in Rust and exposed to Python via [PyO3](https://pyo3.rs), with no dependency on PyTorch.

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
- **Fast MatMul** — Optimized pure-Rust GEMM via `matrixmultiply` crate: **88 GFLOP/s** at 1024×1024 on CPU.
- **Multi-threading** — Automatic parallelism via [rayon](https://github.com/rayon-rs/rayon) with cache-aware chunking for both memory-bound and compute-bound workloads.
- **Native Autograd** — Built-in automatic differentiation engine with operation tracking, backward passes, and `no_grad` context support.
- **Optimized Convolutions** — im2col-based Conv2d with specialized kernels for 1×1, depthwise, and 3×3 convolutions. Also Conv1d, Conv3d, and ConvTranspose2d.
- **PyO3 Python Bindings** — Train and evaluate models from Python with a PyTorch-like API.
- **Smart Data Loading** — Multi-threaded parallel DataLoader with automatic resource tuning. Asymmetric scaling: fast scale-up when data-bound, conservative scale-down to avoid GPU starvation.
- **GPU Acceleration** *(experimental)* — Cross-platform GPU compute via [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12). Basic elementwise ops and tiled matrix multiplication.

---

### Fused Conv+BN+SiLU Performance

Using the optimized fused kernel (`fused_conv_bn_silu`), FastNN achieves **14-25× speedup** over PyTorch's separate Conv2d+BatchNorm2d+SiLU operations on CPU:

| Configuration | PyTorch (separate) | FastNN (fused) | Speedup |
|---------------|-------------------|----------------|---------|
| Conv2d(32→64) + BN + SiLU (64×64) | 81.81 ms | 3.27 ms | **25.0×** |
| Conv2d(64→128) + BN + SiLU (32×32) | 42.55 ms | 2.01 ms | **21.2×** |
| Conv2d(128→256) + BN + SiLU (16×16) | 26.24 ms | 1.82 ms | **14.4×** |

These results demonstrate the effectiveness of operator fusion and SIMD-accelerated 3×3 convolution kernels.

---

## Packed Precision Performance

Native low-bit inference — weights stored packed in u32 words, SIMD/SWAR kernels operating directly on the packed representation without unpacking.

### GEMV (Matrix × Vector) at 4096×4096 — 8 threads

| Implementation         | Time     | GFLOP/s | vs f32 baseline | Memory |
|------------------------|----------|---------|-----------------|--------|
| PyTorch f32 (MKL BLAS) | 4.04ms   | 8.3     | 1.0×            | 64 MB  |
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

---

## CPU Performance

Benchmarks measured on AMD Ryzen 7 3700X (Arch Linux). All times are steady-state medians with warmup.

### MatMul (matrixmultiply — pure Rust GEMM)

| Size | fastnn | GFLOP/s |
|------|--------|---------|
| 256×256 | 0.5 ms | **66.5** |
| 512×512 | 3.1 ms | **87.6** |
| 1024×1024 | 24.2 ms | **88.6** |

### Main Tensor Operations

| Operation | Size | Time |
|---|---|---|
| ReLU | 1024×1024 | 0.30 ms |
| GELU | 1024×1024 | 1.06 ms |
| Conv2d | 2×8×32×32 → 16 | 1.64 ms |
| Transformer forward | (4, 32) | 10.73 ms |
| Transformer train step | (4, 32) | 10.28 ms |

---

## Project Structure

```
fastnn/
├── Cargo.toml                  # Rust dependencies (PyO3, rayon, wgpu, half, ...)
├── pyproject.toml              # Python package configuration (maturin)
├── src/
│   ├── lib.rs                  # Crate root, module declarations, public re-exports
│   ├── python/                 # PyO3 bindings and _core module registration
│   ├── tensor/                 # Tensor struct, shape, factories, ops, reductions
│   ├── storage.rs              # Memory backend, device allocation (CPU/GPU)
│   ├── storage_pool.rs         # Storage pooling for output tensor reuse
│   ├── dispatcher.rs           # Dynamic kernel dispatch (CPU vs GPU)
│   ├── autograd/               # Backward engine and per-family gradient nodes
│   ├── kernels/                # CPU/GPU backend kernels and registration
│   │   ├── cpu/                # Elementwise, reductions, matmul, conv, norm, losses
│   │   ├── gpu/                # WGPU kernels, buffers, sync policy, fusion, optimizers
│   │   └── blas.rs             # BLAS-accelerated matrix multiplication (optional)
│   ├── nn/                     # Neural network layers
│   │   ├── linear.rs           # Linear (fully connected)
│   │   ├── conv.rs             # Conv1d, Conv2d, Conv3d, ConvTranspose2d
│   │   ├── activations.rs      # ReLU, GELU, Sigmoid, Tanh, SiLU, LeakyReLU, Softplus, Hardswish
│   │   ├── attention.rs        # MultiHeadAttention
│   │   ├── transformer.rs      # TransformerBlock, TransformerEncoder
│   │   ├── norm.rs             # LayerNorm, BatchNorm1d, BatchNorm2d, RMSNorm, GroupNorm
│   │   ├── dropout.rs          # Dropout
│   │   ├── embedding.rs        # Embedding
│   │   ├── pooling.rs          # MaxPool2d, AvgPool2d
│   │   └── sequential.rs       # Sequential container
│   ├── optim/                  # Optimizers
│   │   ├── sgd.rs              # SGD with momentum
│   │   ├── adam.rs             # Adam / Adam (AMSGrad variant)
│   │   ├── adamw.rs            # AdamW
│   │   ├── muon.rs             # Muon (orthogonalized momentum)
│   │   └── lion.rs             # Lion (sign-based momentum)
│   ├── train/                  # Training utilities
│   │   ├── loss.rs             # MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, HuberLoss
│   │   ├── metrics.rs          # Accuracy metric
│   │   └── callbacks.rs        # EarlyStopping, ModelCheckpoint, LR Scheduler, CSVLogger
│   ├── io/
│   │   └── serialize.rs        # Model serialization (save/load)
│   ├── packed_tensor.rs        # PackedTensor<T> with scale/zero dequantization
│   ├── packed_layer.rs         # PackedLinear<T> with auto backend selection
│   └── packed_train.rs         # MasterWeightOptimizer for f32 master weights
├── fastnn/                     # Python package
│   ├── __init__.py             # Stable top-level public API facade
│   ├── tensor.py               # Tensor factories and Tensor alias
│   ├── ops.py                  # Tensor operations and reductions
│   ├── nn.py                   # Neural network module aliases
│   ├── losses.py               # Loss functions
│   ├── parallel.py             # DataParallel / DDP (experimental)
│   ├── models/                 # Pre-built models: MLP, Transformer
│   ├── data.py                 # Dataset, TensorDataset, DataLoader, auto-tuning
│   └── callbacks.py            # Training callbacks
├── tests/                      # Python test suite
│   └── conftest.py             # Memory pool isolation fixture
```

## Internal Architecture

FastNN is organized around a stable Python API and modular Rust backend layers:

1. Tensor API and shape/device behavior
2. Dispatcher and backend kernel registration
3. CPU, GPU, packed, and BLAS kernel implementations
4. Autograd nodes
5. PyO3 bindings and Python facade modules

GPU execution is asynchronous by default. Kernel launch paths enqueue work and synchronize only at explicit host readback boundaries such as `.item()`, `.numpy()`, `.to_cpu()`, DLPack export, or test-only barriers. See [Development Architecture](docs/development.md) and [Performance Roadmap](docs/performance-roadmap.md) for contributor workflows.

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
uv pip install -e .
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
c = fnn.matmul(a, b)  # Optimized GEMM

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

model.train()
for epoch in range(100):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loader.reset_sampler()  # Re-shuffle for next epoch
```

### Parallel Data Loading

The DataLoader supports multi-threaded parallel loading to overlap I/O with compute:

```python
# Explicit worker count (2 threads)
loader = fnn.DataLoader(ds, batch_size=32, num_workers=2)

# Auto-tuning: dynamically adjusts workers based on data loading speed
loader = fnn.DataLoader(ds, batch_size=32, num_workers="auto")

for epoch in range(100):
    for batch_x, batch_y in loader:
        # training logic
        pass
    loader.reset_sampler()  # Re-shuffle + auto-tune adjustment
```

| Mode | `num_workers` | Behavior |
|------|---------------|----------|
| Default | `0` | Single-threaded with prefetch |
| Parallel | `> 0` | Thread pool with N workers |
| Auto | `"auto"` | Asymmetric auto-tuning: fast scale-up, conservative scale-down |

The auto-tuner starts at 1 worker and adjusts based on mean wait time:
- **Scale up** immediately when mean wait > 30ms (prefetch first, then workers)
- **Scale down** only after 2 consecutive epochs below 5ms (avoids GPU starvation)
- Tunable via `up_threshold_ms`, `down_threshold_ms`, `scale_down_patience`

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
from fastnn.io import export_pytorch_model, load_fnn_model

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
| `BatchNorm1d` / `BatchNorm2d`            | ✅     |                                    |
| `ReLU`, `GELU`, `SiLU`                  | ✅     |                                    |
| `LayerNorm`, `Embedding`, `Dropout`     | ✅     |                                    |
| `AdaptiveAvgPool2d`                      | ✅     | Output size must be (1,1)          |
| `MaxPool2d`                              | ✅     |                                    |
| Residual/skip connections (e.g. ResNet BasicBlock) | ❌ | Not yet supported |

> **Note:** Skip connections are not currently supported. Plain sequential models (VGG-style, MLP, basic Transformers) work correctly.

---

## API Reference

### Tensor Creation

| Function                        | Description                      |
|---------------------------------|----------------------------------|
| `fnn.tensor(data, shape)`       | Create tensor from a Python list |
| `fnn.zeros(shape, dtype)`       | Tensor of zeros                  |
| `fnn.ones(shape, dtype)`        | Tensor of ones                   |
| `fnn.full(shape, value, dtype)` | Tensor filled with a value       |
| `fnn.eye(n)`                    | Identity matrix                  |
| `fnn.arange(end)`               | Integer range `[0, end)`         |
| `fnn.linspace(start, end, n)`   | Linearly spaced values           |
| `fnn.randn(shape)`              | Random normal (Gaussian)         |
| `fnn.rand(shape)`               | Random uniform `[0, 1)`          |
| `fnn.randint(low, high, shape)` | Random integers in `[low, high)` |

### Tensor Operations

| Method / Function               | Description                      |
|---------------------------------|----------------------------------|
| `a + b`, `a - b`, `a * b`       | Elementwise operations           |
| `fnn.matmul(a, b)`              | Matrix multiplication            |
| `fnn.einsum(eq, [a, b])`        | Einstein summation               |
| `fnn.cat([a, b], dim)`          | Concatenate tensors              |
| `x.repeat(shape)`               | Repeat tensor elements           |
| `x.where_tensor(cond, other)`   | Conditional selection            |
| `x.reshape(shape)`              | Reshape (supports -1)            |
| `x.transpose(d0, d1)`           | Transpose two dimensions         |
| `x.permute(dims)`               | Permute all dimensions           |
| `x.view(shape)`                 | View (no copy)                   |
| `x.squeeze(dim)` / `x.unsqueeze(dim)` | Add/remove dimensions    |
| `x.flip(dim)` | Reverse along dimension |
| `fnn.maximum(a, b)` | Elementwise maximum (with broadcasting) |
| `fnn.minimum(a, b)` | Elementwise minimum (with broadcasting) |
| `fnn.stack([a, b], dim)` | Stack tensors along new dimension |
| `x.sum(dim, keepdim)` | Sum reduction |
| `x.mean(dim, keepdim)`          | Mean reduction                   |
| `x.max(dim, keepdim)`           | Max reduction                    |
| `x.min(dim, keepdim)`           | Min reduction                    |
| `x.softmax(dim)`                | Softmax                          |
| `x.log_softmax(dim)`            | Log softmax                      |

### Neural Network Modules

| Module                                              | Description                  |
|-----------------------------------------------------|------------------------------|
| `fnn.Linear(in, out, bias=True)`                    | Fully connected layer        |
| `fnn.Conv1d(cin, cout, kernel, stride, padding)`    | 1D convolution               |
| `fnn.Conv2d(cin, cout, kernel, stride, padding)`    | 2D convolution               |
| `fnn.Conv3d(cin, cout, kernel, stride, padding)`    | 3D convolution               |
| `fnn.ConvTranspose2d(cin, cout, kernel, stride, padding)` | Transposed 2D conv  |
| `fnn.LayerNorm(shape)`                              | Layer normalization          |
| `fnn.BatchNorm1d(features)`                         | Batch normalization 1D       |
| `fnn.BatchNorm2d(features)`                         | Batch normalization 2D       |
| `fnn.RMSNorm(shape)`                                | RMS normalization              |
| `fnn.GroupNorm(num_groups, num_channels)`            | Group normalization          |
| `fnn.Dropout(p)`                                    | Dropout regularization       |
| `fnn.Embedding(num, dim)`                           | Learned embeddings           |
| `fnn.MultiHeadAttention(d_model, num_heads)`         | Multi-head self-attention    |
| `fnn.TransformerBlock(...)`                         | Transformer encoder block    |
| `fnn.TransformerEncoder(...)`                       | Full transformer encoder     |
| `fnn.ReLU` / `fnn.GELU` / `fnn.Sigmoid` / `fnn.Tanh` / `fnn.SiLU` | Activations |
| `fnn.LeakyReLU(negative_slope)`                     | Leaky ReLU                   |
| `fnn.Softplus(beta, threshold)`                     | Softplus activation          |
| `fnn.Hardswish`                                     | Hard swish activation        |
| `fnn.Sequential(*layers)`                           | Sequential container         |
| `fnn.ModuleList(modules)`                           | Indexable module list        |

### Optimizers

| Optimizer                                                        | Description              |
|------------------------------------------------------------------|--------------------------|
| `fnn.SGD(params, lr, momentum=0, weight_decay=0)`               | Stochastic Gradient Descent |
| `fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8)`            | Adam                     |
| `fnn.AdamW(params, lr, betas=(0.9, 0.999), weight_decay=0.01)`  | AdamW (decoupled L2)     |
| `fnn.Muon(params, lr, momentum=0.95)`                            | Muon (orthogonalized momentum) |
| `fnn.Lion(params, lr, betas=(0.95, 0.98))`                       | Lion (sign-based momentum) |

### Loss Functions

| Function                                 | Description          |
|------------------------------------------|----------------------|
| `fnn.mse_loss(pred, target)`             | Mean squared error   |
| `fnn.cross_entropy_loss(logits, target)` | Cross-entropy loss   |
| `fnn.bce_with_logits(input, target)`     | BCE with logits      |
| `fnn.huber_loss(input, target, delta)`   | Huber (smooth L1)    |

### Model I/O

| Function                                 | Description          |
|------------------------------------------|----------------------|
| `fnn.save_model(model, path)`            | Save model weights   |
| `fnn.load_model(path)`                   | Load model weights   |
| `fnn.save_optimizer(opt, path)`          | Save optimizer state |
| `fnn.load_optimizer(opt, path)`          | Load optimizer state |

### Attention

| Function                                 | Description          |
|------------------------------------------|----------------------|
| `fnn.flash_attention(q, k, v)`           | Memory-efficient attention (O(N) memory) |
| `fnn.flash_attention(q, k, v, causal=True)` | Causal FlashAttention |

FlashAttention is mathematically equivalent to standard attention (max diff < 1e-7) but uses block-wise tiling with online softmax to avoid materializing the full N×N attention scores matrix.

---

## Build Flags

| Feature       | Description                                       | Default |
|---------------|---------------------------------------------------|---------|
| `simd`        | SIMD kernels (AVX2, AVX512, NEON, F16C)           | on      |
| `parallel`    | Rayon multi-threaded parallelism                  | on      |
| `simd-avx512` | AVX-512 kernels (requires AVX-512 CPU)            | on      |
| `blas`        | BLAS-accelerated matmul (requires system cblas)   | off     |
| `openblas`    | Link against OpenBLAS for large matmul            | off     |

> **Note:** The `blas` feature is disabled by default because the system BLAS reference implementation is very slow. The default `matrixmultiply` pure-Rust GEMM is faster on most systems. If you have OpenBLAS installed, enable the `blas` feature for potentially better performance on very large matrices.

---

## Testing & Benchmarking

```bash
# Rust unit + integration tests
cargo test

# Python unit and integration tests
uv run pytest tests/ -v

# Packed precision benchmarks
cargo bench --bench packed_bench
```

---

## Roadmap

- [ ] Residual/skip connection support in PyTorch export
- [ ] GPU training (backward pass on GPU storage)
- [ ] Raspberry Pi benchmark suite (ARM NEON validation)
- [ ] Multi-GPU training
- [ ] ONNX model import
- [ ] FlashAttention SIMD optimization (AVX2/AVX512 block kernels)
- [ ] True process-based multiprocessing for DataLoader (requires PyTensor pickle support or shared memory)

---

## License

fastnn is licensed under the [MIT License](LICENSE).
Copyright © 2026 Petros Fanioudakis
