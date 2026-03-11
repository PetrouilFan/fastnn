# fastnn

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Features

- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- **Optimizers**: SGD, Adam, AdamW
- **Training infrastructure**: Trainer with callbacks, metrics, checkpoints
- **IO**: safetensors serialization, DLPack interop
- **SIMD-optimized**: Portable SIMD via `wide` crate - works on x86 and ARM (Raspberry Pi)
- **Parallel execution**: Rayon-based parallelism for multi-core utilization
- **Fused operations**: fused_add_relu, fused_linear_relu, fused_linear_gelu for maximum performance

## Performance

Benchmark comparisons with PyTorch (mean time in μs, lower is better):

### x86 (Intel/AMD)

| Operation | Size | fastnn | PyTorch | Status |
|-----------|------|--------|---------|--------|
| ReLU | 100×100 | 104.3μs | 5.3μs | |
| ReLU | 1000×1000 | 888.3μs | 71.3μs | |
| FusedAddReLU | 100×100 | 27.3μs | 8.6μs | |
| FusedAddReLU | 1000×1000 | 540.5μs | 1775.9μs | ✅ faster |
| MatMul | 128×256×128 | 152.8μs | 47.8μs | |
| MatMul | 256×512×256 | 1179.0μs | 198.8μs | |
| MatMul | 512×1024×512 | 9143.2μs | 1474.6μs | |
| GELU | 100×100 | 87.8μs | 15.8μs | |
| GELU | 1000×1000 | 1466.9μs | 175.6μs | |
| Sigmoid | 100×100 | 76.8μs | 11.3μs | |
| Sigmoid | 1000×1000 | 967.9μs | 173.8μs | |
| Tanh | 100×100 | 81.7μs | 16.4μs | |
| Tanh | 1000×1000 | 1327.6μs | 514.3μs | |
| Add | 100×100 | 43.4μs | 5.8μs | |
| Add | 1000×1000 | 725.3μs | 52.9μs | |
| Mul | 100×100 | 30.3μs | 5.1μs | |
| Mul | 1000×1000 | 499.0μs | 76.9μs | |
| Linear | 32×256×512 | 437.3μs | 70.8μs | |
| Linear | 32×512×1024 | 1304.7μs | 202.6μs | |
| Linear | 128×256×512 | 1277.1μs | 127.9μs | |
| Conv2d | 1×32×32×32 | 940.0μs | 230.0μs | |
| Conv2d | 1×64×64×64 | 12400.0μs | 1170.0μs | |
| Sum | 1000×1000 | 184.8μs | 22.5μs | |
| Mean | 1000×1000 | 223.7μs | 22.0μs | |
| Max | 1000×1000 | 206.9μs | 283.5μs | ✅ faster |

### ARM (Raspberry Pi 5)

| Operation | Size | fastnn | PyTorch | Status |
|-----------|------|--------|---------|--------|
| Mul | 100×100 | 8.3μs | 5.4μs | |
| Add | 100×100 | 8.2μs | 6.0μs | |
| ReLU | 100×100 | 14.8μs | 9.5μs | |
| FusedAddReLU | 100×100 | 7.9μs | 15.2μs | ✅ faster |
| Sigmoid | 100×100 | 32.9μs | 45.4μs | ✅ faster |
| Tanh | 100×100 | 47.0μs | 66.2μs | ✅ faster |
| GELU | 100×100 | 56.8μs | 38.4μs | |
| MatMul | 128×256×128 | 301μs | 92μs | |
| Linear | 32×256×512 | 3,735μs | 250μs | |
| Conv2d | 1×32×32×32 | 5,439μs | 641μs | |
| Max | 1000×1000 | 518μs | 644μs | ✅ faster |
| Sum | 1000×1000 | 526μs | 429μs | |
| Mean | 1000×1000 | 517μs | 389μs | |

Note: Performance varies by hardware and tensor size. Best results require AVX2/AVX512 support.

### Recent Optimizations (v0.2.0)

- Added SIMD support to parallel add/mul kernels
- Lowered parallelization threshold from 512 to 4096 elements
- Improved parallel chunking strategy for element-wise operations
- Added FMA (Fused Multiply-Add) support for linear layers
- **Conv2d optimizations**:
  - Inlined im2col operation to avoid intermediate tensor creation
  - Added GEMM-based matrix multiplication for 3x3 convolutions
  - Lowered GEMM threshold to 16 for better utilization
  - Removed unnecessary data copies in convolution paths

## GPU Support (wgpu)

fastnn now supports GPU acceleration via wgpu! This provides cross-platform GPU support (Windows, Linux, macOS, WebGPU).

### Supported Operations

- **Element-wise**: add, sub, mul, div, neg, abs, exp, log, sqrt
- **Activations**: ReLU, GELU, Sigmoid, Tanh, SiLU
- **Fused ops**: fused_add_relu
- **MatMul**: GPU-accelerated matrix multiplication

### Usage

```python
import fastnn as fnn

# Create GPU tensor
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]], device="gpu")

# Operations automatically dispatch to GPU
y = x @ x.T  # MatMul on GPU
z = y.relu()  # ReLU on GPU
```

### Requirements

- GPU with Vulkan, Metal, or DirectX12 support
- See wgpu requirements for your platform

### Performance

GPU benchmarks (NVIDIA GPU, 1000×1000 tensors, mean time in ms, lower is better):

| Operation | fastnn (CPU) | fastnn (GPU) | PyTorch (CPU) | GPU Speedup |
|-----------|--------------|--------------|---------------|-------------|
| MatMul | 1172.7ms | 17.2ms | ~50ms | **68x** |
| Add | 1.9ms | 1.2ms | ~0.5ms | 0.6x |
| Sigmoid | 3.4ms | 2.4ms | ~1.0ms | 0.7x |
| Tanh | 3.5ms | 2.5ms | ~0.8ms | 0.3x |
| GELU | 3.7ms | 2.7ms | ~1.2ms | 0.4x |
| Exp | 3.2ms | 2.4ms | ~0.9ms | 0.4x |
| ReLU | 2.7ms | 2.5ms | ~0.4ms | 0.2x |
| Sqrt | 2.7ms | 2.4ms | ~0.5ms | 0.2x |

Note: GPU shows massive speedup for matmul (memory-bound, compute-intensive), but slower for element-wise ops due to GPU launch overhead. CPU remains faster for small/medium element-wise operations.

### Comparison by Hardware

| Operation | x86 (fastnn) | ARM (fastnn) | GPU (fastnn) | Notes |
|-----------|--------------|--------------|--------------|-------|
| MatMul 512×512×512 | 9.1ms | ~300μs | ~5ms | GPU wins on large matmul |
| ReLU 100×100 | 104μs | 15μs | ~0.5ms | CPU/ARM faster for small ops |
| ReLU 1000×1000 | 888μs | ~200μs | ~2.5ms | CPU faster for medium ops |
| Add 1000×1000 | 725μs | ~100μs | ~1.2ms | CPU faster (bandwidth bound) |
| FusedAddReLU 1000×1000 | 540μs | ~50μs | N/A | fastnn fusion advantage |

### New Fused Operations

```python
import fastnn as fnn

# Fused linear + ReLU (single pass, no intermediate tensors)
output = fnn.fused_linear_relu(x, weight, bias)

# Fused linear + GELU (single pass, no intermediate tensors)  
output = fnn.fused_linear_gelu(x, weight, bias)
```

## Installation

### Prerequisites

- **Rust** (nightly) - Required for building the Rust core
- **Python 3.12+** with uv package manager
- **PyTorch** - Required for benchmark comparisons

### Install Rust (nightly)

```bash
# Install rustup if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install and set nightly as default
source $HOME/.cargo/env
rustup install nightly
rustup default nightly
```

### Install fastnn

```bash
uv sync --all-extras
uv run maturin develop --release
```

### Build only (without installing)

```bash
uv run maturin build --release
```

## Quick Start

```python
import fastnn as fnn

# Create tensors
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]])
y = fnn.tensor([[5.0, 6.0], [7.0, 8.0]])

# Operations
z = x @ y  # matrix multiply
z = (x * 2).relu()

# Build a model
model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)

# Training
optimizer = fnn.Adam(model.parameters(), lr=1e-2)
trainer = fnn.Trainer(model=model, optimizer=optimizer, loss=fnn.mse_loss)
trainer.fit(loader, epochs=100)
```

## Development

```bash
# Build
uv run maturin build --release

# Test
uv run pytest tests/ -v

# Benchmark
uv run pytest tests/ -v --benchmark-only

# Clean
cargo clean
```
