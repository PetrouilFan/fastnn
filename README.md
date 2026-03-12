# fastnn

A high-performance deep learning library with a Rust core, featuring SIMD optimization, multi-core parallelism, and GPU acceleration.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Rust Edition 2021](https://img.shields.io/badge/Rust-2021-blue.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://crates.io/crates/fastnn)

## Features

| Category | Capabilities |
|----------|-------------|
| **Tensor Operations** | Full autograd support, 50+ operations, DLPack interop |
| **Neural Networks** | Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations |
| **Optimizers** | SGD, Adam, AdamW with momentum and weight decay |
| **Training** | Trainer with callbacks, metrics tracking, checkpoints |
| **Performance** | SIMD (AVX2/AVX512/NEON), Rayon parallelism, fused operations |
| **IO** | safetensors serialization, PyTorch model loading |

## What Makes fastnn Different

fastnn is architected from the ground up for performance, combining Python's simplicity with Rust's speed:

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Rust Core** | PyO3 bindings | Zero-cost Python interface |
| **SIMD** | AVX2/AVX512/NEON via `wide` crate | 3-4x vectorization |
| **Parallel** | Rayon-based multi-core | Linear scaling with cores |
| **Fused Ops** | Single-pass kernels | No intermediate allocations |
| **GPU** | wGPU compute shaders | 68x speedup on matmul |
| **Memory** | mimalloc custom allocator | Reduced fragmentation |

### Key Differentiators

**Fused Operations** — Eliminates intermediate tensor allocations:

```python
# Single pass, no intermediate tensors
output = fnn.fused_linear_relu(x, weight, bias)
output = fnn.fused_linear_gelu(x, weight, bias)
```

**3.3x faster** than PyTorch for fused operations (1000×1000 tensors).

**Platform-Specific Optimizations:**
- **x86**: AVX2/AVX512 SIMD instructions
- **ARM**: NEON SIMD (Raspberry Pi 5, Apple Silicon)  
- **GPU**: wGPU compute shaders for large operations

### Architecture

```
Python API → PyO3 → Rust Core → SIMD/Parallel/GPU Kernels
```

### Fused Operations

```python
import fastnn as fnn

# Fused linear + ReLU (single pass, no intermediate tensors)
output = fnn.fused_linear_relu(x, weight, bias)

# Fused linear + GELU (single pass, no intermediate tensors)  
output = fnn.fused_linear_gelu(x, weight, bias)
```

## Table of Contents

- [What Makes fastnn Different](#what-makes-fastnn-different)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Performance](#performance)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- **Rust** (nightly) - Required for building the Rust core
- **Python 3.12+** with [uv](https://github.com/astral-sh/uv) package manager
- **OpenBLAS** (optional) - For faster matmul/linear on ARM: `sudo apt-get install libopenblas-dev`

### Install Rust (nightly)

```bash
# Install rustup if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install and set nightly as default
source $HOME/.cargo/env
rustup install nightly
rustup default nightly
```

### Build and Install

```bash
# Install dependencies and build
uv sync --all-extras
uv run maturin develop --release

# Or build only (produces .whl in target/wheels/)
uv run maturin build --release
```

## Quick Start

```python
import fastnn as fnn

# Create tensors
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]])
y = fnn.tensor([[5.0, 6.0], [7.0, 8.0]])

# Tensor operations
z = x @ y                    # matrix multiply
z = (x * 2).relu()           # element-wise with activation

# Build a neural network
model = fnn.models.MLP(
    input_dim=784,
    hidden_dims=[512, 256],
    output_dim=10
)

# Training loop
optimizer = fnn.Adam(model.parameters(), lr=1e-3)
trainer = fnn.Trainer(
    model=model,
    optimizer=optimizer,
    loss=fnn.nn.cross_entropy_loss
)

# Train with data loader
trainer.fit(train_loader, epochs=10)
```

## Core Concepts

### Tensors

The core data structure - n-dimensional arrays with automatic differentiation:

```python
# Create tensors with gradient tracking
x = fnn.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()  # Computes gradients automatically
print(x.grad)    # tensor([2., 4., 6.])
```

### Autograd

Automatic differentiation engine that tracks operations for gradient computation:

```python
# Gradients are computed lazily until backward() is called
z = fnn.relu(x)  # Non-differentiable ops handled automatically
```

### nn.Module

Neural network layers and models:

```python
class MyModel(fnn.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = fnn.nn.Linear(784, 256)
        self.layer2 = fnn.nn.Linear(256, 10)
    
    def forward(self, x):
        x = fnn.relu(self.layer1(x))
        return self.layer2(x)
```

### Optimizers

Built-in optimizers with familiar API:

```python
optimizer = fnn.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Performance

fastnn achieves significant speedups over PyTorch on specific operations:

| Operation | Size | fastnn | PyTorch | Improvement |
|-----------|------|--------|---------|-------------|
| FusedAddReLU | 1000×1000 | 540μs | 1776μs | **3.3x faster** |
| Max | 1000×1000 | 207μs | 284μs | **1.4x faster** |
| MatMul (ARM) | 256×512×256 | 2408μs | 2999μs | **1.2x faster** |
| Linear (ARM) | 32×256×512 | 520μs | 599μs | **1.2x faster** |

### Optimization Highlights

- **SIMD Vectorization**: AVX2/AVX512 on x86, NEON on ARM via `wide` crate
- **Parallel Execution**: Rayon-based multi-core parallelism
- **Fused Operations**: Eliminated intermediate tensor allocations
- **Memory Efficiency**: Custom allocator with mimalloc

See [BENCHMARKS.md](./BENCHMARKS.md) for detailed performance data across x86, ARM, and GPU.

## Architecture

```
┌─────────────────────────────────────┐
│           Python API                │
│   (fastnn/__init__.py, core.py)    │
└──────────────┬──────────────────────┘
               │ PyO3 bindings
┌──────────────▼──────────────────────┐
│         Rust Core (src/)             │
├──────────────────────────────────────┤
│  tensor.rs    │ Autograd engine     │
│  nn/          │ Optimizers          │
│  kernels/     │ SIMD + parallel      │
│  storage.rs   │ Memory management   │
└──────────────────────────────────────┘
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/fastnn.git
cd fastnn
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Build in debug mode
uv run maturin develop
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [PyTorch](https://github.com/pytorch/pytorch) and [NVLabs/vibetensor](https://github.com/NVlabs/vibetensor). Built with [PyO3](https://github.com/pyo3/pyo3), [Rayon](https://github.com/rayon-rs/rayon), and the [wide](https://github.com/huangjj27/wide) SIMD library.
