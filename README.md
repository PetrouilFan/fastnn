# fastnn

**fastnn** is a high-performance, production-grade neural network framework built from scratch in Rust with seamless Python bindings. It delivers hardware-accelerated CPU and GPU compute through a familiar PyTorch-like API, without the overhead of mainstream deep learning stacks.

> **Version:** v0.5.0 — Stability and Performance Fixes

[![CI](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml/badge.svg)](https://github.com/PetrouilFan/fastnn/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Rust: stable](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs)

---

## Overview

fastnn is designed for researchers and engineers who need both the ergonomics of Python and the raw performance of systems-level code. It is implemented entirely in Rust and exposed to Python via [PyO3](https://pyo3.rs), making it a fast, dependency-light alternative for training and inference workloads.

**Core design goals:**
- Zero-compromise performance via hand-written SIMD kernels and GPU compute shaders
- A clean, PyTorch-compatible Python API with no learning curve
- Portable acceleration across x86-64 (AVX2/AVX512), ARM (NEON), and GPU (WebGPU/wgpu)
- First-class autograd with full backward pass support

---

## Features

- **Vectorized CPU Kernels** — Hand-optimized SIMD kernels targeting AVX2, AVX512, and ARM NEON, with runtime dispatch to the best available instruction set. Includes Cephes-style fast approximations for transcendental functions (`exp`, `log`).
- **GPU Acceleration** — Cross-platform GPU compute via [wgpu](https://wgpu.rs) (WebGPU). Vectorized `vec4` shaders for elementwise ops and tiled matrix multiplication with shared memory.
- **Multi-threading** — Automatic parallelism across CPU cores using [rayon](https://github.com/rayon-rs/rayon), with cache-aware chunking for memory-bound and compute-bound workloads.
- **Native Autograd** — Built-in automatic differentiation engine with operation tracking, backward passes, and `no_grad` context support.
- **Multi-GPU Training** — Distributed Data Parallel (DDP) with bucketed AllReduce gradient synchronization and dynamic load balancing across GPUs.
- **PyO3 Python Bindings** — Train and evaluate models from Python with a PyTorch-like API. No Python performance penalty on the hot path.
- **Optimized Convolutions** — im2col-based Conv2d with specialized kernels for 1×1, depthwise, and 3×3 convolutions at various stride/dilation configurations.
- **BLAS Integration** — Optional OpenBLAS backend for matrix multiplication on large tensors.
- **Training Utilities** — Datasets, DataLoaders, and Keras-style Callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger).

---

## GPU Performance

Benchmarks measured against equivalent PyTorch CPU operations on medium-to-large tensors.

| Operation           | Tensor Size      | GPU Speedup | Notes                                 |
|---------------------|------------------|-------------|---------------------------------------|
| MatMul              | 512×1024×512     | **152×**    | Tiled matmul with shared memory       |
| GELU                | 1000×1000        | **14×**     | Vectorized `tanh` computation         |
| Sigmoid             | 1000×1000        | **11×**     | Vectorized shader operations          |
| Add                 | 1000×1000        | **4×**      | `vec4` vectorized elementwise shader  |

> **Note:** GPU acceleration shows the highest gains for medium-to-large tensors (≥ 100×100). Small tensor operations may be bound by kernel launch and data transfer overhead.

---

## Project Structure

```
fastnn/
├── Cargo.toml                  # Rust dependencies (PyO3, rayon, wgpu, ...)
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
│   │   ├── blas.rs             # BLAS-accelerated matrix multiplication
│   │   └── gpu/                # WebGPU compute pipelines and WGSL shaders
│   ├── nn/                     # Neural network layers, activations, attention
│   ├── optim/                  # SGD, Adam, AdamW optimizers
│   ├── train/                  # Trainer, callbacks, metrics, loss functions
│   └── io/                     # Model serialization (safetensors), DLPack
├── fastnn/                     # Python package
│   ├── __init__.py             # Public API surface
│   ├── nn.py                   # Sequential, ModuleList
│   ├── parallel.py             # DataParallel / DDP
│   ├── models/                 # Pre-built models: MLP, Transformer
│   ├── data.py                 # Dataset, TensorDataset, DataLoader
│   └── callbacks.py            # Training callbacks
└── tests/
    ├── bench/                  # Benchmarks vs PyTorch (CPU & GPU)
    └── *.py                    # Unit and integration tests
```

---

## Installation

### Prerequisites

| Tool    | Version   | Purpose                          |
|---------|-----------|----------------------------------|
| Rust    | stable    | Build the core library           |
| Python  | ≥ 3.12    | Python bindings                  |
| uv      | latest    | Python dependency management     |

Install Rust via [rustup](https://rustup.rs):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build & Install

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn

# Install and build (editable mode, with dev dependencies)
uv pip install -e ".[dev]"

# Or using the Makefile
make install
```

The `[dev]` flag installs testing dependencies including `pytest`, `pytest-benchmark`, and `numpy`.

### Platform Notes

fastnn auto-selects the best CPU instruction set at runtime:
- **x86-64:** AVX512 → AVX2 → scalar fallback
- **ARM64:** NEON intrinsics
- **GPU:** WebGPU via `wgpu` (Vulkan, Metal, DX12, or WebGPU backends)

---

## Quick Start

### Basic Usage

```python
import fastnn as fnn

# Create tensors
a = fnn.randn([1000, 1000])
b = fnn.randn([1000, 1000])
c = a @ b  # BLAS/SIMD-accelerated matmul

# Define a model
model = fnn.Sequential(
    fnn.Linear(128, 64),
    fnn.ReLU(),
    fnn.Linear(64, 10),
)

optimizer = fnn.Adam(model.parameters(), lr=0.001)
inputs  = fnn.randn([32, 128])
targets = fnn.randint(low=0, high=10, shape=[32])

# Training step
outputs = model(inputs)
loss    = fnn.cross_entropy_loss(outputs, targets)
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item():.4f}")
```

### GPU Acceleration

```python
import fastnn as fnn

# Switch to WebGPU
fnn.set_default_device("gpu:0")

a = fnn.randn([1000, 1000], device="gpu")
b = fnn.randn([1000, 1000], device="gpu")
c = a @ b  # GPU-accelerated matrix multiplication
```

### Multi-GPU Training (DDP)

```python
import fastnn as fnn

# Create identical model replicas for each GPU
model_gpu0 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)
model_gpu1 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)

# Wrap in DataParallel with optional weighted data splitting
dp_model = fnn.DataParallel(
    [model_gpu0, model_gpu1],
    device_ids=[0, 1],
    weights=[0.6, 0.4],  # Proportional to GPU memory/speed
)

optimizers = [
    fnn.Adam(dp_model.replicas[0].parameters(), lr=1e-3),
    fnn.Adam(dp_model.replicas[1].parameters(), lr=1e-3),
]

for x_batch, y_batch in dataloader:
    loss = dp_model.forward_backward(x_batch, y_batch, fnn.cross_entropy_loss)
    dp_model.sync_gradients()         # Bucketed AllReduce
    for opt in optimizers:
        opt.step()
        opt.zero_grad()

    dp_model.adjust_weights_based_on_performance()  # Dynamic load balancing
```

### Training with Callbacks

```python
import fastnn as fnn

model     = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, activation="relu")
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

ds     = fnn.TensorDataset(X, y)
loader = fnn.DataLoader(ds, batch_size=4, shuffle=True)

callbacks = [
    fnn.EarlyStopping(monitor="loss", patience=10),
    fnn.ModelCheckpoint(dirpath="./checkpoints", monitor="loss", save_best_only=True),
    fnn.LearningRateScheduler(schedule="cosine", lr=1e-2, T_max=100),
]

model.train()
for epoch in range(100):
    total_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Inference without gradient tracking
with fnn.no_grad():
    preds = model(X)
    print(preds.numpy().round(2))
```

### Controlling Parallelism

```python
# Set CPU thread count for parallel kernels
fnn.set_num_threads(8)

# Inspect memory and registered ops
print(fnn.allocator_stats())
print(fnn.list_registered_ops())
```

---

## API Reference

### Tensor Creation

| Function                          | Description                         |
|-----------------------------------|-------------------------------------|
| `fnn.tensor(data, shape)`         | Create tensor from a Python list    |
| `fnn.zeros(shape)`                | Tensor of zeros                     |
| `fnn.ones(shape)`                 | Tensor of ones                      |
| `fnn.full(shape, value)`          | Tensor filled with a value          |
| `fnn.eye(n)`                      | Identity matrix                     |
| `fnn.arange(end)`                 | Integer range `[0, end)`            |
| `fnn.linspace(start, end, n)`     | Linearly spaced values              |
| `fnn.randn(shape)`                | Random normal (Gaussian)            |
| `fnn.rand(shape)`                 | Random uniform `[0, 1)`             |
| `fnn.randint(low, high, shape)`   | Random integers in `[low, high)`    |

### Neural Network Modules

| Module                                              | Description                         |
|-----------------------------------------------------|-------------------------------------|
| `fnn.Linear(in, out, bias=True)`                    | Fully connected layer               |
| `fnn.Conv2d(cin, cout, kernel, stride, padding)`    | 2D convolution                      |
| `fnn.LayerNorm(shape)`                              | Layer normalization                 |
| `fnn.BatchNorm1d(features)`                         | Batch normalization                 |
| `fnn.Dropout(p)`                                    | Dropout regularization              |
| `fnn.Embedding(num, dim)`                           | Learned word embeddings             |
| `fnn.ReLU` / `fnn.GELU` / `fnn.Sigmoid` / `fnn.Tanh` / `fnn.SiLU` | Activation layers |
| `fnn.Sequential(*layers)`                           | Sequential layer container          |
| `fnn.ModuleList(modules)`                           | Indexable module list               |

### Optimizers

| Optimizer                                                         | Description          |
|-------------------------------------------------------------------|----------------------|
| `fnn.SGD(params, lr, momentum=0, weight_decay=0)`                | Stochastic Gradient Descent |
| `fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8)`             | Adam                 |
| `fnn.AdamW(params, lr, betas=(0.9, 0.999), weight_decay=0.01)`   | AdamW (decoupled L2) |

### Loss Functions

| Function                                  | Description             |
|-------------------------------------------|-------------------------|
| `fnn.mse_loss(pred, target)`              | Mean squared error      |
| `fnn.cross_entropy_loss(logits, target)`  | Cross-entropy loss      |

---

## PyTorch Model Export & Import

fastnn supports exporting PyTorch models to the `.fnn` format for inference. This allows you to leverage the performance of fastnn while using familiar PyTorch model architectures.

### Exporting PyTorch Models

```python
import torch
import torchvision.models as models
from fastnn.export import export_pytorch_model

# Load a pretrained PyTorch model
model = models.resnet18(pretrained=True)
model.eval()

# Export to .fnn format
export_pytorch_model(model, "resnet18.fnn")
```

### Loading .fnn Models

```python
from fastnn.export import load_fnn_model
import fastnn as fnn

# Load the exported model
fnn_model = load_fnn_model("resnet18.fnn")
fnn_model.eval()

# Run inference
input_tensor = fnn.tensor(data, shape)
output = fnn_model(input_tensor)
```

### Supported PyTorch Layers

| Layer Type | Status | Notes |
|------------|--------|-------|
| `Conv2d` | ✅ | Full support |
| `BatchNorm1d` / `BatchNorm2d` | ✅ | BatchNorm2d mapped to BatchNorm1d |
| `Linear` | ✅ | Weight transpose handled automatically |
| `ReLU`, `GELU`, `SiLU` | ✅ | Activation functions |
| `LayerNorm` | ✅ | |
| `Embedding` | ✅ | |
| `Dropout` | ✅ | |
| `AdaptiveAvgPool2d` | ✅ | Output size must be (1,1) |
| `MaxPool2d` | ⚠️ | Placeholder implementation (uses mean pooling) |
| `BasicBlock` (ResNet) | ⚠️ | Skip connections not supported |

### Limitations

- **MaxPool2d**: Currently uses mean pooling as a placeholder. This affects accuracy for models using MaxPool2d (e.g., ResNet).
- **Skip Connections**: BasicBlock and other skip connections are not supported. The export skips container layers and only exports individual layers.
- **Model Accuracy**: Due to the MaxPool2d limitation and missing skip connections, exported models may have different outputs compared to PyTorch.

### Benchmarking

```python
from fastnn.export import export_pytorch_model, load_fnn_model
import fastnn as fnn
import torch
import time

# Export and load model
export_pytorch_model(pytorch_model, "model.fnn")
fnn_model = load_fnn_model("model.fnn")

# Benchmark PyTorch
start = time.time()
for _ in range(100):
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
pytorch_time = time.time() - start

# Benchmark fastnn
start = time.time()
for _ in range(100):
    with fnn.no_grad():
        fnn_output = fnn_model(input_tensor)
fnn_time = time.time() - start

print(f"PyTorch: {pytorch_time:.4f}s, fastnn: {fnn_time:.4f}s")
print(f"Speedup: {pytorch_time / fnn_time:.2f}x")
```

---

## Testing & Benchmarking

```bash
# Run unit and integration tests
pytest

# Run benchmarks only
pytest --benchmark-only

# CPU benchmark suite (fastnn vs PyTorch)
python tests/bench/fastnn.py

# GPU vs CPU comparison (quick)
python tests/bench/bench_gpu_simple.py

# Full GPU benchmark suite
python tests/bench/bench_gpu.py
```

---

## Building from Source

```bash
# Development build (faster compile, unoptimized)
maturin develop

# Release build (full optimizations: LTO, codegen-units=1, opt-level=3)
maturin build --release

# Or via Makefile
make build
```

Cargo feature flags:

| Feature       | Description                                      |
|---------------|--------------------------------------------------|
| `simd`        | Enable SIMD kernels (AVX2, AVX512, NEON)         |
| `parallel`    | Enable Rayon multi-threaded parallelism          |
| `simd-avx512` | Enable AVX-512 kernels (requires AVX512-capable CPU) |
| `openblas`    | Link against OpenBLAS for large matmul           |
| `prefetch`    | Enable software prefetching in matmul kernels    |

---

## License

fastnn is licensed under the [MIT License](LICENSE).  
Copyright © 2026 Petros Fanioudakis
