# fastnn

**fastnn** is a high-performance, lightweight neural network framework built from scratch in Rust with seamless Python bindings. It is designed to be a fast, hardware-efficient alternative to mainstream deep learning libraries, providing hardware-accelerated CPU and GPU compute via a familiar PyTorch-like Python API.

**Version:** v0.3.0 - GPU Performance Update with Vectorized Shaders


## Features


- **Blazing Fast CPU Kernels:** Hand-written SIMD instructions (AVX2, AVX512 for x86_64, and NEON for ARM) with optimized approximations for transcendental functions (e.g., Cephes-style fast `exp` and `log`).
- **Multi-Threading:** Automatic work distribution across CPU cores using `rayon` for parallelized tensor computations.
- **GPU Acceleration:** Cross-platform hardware acceleration using `wgpu` (WebGPU) to run compute shaders with vectorized operations (ADD, GELU, MUL, DIV) and optimized matrix multiplication kernels. GPU speedups include **152x for MatMul** (512×1024×512) and **14x for GELU** (1000×1000).
- **Native Autograd:** Built-in automatic differentiation engine supporting tracking, backward passes, and context managers (`no_grad()`).
- **Python Bindings (`PyO3`):** Train and evaluate models natively in Python without sacrificing the performance of the underlying Rust implementation.
- **Optimized Convolutions:** Supports `im2col` transforms and specialized kernels (e.g., 1x1, depthwise, 3x3) optimized for specific stride/dilation configurations.


## Core Components


The framework ships with the essential building blocks for modern deep learning:


- **Tensors & Math:** Full support for multidimensional tensors, broadcasting, matrix multiplication (BLAS-accelerated), and standard mathematical operations.
- **Layers:** `Linear`, `Conv2d`, `Embedding`, `LayerNorm`, `BatchNorm1d`, `Dropout`.
- **Activations:** `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `SiLU`, `Softmax`, `LogSoftmax`.
- **Loss Functions:** `MSE`, `CrossEntropy`.
- **Optimizers:** `SGD`, `Adam`, `AdamW`.


## Project Structure


```text
fastnn/
├── Cargo.toml            # Rust dependencies (PyO3, rayon, wgpu, etc.)
├── uv.lock               # Python dependency management (uv)
└── src/
    ├── lib.rs            # Python module export & PyO3 bindings
    ├── tensor.rs         # Core Tensor struct, dimensions, and striding logic
    ├── autograd/         # Automatic differentiation and tape tracking
    ├── nn/               # Neural network layers (Linear, Conv2d, etc.)
    ├── optim/            # Model optimizers
    ├── dispatcher/       # Dynamic kernel dispatch (CPU vs GPU)
    ├── storage/          # Memory backend management (Device allocation)
    └── kernels/
        ├── cpu.rs        # Highly optimized SIMD and scalar CPU operations
        ├── blas.rs       # Matrix multiplication routines
        └── gpu/          # WebGPU contexts, compute pipelines, and vectorized shaders
```

## GPU Performance

fastnn includes highly optimized GPU kernels with vectorized shader operations:

| Operation | Size | GPU Speedup | Notes |
|-----------|------|-------------|-------|
| MatMul | 512×1024×512 | **152x** | Tiled matrix multiplication |
| GELU | 1000×1000 | **14x** | Vectorized tanh computation |
| Sigmoid | 1000×1000 | **11x** | Vectorized operations |
| Add | 1000×1000 | **4x** | Vectorized vec4 operations |

**Key GPU Optimizations:**
- Vectorized ADD shader: Changed from scalar to vec4 operations (4x processing per thread)
- Vectorized GELU shader: Implemented vectorized tanh and GELU computations
- Shader consistency: All binary operations now use vectorized shaders
- Optimized MatMul: Tiled matrix multiplication with shared memory

For detailed benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).

## Installation

### Prerequisites

- **Rust**: The latest stable Rust toolchain (via rustup).
- **Python**: Python 3.12+
- **uv**: For Python dependency management (recommended).

### Building the Project

Clone the repository and install it as an editable Python package using uv (or pip):

```bash
git clone https://github.com/PetrouilFan/fastnn.git
cd fastnn
uv pip install -e .[dev]
```

**Note**: The `[dev]` flag installs testing dependencies like pytest, pytest-benchmark, and numpy.

## Quick Start (Python)

Because fastnn exposes a PyTorch-like API, you can easily define models and execute operations directly from Python:

```python
import fastnn as fn


# Define a simple multi-layer perceptron
model = fn.Sequential(
    fn.Linear(128, 64),
    fn.ReLU(),
    fn.Linear(64, 10)
)


# Initialize optimizer
optimizer = fn.PyAdam(model.parameters(), lr=0.001)


# Forward pass
inputs = fn.randn() # Batch of 32
targets = fn.randint(low=0, high=10, shape=)


# Compute predictions
outputs = model(inputs)


# Calculate loss & step
loss = fn.cross_entropy_loss(outputs, targets)
loss.backward()
optimizer.step()


print(f"Loss: {loss.item()}")
```

## Advanced Configuration

### Device Management

By default, operations run on the CPU. You can change the global device configuration via the API:

```python
# Switch to WebGPU Compute
fn._set_default_device("gpu:0")

# Or specify device per tensor
a = fn.randn([1000, 1000], device="gpu")
b = fn.randn([1000, 1000], device="gpu")
c = a @ b  # GPU-accelerated matrix multiplication
```

### Multithreading

fastnn leverages rayon for heavy parallel lifting. You can adjust the number of threads allocated to CPU compute:

```python
fn._set_num_threads(8)
```

## Testing and Benchmarking

Tests are built using pytest. You can run the testing and benchmarking suites using:

```bash
pytest
pytest --benchmark-only
```

### GPU Benchmarking

To benchmark GPU performance:

```bash
python tests/bench_gpu_simple.py  # Quick GPU vs CPU comparison
python tests/bench_gpu.py         # Comprehensive GPU benchmark suite
```

**Performance Note:** GPU acceleration shows best results for medium-to-large tensors (>100×100). Small tensor operations may have overhead from kernel launches and data transfers.
