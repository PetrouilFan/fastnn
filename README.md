# fastnn

A Python deep learning library with a Rust core, inspired by PyTorch and NVLabs/vibetensor.

## Features

- Tensor operations with full autograd support
- Neural network modules: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- Optimizers: SGD, Adam, AdamW
- Training infrastructure with callbacks, metrics, and checkpoints
- Fused operations: fused_add_relu, fused_linear_relu, fused_linear_gelu
- SIMD-optimized via `wide` crate (x86 and ARM)
- Parallel execution with Rayon
- GPU support via wgpu (Vulkan, Metal, DirectX12)

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

## Installation

```bash
uv sync --all-extras
uv run maturin develop --release
```

## Performance

fastnn outperforms PyTorch on specific operations:

| Operation | Size | fastnn | PyTorch |
|-----------|------|--------|---------|
| FusedAddReLU | 1000×1000 | 540μs | 1776μs |
| Max | 1000×1000 | 207μs | 284μs |

GPU acceleration provides **68x speedup** on large matrix multiplications.

See [BENCHMARKS.md](./BENCHMARKS.md) for detailed performance data across x86, ARM, and GPU.

## Development

```bash
# Build
uv run maturin build --release

# Test
uv run pytest tests/ -v
```
