# fastnn

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Features

- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- **Optimizers**: SGD, Adam, AdamW
- **Training infrastructure**: Trainer with callbacks, metrics, checkpoints
- **IO**: safetensors serialization, DLPack interop
- **SIMD-optimized**: AVX2/AVX512 vectorization for maximum performance

## Performance

Benchmark comparisons with PyTorch (lower is better):

| Operation | Size | fastnn | PyTorch | Status |
|-----------|------|--------|---------|--------|
| ReLU | 2×100 | 2.3μs | 2.5μs | ✅ faster |
| FusedAddReLU | 2×100 | 3.6μs | 5.2μs | ✅ faster |
| MatMul | 106×64×128 | 106μs | 163μs | ✅ faster |
| GELU | 2×100 | 42μs | 42μs | competitive |
| Sigmoid | 2×100 | 28μs | 15μs | |
| Tanh | 2×100 | 39μs | 7.9ms | ✅ faster |
| Add | 2×100 | 40μs | 2.9μs | |
| Mul | 2×100 | 43μs | 3.2μs | |
| Conv2d | 1×64×56×56 | 873μs | 135μs | |
| Linear | 1×256 | 679μs | 41μs | |

Note: Performance varies by hardware and tensor size. Best results require AVX2/AVX512 support.

## Installation

```bash
make install
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
make build

# Test
make test

# Benchmark
make bench

# Clean
make clean
```
