# fastnn

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Features

- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- **Optimizers**: SGD, Adam, AdamW
- **Training infrastructure**: Trainer with callbacks, metrics, checkpoints
- **IO**: safetensors serialization, DLPack interop
- **SIMD-optimized**: AVX2/AVX512 vectorization for maximum performance
- **Parallel execution**: Rayon-based parallelism for multi-core utilization

## Performance

Benchmark comparisons with PyTorch (mean time in μs, lower is better):

| Operation | Size | fastnn | PyTorch | Status |
|-----------|------|--------|---------|--------|
| ReLU | 100×100 | 93.6μs | 2.7μs | |
| ReLU | 1000×1000 | 551.7μs | 35.2μs | |
| FusedAddReLU | 100×100 | 94.7μs | 5.5μs | |
| FusedAddReLU | 1000×1000 | 639.5μs | 113.7μs | |
| MatMul | 128×256×128 | 114.7μs | 27.5μs | |
| MatMul | 256×512×256 | 868.5μs | 167.3μs | |
| MatMul | 512×1024×512 | 7099.8μs | 1056.6μs | |
| GELU | 100×100 | 105.4μs | 19.2μs | |
| GELU | 1000×1000 | 895.6μs | 193.9μs | |
| Sigmoid | 100×100 | 104.2μs | 8.5μs | |
| Sigmoid | 1000×1000 | 689.1μs | 164.8μs | |
| Tanh | 100×100 | 108.2μs | 2493.7μs | ✅ faster |
| Tanh | 1000×1000 | 1025.9μs | 913.8μs | |
| Add | 100×100 | 123.8μs | 4.3μs | |
| Add | 1000×1000 | 1239.4μs | 109.5μs | |
| Mul | 100×100 | 97.7μs | 3.3μs | |
| Mul | 1000×1000 | 571.1μs | 59.5μs | |
| Linear | 32×256×512 | 402.0μs | 371.5μs | |
| Linear | 32×512×1024 | 989.0μs | 1368.9μs | ✅ faster |
| Linear | 128×256×512 | 728.8μs | 176.0μs | |
| Conv2d | 1×32×32×32 | 962.8μs | 173.7μs | |
| Conv2d | 1×64×64×64 | 13810.2μs | 645.7μs | |
| Sum | 1000×1000 | 138.9μs | 29.3μs | |
| Mean | 1000×1000 | 205.0μs | 2885.0μs | ✅ faster |
| Max | 1000×1000 | 302.4μs | 106.9μs | |

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
