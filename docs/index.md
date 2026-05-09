# FastNN Documentation

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Overview

FastNN provides:
- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- **Optimizers**: SGD, Adam, AdamW, Muon, Lion, RMSprop
- **Training infrastructure**: DataLoader with callbacks, metrics, checkpoints
- **IO**: Custom binary format (.fnn), DLPack interop (Rust only)

## Sections

- [Getting Started](getting-started.md) - Installation and quick start
- [Tensors](tensors.md) - Tensor creation, operations, and autograd
- [Neural Network Modules](nn-modules.md) - Building blocks for neural networks
- [Optimizers](optimizers.md) - Training optimization algorithms
- [Training](training.md) - Data loaders, callbacks, and training loops
- [Models](models.md) - Pre-built model architectures
- [IO & Serialization](io.md) - Saving and loading models
- [API Reference](api-reference.md) - Complete API documentation
- [Development Architecture](development.md) - Internal module layout and contribution workflow
- [Performance Roadmap](performance-roadmap.md) - GPU, fusion, optimizer, and packed precision roadmap

## Requirements

- Python 3.12+
- numpy >= 1.24

## Quick Example

```python
import fastnn as fnn

# Create tensors
x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
y = fnn.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

# Operations with autograd
z = fnn.matmul(x, y)  # matrix multiply
z = (x * 2).relu()

# Build a model
model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)

# Training
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
