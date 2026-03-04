# fastnn

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Features

- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations
- **Optimizers**: SGD, Adam, AdamW
- **Training infrastructure**: Trainer with callbacks, metrics, checkpoints
- **IO**: safetensors serialization, DLPack interop

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

# Clean
make clean
```
