# FastNN Documentation

A Python deep learning library with a Rust core, inspired by NVLabs/vibetensor and PyTorch.

## Overview

FastNN provides:
- **AOT compiler pipeline (v2.0)** — Compile computation graphs through shape inference, operator fusion, weight quantization, and memory planning before execution
- **Native weight quantization** — 4-bit (U4x8) and 8-bit (U8x4) with per-channel scales, fused into GEMM/conv kernels
- **IR-based execution** — `ComputeGraph` with 30+ opcodes, `GraphBuilder` fluent API, `GraphExecutor` for compile+run
- **Tensor operations** with full autograd support
- **Neural network modules**: Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding, activations, fused layers
- **Optimizers**: SGD, Adam, AdamW, Muon, Lion, RMSprop
- **Training infrastructure**: DataLoader with callbacks, metrics, checkpoints
- **Packed precision**: 4-bit, 8-bit, 16-bit quantized inference with SIMD/SWAR kernels; packed linear, conv, and fused layers
- **IO**: Custom binary format (.fnn, v2/v3), PyTorch model export, ONNX model import (Q/DQ folding, quantized runtime)
- **YOLO object detection**: Full pipeline for YOLOv5/v8/v10/v11 ONNX models with NMS post-processing

## Sections

- [Getting Started](getting-started.md) - Installation and quick start
- [Tensors](tensors.md) - Tensor creation, operations, and autograd
- [Neural Network Modules](nn-modules.md) - Building blocks for neural networks
- [Optimizers](optimizers.md) - Training optimization algorithms
- [Training](training.md) - Data loaders, callbacks, and training loops
- [Models](models.md) - Pre-built model architectures
- [IO & Serialization](io.md) - Saving and loading models
- [API Reference](api-reference.md) - Complete API documentation
- [Architecture](architecture.md) - v2.0 AOT compiler pipeline overview
- [Development Architecture](development.md) - Internal module layout and contribution workflow
- [Performance Roadmap](performance-roadmap.md) - GPU, fusion, optimizer, and packed precision roadmap

## Requirements

- Python 3.12+
- numpy >= 1.24

## Quick Start — v2.0 AOT Compiler Pipeline

The AOT pipeline compiles models through shape inference, operator fusion, optional weight quantization, and memory planning before execution — zero runtime graph traversal overhead.

### Python: ONNX Import with Quantization

```python
import fastnn as fnn

# Import an ONNX model and compile with 4-bit weight quantization
executor = fnn.AotExecutor(
    nodes=model_nodes,       # from onnx import
    params=model_params,     # weight tensors
    input_names=["input"],
    output_names=["output"],
    quantize=4,              # 4-bit or 8-bit quantization (None for f32)
)

# Run inference
outputs = executor.forward({"input": input_tensor})
print(outputs["output"])
```

### Rust: GraphBuilder API

```rust
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType};
use fastnn::backend::cpu::CpuBackend;

let gb = GraphBuilder::new();
let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
let weight = gb.parameter(&[784, 10], IrDType::F32);
let bias = gb.parameter(&[10], IrDType::F32);
let mm = gb.matmul(&input, &weight);
let output = gb.add(&mm, &bias);

// Compile with 8-bit quantization and execute
let result = gb.compile_and_execute_with_quantize(
    &[&output], CpuBackend, Some(8),
    &[input_data, weight_data, bias_data],
);
```

### v1.x Eager Mode (Legacy)

The v1.x eager-mode API is still available for backward compatibility:

```python
import fastnn as fnn

# Tensor operations with autograd
x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
y = fnn.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
z = fnn.matmul(x, y)

# Training
model = fnn.models.MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
optimizer = fnn.Adam(model.parameters(), lr=1e-2)
for epoch in range(100):
    pred = model(batch_x)
    loss = fnn.mse_loss(pred, batch_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
