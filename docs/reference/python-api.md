# Python API Reference

Python-facing APIs for compiled execution, flash attention, and GPU backend.

## Compiled Training

Compiles a forward+backward+optimizer pipeline into a single-dispatch `PyCompiledTrainingModel`.

### compile_train_model()

```python
model = fnn.compile_train_model(
    graph_bytes,            # Serialized ComputeGraph (bytes)
    loss_node_id,           # Node ID of the loss tensor
    param_ids,              # Node IDs of trainable parameters
    param_data,             # Raw weight data per parameter
    batch_input_ids,        # Node IDs of batch inputs
    optimizer,              # "sgd" | "adamw" | "muon" | "lion" | "rmsprop"
    lr=0.001,
    weight_decay=0.0,
    beta1=None,             # Adam/AdamW/Lion beta1
    beta2=None,             # Adam/AdamW/Lion beta2
    eps=None,               # Adam/AdamW/RMSprop epsilon
    quantize=None,          # Quantization: 4 or 8 for standard, "i4cb" for codebook INT4
) -> PyCompiledTrainingModel
```

**Supported optimizer strings:**

| String | Required Params |
|--------|-----------------|
| `"sgd"` | lr, weight_decay |
| `"adamw"` | lr, beta1, beta2, eps, weight_decay |
| `"muon"` | lr, weight_decay |
| `"lion"` | lr, beta1, beta2 |
| `"rmsprop"` | lr, beta1, eps |

### train_step()

```python
loss: float = model.train_step(inputs: list[bytes])
```

Executes forward + backward + optimizer update in a single dispatch call. Returns scalar loss.

### Example

```python
graph_bytes = build_my_graph()
model = fnn.compile_train_model(
    graph_bytes=graph_bytes, loss_node_id=3,
    param_ids=[1, 2], param_data=[w_bytes, b_bytes],
    batch_input_ids=[0], optimizer="lion",
    lr=0.0001, beta1=0.95, beta2=0.98,
)
for epoch in range(100):
    for batch_x in data:
        loss = model.train_step([batch_x])
```

## FlashAttention

Memory-efficient attention using tiled online-softmax with SIMD tile matmul (AVX2/AVX-512). 2-4x speedup over baseline.

```python
output = fnn.flash_attention(
    q,                        # [batch, heads, seq_len, head_dim]
    k,                        # [batch, heads, seq_len, head_dim]
    v,                        # [batch, heads, seq_len, head_dim]
    scale=None,               # Default: 1/sqrt(head_dim)
    causal=False,             # Causal masking for autoregressive models
) -> Tensor
```

## WGPU Backend

### Device Selection

```python
fnn.set_default_device("wgpu:0")   # Set default GPU
x_gpu = x_cpu.to_gpu(0)            # Move tensor to GPU
x_cpu = x_gpu.to_cpu()             # Move tensor back to CPU
```

### Quantized GPU Inference

WGPU supports U4/U8 quantized inference via WGSL compute shaders with per-channel dequantization. Models run entirely on GPU with no CPU copy during dispatch.

```python
executor = fnn.AotExecutor(nodes, params, input_names, output_names, quantize=4)
outputs = executor.forward({"input": x_gpu})
```

## See also

- [API Reference](api-reference.md) -- Complete Python API table
- [Tensors](tensors.md) -- Tensor operations, autograd, GPU support
- [Optimizers](optimizers.md) -- Optimizer parameter details
- [ONNX Support](../models/onnx.md) -- Quantized model inference
- [Index](../index.md) -- Full documentation index
