# Python API Reference (v2.2)

## Compiled Training API

### `compile_train_model()`

Compiles a forward+backward+optimizer pipeline into an `ExecutablePlan` that executes with a single dispatch call.

```python
model = fnn.compile_train_model(
    graph_bytes: bytes,           # Serialized ComputeGraph
    loss_node_id: int,            # Node ID of the loss tensor
    param_ids: list[int],         # Node IDs of trainable parameters
    param_data: list[bytes],      # Raw weight data for each parameter
    batch_input_ids: list[int],   # Node IDs of batch input tensors
    optimizer: str,               # "sgd", "adamw", "muon", "lion", "rmsprop"
    lr: float = 0.001,
    weight_decay: float = 0.0,
    beta1: float | None = None,   # Adam/AdamW/Lion beta1
    beta2: float | None = None,   # Adam/AdamW/Lion beta2
    eps: float | None = None,     # Adam/AdamW/RMSprop epsilon
    quantize: int | None = None,  # Optional quantization bit-width (4 or 8)
) -> PyCompiledTrainingModel
```

**Supported optimizer strings:**

| String  | Optimizer | Required params |
|---------|-----------|----------------|
| `"sgd"` | SGD | lr, weight_decay |
| `"adamw"` | AdamW | lr, beta1, beta2, eps, weight_decay |
| `"muon"` | Muon | lr, weight_decay |
| `"lion"` | Lion | lr, beta1, beta2 |
| `"rmsprop"` | RMSprop | lr, beta, eps |

### `PyCompiledTrainingModel.train_step()`

Executes one training step (forward + backward + optimizer update).

```python
loss: float = model.train_step(inputs: list[bytes])
```

- `inputs`: List of serialized batch data, one per `batch_input_id`.
- Returns the scalar loss value as a Python float.

### Example

```python
import fastnn as fnn

# Build graph and serialize
graph_bytes = build_my_graph()
params = [w_bytes, b_bytes]

# Compile
model = fnn.compile_train_model(
    graph_bytes=graph_bytes,
    loss_node_id=3,
    param_ids=[1, 2],
    param_data=params,
    batch_input_ids=[0],
    optimizer="lion",
    lr=0.0001,
    beta1=0.95,
    beta2=0.98,
)

# Train
for epoch in range(100):
    for batch_x in data:
        loss = model.train_step([batch_x])
        print(f"loss = {loss:.4f}")
```

## FlashAttention

### `flash_attention()`

Memory-efficient attention using tiled online-softmax with SIMD-accelerated tile matmul (AVX2/AVX-512 in v2.2).

```python
output = fnn.flash_attention(
    q: Tensor,           # [batch, heads, seq_len, head_dim]
    k: Tensor,           # [batch, heads, seq_len, head_dim]
    v: Tensor,           # [batch, heads, seq_len, head_dim]
    scale: float | None = None,   # Scale factor (default: 1/sqrt(head_dim))
    causal: bool = False,         # Causal masking for autoregressive models
) -> Tensor
```

**Performance:** 2-4× speedup over the baseline with AVX-512/AVX2 tile matmul (v2.2).

## WGPU Backend

### Device Selection

```python
import fastnn as fnn

# Set default device to GPU
fnn.set_default_device("wgpu:0")

# Or move specific tensors
x_gpu = x_cpu.to_gpu(0)
x_cpu = x_gpu.to_cpu()
```

### Quantized GPU Inference (v2.2)

WGPU backend supports U4/U8 quantized inference via WGSL compute shaders with per-channel dequantization. Quantized models run entirely on GPU — no CPU copy during dispatch.

```python
# Compile with quantization for GPU execution
executor = fnn.AotExecutor(
    nodes, params, input_names, output_names,
    quantize=4,  # or 8
)
outputs = executor.forward({"input": x_gpu})
```
