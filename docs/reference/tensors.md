# Tensors

Tensors are the fundamental data structure in fastnn, similar to numpy arrays but with automatic differentiation support and GPU acceleration.

## Creating Tensors

### From Python Data

```python
import fastnn as fnn

x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
print(x.shape)  # [2, 2]
```

### From NumPy

```python
import numpy as np

arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x = fnn.tensor_from_numpy(arr)
```

### Factory Functions

```python
z = fnn.zeros([3, 4], dtype='f32')        # Zeros
o = fnn.ones([2, 3], dtype='f32')         # Ones
f = fnn.full([2, 2], 5.0, dtype='f32')    # Constant fill
I = fnn.eye(3)                            # Identity
r = fnn.arange(10)                        # [0..9]
r = fnn.arange(2, 10, 2)                  # [2, 4, 6, 8]
l = fnn.linspace(0, 1, 10)                # 10 values 0..1

rand = fnn.randn([3, 4])                  # Gaussian
rand = fnn.rand([3, 4])                   # Uniform [0, 1)
rand = fnn.randint(low=0, high=10, shape=[3, 4])  # Uniform ints

z = fnn.zeros_like(x)
o = fnn.ones_like(x)
f = fnn.full_like(x, 3.0)
```

## Data Types

| dtype | Description | Size |
|-------|-------------|------|
| `f32` | 32-bit float | 4 bytes |
| `f64` | 64-bit float | 8 bytes |
| `f16` | 16-bit float (half) | 2 bytes |
| `bf16` | 16-bit bfloat | 2 bytes |
| `i32` | 32-bit integer | 4 bytes |
| `i64` | 64-bit integer | 8 bytes |
| `u8` | Unsigned 8-bit integer | 1 byte |
| `bool` | Boolean | 1 byte |

Dtype values live under `fastnn.dtypes`. See the [performance roadmap](../internals/performance-roadmap.md) for packed precision types (`I4x8`, `I8x4`, `F16x2`, `F32x1`) used in the Rust API.

## Tensor Operations

### Arithmetic

```python
c = a + b       # add
c = a - b       # subtract
c = a * b       # multiply
c = a / b       # divide
c = -a          # negate
c = a @ b       # matrix multiply (2D only)
```

### Math Functions

```python
y = x.abs()     # Absolute value
y = x.exp()     # Exponential
y = x.log()     # Natural log
y = x.sqrt()    # Square root
y = x.pow(2)    # Power
y = x.clamp(0, 1)  # Clamp values
```

### Reductions

```python
x.sum()                         # Sum all elements
x.mean()                        # Mean of all elements
x.max(dim=0, keepdim=False)     # Max along axis 0
x.min(dim=1, keepdim=False)     # Min along axis 1
fnn.softmax(x, dim=-1)          # Softmax
fnn.log_softmax(x, dim=-1)      # Log softmax
fnn.cumsum(x, dim=0)            # Cumulative sum
```

### Element-wise Comparison

```python
a = fnn.tensor([1.0, 2.0, 3.0], [3])
b = fnn.tensor([2.0, 2.0, 2.0], [3])

mask = a.gt_scalar(1.5)         # [0.0, 1.0, 1.0]
mask = a.lt_scalar(2.5)         # [1.0, 1.0, 0.0]
c = fnn.maximum(a, b)           # Element-wise max: [2.0, 2.0, 3.0]
d = fnn.minimum(a, b)           # Element-wise min: [1.0, 2.0, 2.0]
```

### Advanced Operations

```python
fnn.einsum('ij,jk->ik', [a, b])        # Einstein summation
values, indices = fnn.topk(x, k=2, dim=1)  # Top-k
result = fnn.gather(x, dim=1, index=indices)  # Gather along dim
y = x.repeat([2, 3])                   # Repeat along dims
y = x.expand([10, 2, 3])               # Expand without copy
y = fnn.slice(x, dim=0, start=0, end=2)  # Slice along dim
y = fnn.erf(x)                         # Error function
result = on_true.where_tensor(cond, on_false)  # Conditional selection
```

### Fused Linear + Activation

```python
result = x.fused_linear_relu(weight, bias)   # Matmul + ReLU
result = x.fused_linear_gelu(weight, bias)   # Matmul + GELU
```

## Reshaping and Indexing

```python
y = x.reshape([6, 4])             # Reshape (supports -1)
y = x.view([24])                  # View without copy
y = x.squeeze(1)                  # Remove dim if size=1
y = x.unsqueeze(0)                # Add dim at position 0
y = x.permute([2, 0, 1])         # Permute dimensions
y = x.transpose(0, 2)            # Swap two dimensions
y = x.flip(1)                     # Reverse along dim 1

y = fnn.cat([a, b], dim=0)       # Concatenate along dim
y = fnn.stack([x, y, z], dim=0)  # Stack along new dim
x[0:2, :, :]                     # Standard Python slicing
```

## Broadcasting Rules

Two dimensions are compatible when they are equal or one of them is 1. Tensors with fewer dims have 1 prepended automatically.

```python
a = fnn.randn([3, 1])            # [3, 1]
b = fnn.randn([5])               # [5] -> [1, 5]
c = a + b                        # result [3, 5]
```

## Autograd

### Computing Gradients

```python
x = fnn.tensor([1.0, 2.0])
x.requires_grad_(True)

y = x * 2
z = y.sum()
z.backward()                     # Compute gradients

print(x.grad)                    # tensor([2., 2.])
x.set_grad(None)                 # Clear gradients
```

### Disabling Gradient Tracking

```python
with fnn.no_grad():              # Context manager
    result = model(x)            # No gradient tracked

y = x.detach()                   # Detach from computation graph
```

### Numerical Gradient Checks

Run finite-difference verification for eager autograd on CPU f32:

```bash
cargo +stable test --test autograd_gradient_checks
```

Tolerances: default ops `atol=2e-2, rtol=2e-2`; `silu` `3e-2`; `gelu` `4e-2`.

## Tensor Properties

```python
x.shape              # [3, 4, 5]
x.ndim               # 3
x.dtype              # 'f32'
x.device             # 'cpu' or Device.Wgpu(0)
x.numel              # 60
x.stride             # [20, 5, 1]
x.is_contiguous      # True
x.requires_grad      # True/False
x.grad               # Gradient tensor (after backward())
```

## Device Management

Move tensors between CPU and GPU (WGPU backend):

```python
x_cpu = fnn.randn([3, 4])
x_gpu = x_cpu.to_gpu(0)          # CPU to GPU
x_cpu = x_gpu.to_cpu()           # GPU to CPU
fnn.set_default_device("wgpu:0") # Set default device globally
```

## Gradient Clipping

```python
# Clip by norm
fnn.clip_grad_norm_(params, max_norm=1.0, norm_type=2.0)

# Clip by value
fnn.clip_grad_value_(params, clip_value=0.1)
```

## FlashAttention

Memory-efficient attention with O(N) memory using tiled online-softmax:

```python
q = fnn.randn([4, 8, 64, 32])     # [batch, heads, seq_len, head_dim]
k = fnn.randn([4, 8, 64, 32])
v = fnn.randn([4, 8, 64, 32])

output = fnn.flash_attention(q, k, v)               # Standard
output = fnn.flash_attention(q, k, v, causal=True)  # Causal masking
```

## Reproducibility

```python
fnn.set_seed(42)
fnn.set_num_threads(4)            # Control CPU thread count
```

## NumPy Interop

```python
# Tensor to numpy
arr = x.numpy()

# Numpy to tensor
import numpy as np
arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x = fnn.tensor_from_numpy(arr)
```

## See also

- [API Reference](api-reference.md) -- Complete catalog of tensor functions
- [Neural Network Modules](nn-modules.md) -- Layers that consume tensors
- [Optimizers](optimizers.md) -- Optimizers that update tensor parameters
- [Python API](python-api.md) -- Compiled training with tensors
- [Index](../index.md) -- Full documentation index
