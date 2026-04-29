# Tensors

Tensors are the fundamental data structure in FastNN, similar to numpy arrays but with automatic differentiation support.

## Creating Tensors

### From Python Data

```python
import fastnn as fnn

# From list + shape
x = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
print(x.shape)  # [2, 2]
```

### Utility Functions

```python
# Zeros
z = fnn.zeros([3, 4], dtype='f32')

# Ones
o = fnn.ones([2, 3], dtype='f32')

# Full with constant
f = fnn.full([2, 2], 5.0, dtype='f32')

# Identity matrix
I = fnn.eye(3)

# Range
r = fnn.arange(10)           # [0, 1, 2, ..., 9]
r = fnn.arange(2, 10, 2)     # [2, 4, 6, 8]

# Linear spacing
l = fnn.linspace(0, 1, 10)   # 10 values from 0 to 1

# Random
rand = fnn.randn([3, 4])                    # Gaussian (mean=0, std=1)
rand = fnn.rand([3, 4])                     # Uniform [0, 1)
rand = fnn.randint(low=0, high=10, shape=[3, 4])  # Uniform integers [0, 10)

# Like functions
z = fnn.zeros_like(x)
o = fnn.ones_like(x)
f = fnn.full_like(x, 3.0)
```

## Data Types

| dtype | Description | Size |
|-------|-------------|------|
| `f32` | 32-bit float | 4 bytes |
| `f64` | 64-bit float | 8 bytes |
| `f16` | 16-bit float (half precision) | 2 bytes |
| `bf16` | 16-bit bfloat | 2 bytes |
| `i32` | 32-bit integer | 4 bytes |
| `i64` | 64-bit integer | 8 bytes |
| `bool` | Boolean | 1 byte |

## Tensor Operations

### Arithmetic

```python
a = fnn.tensor([1.0, 2.0])
b = fnn.tensor([3.0, 4.0])

c = a + b       # add
c = a - b       # subtract
c = a * b       # multiply
c = a / b       # divide
c = -a          # negate
c = a @ b       # matrix multiply (2D only)
```

### Math Functions

```python
x = fnn.tensor([1.0, 2.0, 3.0])

y = x.abs()           # absolute value
y = x.exp()           # exponential
y = x.log()           # natural logarithm
y = x.sqrt()          # square root
y = x.pow(2)          # power
y = x.clamp(0, 1)     # clamp values to [0, 1]
y = x.sign()          # sign function
```

### Activation Functions

```python
x = fnn.tensor([[-1.0, 0.0, 1.0]])

y = x.relu()          # ReLU: max(0, x)
y = x.sigmoid()       # Sigmoid: 1/(1+exp(-x))
y = x.tanh()          # Tanh
y = x.gelu()          # GELU
y = x.silu()          # SiLU (Swish): x * sigmoid(x)
y = x.leaky_relu(0.01)  # Leaky ReLU
y = x.elu(1.0)        # ELU
y = x.softplus()      # Softplus
y = x.hardswish()     # Hard swish
y = x.softmax(dim=-1) # Softmax
y = x.log_softmax(dim=-1)  # Log softmax
```

### Reduction Operations

```python
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]])

s = x.sum()                    # sum all elements
m = x.mean()                   # mean of all elements
mx = x.max(dim=0, keepdim=False)  # max along axis 0
mn = x.min(dim=1, keepdim=False)  # min along axis 1

# Softmax
sm = fnn.softmax(x, dim=-1)
lsm = fnn.log_softmax(x, dim=-1)
```

### Reshaping

```python
x = fnn.randn([2, 3, 4])

# Reshape (supports -1 for auto-infer)
y = x.reshape([6, 4])
y = x.reshape([-1, 4])  # Infer first dimension

# View (no copy)
y = x.view([24])

# Squeeze/Unsqueeze
y = x.squeeze(1)        # Remove dim 1 if size=1
y = x.unsqueeze(0)      # Add dim at position 0

# Permute
y = x.permute([2, 0, 1])

# Transpose
y = x.transpose(0, 2)

# Flip
y = x.flip(1)           # Reverse along dim 1
```

### Concatenation and Stacking

```python
a = fnn.randn([2, 3])
b = fnn.randn([4, 3])

# Concatenate (along existing dimension)
c = fnn.cat([a, b], dim=0)  # [6, 3]

# Stack (concatenate along NEW dimension)
x = fnn.tensor([1, 2, 3], [3])
y = fnn.tensor([4, 5, 6], [3])
z = fnn.tensor([7, 8, 9], [3])

# Stack along dim=0 creates new dimension at front
stacked = fnn.stack([x, y, z], dim=0)  # Shape: [3, 3]
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

# Stack along dim=1 creates new dimension at position 1
stacked_t = fnn.stack([x, y, z], dim=1)  # Shape: [3, 3]
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]

# Repeat
x = fnn.randn([2, 3])
y = x.repeat([2, 3])  # [4, 9]

# Where (conditional selection)
cond = fnn.tensor([1.0, 0.0, 1.0], [3])
on_true = fnn.tensor([10.0, 20.0, 30.0], [3])
on_false = fnn.tensor([0.0, 0.0, 0.0], [3])
result = on_true.where_tensor(cond, on_false)  # [10.0, 0.0, 30.0]
```

### Einstein Summation

```python
a = fnn.randn([3, 4])
b = fnn.randn([4, 5])

# Matrix multiply
c = fnn.einsum('ij,jk->ik', [a, b])

# Dot product
a = fnn.tensor([1.0, 2.0, 3.0], [3])
b = fnn.tensor([4.0, 5.0, 6.0], [3])
dot = fnn.einsum('i,i->', [a, b])
```

### Comparison

```python
a = fnn.tensor([1.0, 2.0, 3.0], [3])
b = fnn.tensor([2.0, 2.0, 2.0], [3])

mask = a.gt_scalar(1.5)  # [0.0, 1.0, 1.0]
mask = a.lt_scalar(2.5)  # [1.0, 1.0, 0.0]
mask = a.logical_not()  # Invert boolean mask

# Element-wise maximum/minimum (with broadcasting)
c = fnn.maximum(a, b)  # Element-wise max: [2.0, 2.0, 3.0]
d = fnn.minimum(a, b)  # Element-wise min: [1.0, 2.0, 2.0]

# Use in RL for clipped double Q-learning
q1 = fnn.tensor([10.0, 20.0, 30.0], [3])
q2 = fnn.tensor([15.0, 18.0, 35.0], [3])
q_clipped = fnn.minimum(q1, q2)  # [10.0, 18.0, 30.0]
```

## Autograd

FastNN tracks operations for automatic differentiation.

### Computing Gradients

```python
import fastnn as fnn

# Create tensor (requires_grad=False by default)
x = fnn.tensor([1.0, 2.0])
x.requires_grad_(True)

# Operations create computation graph
y = x * 2
z = y.sum()

# Backward computes gradients
z.backward()

# Access gradients
print(x.grad)  # tensor([2., 2.])

# Clear gradients
x.set_grad(None)
```

### Disable Gradient Tracking

```python
# Using context manager
with fnn.no_grad():
    result = model(x)  # No gradient tracked

# Or detach
x = fnn.randn([3, 4], requires_grad=True)
y = x.detach()  # New tensor without gradient tracking
```

## Tensor Properties

```python
x = fnn.randn([3, 4, 5])

print(x.shape)        # [3, 4, 5]
print(x.ndim)         # 3
print(x.dtype)        # 'f32'
print(x.device)       # 'cpu' or Device.Wgpu(0)
print(x.numel)        # 60
print(x.stride)       # [20, 5, 1]
print(x.is_contiguous)  # True
```

## Device Management

### Moving Tensors

```python
# CPU to GPU
x_cpu = fnn.randn([3, 4])
x_gpu = x_cpu.to_gpu(0)

# GPU to CPU
x_cpu = x_gpu.to_cpu()

# Check device
print(x_cpu.device)  # 'cpu'
print(x_gpu.device)  # Device.Wgpu(0)
```

## NumPy Interop

```python
# Tensor to numpy
x = fnn.randn([3, 4])
arr = x.numpy()

# Numpy to tensor
import numpy as np
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
x = fnn.tensor(arr.flatten().tolist(), list(arr.shape))
```

## FlashAttention

Memory-efficient attention with O(N) memory:

```python
q = fnn.randn([4, 8, 64, 32])  # [batch, heads, seq_len, head_dim]
k = fnn.randn([4, 8, 64, 32])
v = fnn.randn([4, 8, 64, 32])

# Standard attention
output = fnn.flash_attention(q, k, v)

# Causal attention (for autoregressive models)
output = fnn.flash_attention(q, k, v, causal=True)
```

## Gradient Clipping

```python
params = model.parameters()

# Clip by norm
fnn.clip_grad_norm_(params, max_norm=1.0, norm_type=2.0)

# Clip by value
fnn.clip_grad_value_(params, clip_value=0.1)
```

## Setting Seeds

For reproducible results:

```python
fnn.set_seed(42)
x = fnn.randn([3, 3])

# Reset for same results
fnn.set_seed(42)
y = fnn.randn([3, 3])

# x and y are identical
```

## Threading

Control CPU thread usage:

```python
fnn.set_num_threads(4)  # Use 4 threads for operations
```
