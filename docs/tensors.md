# Tensors

Tensors are the fundamental data structure in FastNN, similar to numpy arrays but with automatic differentiation support.

## Creating Tensors

### From Python Data

```python
import fastnn as fnn

# From nested lists
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]])

# Shape is inferred automatically
print(x.shape)  # (2, 2)
```

### Utility Functions

```python
# Zeros
z = fnn.zeros(3, 4)          # 3x4 tensor of zeros

# Ones
o = fnn.ones(2, 3)           # 2x3 tensor of ones

# Full with constant
f = fnn.full((2, 2), 5.0)    # 2x2 tensor filled with 5.0

# Identity matrix
I = fnn.eye(3)               # 3x3 identity matrix

# Range
r = fnn.arange(10)           # [0, 1, 2, ..., 9]

# Linear spacing
l = fnn.linspace(0, 1, 10)   # 10 values from 0 to 1

# Random
rand = fnn.randn(3, 4)       # Gaussian (mean=0, std=1)
rand = fnn.rand(3, 4)        # Uniform [0, 1)
rand = fnn.randint(0, 10, (3, 4))  # Uniform integers [0, 10)

# Like functions
z = fnn.zeros_like(x)
o = fnn.ones_like(x)
f = fnn.full_like(x, 3.0)
```

## Tensor Operations

### Arithmetic

```python
a = fnn.tensor([1.0, 2.0])
b = fnn.tensor([3.0, 4.0])

c = a + b       # add(a, b)
c = a - b       # sub(a, b)
c = a * b       # mul(a, b)
c = a / b       # div(a, b)
c = -a          # neg(a)
c = a @ b       # matmul (matrix multiplication)
```

### Math Functions

```python
x = fnn.tensor([1.0, 2.0, 3.0])

y = x.abs()           # absolute value
y = x.exp()           # exponential
y = x.log()           # natural logarithm
y = x.sqrt()          # square root
y = x.pow(2)          # power
y = x.clamp(0, 1)    # clamp values to [0, 1]
```

### Activation Functions

```python
x = fnn.tensor([[-1.0, 0.0, 1.0]])

y = x.relu()          # ReLU: max(0, x)
y = x.sigmoid()       # Sigmoid: 1/(1+exp(-x))
y = x.tanh()          # Tanh: (exp(x)-exp(-x))/(exp(x)+exp(-x))
y = x.gelu()          # GELU: x * Φ(x)
y = x.silu()          # SiLU (Swish): x * sigmoid(x)
```

### Reduction Operations

```python
x = fnn.tensor([[1.0, 2.0], [3.0, 4.0]])

s = x.sum()           # sum all elements
m = x.mean()         # mean of all elements
mx = x.max()         # maximum
mn = x.min()         # minimum

# Along specific axis
s0 = x.sum(axis=0)   # sum along axis 0
m1 = x.mean(axis=1) # mean along axis 1

# Softmax
sm = fnn.softmax(x, axis=-1)
lsm = fnn.log_softmax(x, axis=-1)

# Argmax/Argmin
am = fnn.argmax(x)   # index of max
ami = fnn.argmin(x)  # index of min
```

## Autograd

FastNN tracks operations for automatic differentiation.

### Computing Gradients

```python
import fastnn as fnn

# Create tensor (requires_grad=False by default)
x = fnn.tensor([1.0, 2.0])

# Operations create computation graph
y = x * 2
z = y.sum()

# Backward computes gradients
z.backward()

# Access gradients
print(x.grad)  # tensor([2., 2.])
```

### Disable Gradient Tracking

```python
# Using context manager
with fnn.no_grad():
    result = model(x)  # No gradient tracked
    
# Or inference mode
result = model(x)  # gradients still tracked
```

## Tensor Properties

```python
x = fnn.randn(3, 4, 5)

print(x.shape)      # (3, 4, 5)
print(x.dtype)      # float32
print(x.device)     # cpu
print(x.requires_grad)  # False
```

## Converting to NumPy

```python
x = fnn.randn(3, 4)

# Convert to numpy array
arr = x.numpy()
print(type(arr))  # <class 'numpy.ndarray'>
```

## Setting Seeds

For reproducible results:

```python
fnn.set_seed(42)
x = fnn.randn(3, 3)

# Reset for same results
fnn.set_seed(42)
y = fnn.randn(3, 3)

# x and y are identical
```

## Threading

Control CPU thread usage:

```python
fnn.set_num_threads(4)  # Use 4 threads for operations
```
