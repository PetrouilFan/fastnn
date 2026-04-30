# API Reference

The stable public API remains available from `import fastnn as fnn`. New code may also import narrower facade modules:

```python
from fastnn.tensor import tensor, zeros, randn
from fastnn.ops import matmul, relu, softmax
from fastnn.nn import Linear, Conv2d, Sequential
from fastnn.losses import mse_loss, cross_entropy_loss
```

## Tensor Creation

| Function | Description |
|----------|-------------|
| `fnn.tensor(data)` | Create tensor from Python list |
| `fnn.zeros(*shape)` | Create tensor of zeros |
| `fnn.ones(*shape)` | Create tensor of ones |
| `fnn.full(shape, value)` | Create tensor filled with value |
| `fnn.eye(n)` | Create identity matrix |
| `fnn.arange(end)` | Create range [0, end) |
| `fnn.linspace(start, end, n)` | Create n values from start to end |
| `fnn.randn(*shape)` | Random normal (Gaussian) |
| `fnn.rand(*shape)` | Random uniform [0, 1) |
| `fnn.randint(low, high, shape)` | Random integers [low, high) |

## Tensor Operations

### Arithmetic
```python
fnn.add(a, b)      # Addition
fnn.sub(a, b)      # Subtraction  
fnn.mul(a, b)      # Multiplication
fnn.div(a, b)     # Division
fnn.matmul(a, b)  # Matrix multiplication
fnn.neg(a)        # Negation
```

### Math
```python
x.abs()           # Absolute value
x.exp()           # Exponential
x.log()           # Natural log
x.sqrt()          # Square root
x.pow(n)          # Power
x.clamp(min, max) # Clamp values
```

### Activations
```python
fnn.relu(x)       # ReLU
fnn.sigmoid(x)    # Sigmoid
fnn.tanh(x)       # Tanh
fnn.gelu(x)       # GELU
fnn.silu(x)       # SiLU (Swish)
```

### Reductions
```python
fnn.sum(x)          # Sum all elements
fnn.mean(x)        # Mean
fnn.max(x)          # Maximum (reduction)
fnn.min(x)          # Minimum (reduction)
fnn.argmax(x)       # Index of max
fnn.argmin(x)       # Index of min
fnn.softmax(x, axis)    # Softmax
fnn.log_softmax(x, axis) # Log softmax
```

### Element-wise Maximum/Minimum
```python
# Element-wise maximum/minimum with broadcasting
fnn.maximum(a, b)   # Element-wise max (like np.maximum)
fnn.minimum(a, b)   # Element-wise min (like np.minimum)

# Example: Clipped double Q-learning in SAC
q1 = fnn.randn([batch_size, action_dim])
q2 = fnn.randn([batch_size, action_dim])
q_clipped = fnn.minimum(q1, q2)  # Conservative Q estimate
```

### Concatenation and Stacking
```python
fnn.cat([a, b], dim)      # Concatenate along existing dimension
fnn.stack([a, b], dim)    # Stack along NEW dimension (like np.stack)

# Example:
x = fnn.tensor([1, 2, 3], [3])
y = fnn.tensor([4, 5, 6], [3])

# cat requires same dimensions except concat dim
z = fnn.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)  # [2, 3]

# stack creates new dimension
z = fnn.stack([x, y], dim=0)  # [2, 3] - cleaner!
```

## Loss Functions

```python
fnn.mse_loss(pred, target)           # Mean squared error
fnn.cross_entropy_loss(logits, target) # Cross entropy
```

## Neural Network Modules

| Module | Description |
|--------|-------------|
| `fnn.Linear(in, out, bias)` | Fully connected layer |
| `fnn.Conv2d(cin, cout, kernel, stride, padding, bias)` | 2D convolution |
| `fnn.BatchNorm1d(features)` | Batch normalization |
| `fnn.LayerNorm(shape)` | Layer normalization |
| `fnn.Dropout(p)` | Dropout |
| `fnn.Embedding(num, dim)` | Word embeddings |
| `fnn.ReLU()` | ReLU activation |
| `fnn.GELU()` | GELU activation |
| `fnn.Sigmoid()` | Sigmoid activation |
| `fnn.Tanh()` | Tanh activation |
| `fnn.SiLU()` | SiLU activation |
| `fnn.Sequential(layers)` | Sequential container |
| `fnn.ModuleList(modules)` | Module list container |

## Optimizers

```python
fnn.SGD(params, lr, momentum=0, weight_decay=0)
fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
fnn.AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

## Data

```python
fnn.Dataset        # Base dataset class
fnn.TensorDataset(*tensors)  # Dataset from tensors
fnn.DataLoader(dataset, batch_size, shuffle, drop_last, num_workers)
```

## Callbacks

```python
fnn.EarlyStopping(monitor, patience, min_delta, restore_best_weights)
fnn.ModelCheckpoint(dir_path, monitor, mode, save_best_only, verbose)
fnn.LearningRateScheduler(schedule, lr, step_size, gamma, T_max, eta_min)
fnn.CSVLogger(filepath)
```

## Model I/O

```python
fnn.save_model(model, path)  # Save to safetensors
fnn.load_model(model, path)   # Load from safetensors
```

## Utilities

```python
fnn.no_grad()              # Context manager to disable gradients
fnn.set_seed(n)            # Set random seed
fnn.set_num_threads(n)     # Set CPU thread count
fnn.set_default_device(device)  # Set default device
fnn.allocator_stats()      # Get memory stats
fnn.list_registered_ops() # List all operations
```

## Models

```python
fnn.models.MLP(input_dim, hidden_dims, output_dim, activation, dropout, batch_norm)
# Transformer: Not yet implemented
```

## Tensor Methods

Each tensor supports:

```python
tensor.shape       # Tuple of dimensions
tensor.dtype       # Data type
tensor.device      # Device (cpu/cuda)
tensor.requires_grad  # Gradient tracking
tensor.grad        # Gradient (after backward())
tensor.numpy()     # Convert to numpy
tensor.item()      # Get scalar value
tensor.backward()  # Compute gradients
```
