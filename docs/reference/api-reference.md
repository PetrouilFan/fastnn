# API Reference

The stable public Python API. All symbols accessible from `import fastnn as fnn` unless otherwise noted.

## Tensor Creation

| Function | Description |
|----------|-------------|
| `fnn.tensor(data, shape, device=None)` | Tensor from flat Python list + shape |
| `fnn.tensor_from_numpy(arr, device=None)` | Tensor from numpy array |
| `fnn.zeros(shape, dtype=None, device=None)` | Tensor of zeros |
| `fnn.ones(shape, dtype=None, device=None)` | Tensor of ones |
| `fnn.full(shape, value, dtype=None, device=None)` | Tensor filled with constant |
| `fnn.eye(n, device=None)` | Identity matrix (n x n) |
| `fnn.arange(start, end, step=1, device=None)` | 1-D range |
| `fnn.linspace(start, end, n, device=None)` | n evenly spaced values |
| `fnn.randn(shape, device=None)` | Random normal (Gaussian) |
| `fnn.rand(shape, device=None)` | Random uniform [0, 1) |
| `fnn.randint(low, high, shape, device=None)` | Random integers [low, high) |

**Like functions:** `fnn.zeros_like(x)`, `fnn.ones_like(x)`, `fnn.full_like(x, val)`

## Tensor Operations

### Arithmetic

```python
fnn.add(a, b)            # a + b
fnn.sub(a, b)            # a - b
fnn.mul(a, b)            # a * b
fnn.div(a, b)            # a / b
fnn.matmul(a, b)         # a @ b
fnn.neg(a)               # -a
fnn.im2col(a, k, s, p)   # im2col transformation
fnn.einsum(expr, [a,b])  # Einstein summation
```

### Math & Activations

```python
x.abs() | x.exp() | x.log() | x.sqrt()
x.pow(n) | x.clamp(min, max) | x.erf()

fnn.relu(x)   | x.relu()      # ReLU
fnn.gelu(x)   | x.gelu()      # GELU
fnn.silu(x)   | x.silu()      # SiLU (Swish)
fnn.tanh(x)   | x.tanh()      # Tanh
fnn.sigmoid(x)| x.sigmoid()   # Sigmoid
fnn.softmax(x, dim)           # Softmax
fnn.log_softmax(x, dim)       # Log softmax
fnn.leaky_relu(x, slope)      # Leaky ReLU
fnn.softplus(x)               # Softplus
fnn.hardswish(x)              # HardSwish
```

### Reductions

```python
fnn.sum(x, dim, keepdim)      # Sum
fnn.mean(x, dim, keepdim)     # Mean
fnn.max(x, dim, keepdim)      # Max
fnn.min(x, dim, keepdim)      # Min
fnn.argmax(x, dim)            # Index of max
fnn.argmin(x, dim)            # Index of min
fnn.cumsum(x, dim)            # Cumulative sum
fnn.topk(x, k, dim)           # Top-k values and indices
```

### Element-wise Operations

```python
fnn.maximum(a, b)             # Element-wise max (broadcasting)
fnn.minimum(a, b)             # Element-wise min (broadcasting)
```

### Shape Operations

```python
x.reshape(shape)     | x.view(shape)      # Reshape / view
x.squeeze(dim)       | x.unsqueeze(dim)   # Squeeze / unsqueeze
x.permute(dims)      | x.transpose(d0,d1) # Permute / transpose
x.flip(dim)          | x.repeat(repeats)  # Flip / repeat
x.expand(sizes)                           # Broadcast without copy
```

### Composition

```python
fnn.cat([a, b], dim)          # Concatenate
fnn.stack([a, b], dim)        # Stack along new dimension
fnn.gather(x, dim, index)     # Gather along dimension
x.where_tensor(cond, other)   # Conditional selection
fnn.flash_attention(q,k,v,causal=None)  # Memory-efficient attention
```

### Fused Operations

Single-pass fused kernels for performance:

```python
x.fused_linear_relu(weight, bias)         # Matmul + ReLU
x.fused_linear_gelu(weight, bias)         # Matmul + GELU
fnn.fused_add_relu(a, b)                  # Add + ReLU
fnn.fused_conv_bn_silu(conv, bn, x)       # Conv + BN + SiLU
```

### Gradient Clipping

```python
fnn.clip_grad_norm_(params, max_norm, norm_type=2.0)
fnn.clip_grad_value_(params, clip_value)
```

## Loss Functions

```python
fnn.mse_loss(pred, target)              # Mean squared error
fnn.cross_entropy_loss(logits, target)  # Cross entropy
fnn.bce_with_logits(pred, target)       # Binary CE with logits
fnn.huber_loss(pred, target, delta=1.0) # Huber (smooth L1)
```

## Neural Network Modules

| Module | Description |
|--------|-------------|
| `fnn.Linear(in, out, bias)` | Fully connected layer |
| `fnn.Conv1d/2d/3d(cin, cout, k, s, p, bias)` | Convolutional layers |
| `fnn.ConvTranspose2d(cin, cout, k, s, p, bias)` | Transposed 2D convolution |
| `fnn.BatchNorm1d/2d(features)` | Batch normalization |
| `fnn.LayerNorm(shape)` / `fnn.RMSNorm(shape)` | Layer / RMS normalization |
| `fnn.GroupNorm(num_groups, channels)` | Group normalization |
| `fnn.Dropout(p)` / `fnn.Dropout2d(p)` | Dropout layers |
| `fnn.Embedding(num, dim)` | Embedding layer |
| `fnn.Upsample(scale_factor, mode)` | Upsampling layer |
| `fnn.MaxPool1d/2d(k, s, p)` | Max pooling |
| `fnn.AvgPool1d/2d(k, s, p)` | Average pooling |
| `fnn.AdaptiveAvgPool2d(h, w)` | Adaptive average pooling |
| `fnn.Flatten(start, end)` | Flatten layer |
| `fnn.ResidualBlock(...)` | ResNet BasicBlock |
| `fnn.FusedConvBn(conv, bn)` | Fused Conv + BN (inference) |
| `fnn.RNN(input_size, hidden_size, ...)` | RNN layer |
| `fnn.LSTM(input_size, hidden_size, ...)` | LSTM layer |
| `fnn.GRU(input_size, hidden_size, ...)` | GRU layer |
| `fnn.Sequential([...])` | Sequential container |
| `fnn.ModuleList([...])` | Module list container |

**Activation layer classes:** `fnn.ReLU()`, `fnn.GELU()`, `fnn.Sigmoid()`, `fnn.Tanh()`, `fnn.SiLU()` (Swish), `fnn.LeakyReLU(negative_slope)`, `fnn.Softplus(beta, threshold)`, `fnn.Hardswish()`, `fnn.Elu(alpha)`, `fnn.Mish()`, `fnn.PReLU()`

**Recurrent layers:** `fnn.RNN(input_size, hidden_size, num_layers, ...)`, `fnn.LSTM(...)`, `fnn.GRU(...)` all support `batch_first`, `dropout`, `bidirectional`, and return `(output, hidden)` tuples.

## Optimizers

```python
fnn.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)
fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
fnn.AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
fnn.Muon(params, lr, momentum=0.95, weight_decay=0, nesterov=True)
fnn.Lion(params, lr, betas=(0.95, 0.98), weight_decay=0)
fnn.RMSprop(params, lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
```

See [Optimizers](optimizers.md) for details.

## LR Schedulers

```python
fnn.StepLR(optimizer, step_size, gamma)
fnn.CosineAnnealingLR(optimizer, T_max, eta_min)
fnn.ExponentialLR(optimizer, gamma)
fnn.ReduceLROnPlateau(optimizer, mode, factor, patience, min_lr)
```

## Data

```python
fnn.Dataset                              # Base dataset class
fnn.TensorDataset(*tensors)              # Dataset from tensors
fnn.DataLoader(dataset, batch_size, shuffle, drop_last, num_workers)
```

## Callbacks

```python
fnn.EarlyStopping(monitor, patience, min_delta, restore_best_weights)
fnn.ModelCheckpoint(dir_path, monitor, mode, save_best_only, verbose)
fnn.LearningRateScheduler(scheduler, **kwargs)
fnn.CSVLogger(filepath)
```

## Model I/O

```python
fnn.io.save(model, "model.fnn")
model = fnn.io.load("model.fnn")
fnn.io.convert_from_pytorch(torch_model, "model.fnn")
info = fnn.io.convert_from_onnx("model.onnx", "model.fnn")
```

See [IO & Serialization](io.md) for format details.

## Models (fastnn.models)

```python
fnn.models.MLP(input_dim, hidden_dims, output_dim, ...)
fnn.models.Transformer(vocab_size, max_seq_len, d_model, ...)
```

## YOLO & NMS Utilities

```python
fnn.YOLO("model.onnx")                            # YOLO object detection model
fnn.nms(boxes, scores, iou_threshold)              # Non-maximum suppression
fnn.yolo_decode(output, conf_thresh)               # YOLO output decoding
fnn.yolo_dfl_decode(output, conf_thresh)           # YOLOv8+ DFL decoding
fnn.xywh2xyxy(boxes)                               # Convert bbox format
fnn.scale_boxes(img1_shape, boxes, img0_shape)     # Rescale boxes
```

## Utilities

```python
fnn.no_grad()                   # Disable autograd
fnn.set_seed(n)                 # Set random seed
fnn.set_num_threads(n)          # Set CPU thread count
fnn.set_default_device(device)  # Set default device (cpu/wgpu)
fnn.allocator_stats()           # Memory stats
fnn.list_registered_ops()       # List all tensor operations
fnn.load_state_dict(model, d)   # Load parameter state
```

## Tensor Methods

Every tensor exposes these properties and methods:

**Metadata:** `tensor.shape`, `.dtype`, `.device`, `.ndim`, `.numel`, `.stride`, `.is_contiguous`, `.requires_grad`, `.grad`

**Conversion:** `.numpy()` to numpy array, `.item()` to Python scalar, `.to_cpu()` to CPU, `.to_gpu(id)` to GPU

**Autograd:** `.backward()` computes gradients, `.detach()` detaches from graph, `.requires_grad_(bool)` sets gradient tracking

**Indexing:** Standard Python slicing via `tensor[slice]`

## See also

- [Tensors](tensors.md) -- Tensor creation, ops, autograd
- [Neural Network Modules](nn-modules.md) -- Module catalog with examples
- [Optimizers](optimizers.md) -- Optimizer and scheduler details
- [IO & Serialization](io.md) -- Save/load and format conversion
- [Python API](python-api.md) -- Compiled training and GPU backend
- [Index](../index.md) -- Full documentation index
