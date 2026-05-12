# API Reference

The stable public API is available from `import fastnn as fnn`. All tensor operations, layers, and losses are accessible directly from the `fastnn` module:

```python
import fastnn as fnn

# Tensor creation
x = fnn.tensor([1, 2, 3])
y = fnn.zeros([10, 20])
z = fnn.randn([5, 5])

# Operations
result = fnn.matmul(a, b)
activated = fnn.relu(x)
probs = fnn.softmax(logits)

# Layers
layer = fnn.Linear(10, 20)
conv = fnn.Conv2d(3, 16, 3)

# Losses
loss = fnn.mse_loss(pred, target)
```

## Tensor Creation

| Function | Description |
|----------|-------------|
| `fnn.tensor(data, shape, device=None)` | Create tensor from flat Python list + shape |
| `fnn.from_numpy(arr, device=None)` | Create tensor from numpy array |
| `fnn.zeros(shape, dtype=None, device=None)` | Create tensor of zeros |
| `fnn.ones(shape, dtype=None, device=None)` | Create tensor of ones |
| `fnn.full(shape, value, dtype=None, device=None)` | Create tensor filled with value |
| `fnn.eye(n, device=None)` | Create identity matrix |
| `fnn.arange(start, end, step=1, device=None)` | Create range |
| `fnn.linspace(start, end, n, device=None)` | Create n values from start to end |
| `fnn.randn(shape, device=None)` | Random normal (Gaussian) |
| `fnn.rand(shape, device=None)` | Random uniform [0, 1) |
| `fnn.randint(low, high, shape, device=None)` | Random integers [low, high) |

## Tensor Operations

### Arithmetic
```python
fnn.add(a, b)      # Addition
fnn.sub(a, b)      # Subtraction  
fnn.mul(a, b)      # Multiplication
fnn.div(a, b)     # Division
fnn.matmul(a, b)  # Matrix multiplication
fnn.neg(a)        # Negation
fnn.im2col(a, kernel_size, stride, padding)  # im2col transformation
```

### Math
```python
x.abs()           # Absolute value
x.exp()           # Exponential
x.log()           # Natural log
x.sqrt()          # Square root
x.pow(n)          # Power
x.clamp(min, max) # Clamp values
fnn.pow(a, n)      # Power (function form)
```

### Activations
```python
fnn.relu(x)       # ReLU
fnn.sigmoid(x)    # Sigmoid
fnn.tanh(x)       # Tanh
fnn.gelu(x)       # GELU
fnn.silu(x)       # SiLU (Swish)
fnn.leaky_relu(x, negative_slope)  # Leaky ReLU
fnn.softplus(x)   # Softplus
fnn.hardswish(x)   # HardSwish
```

### Reductions
```python
fnn.sum(x, dim=None, keepdim=False)      # Sum all elements or along dim
fnn.mean(x, dim=None, keepdim=False)    # Mean
fnn.max(x, dim=None, keepdim=False)    # Maximum (reduction)
fnn.min(x, dim=None, keepdim=False)    # Minimum (reduction)
fnn.argmax(x, dim)                     # Index of max
fnn.argmin(x, dim)                     # Index of min
fnn.softmax(x, dim)                    # Softmax
fnn.log_softmax(x, dim)                # Log softmax
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

### Advanced Operations
```python
fnn.einsum('ij,jk->ik', [a, b])  # Einstein summation
fnn.flash_attention(q, k, v, causal=None)  # Memory-efficient attention
fnn.fused_add_relu(a, b)  # Fused add + ReLU
fnn.fused_conv_bn_silu(conv, bn, x)  # Fused Conv + BN + SiLU
fnn.fused_linear_relu(x, weight, bias)  # Fused matmul + ReLU
fnn.fused_linear_gelu(x, weight, bias)  # Fused matmul + GELU
fnn.cumsum(x, dim)        # Cumulative sum
fnn.erf(x)                # Error function
fnn.topk(x, k, dim)       # Top-k values and indices
fnn.gather(x, dim, index)  # Gather values along dimension
```

### Concatenation and Stacking
```python
fnn.cat([a, b], dim)      # Concatenate along existing dimension
fnn.stack([a, b], dim)    # Stack along NEW dimension (like np.stack)
fnn.repeat(x, repeats)    # Repeat tensor along dimensions
fnn.flip(x, dim)         # Reverse along dimension
x.where_tensor(cond, other)  # Conditional selection
x.squeeze(dim)             # Remove dimensions of size 1
x.unsqueeze(dim)           # Add dimension of size 1
x.permute(dims)           # Permute dimensions
x.transpose(d0, d1)        # Transpose two dimensions
x.view(shape)             # View (no copy)
x.reshape(shape)           # Reshape (supports -1)

# Example:
x = fnn.tensor([1, 2, 3], [3])
y = fnn.tensor([4, 5, 6], [3])

# cat requires same dimensions except concat dim
z = fnn.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)  # [2, 3]

# stack creates new dimension
z = fnn.stack([x, y], dim=0)  # [2, 3] - cleaner!
```

### Gradient Clipping
```python
fnn.clip_grad_norm_(params, max_norm, norm_type=2.0)
fnn.clip_grad_value_(params, clip_value)
```

## Loss Functions

```python
fnn.mse_loss(pred, target)                # Mean squared error
fnn.cross_entropy_loss(logits, target)      # Cross entropy
fnn.bce_with_logits(pred, target)           # Binary cross-entropy with logits
fnn.huber_loss(pred, target, delta=1.0)    # Huber loss (smooth L1)
```

## Neural Network Modules

| Module | Description |
|--------|-------------|
| `fnn.Linear(in, out, bias)` | Fully connected layer |
| `fnn.Conv1d(cin, cout, kernel, stride, padding, bias)` | 1D convolution |
| `fnn.Conv2d(cin, cout, kernel, stride, padding, bias)` | 2D convolution |
| `fnn.Conv3d(cin, cout, kernel, stride, padding, bias)` | 3D convolution |
| `fnn.ConvTranspose2d(cin, cout, kernel, stride, padding, bias)` | Transposed 2D convolution |
| `fnn.BatchNorm1d(features)` | Batch normalization 1D |
| `fnn.BatchNorm2d(features)` | Batch normalization 2D |
| `fnn.LayerNorm(shape)` | Layer normalization |
| `fnn.RMSNorm(shape)` | RMS normalization |
| `fnn.GroupNorm(num_groups, num_channels)` | Group normalization |
| `fnn.Dropout(p)` | Dropout |
| `fnn.Dropout2d(p)` | Dropout2d (channel-wise) |
| `fnn.Embedding(num, dim)` | Word embeddings |
| `fnn.Upsample(scale_factor, mode)` | Upsampling layer |
| `fnn.ReLU()` | ReLU activation |
| `fnn.GELU()` | GELU activation |
| `fnn.Sigmoid()` | Sigmoid activation |
| `fnn.Tanh()` | Tanh activation |
| `fnn.SiLU()` | SiLU activation |
| `fnn.LeakyReLU(negative_slope)` | Leaky ReLU |
| `fnn.Softplus(beta, threshold)` | Softplus |
| `fnn.Hardswish()` | HardSwish |
| `fnn.Elu(alpha)` | ELU |
| `fnn.Mish()` | Mish |
| `fnn.MaxPool2d(kernel_size, stride, padding)` | Max pooling 2D |
| `fnn.AdaptiveAvgPool2d(output_h, output_w)` | Adaptive average pooling |
| `fnn.MaxPool1d(kernel_size, stride, padding)` | Max pooling 1D |
| `fnn.AvgPool1d(kernel_size, stride, padding)` | Average pooling 1D |
| `fnn.AvgPool2d(kernel_size, stride, padding)` | Average pooling 2D |
| `fnn.PReLU()` | Parametric ReLU |
| `fnn.Flatten(start_dim, end_dim)` | Flatten layer |
| `fnn.ResidualBlock(conv1, bn1, relu, conv2, bn2, downsample)` | ResNet BasicBlock |
| `fnn.FusedConvBn(conv, bn)` | Fused Conv2d+BatchNorm2d for inference |
| `fnn.FusedConvBnRelu(conv, bn)` | Fused Conv+BN+ReLU |
| `fnn.FusedConvBnGelu(conv, bn)` | Fused Conv+BN+GELU |
| `fnn.Sequential(layers)` | Sequential container |
| `fnn.ModuleList(modules)` | Module list container |

## Optimizers

```python
fnn.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)
fnn.Adam(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
fnn.AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
fnn.Muon(params, lr, momentum=0.95, weight_decay=0, nesterov=True)
fnn.Lion(params, lr, betas=(0.95, 0.98), weight_decay=0)
fnn.RMSprop(params, lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
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
fnn.LearningRateScheduler(scheduler, **kwargs)  # scheduler from fnn.schedulers, or string with optimizer
fnn.CSVLogger(filepath)
```

## Model I/O

### Unified API (fastnn.io)

```python
import fastnn as fnn

# Save/Load models (custom binary format .fnn)
fnn.io.save(model, "model.fnn")        # Save model
model = fnn.io.load("model.fnn")        # Load model

# Convert from other formats
fnn.io.convert_from_pytorch(torch_model, "model.fnn")
info = fnn.io.convert_from_onnx("model.onnx", "model.fnn")
```

## Packed Precision Layers

```python
# Packed linear layers
fnn.Linear4(in, out, bias)      # 4-bit packed linear
fnn.Linear8(in, out, bias)      # 8-bit packed linear
fnn.Linear16(in, out, bias)     # 16-bit packed linear
fnn.Linear32(in, out, bias)     # 32-bit packed linear

# Packed fused layers
fnn.PackedConvRelu4(ci, co, k, ...)   # Packed conv + ReLU (4-bit)
fnn.PackedConvRelu8(ci, co, k, ...)   # Packed conv + ReLU (8-bit)
fnn.PackedLinearGelu4(in, out)         # Packed linear + GELU (4-bit)
fnn.PackedLinearGelu8(in, out)         # Packed linear + GELU (8-bit)

# Packed conv2d
fnn.PackedConv2d4(ci, co, k, ...)     # 4-bit packed conv
fnn.PackedConv2d8(ci, co, k, ...)     # 8-bit packed conv

# Packed tensors
fnn.packed_tensor(data, shape, scale, zero, dtype="u4")
fnn.packed_tensor_from_f32(data, shape, dtype="u4")
fnn.PackedTensor4(data, shape, scale, zp)    # 4-bit packed tensor
fnn.PackedTensor8(data, shape, scale, zp)    # 8-bit packed tensor
fnn.QuantizedTensor(data, scale, zp, qdtype) # Quantized tensor

# Packed multi-head attention & transformer
fnn.PackedMultiHeadAttention4(d_model, n_heads)   # 4-bit packed MHA
fnn.PackedMultiHeadAttention8(d_model, n_heads)   # 8-bit packed MHA
fnn.PackedTransformerEncoder4(vocab, max_seq, ...) # 4-bit packed transformer
fnn.PackedTransformerEncoder8(vocab, max_seq, ...) # 8-bit packed transformer

# Master weight optimizers
fnn.MasterWeightOptimizer4(params, lr)
fnn.MasterWeightOptimizer8(params, lr)

# Backend control
fnn.use_wgpu()   # Switch to GPU backend
fnn.use_cpu()    # Switch to CPU backend
fnn.is_wgpu()    # Check if GPU backend is active
```

## YOLO & NMS Utilities

```python
fnn.YOLO("model.onnx")                    # YOLO object detection model
fnn.nms(boxes, scores, iou_threshold)     # Non-maximum suppression
fnn.yolo_decode(output, conf_thresh)      # YOLO output decoding
fnn.yolo_dfl_decode(output, conf_thresh)  # YOLOv8+ DFL decoding
fnn.xywh2xyxy(boxes)                      # Convert bbox format
fnn.scale_boxes(img1_shape, boxes, img0_shape)  # Rescale boxes
```

## DAG Executor

```python
fnn.DAGExecutor(params, ops, input_names, output_names)
# Executes ONNX model graphs natively in Rust
executor.forward({"input_name": tensor})
executor(tensor)  # For single-input/single-output
```

## Utilities

```python
fnn.no_grad()              # Context manager to disable gradients
fnn.set_seed(n)            # Set random seed
fnn.set_num_threads(n)     # Set CPU thread count
fnn.set_default_device(device)  # Set default device (cpu/wgpu)
fnn.allocator_stats()      # Get memory stats
fnn.list_registered_ops()  # List all registered operations
fnn.batched_mlp_forward()  # Batched MLP inference
fnn.clear_storage_pool()   # Clear storage pool cache
fnn.import_onnx(onnx_path, fnn_path)  # Import ONNX model
fnn.load_state_dict(model, state_dict)  # Load state dict
fnn.from_numpy(arr)        # Create tensor from numpy array
```

## Parallel Training

```python
from fastnn.parallel import DataParallel

# Create model replicas for each GPU
model_gpu0 = fnn.models.MLP(...)
model_gpu1 = fnn.models.MLP(...)

dp_model = DataParallel(
    [model_gpu0, model_gpu1],
    device_ids=[0, 1],
    weights=[0.6, 0.4]  # Optional: data distribution weights
)

# Training
loss = dp_model.forward_backward(x, y, fnn.cross_entropy_loss)
dp_model.sync_gradients()
for opt in optimizers:
    opt.step()
    opt.zero_grad()
```

## Models

```python
fnn.models.MLP(input_dim, hidden_dims, output_dim, activation, dropout, batch_norm)
fnn.models.Transformer(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, num_classes, dropout_p)
```

## Tensor Methods

Each tensor supports:

```python
# Shape and metadata
tensor.shape       # Tuple of dimensions
tensor.dtype       # Data type
tensor.device      # Device (cpu/wgpu:0)
tensor.requires_grad  # Gradient tracking
tensor.grad        # Gradient (after backward())
tensor.numel       # Total number of elements
tensor.stride      # Stride tuple
tensor.is_contiguous  # Whether tensor is contiguous

# Conversion
tensor.numpy()     # Convert to numpy
tensor.item()      # Get scalar value
tensor.to_cpu()     # Move to CPU
tensor.to_gpu(id)   # Move to GPU

# Gradient
tensor.backward()  # Compute gradients
tensor.detach()    # Detach from computation graph
tensor.requires_grad_(requires_grad)  # Set grad requirement

# Math operations (also available as functions)
tensor.abs()
tensor.exp()
tensor.log()
tensor.sqrt()
tensor.pow(n)
tensor.clamp(min, max)
tensor.neg()

# Activations (tensor methods)
tensor.relu()
tensor.gelu()
tensor.sigmoid()
tensor.tanh()
tensor.silu()
tensor.leaky_relu(negative_slope)
tensor.softplus(beta, threshold)
tensor.hardswish()
tensor.mish()
tensor.softmax(dim)
tensor.log_softmax(dim)
tensor.erf()              # Error function

# Fused linear + activation
tensor.fused_linear_relu(weight, bias=None)   # matmul + ReLU in one pass
tensor.fused_linear_gelu(weight, bias=None)   # matmul + GELU in one pass

# Note: ELU is available as a layer (fnn.Elu(alpha)) but NOT as a tensor method

# Reductions
tensor.sum(dim=None, keepdim=False)
tensor.mean(dim=None, keepdim=False)
tensor.max(dim=None, keepdim=False)
tensor.min(dim=None, keepdim=False)
tensor.cumsum(dim=None)   # Cumulative sum

# Top-k
tensor.topk(k, dim=None)  # Returns (values, indices)

# Shape operations
tensor.view(shape)
tensor.reshape(shape)
tensor.squeeze(dim=None)
tensor.unsqueeze(dim)
tensor.permute(dims)
tensor.transpose(d0, d1)
tensor.flip(dim)
tensor.repeat(repeats)
tensor.expand(sizes)       # Expand without copy
tensor.gather(dim, index)  # Gather values along dim
tensor.where_tensor(cond, other)

# Slicing
tensor[slice]              # Standard Python slicing

# Indexing
tensor[slice]       # Indexing/slicing
```
