# Neural Network Modules

Layer and module catalog for building neural networks. All modules are callable objects from `import fastnn as fnn`.

## Core Layers

### Linear

Fully connected layer: `y = xW + b`

```python
linear = fnn.Linear(in_features=128, out_features=64, bias=True)
output = linear(input_tensor)  # (batch_size, 64)
```

**Parameters:** `in_features`, `out_features`, `bias` (default True)

### Embedding

```python
embedding = fnn.Embedding(num_embeddings=10000, embedding_dim=256)
embeddings = embedding(fnn.randint(low=0, high=10000, shape=[2, 3]))
# (2, 3, 256)
```

## Convolutional Layers

### Conv1d

```python
conv1d = fnn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,
                     stride=1, padding=1, dilation=1, bias=True)
# Input: (batch, channels, length) -> Output: (8, 256, 64)
output = conv1d(fnn.randn([8, 128, 64]))
```

### Conv2d

```python
conv = fnn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                   stride=1, padding=1, dilation=1, groups=1, bias=True)
# Input: (8, 3, 32, 32) -> Output: (8, 64, 32, 32)
output = conv(fnn.randn([8, 3, 32, 32]))
```

### Conv3d

```python
conv3d = fnn.Conv3d(in_channels=16, out_channels=32, kernel_size=3,
                     stride=1, padding=1, dilation=1, bias=True)
# Input: (2, 16, 8, 32, 32) -> Output: (2, 32, 8, 32, 32)
```

### ConvTranspose2d

```python
conv_t = fnn.ConvTranspose2d(in_channels=64, out_channels=32,
                              kernel_size=3, stride=2, padding=1, bias=True)
# Input: (4, 64, 8, 8) -> Output: (4, 32, 16, 16)
```

## Normalization Layers

### BatchNorm1d / BatchNorm2d

```python
bn = fnn.BatchNorm1d(num_features=64, momentum=0.1, eps=1e-5)
bn.train()    # Track running statistics
bn.eval()     # Use running statistics

bn2d = fnn.BatchNorm2d(num_features=64, momentum=0.1, eps=1e-5)
```

### LayerNorm

```python
ln = fnn.LayerNorm(normalized_shape=64, eps=1e-5)
output = ln(fnn.randn([8, 32, 64]))  # Normalizes over last dim
```

### RMSNorm

RMS normalization (used in Llama and Mistral architectures):

```python
rms = fnn.RMSNorm(normalized_shape=64, eps=1e-5)
output = rms(fnn.randn([8, 32, 64]))
```

### GroupNorm

```python
gn = fnn.GroupNorm(num_groups=8, num_channels=64, eps=1e-5)
output = gn(fnn.randn([8, 64, 32, 32]))
```

## Activation Functions

Available as both layer objects and tensor methods:

```python
# As layers
relu = fnn.ReLU()
gelu = fnn.GELU()
sigmoid = fnn.Sigmoid()
tanh = fnn.Tanh()
silu = fnn.SiLU()                              # Swish: x * sigmoid(x)
leaky_relu = fnn.LeakyReLU(negative_slope=0.01)
softplus = fnn.Softplus(beta=1.0, threshold=20.0)
hardswish = fnn.Hardswish()
elu = fnn.Elu(alpha=1.0)
mish = fnn.Mish()
prelu = fnn.PReLU()
```

```python
# As tensor methods
output = input_tensor.relu()
output = input_tensor.gelu()
output = input_tensor.sigmoid()
output = input_tensor.tanh()
output = input_tensor.silu()
output = input_tensor.leaky_relu(0.01)
output = input_tensor.softmax(dim=-1)
output = input_tensor.log_softmax(dim=-1)
```

## Pooling Layers

```python
# MaxPool1d: (8, 64, 32) -> (8, 64, 16)
pool1d = fnn.MaxPool1d(kernel_size=2, stride=2)

# MaxPool2d: (8, 64, 32, 32) -> (8, 64, 16, 16)
pool2d = fnn.MaxPool2d(kernel_size=2, stride=2)

# AvgPool1d: (8, 64, 32) -> (8, 64, 16)
avg1d = fnn.AvgPool1d(kernel_size=2, stride=2)

# AvgPool2d: (8, 64, 32, 32) -> (8, 64, 16, 16)
avg2d = fnn.AvgPool2d(kernel_size=2, stride=2)

# AdaptiveAvgPool2d: (8, 64, 32, 32) -> (8, 64, 1, 1)
adapt = fnn.AdaptiveAvgPool2d(output_h=1, output_w=1)
```

## Dropout

```python
dropout = fnn.Dropout(p=0.5)
dropout.train()    # Applies dropout
dropout.eval()     # Inference: no dropout

# Channel-wise dropout for 2D (batch, ch, h, w)
dropout2d = fnn.Dropout2d(p=0.5)
```

## Upsample

```python
up = fnn.Upsample(scale_factor=2.0, mode='nearest')
output = up(fnn.randn([1, 2, 4, 4]))       # (1, 2, 8, 8)
```

## Flatten

```python
flatten = fnn.Flatten(start_dim=1, end_dim=-1)
output = flatten(fnn.randn([32, 3, 32, 32]))  # [32, 3072]
```

## Recurrent Layers

```python
lstm = fnn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                bias=True, batch_first=True, dropout=0.0, bidirectional=False)
rnn = fnn.RNN(input_size=128, hidden_size=256, num_layers=2, ...)
gru = fnn.GRU(input_size=128, hidden_size=256, num_layers=2, ...)

# Input: (batch, seq_len, input_size) with batch_first=True
output, hidden = lstm(fnn.randn([8, 32, 128]))
```

## Container

### Sequential

```python
model = fnn.Sequential([
    fnn.Linear(784, 256),
    fnn.ReLU(),
    fnn.Linear(256, 128),
    fnn.ReLU(),
    fnn.Linear(128, 10),
])
output = model(input_tensor)
```

### ModuleList

```python
layers = fnn.ModuleList([fnn.Linear(10, 10), fnn.Linear(10, 1)])
for layer in layers:
    input_tensor = layer(input_tensor)
```

## ResidualBlock

ResNet-style skip connection block:

```python
block = fnn.ResidualBlock(
    conv1_in=64, conv1_out=64, conv1_kernel=3, conv1_stride=1, conv1_padding=1,
    bn1_features=64,
    conv2_in=64, conv2_out=64, conv2_kernel=3, conv2_stride=1, conv2_padding=1,
    bn2_features=64,
    downsample=None  # (ds_in, ds_out, ds_k, ds_s, ds_p, ds_bn) for stride > 1
)
output = block(fnn.randn([4, 64, 32, 32]))  # (4, 64, 32, 32)
```

## Fused Inference Layers

Fused Conv2d + BatchNorm2d for single-pass inference:

```python
conv = fnn.Conv2d(3, 64, 3, padding=1)
bn = fnn.BatchNorm2d(64)

fused = fnn.FusedConvBn(conv, bn)            # Conv + BN
fused_relu = fnn.FusedConvBnRelu(conv, bn)   # Conv + BN + ReLU
fused_gelu = fnn.FusedConvBnGelu(conv, bn)   # Conv + BN + GELU
```

## Transformer Encoder

```python
encoder = fnn.models.Transformer(
    vocab_size=10000, max_seq_len=512, d_model=512,
    n_head=8, n_layers=6, ff_dim=2048, n_classes=10, dropout=0.1
)
```

For custom attention, use `fnn.flash_attention()`. See [tensors](tensors.md#flashattention).

## Loss Functions

```python
fnn.mse_loss(pred, target)                  # Mean squared error
fnn.cross_entropy_loss(logits, target)       # Cross entropy
fnn.bce_with_logits(pred, target)            # Binary CE with logits
fnn.huber_loss(pred, target, delta=1.0)     # Huber (smooth L1)
fnn.l1_loss(pred, target)                   # L1 loss
```

## Module Utilities

```python
params = model.parameters()
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

model.train()   # Enable dropout, BN tracking
model.eval()    # Disable dropout, freeze BN
```

## See also

- [Tensors](tensors.md) -- Tensor creation, operations, autograd
- [Optimizers](optimizers.md) -- Optimizer APIs for training modules
- [API Reference](api-reference.md) -- Complete module signatures
- [Python API](python-api.md) -- Compiled training API
- [Training Basics](../guides/training/training-basics.md) -- Training loop walkthrough
- [ONNX Support](../models/onnx.md) -- Quantized inference
- [Index](../index.md) -- Full documentation index
