# Neural Network Modules

FastNN provides a variety of neural network layers (modules) for building neural networks.

## Linear Layer

Fully connected layer: `y = xW + b`

```python
import fastnn as fnn

linear = fnn.Linear(in_features=128, out_features=64, bias=True)
output = linear(input_tensor)
print(output.shape)  # (batch_size, 64)
```

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension
- `bias`: Whether to include bias (default: True)

## Convolutional Layers

### Conv2d

```python
conv = fnn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3x3 kernel
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True
)

input_tensor = fnn.randn([8, 3, 32, 32])  # (batch, channels, height, width)
output = conv(input_tensor)
print(output.shape)  # (8, 64, 32, 32)
```

### Conv1d

```python
conv1d = fnn.Conv1d(
    in_channels=128,
    out_channels=256,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    bias=True
)

# Input: (batch, channels, length)
input_tensor = fnn.randn([8, 128, 64])
output = conv1d(input_tensor)
print(output.shape)  # (8, 256, 64)
```

### Conv3d

```python
conv3d = fnn.Conv3d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    bias=True
)

# Input: (batch, channels, depth, height, width)
input_tensor = fnn.randn([2, 16, 8, 32, 32])
output = conv3d(input_tensor)
print(output.shape)  # (2, 32, 8, 32, 32)
```

### ConvTranspose2d

Transposed convolution (upsampling convolution):

```python
conv_t = fnn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=True
)

input_tensor = fnn.randn([4, 64, 8, 8])
output = conv_t(input_tensor)
print(output.shape)  # (4, 32, 16, 16)
```

## Normalization Layers

### BatchNorm1d

```python
bn = fnn.BatchNorm1d(num_features=64, momentum=0.1, eps=1e-5)

# Training mode (tracks running statistics)
bn.train()
output = bn(input_tensor)

# Inference mode (uses running statistics)
bn.eval()
output = bn(input_tensor)
```

### BatchNorm2d

```python
bn2d = fnn.BatchNorm2d(num_features=64, momentum=0.1, eps=1e-5)

input_tensor = fnn.randn([8, 64, 32, 32])
output = bn2d(input_tensor)
```

### LayerNorm

```python
ln = fnn.LayerNorm(normalized_shape=64, eps=1e-5)

# Normalizes over last dimension
input_tensor = fnn.randn([8, 32, 64])
output = ln(input_tensor)  # Normalizes over dim 64
```

### RMSNorm

RMS normalization (used in Llama/Mistral):

```python
rms = fnn.RMSNorm(normalized_shape=64, eps=1e-5)

input_tensor = fnn.randn([8, 32, 64])
output = rms(input_tensor)
```

### GroupNorm

```python
gn = fnn.GroupNorm(num_groups=8, num_channels=64, eps=1e-5)

input_tensor = fnn.randn([8, 64, 32, 32])
output = gn(input_tensor)
```

## Dropout

### Dropout

```python
dropout = fnn.Dropout(p=0.5)

# Training mode: applies dropout
dropout.train()
output = dropout(input_tensor)

# Inference mode: no dropout
dropout.eval()
output = dropout(input_tensor)
```

### Dropout2d

Channel-wise dropout for 2D inputs:

```python
dropout2d = fnn.Dropout2d(p=0.5)

# Input: (batch, channels, height, width)
input_tensor = fnn.randn([8, 64, 32, 32])
output = dropout2d(input_tensor)
```

## Upsample

```python
# Nearest neighbor upsampling
up = fnn.Upsample(scale_factor=2.0, mode='nearest')
input_tensor = fnn.randn([1, 2, 4, 4])
output = up(input_tensor)
print(output.shape)  # (1, 2, 8, 8)

# Bilinear upsampling
up2 = fnn.Upsample(scale_factor=2.0, mode='bilinear')
output2 = up2(input_tensor)
```

## Embedding

```python
embedding = fnn.Embedding(
    num_embeddings=10000,
    embedding_dim=256
)

token_ids = fnn.randint(low=0, high=10000, shape=[2, 3])
embeddings = embedding(token_ids)
print(embeddings.shape)  # (2, 3, 256)
```

## Activation Functions

### As Layers

```python
relu = fnn.ReLU()
gelu = fnn.GELU()
sigmoid = fnn.Sigmoid()
tanh = fnn.Tanh()
silu = fnn.SiLU()
leaky_relu = fnn.LeakyReLU(negative_slope=0.01)
softplus = fnn.Softplus(beta=1.0, threshold=20.0)
hardswish = fnn.Hardswish()
elu = fnn.Elu(alpha=1.0)
mish = fnn.Mish()
prelu = fnn.PReLU()

output = relu(input_tensor)
```

### As Tensor Methods

```python
output = input_tensor.relu()
output = input_tensor.gelu()
output = input_tensor.sigmoid()
output = input_tensor.tanh()
output = input_tensor.silu()
output = input_tensor.leaky_relu(0.01)
output = input_tensor.softplus(beta, threshold)
output = input_tensor.hardswish()
output = input_tensor.softmax(dim=-1)
output = input_tensor.log_softmax(dim=-1)
```

## Pooling

### MaxPool2d

```python
pool = fnn.MaxPool2d(kernel_size=2, stride=2)
input_tensor = fnn.randn([8, 64, 32, 32])
output = pool(input_tensor)
print(output.shape)  # (8, 64, 16, 16)
```

### MaxPool1d

```python
pool = fnn.MaxPool1d(kernel_size=2, stride=2)
input_tensor = fnn.randn([8, 64, 32])
output = pool(input_tensor)
```

### AvgPool2d

```python
pool = fnn.AvgPool2d(kernel_size=2, stride=2)
input_tensor = fnn.randn([8, 64, 32, 32])
output = pool(input_tensor)
print(output.shape)  # (8, 64, 16, 16)
```

### AvgPool1d

```python
pool = fnn.AvgPool1d(kernel_size=2, stride=2)
input_tensor = fnn.randn([8, 64, 32])
output = pool(input_tensor)
```

### AdaptiveAvgPool2d

```python
pool = fnn.AdaptiveAvgPool2d(output_h=1, output_w=1)
input_tensor = fnn.randn([8, 64, 32, 32])
output = pool(input_tensor)
print(output.shape)  # (8, 64, 1, 1)
```

## Flatten

Flattens the input tensor starting from a given dimension.

```python
flatten = fnn.Flatten(start_dim=1, end_dim=-1)

input_tensor = fnn.randn([32, 3, 32, 32])
output = flatten(input_tensor)  # shape [32, 3072]
```

Parameters:
- `start_dim`: First dimension to flatten (default: 1)
- `end_dim`: Last dimension to flatten (default: -1)

## ResidualBlock

Skip connection block for ResNet-style architectures:

```python
block = fnn.ResidualBlock(
    conv1_in=64, conv1_out=64, conv1_kernel=3, conv1_stride=1, conv1_padding=1,
    bn1_features=64,
    conv2_in=64, conv2_out=64, conv2_kernel=3, conv2_stride=1, conv2_padding=1,
    bn2_features=64,
    downsample=None  # Or (ds_in, ds_out, ds_k, ds_s, ds_p, ds_bn) for stride > 1
)

input_tensor = fnn.randn([4, 64, 32, 32])
output = block(input_tensor)
print(output.shape)  # (4, 64, 32, 32)
```

## Fused Conv + BN Layers

Fused inference-only layers that combine Conv2d and BatchNorm2d into a single pass:

```python
conv = fnn.Conv2d(3, 64, 3, padding=1)
bn = fnn.BatchNorm2d(64)

# FusedConvBn — Conv+BN fused
fused = fnn.FusedConvBn(conv, bn)

# FusedConvBnRelu — Conv+BN+ReLU fused
fused_relu = fnn.FusedConvBnRelu(conv, bn)

# FusedConvBnGelu — Conv+BN+GELU fused  
fused_gelu = fnn.FusedConvBnGelu(conv, bn)
```

## Packed Precision Layers

Quantized layers with packed weight storage (U4, U8, F16, F32 precisions):

```python
# Packed linear layers
fnn.Linear4(128, 64)    # 4-bit packed linear
fnn.Linear8(128, 64)    # 8-bit packed linear
fnn.Linear16(128, 64)   # 16-bit packed linear
fnn.Linear32(128, 64)   # 32-bit (f32) packed linear

# Packed fused conv + ReLU
fnn.PackedConvRelu4(3, 64, 3, padding=1)
fnn.PackedConvRelu8(3, 64, 3, padding=1)
fnn.PackedConvRelu16(3, 64, 3, padding=1)
fnn.PackedConvRelu32(3, 64, 3, padding=1)

# Packed fused linear + GELU
fnn.PackedLinearGelu4(128, 64)
fnn.PackedLinearGelu8(128, 64)
fnn.PackedLinearGelu16(128, 64)
fnn.PackedLinearGelu32(128, 64)

# Packed convolution layers
fnn.PackedConv2d4(3, 64, 3, padding=1)
fnn.PackedConv2d8(3, 64, 3, padding=1)
fnn.PackedConv2d16(3, 64, 3, padding=1)
fnn.PackedConv2d32(3, 64, 3, padding=1)
```

## Multi-Head Attention

```python
# Standard attention
mha = fnn.MultiHeadAttention(d_model=512, n_heads=8)
output = mha(query, key, value)

# Packed quantized attention (U4/U8)
from fastnn import PackedMultiHeadAttention4, PackedMultiHeadAttention8
mha4 = PackedMultiHeadAttention4(d_model=512, n_heads=8)
mha8 = PackedMultiHeadAttention8(d_model=512, n_heads=8)
```

## Transformer

```python
# Standard transformer block
block = fnn.TransformerBlock(d_model=512, n_heads=8, ff_dim=2048, dropout=0.1)

# Full transformer encoder
encoder = fnn.TransformerEncoder(
    vocab_size=10000, max_seq_len=512,
    d_model=512, n_heads=8, n_layers=6,
    ff_dim=2048, n_classes=10, dropout=0.1
)

# Packed quantized transformer encoder (U4/U8)
from fastnn import PackedTransformerEncoder4, PackedTransformerEncoder8
enc4 = PackedTransformerEncoder4(10000, 512, 512, 8, 6, 2048, 10)
enc8 = PackedTransformerEncoder8(10000, 512, 512, 8, 6, 2048, 10)
```

## Sequential

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

## ModuleList

```python
layers = fnn.ModuleList([
    fnn.Linear(10, 10),
    fnn.Linear(10, 10),
    fnn.Linear(10, 1),
])

for layer in layers:
    input_tensor = layer(input_tensor)
```

## Accessing Parameters

```python
params = model.parameters()
print(f"Number of parameter tensors: {len(params)}")

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## Training/Evaluation Mode

```python
model = fnn.Sequential([
    fnn.Linear(10, 10),
    fnn.BatchNorm1d(10),
    fnn.Dropout(0.5),
])

model.train()  # Training mode
model.eval()   # Inference mode
```
