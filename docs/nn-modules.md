# Neural Network Modules

FastNN provides a variety of neural network layers (modules) for building neural networks.

## Linear Layer

Fully connected layer: `y = xW^T + b`

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

## Convolutional Layer

```python
# Conv2d: 2D convolution
conv = fnn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3x3 kernel
    stride=1,          # Convolution stride
    padding=1,         # Zero padding
    bias=True
)

# Input: (batch, channels, height, width)
input_tensor = fnn.randn(8, 3, 32, 32)
output = conv(input_tensor)
print(output.shape)  # (8, 64, 32, 32)
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

### LayerNorm

```python
ln = fnn.LayerNorm(normalized_shape=64, eps=1e-5)

# Normalizes over last dimension
input_tensor = fnn.randn(8, 32, 64)
output = ln(input_tensor)  # Normalizes over dim 64
```

## Dropout

```python
dropout = fnn.Dropout(p=0.5)  # p = probability of zeroing

# Training mode: applies dropout
dropout.train()
output = dropout(input_tensor)  # ~50% values zeroed

# Inference mode: no dropout
dropout.eval()
output = dropout(input_tensor)  # Identity operation
```

## Embedding

```python
# Word embeddings
embedding = fnn.Embedding(
    num_embeddings=10000,  # Vocabulary size
    embedding_dim=256      # Embedding dimension
)

# Input: batch of token indices
token_ids = fnn.tensor([[1, 2, 3], [4, 5, 6]])  # (batch, seq_len)
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

output = relu(input_tensor)
```

### As Tensor Methods

```python
output = input_tensor.relu()
output = input_tensor.gelu()
output = input_tensor.sigmoid()
output = input_tensor.tanh()
output = input_tensor.silu()
```

## Sequential

Combine layers in sequence:

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

Store list of modules:

```python
layers = fnn.ModuleList([
    fnn.Linear(10, 10),
    fnn.Linear(10, 10),
    fnn.Linear(10, 1),
])

# Access by index
output = layers[0](input_tensor)

# Iterate
for layer in layers:
    input_tensor = layer(input_tensor)
```

## Accessing Parameters

All modules with learnable parameters implement `parameters()`:

```python
model = fnn.Sequential([
    fnn.Linear(784, 256),
    fnn.ReLU(),
    fnn.Linear(256, 10),
])

# Get all parameters
params = model.parameters()
print(f"Number of parameter tensors: {len(params)}")

# Get named parameters (if supported)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## Training/Evaluation Mode

Some layers behave differently in training vs. eval mode:

```python
model = fnn.Sequential([
    fnn.Linear(10, 10),
    fnn.BatchNorm1d(10),
    fnn.Dropout(0.5),
])

# Training mode
model.train()

# Inference mode
model.eval()
```
