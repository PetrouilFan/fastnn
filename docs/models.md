# Models

FastNN provides pre-built model architectures.

## MLP (Multi-Layer Perceptron)

A fully-connected neural network.

```python
import fastnn as fnn

model = fnn.models.MLP(
    input_dim=784,           # Input features
    hidden_dims=[256, 128],  # Hidden layer sizes
    output_dim=10,           # Output features
    activation="relu",       # Activation: "relu", "gelu", or "silu"
    dropout=0.0,             # Dropout probability (0 = no dropout)
    batch_norm=False         # Whether to use batch normalization
)

# Forward pass
output = model(input_tensor)
print(output.shape)  # [batch_size, 10]
```

### MLP with BatchNorm and Dropout

```python
model = fnn.models.MLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    activation="gelu",
    dropout=0.2,
    batch_norm=True
)

# Training vs. Inference
model.train()  # Enables dropout and batch norm training mode
model.eval()   # Disables dropout, uses batch norm running stats
```

### Using `create_mlp` Builder

```python
from fastnn.models.builder import create_mlp

model = create_mlp(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    activation="relu",
    dropout=0.2,
    batch_norm=True
)
# Returns a PySequential container
```

## Transformer

A full transformer encoder for sequence classification.

```python
model = fnn.models.Transformer(
    vocab_size=10000,      # Vocabulary size
    max_seq_len=512,       # Maximum sequence length
    d_model=512,           # Model dimension
    num_heads=8,           # Number of attention heads
    num_layers=6,          # Number of transformer layers
    ff_dim=2048,           # Feed-forward hidden dimension
    num_classes=10,        # Number of output classes
    dropout_p=0.1          # Dropout probability
)

# Forward pass
# Input: token IDs [batch, seq_len]
token_ids = fnn.randint(low=0, high=10000, shape=[32, 128])
logits = model(token_ids)
print(logits.shape)  # [32, 10]
```

### Transformer Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Size of the token vocabulary | Required |
| `max_seq_len` | Maximum sequence length | Required |
| `d_model` | Model/embedding dimension | Required |
| `num_heads` | Number of attention heads | Required |
| `num_layers` | Number of transformer encoder layers | Required |
| `ff_dim` | Feed-forward hidden dimension | Required |
| `num_classes` | Number of output classes | Required |
| `dropout_p` | Dropout probability | 0.1 |

## YOLO Object Detection

FastNN can load YOLO models (v5/v8/v10/v11) exported to ONNX format and run object detection:

```python
import fastnn as fnn

# Load YOLO model from ONNX
model = fnn.YOLO("yolov8n.onnx")

# Run inference on image
detections = model("image.jpg")
# Returns list of [N, 6] arrays: [x1, y1, x2, y2, confidence, class_id]

# With custom thresholds
model = fnn.YOLO("yolov8n.onnx", conf_threshold=0.5, iou_threshold=0.5)

# Per-inference overrides
detections = model("image.jpg", conf_threshold=0.3)
```

### NMS Utilities

```python
from fastnn import nms, yolo_decode, yolo_dfl_decode, xywh2xyxy, scale_boxes

# Standard NMS
keep = nms(boxes, scores, iou_threshold=0.5)

# YOLO format conversion
boxes_xyxy = xywh2xyxy(boxes_xywh)

# Scale boxes back to original image
boxes_scaled = scale_boxes((640, 640), boxes_xyxy, (1080, 1920))

# Full YOLO output decoding
detections = yolo_decode(model_output, conf_threshold=0.25)
detections = yolo_dfl_decode(model_output, conf_threshold=0.25)  # YOLOv8/v10/v11
```

## BaseModel

All built-in models inherit from `BaseModel`, which provides `save()` and `load()` methods with automatic metadata serialization:

```python
# Save model (includes class name and config metadata)
model.save("model.fnn")

# Load model (auto-reconstructs from metadata)
loaded = fnn.models.MLP.load("model.fnn")
```

## BasicBlock (PySequential)

A simple sequential wrapper for Python-side layer composition:

```python
from fastnn.layers import BasicBlock

model = BasicBlock([
    fnn.Linear(784, 256),
    fnn.ReLU(),
    fnn.Linear(256, 128),
    fnn.ReLU(),
    fnn.Linear(128, 10),
])
```

## Creating Custom Models

Use `fnn.Sequential` directly for custom architectures:

```python
model = fnn.Sequential([
    fnn.Linear(784, 256),
    fnn.BatchNorm1d(256),
    fnn.GELU(),
    fnn.Dropout(0.3),
    fnn.Linear(256, 128),
    fnn.BatchNorm1d(128),
    fnn.GELU(),
    fnn.Dropout(0.3),
    fnn.Linear(128, 10),
])
```

## ResNet-style Models

Use `ResidualBlock` for skip connections:

```python
# Building blocks for ResNet
block = fnn.ResidualBlock(
    conv1_in=64, conv1_out=64, conv1_kernel=3, conv1_stride=1, conv1_padding=1,
    bn1_features=64,
    conv2_in=64, conv2_out=64, conv2_kernel=3, conv2_stride=1, conv2_padding=1,
    bn2_features=64,
    downsample=None  # Or (ds_in, ds_out, ds_k, ds_s, ds_p, ds_bn) for stride > 1
)
```

## Pre-Built Model Summary

| Model | Constructor | Description |
|-------|-------------|-------------|
| `MLP` | `fnn.models.MLP(...)` | Multi-layer perceptron |
| `Transformer` | `fnn.models.Transformer(...)` | Transformer encoder for classification |
| `YOLO` | `fnn.YOLO("model.onnx")` | YOLO object detection (v5/v8/v10/v11) |
| `create_mlp` | `fnn.models.create_mlp(...)` | Builder for sequential MLP |

## Complete Training Example

```python
import fastnn as fnn

# Create data
X_train = fnn.randn([1000, 784])
y_train = fnn.randint(low=0, high=10, shape=[1000])

X_test = fnn.randn([200, 784])
y_test = fnn.randint(low=0, high=10, shape=[200])

# Build model
model = fnn.models.MLP(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    activation="relu",
    dropout=0.2,
    batch_norm=True
)

# Optimizer
optimizer = fnn.Adam(model.parameters(), lr=1e-3)

# LR Scheduler
scheduler = fnn.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Data loaders
train_ds = fnn.TensorDataset(X_train, y_train)
test_ds = fnn.TensorDataset(X_test, y_test)

train_loader = fnn.DataLoader(train_ds, batch_size=32, shuffle=True, prefetch_size=2)
test_loader = fnn.DataLoader(test_ds, batch_size=32, shuffle=False)

# Training
model.train()
for epoch in range(50):
    train_loss = 0
    train_loader.reset_sampler()
    for x, y in train_loader:
        pred = model(x)
        loss = fnn.cross_entropy_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        fnn.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

    # Update LR
    scheduler.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with fnn.no_grad():
        for x, y in test_loader:
            pred = model(x)
            preds = pred.argmax(1, False)
            correct += (preds.numpy() == y.numpy()).sum()
            total += y.shape[0]

    accuracy = correct / total
    print(f"Epoch {epoch}: loss={train_loss/len(train_loader):.4}, acc={accuracy:.4}")

    model.train()

    # Save checkpoint
    if epoch % 10 == 0:
        fnn.io.save(model, f'checkpoint_epoch_{epoch}.fnn')
```

## Model I/O

```python
# Save model (custom binary format)
fnn.io.save(model, 'model.fnn')

# Load model
loaded_model = fnn.io.load('model.fnn')

# Save/Load with metadata (BaseModel subclasses)
model.save('model.fnn')
loaded = fnn.models.MLP.load('model.fnn')
```

## ONNX Import

Import models from PyTorch via ONNX:

```python
# Export from PyTorch
import torch
torch_model = torch.nn.Linear(784, 10)
torch.onnx.export(torch_model, torch.randn(1, 784), 'model.onnx')

# Import to fastnn
info = fnn.import_onnx('model.onnx', 'model.fnn')
print(f"Imported {info['parameters']} parameters")

# Or use the unified API
from fastnn.io import convert_from_onnx
info = convert_from_onnx('model.onnx', 'model.fnn')
```
