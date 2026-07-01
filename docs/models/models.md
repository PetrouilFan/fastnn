# Models

FastNN provides pre-built model architectures for common deep learning tasks, along with tools to load, run, and customize models. All built-in models inherit from `BaseModel` and support serialization to the `.fnn` binary format.

## Supported Architectures

### MLP (Multi-Layer Perceptron)

A fully-connected neural network configurable with activation, dropout, and batch normalization.

```python
import fastnn as fnn

model = fnn.models.MLP(
    input_dim=784,           # Input features
    hidden_dims=[256, 128],  # Hidden layer sizes
    output_dim=10,           # Output features
    activation="relu",       # "relu", "gelu", or "silu"
    dropout=0.0,             # Dropout probability
    batch_norm=False         # Use batch normalization
)

output = model(input_tensor)
```

Use `model.train()` to enable dropout and batch norm training mode; `model.eval()` to disable them.

**Builder helper:**

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

### Transformer

A full transformer encoder for sequence classification.

```python
model = fnn.models.Transformer(
    vocab_size=10000,      # Vocabulary size
    max_seq_len=512,       # Maximum sequence length
    d_model=512,           # Model/embedding dimension
    num_heads=8,           # Number of attention heads
    num_layers=6,          # Number of encoder layers
    ff_dim=2048,           # Feed-forward hidden dimension
    num_classes=10,        # Output classes
    dropout_p=0.1          # Dropout probability
)

# Input: token IDs [batch, seq_len]
logits = model(token_ids)
```

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

### YOLO Object Detection

FastNN can load YOLO models (v5/v8/v10/v11) exported to ONNX and run object detection.

```python
model = fnn.YOLO("yolov8n.onnx")

# Run inference on image
detections = model("image.jpg")
# Returns list of [N, 6] arrays: [x1, y1, x2, y2, confidence, class_id]

# With custom thresholds
model = fnn.YOLO("yolov8n.onnx", conf_threshold=0.5, iou_threshold=0.5)
detections = model("image.jpg", conf_threshold=0.3)
```

**NMS and decoding utilities:**

```python
from fastnn import nms, yolo_decode, yolo_dfl_decode, xywh2xyxy, scale_boxes

keep = nms(boxes, scores, iou_threshold=0.5)
boxes_xyxy = xywh2xyxy(boxes_xywh)
boxes_scaled = scale_boxes((640, 640), boxes_xyxy, (1080, 1920))
detections = yolo_decode(model_output, conf_threshold=0.25)
detections = yolo_dfl_decode(model_output, conf_threshold=0.25)  # v8/v10/v11
```

### ResNet-style Models

Use `ResidualBlock` for skip connections:

```python
block = fnn.ResidualBlock(
    conv1_in=64, conv1_out=64, conv1_kernel=3, conv1_stride=1, conv1_padding=1,
    bn1_features=64,
    conv2_in=64, conv2_out=64, conv2_kernel=3, conv2_stride=1, conv2_padding=1,
    bn2_features=64,
    downsample=None  # Or (ds_in, ds_out, ds_k, ds_s, ds_p, ds_bn) for stride > 1
)
```

## Loading and Running Models

### From `.fnn` format

Models saved in fastnn's custom binary format load quickly with full reconstruction:

```python
# Save (BaseModel subclass)
model.save("model.fnn")

# Load with automatic reconstruction from metadata
loaded = fnn.models.MLP.load("model.fnn")
```

### From ONNX via the AOT Pipeline

The v2.0 AOT compiler converts ONNX models through a four-stage pipeline and produces an optimized executable plan.

```python
# Stage 1: Import ONNX to .fnn format
info = fnn.convert_from_onnx("model.onnx", "model.fnn")

# Stage 2: Build and compile (auto-detect provenance)
model = fnn.build_model_from_fnn("model.fnn")

# Stage 3: Execute
outputs = model.forward({"input": input_tensor})
```

For full control over compilation options such as quantization:

```python
from fastnn import AotExecutor
from fastnn.io import build_dag_model

# Simple interface with quantization
model = build_dag_model("model.fnn", quantize=4)
result = model.forward({"input": x})

# Direct AotExecutor
executor = AotExecutor(
    nodes=nodes,
    params=params,
    input_names=["input"],
    output_names=["output"],
    input_shapes={"input": [-1, 3, 224, 224]},  # dynamic batch
    quantize=8,
)
outputs = executor.forward({"input": input_tensor})
```

The AOT compiler runs four passes:
1. **Shape inference** -- resolves symbolic dimensions
2. **Operator fusion** -- merges MatMul+Add+ReLU, Conv2d+Add+ReLU
3. **Weight quantization** (optional) -- replaces f32 weights with packed U4/U8
4. **Memory planning** -- allocates arena slots with live-range analysis

See [onnx.md](onnx.md) for the full ONNX pipeline reference and [io.md](../reference/io.md) for serialization details.

### From PyTorch

```python
fnn.io.convert_from_pytorch(torch_model, "model.fnn")
```

See [io.md](../reference/io.md) for the complete serialization API.

## Custom Model Guide

### Using `fnn.Sequential`

Build custom architectures directly with `fnn.Sequential`:

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

### Using `BasicBlock` (PySequential)

A simpler sequential wrapper for Python-side layer composition:

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

### Serialization for Custom Models

Custom models built with `fnn.Sequential` or `BasicBlock` can use the generic save/load API:

```python
fnn.io.save(model, "custom_model.fnn")
loaded = fnn.io.load("custom_model.fnn")
```

For full-featured save/load with automatic class reconstruction, subclass `BaseModel`. See `reference/` for the `BaseModel` API reference.

## Model Summary

| Model | Constructor | Description |
|-------|-------------|-------------|
| `MLP` | `fnn.models.MLP(...)` | Multi-layer perceptron |
| `Transformer` | `fnn.models.Transformer(...)` | Transformer encoder for classification |
| `YOLO` | `fnn.YOLO("model.onnx")` | YOLO object detection (v5/v8/v10/v11) |
| `create_mlp` | `fnn.models.create_mlp(...)` | Builder for sequential MLP |

## Training Example

See [Training Basics](../guides/training/training-basics.md) for a complete walkthrough of the compiled training pipeline, including data loading, optimizer setup, and evaluation.

## See also

- [onnx.md](onnx.md) -- ONNX model import and execution
- [onnx-training-export.md](onnx-training-export.md) -- ONNX training export contract
- [io.md](../reference/io.md) -- serialization and format conversion
- [Training Basics](../guides/training/training-basics.md) -- training API and compiled pipeline
- [Getting Started](../guides/getting-started.md) -- quickstart guide
- [Guides](../guides/getting-started.md) -- detailed usage guides
- [Reference Docs](../reference/tensors.md) -- API reference
