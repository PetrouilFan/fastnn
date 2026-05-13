# FastNN ONNX Pipeline Examples

This directory contains examples demonstrating FastNN's ONNX model support.

## Prerequisites

Install FastNN and ONNX support:

```bash
pip install onnx pillow  # ONNX parser + image processing
```

## Examples

### 1. YOLO Object Detection (`yolo_inference.py`)

Load a YOLO ONNX model and run object detection on images.

```bash
# Download a YOLO model (example with YOLOv8n)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx

# Run detection
python yolo_inference.py --model yolov8n.onnx --image image.jpg

# With custom thresholds
python yolo_inference.py --model yolov8n.onnx --image image.jpg --conf 0.5 --iou 0.5

# Filter specific classes (e.g., person=0, car=2)
python yolo_inference.py --model yolov8n.onnx --image image.jpg --classes 0 2

# Draw boxes on output image
python yolo_inference.py --model yolov8n.onnx --image image.jpg --output result.jpg

# Benchmark mode
python yolo_inference.py --model yolov8n.onnx --benchmark
```

### 2. ONNX Pipeline Low-Level API (`onnx_pipeline.py`)

Demonstrates the building blocks of the ONNX pipeline without the high-level YOLO wrapper.

```bash
# Basic import + inference
python onnx_pipeline.py --model yolov8n.onnx

# Inspect model structure
python onnx_pipeline.py --model yolov8n.onnx --inspect

# Apply graph optimizations
python onnx_pipeline.py --model yolov8n.onnx --inspect --optimize
```

### 3. Using the Python API Directly

```python
import fastnn as fnn

# High-level YOLO wrapper
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")

# Low-level pipeline
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
model = fnn.build_model_from_fnn("model.fnn")

import numpy as np
x = fnn.tensor(np.random.randn(1, 3, 640, 640).astype(np.float32), [1, 3, 640, 640])
if hasattr(model, "forward"):
    outputs = model.forward({"images": x})
else:
    outputs = model(x)

# NMS utilities
from fastnn import nms, yolo_decode
boxes = np.array([[10, 10, 100, 100], [20, 20, 120, 120]])
scores = np.array([0.9, 0.8])
keep = nms(boxes, scores, iou_threshold=0.5)

# Shape inference
from fastnn.io.shape_inference import infer_shape
out_shapes = infer_shape("Conv", [[1, 3, 224, 224]], {"kernel_shape": [3, 3], "stride": 2, "padding": 1})
print(out_shapes)  # [[1, 3, 112, 112]]

# Graph optimization
from fastnn.io.graph_optimizer import optimize_graph
optimized_header = optimize_graph(header)
```

## Supported Model Formats

- **YOLOv5** — Standard output: `[batch, num_dets, 4 + num_classes]`
- **YOLOv8/v10/v11** — DFL output: `[batch, num_dets, 4*reg_max + num_classes]`
- **Other ONNX models** — Models with standard ONNX ops (Conv, Relu, Gemm, etc.)

## Performance Notes

- All ONNX models run through the v2.0 AOT compiler pipeline (`AotExecutor`) which compiles the graph through shape inference, operator fusion, optional quantization, and memory planning before execution
- Conv2d currently runs on CPU; GPU support requires WGPU conv kernels (planned)
