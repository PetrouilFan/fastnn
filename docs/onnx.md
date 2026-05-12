# ONNX Model Support in FastNN

FastNN can load and execute ONNX models through a multi-stage pipeline:

## Pipeline

```
ONNX (.onnx)
    [import_onnx]  — parse ONNX graph, extract weights
.fnn (JSON header + binary params, v2 or v3)
    [build_model_from_fnn]  — auto-detect provenance
DAGExecutor (Rust) or Sequential (Python)
    [forward]  — execute graph topologically
Output tensors
```

## Stage 1: Import

```python
import fastnn as fnn

# Convert ONNX to .fnn format
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
# info contains: layers, parameters, input_shape, output_shape, graph
```

### Supported Ops (50 operator types)

The importer supports 50 ONNX operator types:

- **NN layers**: Conv, Gemm/Linear, BatchNormalization, MaxPool, AveragePool, GlobalAveragePool
- **Activations**: Relu, Sigmoid, Tanh, SiLU, LeakyRelu, Elu, Softmax, Clip, etc.
- **Arithmetic**: Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt, Neg, MatMul
- **Shape ops**: Reshape, Flatten, Transpose, Concat, Split, Slice, Pad (v2), Tile, Squeeze, Unsqueeze
- **Reductions**: ReduceMean, ReduceSum
- **Data**: Gather, Where, TopK, NonMaxSuppression, Constant, Identity, Resize
- **Quantized ops**: QuantizeLinear, DequantizeLinear, QLinearConv, MatMulInteger
- **Advanced**: NonZero, Unique, Tril, Triu

### Quantized ONNX Import (v1.3)

The ONNX importer detects `QuantizeLinear → DequantizeLinear` patterns around `Conv`/`MatMul` nodes and **folds them**, converting weights to packed storage during import for zero-cost quantized inference.

```python
from fastnn.precision import PrecisionConfig

# Import with automatic Q/DQ folding
info = fnn.convert_from_onnx("model.onnx", "model.fnn")

# Or specify precision config for custom quantization
config = PrecisionConfig.uniform("u4")
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
```

Key quantized ops supported:

| Op | Description |
|----|-------------|
| `QuantizeLinear` | Scale + round + clamp quantization (q = round(v/scale) + zp) |
| `DequantizeLinear` | Scale-based dequantization (v = (q − zp) × scale) |
| `QLinearConv` | Dequantizes quantized input + weight, runs f32 conv, requantizes output |
| `MatMulInteger` | Dequantizes both quantized inputs, runs f32 matmul |

## Stage 2: Build

```python
# Auto-detects whether the model came from ONNX or PyTorch
model = fnn.build_model_from_fnn("model.fnn")
# Returns Rust DAGExecutor (ONNX) or Python Sequential (PyTorch)
```

## Stage 3: Execute

### Rust DAGExecutor (production)

The `DAGExecutor` is a native Rust implementation that:
- Executes nodes in topological order using a HashMap buffer
- Dispatches to optimized CPU kernels via the dispatcher system
- Supports 30+ operation types natively
- Passes through unknown ops (returns first input)
- Includes fused QuantizeLinear / DequantizeLinear support in graph execution

```python
# The DAGExecutor takes input names matching the model's input
x = fnn.tensor(numpy_array, list(numpy_array.shape))
outputs = executor.forward({"images": x})
# Returns dict of output name -> tensor
```

### Input/Output format

The executor accepts a dict of `{name: tensor}` and returns a dict of `{name: tensor}`.
For single-input/single-output models, `Module::forward()` is also implemented:

```python
result = executor(input_tensor)  # Module trait forward
```

## Graph Optimization

```python
from fastnn.io.graph_optimizer import optimize_graph

with open("model.fnn", "rb") as f:
    from fastnn.io import read_fnn_header
    _, _, header, _ = read_fnn_header(f)

optimized = optimize_graph(header)
# Runs: 1. Dead node elimination, 2. Conv+BN fusion, 3. Constant folding (framework)
```

## Shape Inference

```python
from fastnn.io.shape_inference import infer_shape

# Get output shape for any ONNX op
out_shapes = infer_shape("Conv", [[1, 3, 224, 224]], {
    "kernel_shape": [3, 3], "stride": 2, "padding": 1
})  # [[1, 3, 112, 112]]
```

## YOLO Object Detection

```python
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")
# Returns list of [N, 6] arrays: [x1, y1, x2, y2, confidence, class_id]
```

See `examples/yolo_inference.py` for a complete example.

## Post-Processing

```python
from fastnn import nms, xywh2xyxy, scale_boxes, yolo_decode

# NMS on raw boxes
keep = nms(boxes, scores, iou_threshold=0.5)

# Convert YOLO format
boxes_xyxy = xywh2xyxy(boxes_xywh)

# Scale from model input to original image
boxes_scaled = scale_boxes((640, 640), boxes_xyxy, (1080, 1920))

# Full YOLO output decoding (YOLOv5/v8/v10/v11)
detections = yolo_decode(model_output, conf_threshold=0.25)
detections = yolo_dfl_decode(model_output, conf_threshold=0.25)  # YOLOv8+ DFL
```

## Architecture

```
fastnn/
io/
    onnx.py              # ONNX importer (50 ops)
    graph_builder.py     # Model building (auto-detect)
    dag_model.py         # Python DAG prototype
    shape_inference.py   # Shape inference (50 ops)
    graph_optimizer.py   # Graph optimization (3 passes)
    calibrate.py         # Calibration infrastructure
    act_calibrate.py     # Activation calibration (KL-divergence)
    profiler.py          # Precision profiling & sensitivity analysis
    validate.py          # Model validation
models/
    yolo.py              # YOLO model wrapper
utils/
    nms.py               # NMS post-processing
src/
nn/
    dag.rs               # Rust DAGExecutor (~35 ops)
kernels/
    cpu/
        pooling.rs       # CPU kernels (max_pool2d, avg_pool2d)
backends/
    wgpu/
        shaders/
            conv_packed.wgsl  # WGPU packed convolution shader
```

## Current Limitations

- **Conv2d**: CPU-only for general conv; WGPU packed conv shader exists (`conv_packed.wgsl`) but is experimental
- **Control flow**: Loop, If ops recorded but not executed
- **Training**: ONNX models are inference-only; autograd preserved but not optimized for ONNX graphs
- **GPU**: Elementwise ops use WGPU dispatcher; Conv/pooling ops use CPU fallback
