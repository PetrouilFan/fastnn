# ONNX Model Support in FastNN

FastNN loads and executes ONNX models through the v2.0 AOT compiler pipeline:

## Pipeline

```
ONNX (.onnx)
    [import_onnx]  — parse ONNX graph, extract weights (Python)
.fnn (JSON header + binary params)
    [OnnxConverter]  — convert to ComputeGraph IR (Rust)
ComputeGraph IR
    [compiler passes]  — shape inference → fusion → quantization → memory planning
ExecutablePlan
    [GraphExecutor::run]  — execute on CPU (or WGPU fallback)
Output tensors
```

## Stage 1: Import

```python
import fastnn as fnn

# Convert ONNX to .fnn format
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
```

### Supported Ops (90+ operator types)

The importer supports 90+ ONNX operator types, of which 60+ have native IR handlers:

- **NN layers**: Conv, ConvTranspose, Gemm/MatMul, BatchNormalization, LayerNormalization, RMSNormalization, MaxPool, AveragePool, GlobalAveragePool
- **Activations**: Relu, Gelu, Silu/Swish, Sigmoid, Tanh, LeakyRelu, Elu, Softplus, HardSwish, Softmax, LogSoftmax, Clip, Selu, HardSigmoid, Mish, PReLU
- **Arithmetic**: Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt, Neg, Abs, Sign, Not
- **Shape ops**: Reshape, Flatten, Transpose, Concat, Slice, Pad, Squeeze, Unsqueeze, Gather, ScatterND
- **Reductions**: ReduceMean, ReduceSum, ReduceMax, ArgMax
- **Data**: Where, CumSum, Erf, Embedding, Constant, Identity, Tile, Expand, Cast, Shape
- **Quantized ops** (decomposed to f32): QuantizeLinear, DequantizeLinear, QLinearMatMul, QLinearConv
- **Advanced**: NonMaxSuppression, NonZero, Unique

## Stage 2: Build & Compile

```python
# Auto-detect provenance, compile through AOT pipeline
model = fnn.build_model_from_fnn("model.fnn")
# Returns Rust AotExecutor with compiled plan

# Or compile explicitly with quantization:
executor = fnn.AotExecutor(
    nodes=model_nodes,
    params=model_params,
    input_names=["input"],
    output_names=["output"],
    quantize=4,  # 4-bit or 8-bit weight quantization
)
```

The v2.0 AOT compiler runs four passes:
1. **Shape inference** — resolves symbolic dimensions
2. **Operator fusion** — merges MatMul+Add+ReLU, Conv2d+Add+ReLU
3. **Weight quantization** (optional) — replaces f32 weights with packed U4/U8
4. **Memory planning** — allocates arena slots with live-range analysis

## Stage 3: Execute

```python
# Input as dict of {name: tensor}
outputs = executor.forward({"input": input_tensor})

# Or for single-input models:
output = executor(input_tensor)
```

## Input/Output format

The executor accepts a dict of `{name: tensor}` and returns a dict of `{name: tensor}`.

## Python API for ONNX Models

```python
from fastnn.io import build_dag_model
from fastnn import AotExecutor

# Simple interface
model = build_dag_model("model.fnn", quantize=4)
result = model.forward({"input": x})

# Direct AotExecutor for full control
executor = AotExecutor(
    nodes=nodes,
    params=params,
    input_names=["input"],
    output_names=["output"],
    input_shapes={"input": [-1, 3, 224, 224]},  # dynamic batch
    quantize=8,
)
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

# Full YOLO output decoding (YOLOv5/v8/v10/v11)
detections = yolo_decode(model_output, conf_threshold=0.25)
```

## Architecture

```
fastnn/
io/
    onnx.py              # ONNX importer (90+ ops)
    graph_builder.py     # Model building (auto-detect)
    dag_model.py         # DAGModel → AotExecutor bridge
    calibrate.py         # Calibration infrastructure
    act_calibrate.py     # Activation calibration
    profiler.py          # Precision profiling
    validate.py          # Model validation
models/
    yolo.py              # YOLO model wrapper
src/
    ir/                  # ComputeGraph IR, GraphBuilder
      node.rs            # Opcode (72 variants), IrDType, DimExpr, TensorType
      builder.rs         # GraphBuilder — fluent IR construction API
    compiler/passes/     # AOT compiler passes
      shape_inference.rs # Symbolic shape resolution
      operator_fusion.rs # MatMul+Add+ReLU, Conv2d+Add+ReLU
      quantization.rs    # U4/U8 weight quantization
      memory_planning.rs # Arena-based memory planning
    backend/
      cpu/mod.rs         # CpuBackend — dispatch all ops including quantized
    onnx/
      converter.rs       # OnnxConverter — ONNX nodes → ComputeGraph
```

## Current Limitations

- **Conv2d**: CPU-only for quantized; WGPU f32 conv supported
- **Control flow**: Loop, If ops recorded but not supported
- **GPU quantization**: U4/U8 quantized kernels are CPU-only (WGPU fallback); WGSL shaders planned for v2.2
- **Training**: ONNX models are inference-only; IR training pipeline is experimental (v2.1)
