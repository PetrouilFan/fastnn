# ONNX Model Support in FastNN

FastNN loads and executes ONNX models through the v2.0 AOT compiler pipeline. The pipeline supports 90+ ONNX operator types (60+ with native IR handlers) and produces an optimized executable plan for CPU inference (with WGPU GPU fallback).

## Pipeline Overview

```
ONNX (.onnx)
    [import_onnx] -- parse ONNX graph, extract weights (Python)
.fnn (JSON header + binary params)
    [OnnxConverter] -- convert to ComputeGraph IR (Rust)
ComputeGraph IR
    [compiler passes] -- shape inference > fusion > quantization > memory planning
ExecutablePlan
    [GraphExecutor::run] -- execute on CPU (or WGPU fallback)
Output tensors
```

## Stage 1: Import

```python
import fastnn as fnn

# Convert ONNX to .fnn format
info = fnn.convert_from_onnx("model.onnx", "model.fnn")
print(f"Imported {info['parameters']} parameters")

# Or use the unified API
from fastnn.io import convert_from_onnx
info = convert_from_onnx("model.onnx", "model.fnn")
```

### Supported Ops (90+ operator types)

The importer supports 90+ ONNX operator types. Over 60 have native IR handlers; the remainder are decomposed into supported primitives.

- **NN layers**: Conv, ConvTranspose, Gemm/MatMul, BatchNormalization, LayerNormalization, RMSNormalization, MaxPool, AveragePool, GlobalAveragePool
- **Activations**: Relu, Gelu, Silu/Swish, Sigmoid, Tanh, LeakyRelu, Elu, Softplus, HardSwish, Softmax, LogSoftmax, Clip, Selu, HardSigmoid, Mish, PReLU
- **Arithmetic**: Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt, Neg, Abs, Sign, Not
- **Shape ops**: Reshape, Flatten, Transpose, Concat, Slice, Pad, Squeeze, Unsqueeze, Gather, ScatterND
- **Reductions**: ReduceMean, ReduceSum, ReduceMax, ArgMax
- **Data**: Where, CumSum, Erf, Embedding, Constant, Identity, Tile, Expand, Cast, Shape
- **Quantized ops** (decomposed to f32): QuantizeLinear, DequantizeLinear, QLinearMatMul, QLinearConv
- **Advanced**: NonMaxSuppression, NonZero, Unique

## Stage 2: Build and Compile

```python
# Auto-detect provenance, compile through AOT pipeline
model = fnn.build_model_from_fnn("model.fnn")
# Returns Rust AotExecutor with compiled plan
```

The v2.0 AOT compiler runs four passes:

1. **Shape inference** -- resolves symbolic dimensions in the compute graph
2. **Operator fusion** -- merges common patterns: MatMul+Add+ReLU, Conv2d+Add+ReLU, residual+add+norm
3. **Weight quantization** (optional) -- replaces f32 weights with packed U4/U8
4. **Memory planning** -- allocates arena slots with live-range analysis

### Python API for ONNX Models

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

## Stage 3: Execute

```python
# Input as dict of {name: tensor}
outputs = executor.forward({"input": input_tensor})

# Or for single-input models:
output = executor(input_tensor)
```

The executor accepts a dict of `{name: tensor}` and returns a dict of `{name: tensor}`.

## YOLO Object Detection

YOLO models (v5/v8/v10/v11) benefit from special integration:

```python
model = fnn.YOLO("yolov8n.onnx")
detections = model("image.jpg")
# Returns list of [N, 6] arrays: [x1, y1, x2, y2, confidence, class_id]

model = fnn.YOLO("yolov8n.onnx", conf_threshold=0.5, iou_threshold=0.5)
detections = model("image.jpg", conf_threshold=0.3)
```

See `examples/yolo_inference.py` for a complete example.

## Post-Processing

```python
from fastnn import nms, xywh2xyxy, scale_boxes, yolo_decode, yolo_dfl_decode

keep = nms(boxes, scores, iou_threshold=0.5)
boxes_xyxy = xywh2xyxy(boxes_xywh)
detections = yolo_decode(model_output, conf_threshold=0.25)
detections = yolo_dfl_decode(model_output, conf_threshold=0.25)  # YOLOv8/v10/v11
```

## Build Instructions (Rust `onnx` Feature)

The ONNX pipeline requires the `onnx` feature flag when building the Rust crate:

```toml
# Cargo.toml
[dependencies]
fastnn = { version = "2.4", features = ["onnx"] }
```

Python wheels built from source: `pip install fastnn[onnx]`.

The import stage runs in Python (`io/onnx.py`). The `OnnxConverter` and compiler passes are in Rust (`src/onnx/converter.rs`, `src/ir/`, `src/compiler/passes/`).

## Current Limitations

- **Conv2d**: CPU-only for quantized; WGPU f32 conv supported
- **Control flow**: Loop, If ops are recorded but not supported at execution time
- **GPU quantization**: U4/U8 quantized kernels run on GPU via WGSL compute shaders
- **Training**: ONNX models can execute forward passes within the compiled training pipeline, but the training graph itself cannot be exported to ONNX (see [onnx-training-export.md](onnx-training-export.md))

## Unsupported Ops

The following operations are encountered in practice but lack full support:

- Loop, If -- control flow ops (recorded but not executed)
- SequenceType and MapType operations
- Custom operators defined outside the standard ONNX opset

If you encounter an unsupported op, open an issue with the ONNX model and opset version.

## Architecture

```
fastnn/
io/
    onnx.py              # ONNX importer (90+ ops)
    graph_builder.py     # Model building (auto-detect)
    dag_model.py         # DAGModel > AotExecutor bridge
    calibrate.py         # Calibration infrastructure
    act_calibrate.py     # Activation calibration
    profiler.py          # Precision profiling
    validate.py          # Model validation
models/
    yolo.py              # YOLO model wrapper
src/
    ir/                  # ComputeGraph IR, GraphBuilder
        node.rs            # Opcode (91 variants), IrDType, DimExpr, TensorType
        builder.rs         # GraphBuilder -- fluent IR construction API
    compiler/passes/     # AOT compiler passes
        shape_inference.rs
        operator_fusion.rs
        quantization.rs
        memory_planning.rs
    backend/
        cpu/mod.rs         # CpuBackend -- dispatch all ops including quantized
    onnx/
        converter.rs       # OnnxConverter -- ONNX nodes to ComputeGraph
```

## See also

- [models.md](models.md) -- model architectures and loading guide
- [onnx-training-export.md](onnx-training-export.md) -- training export contract
- [io.md](../reference/io.md) -- serialization and format conversion
- [Training Basics](../guides/training/training-basics.md) -- compiled training pipeline
- [Architecture](../internals/architecture.md) -- system architecture overview
