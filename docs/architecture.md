# v2.0.0 AOT Compiler Pipeline Architecture

## Overview

v2.0.0 replaces the v1.x DAG/layer dispatch with a complete IR-based AOT (ahead-of-time) compiler pipeline. The core components are:

- **ComputeGraph IR** — the intermediate representation that all compilation stages operate on
- **Compiler passes** — transformations that optimize, fuse, and plan the graph
- **Backend dispatch** — maps IRNodes to concrete backend implementations
- **GraphExecutor** — runtime execution over a pre-planned arena memory layout

The pipeline compiles a model once into an `ExecutablePlan`, then runs it repeatedly with zero overhead from graph traversal or dynamic dispatch.

## Pipeline Stages

### Stage 1 — Graph Construction

Raw model definitions are converted into a `ComputeGraph`:

- **ONNX import**: `OnnxConverter` parses ONNX protobuf and emits `ComputeGraph` nodes.
  Quantized ONNX ops (`QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul`, `QLinearConv`)
  are decomposed into primitive f32 operations (`sub`, `mul`, `div`, `matmul`, `conv2d`, `clamp`)
  so the compiler pipeline sees only native f32 IR nodes.
- **Direct API**: `GraphBuilder` provides a programmatic Rust API for constructing graphs without ONNX

The output is a flat list of `IRNode`s with explicit dataflow edges via input references.

### Stage 1.5 — Shape Inference

Symbolic dimensions are resolved before fusion or memory planning can proceed. `DimExpr::evaluate_with_env` substitutes concrete values from a `ShapeEnv` for every `Symbol(String, ..)` in the graph. This allows the compiler to compute exact tensor shapes and byte sizes at compile time for fixed-shape models, and to defer resolution for dynamically-shaped models via `Bounded` dims.

### Stage 2 — Operator Fusion

The fusion pass scans the `ComputeGraph` for adjacent ops that can be merged:

| Pattern | Fused Opcode |
|---------|-------------|
| `MatMul → Add` | `FusedMatMulAdd` |
| `Conv2d → Add` | `FusedConv2dAdd` |
| `FusedMatMulAdd → ReLU` | `FusedMatMulAddRelu` |
| `FusedConv2dAdd → ReLU` | `FusedConv2dAddRelu` |

Fusion eliminates intermediate tensor materialization and reduces kernel launch overhead.

### Stage 2.5 — Weight Quantization (Optional)

When enabled, `quantize_weights` transforms `F32` weight tensors into `U4` or `U8` per-channel quantized format. This is a compiler pass, not a runtime operation — the weights are packed at compile time and embedded directly into the `ExecutablePlan`.

### Stage 3 — Memory Planning

A greedy first-fit arena allocator with live-range analysis assigns each tensor a slot and an arena offset. Tensors whose lifetimes do not overlap share the same slot. Alignment constraints are respected per dtype.

### Stage 4 — Backend Dispatch

Each `IRNode` is mapped to a concrete backend implementation:

- **CpuBackend** — handles all opcodes including quantized `matmul_u4`, `matmul_u8`, `conv2d_u4`, `conv2d_u8`
- **WgpuBackend** — handles `f32` ops but returns `UnsupportedOp` for quantized opcodes, causing fallback to CPU

Both backends receive an `ExecutablePlan` with pre-planned arena offsets, eliminating runtime allocation.

### Stage 5 — Execution

`GraphExecutor::run()` executes the `ExecutablePlan` step by step, reading and writing arena-based tensor storage. No graph traversal or dynamic dispatch occurs at runtime.

## IR Core Types

Key types from `ir/node.rs`:

### ComputeGraph

The top-level IR container. Holds:

- **`nodes`** — `Vec<IRNode>`, the flat list of operations
- **`inputs`** — ordered list of graph input names
- **`outputs`** — ordered list of graph output names
- **`name_to_node`** — `HashMap<String, NodeIdx>` for O(1) lookup by name

### IRNode

A single operation in the graph:

- **`opcode`** — the `Opcode` variant
- **`inputs`** — `Vec<String>` referencing other nodes by name
- **`output_type`** — `TensorType` describing shape and dtype
- **`attrs`** — `HashMap<String, AttrValue>` for per-op constants
- **`name`** — unique identifier used as input reference target

### Opcode Enum

30+ variants including `MatMul`, `Conv2d`, `Softmax`, `FusedMatMulAdd`, `FusedConv2dAddRelu`, and all fused variants. Each opcode corresponds to exactly one kernel implementation per backend.

### IrDType

| Variant | Description |
|---------|-------------|
| `F32` | 32-bit float |
| `F16` | 16-bit float |
| `I64` | 64-bit signed integer |
| `Bool` | Boolean |
| `U4{scales,zero_points,shape}` | 4-bit quantized with per-channel metadata |
| `U8{scales,zero_points,shape}` | 8-bit quantized with per-channel metadata |

### DimExpr

Symbolic and concrete dimension expressions:

- `Known(u64)` — fully resolved dimension
- `Symbol(String, Option<u64>)` — named dimension with optional default
- `Bounded(min, max)` — dimension known to fall within a range
- `Product(vals)` — product of sub-expressions
- `Sum(vals)` — sum of sub-expressions
- `Div(lhs, rhs)` — integer division of sub-expressions

### TensorType

Combines shape (`Vec<DimExpr>`) and dtype (`IrDType`) to fully describe a tensor's layout.

### ShapeEnv

Maps symbolic dimension names to concrete values at runtime. Used by `DimExpr::evaluate_with_env` to resolve `Symbol` dims during shape inference.

### QuantizedWeightMeta

Embedded within `IrDType::U4` and `IrDType::U8`:

- **`packed_dtype`** — the quantized element type (`U4` or `U8`)
- **`scales`** — `Vec<f32>` per-channel scale factors
- **`zero_points`** — `Vec<f32>` per-channel zero points
- **`shape`** — `Vec<DimExpr>` original (unpacked) weight shape

## Quantization Architecture

### Per-Channel Metadata

`IrDType::U4` and `IrDType::U8` carry per-channel `Vec<f32>` for `scales` and `zero_points`. During dequantization, each output channel is reconstructed as:

```
value = (packed_value - zero_point[channel]) * scale[channel]
```

### PackedTensor Layout

- **U4**: 8 values packed per `u32` word (`U4x8`)
- **U8**: 4 values packed per `u32` word (`U8x4`)

The packed layout eliminates wasted padding and enables SIMD-friendly access patterns.

### Weight Reshaping Before Packing

| Weight type | Original shape | Reshaped before packing |
|-------------|---------------|------------------------|
| 2D MatMul | `[K, N]` | Transposed to `[N, K]` to match GEMM convention |
| 4D Conv2d | `[OC, IC, KH, KW]` | Flattened to `[OC, IC*KH*KW]` for im2col+GEMM |

### SIMD_MARGIN

`packed_byte_size` includes a `SIMD_MARGIN` of 16 extra `u32` words per `PackedTensor`. This padding prevents out-of-bounds reads in vectorized kernels that process data in wider strides than the logical element count.

### Memory Planning Integration

The memory planner uses `packed_byte_size()` — not `unpacked_byte_size()` — when computing arena slot sizes for `U4`/`U8` tensors. This ensures arena offsets are correctly sized for the packed layout, including the SIMD margin.

## Backend Dispatch

### CpuBackend

Handles all opcodes:

- Standard `f32` ops: `MatMul`, `Conv2d`, `Softmax`, `ReLU`, etc.
- Quantized ops: `matmul_u4`, `matmul_u8`, `conv2d_u4`, `conv2d_u8`

Quantized kernels read from packed arena slots and dequantize on the fly during accumulation.

### WgpuBackend

Handles `f32` ops with GPU acceleration. Returns `UnsupportedOp` for quantized opcodes, which triggers transparent CPU fallback. No unpacking or de-quantization occurs on GPU.

### ExecutablePlan

Both backends receive an `ExecutablePlan` containing:

- Pre-planned arena offsets for every tensor
- The ordered list of instructions to execute
- Slot-to-offset mapping for arena memory

This means the backend never allocates memory — it operates entirely within the pre-planned arena.

## Memory Planning

The greedy first-fit arena allocator works as follows:

1. Each instruction is assigned a **slot index**
2. Slots are assigned arena offsets respecting dtype alignment
3. When a tensor becomes dead (no future consumers), its slot is freed for reuse
4. Non-overlapping tensors share the same slot, minimizing total arena size

### Runtime Shape Resolution

For models with symbolic dimensions, `tightened()` recomputes arena offsets from a concrete `ShapeEnv` at runtime. This allows a single compiled graph to handle multiple batch sizes without recompilation.

## v1 vs v2 Comparison

| Aspect | v1.x (removed) | v2.0.0 (AOT) |
|--------|-----------|---------------|
| Graph representation | Python DAG + Rust layers | ComputeGraph IR |
| Quantization | Layer-level (_removed_) | Compiler pass |
| Memory | Per-tensor allocation | Arena-based memory plan |
| Fusion | None (sequential ops) | MatMul+Add, Conv2d+Add, +ReLU |
| Shape inference | Runtime only | Compile-time with DimExpr |
| Backend dispatch | Dynamic at runtime | Compiled into ExecutablePlan |
| ONNX import | Python DAG (_removed_) | Rust OnnxConverter → ComputeGraph |