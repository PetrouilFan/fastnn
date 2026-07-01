# AOT Compiler Pipeline Architecture

The fastnn AOT (ahead-of-time) compiler replaces traditional per-layer dispatch with a complete IR-based compilation pipeline that produces an `ExecutablePlan` for zero-overhead runtime execution.

## Pipeline Overview

The compilation pipeline converts a model definition into a pre-planned executable:

```
Model (ONNX / GraphBuilder API)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              ComputeGraph IR (ir/node.rs)                │
│   Flat list of IRNode with explicit dataflow edges       │
│   91 Opcode variants, DimExpr symbolic shapes            │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Compiler Passes                             │
│                                                          │
│   1. Shape Inference    ─── Resolve DimExpr symbols      │
│   2. Auto-Cast          ─── Insert type conversion      │
│   3. Type Inference     ─── Propagate dtypes            │
│   4. Operator Fusion    ─── Merge adjacent op patterns  │
│   5. Quantization       ─── Replace weights with U4/U8/I4Codebook  │
│   6. Constant Folding   ─── Evaluate at compile time    │
│   7. Dead Code Elim.    ─── Remove unused nodes         │
│   8. Memory Planning    ─── Arena offsets via live-range│
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Backend Code Generation                     │
│                                                          │
│   CpuBackend (7326-line mod.rs)     WgpuBackend (exp.)  │
│   ISA dispatch: AVX-512 → AVX2 →    WGSL compute        │
│   NEON → scalar fallback             shaders            │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              ExecutablePlan                              │
│   Pre-planned arena offsets, instruction list,           │
│   slot-to-offset mapping. No runtime allocation.         │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              GraphExecutor::run()                       │
│   Step-by-step execution on pre-allocated arena.         │
│   No graph traversal or dynamic dispatch at runtime.     │
└─────────────────────────────────────────────────────────┘
```

## Stage 1 — Graph Construction

Raw model definitions are converted into a `ComputeGraph`:

- **ONNX import**: `OnnxConverter` parses ONNX protobuf and emits `ComputeGraph` nodes (~67 ops supported). Quantized ONNX ops (`QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul`, `QLinearConv`) are decomposed into primitive f32 operations so the compiler pipeline sees only native IR nodes.
- **Direct API**: `GraphBuilder` provides a programmatic Rust API for constructing graphs without ONNX.

The output is a flat list of `IRNode`s with explicit dataflow edges via input references.

### Stage 1.5 — Shape Inference

Symbolic dimensions are resolved before fusion or memory planning. `DimExpr::evaluate_with_env` substitutes concrete values from a `ShapeEnv` for every `Symbol(String)` in the graph. This allows the compiler to compute exact tensor shapes and byte sizes at compile time for fixed-shape models, and to defer resolution via `Bounded` dims for dynamic shapes.

## Stage 2 — Operator Fusion

The fusion pass scans the `ComputeGraph` for adjacent ops that can be merged:

| Pattern | Fused Opcode |
|---------|-------------|
| `MatMul → Add` | `FusedMatMulAdd` |
| `Conv2d → Add` | `FusedConv2dAdd` |
| `FusedMatMulAdd → ReLU` | `FusedMatMulAddRelu` |
| `FusedConv2dAdd → ReLU` | `FusedConv2dAddRelu` |
| `Residual → Add → LayerNorm` | `FusedResidualAddNorm` |
| Backward fusion | Eliminates 3 intermediate allocations per fused backward chain |

Fusion eliminates intermediate tensor materialization and reduces kernel launch overhead. Backward fusion combines `dRelu+Transpose+MatMul` into single fused kernels.

### Stage 2.5 — Weight Quantization (Optional)

When enabled, `quantize_weights` transforms `F32` weight tensors into `U4`, `U8`, or `I4Codebook` per-channel quantized format. `I4Codebook` uses per-block codebook INT4 quantization: each block is normalized by its max_abs, a single data-adaptive K-means codebook (16 centroids in [-1,1]) is learned across all normalized values, and weights are packed as 4-bit codebook indices. Dequant: `scales[blk] * codebook[nibble]`. This is a compiler pass: the weights are packed at compile time and embedded directly into the `ExecutablePlan`.

### Additional Passes

The compiler pipeline also runs:
- **Auto-cast** — Inserts type conversion nodes for mixed-precision graphs
- **Type inference** — Propagates dtypes through the graph
- **Constant folding** — Evaluates constant subexpressions at compile time
- **Dead code elimination** — Removes unused nodes and no-op operations

## Stage 3 — Memory Planning

A greedy first-fit arena allocator with live-range analysis assigns each tensor a slot and an arena offset. Tensors whose lifetimes do not overlap share the same slot. Alignment constraints are respected per dtype.

The memory planner uses `packed_byte_size()` (which includes a 16-`u32` `SIMD_MARGIN`) when computing arena slot sizes for `U4`/`U8` tensors.

For models with symbolic dimensions, `tighten()` recomputes arena offsets from a concrete `ShapeEnv` at runtime, allowing a single compiled graph to handle multiple batch sizes without recompilation.

## Stage 4 — Backend Code Generation

Each `IRNode` is mapped to a concrete backend implementation:

- **CpuBackend** (7326-line `mod.rs`) — Handles all opcodes including quantized matmul/conv (U4/U8, I4Codebook) with runtime ISA dispatch: AVX-512 → AVX2 → NEON → scalar fallback. I4Codebook dequantizes to f32 before GEMM. Leverages BLAS (`blas.rs`), im2col (`im2col.rs`), microkernels (`microkernels.rs`), optimized reductions (`reductions_fast.rs`), and FlashAttention SIMD (`flash_attn.rs`).
- **WgpuBackend** (experimental) — Handles f32 ops and U4/U8 quantized ops via WGSL compute shaders. Manages a pipeline cache (`pipeline.rs`) and GPU context (`context.rs`).

Both backends receive an `ExecutablePlan` with pre-planned arena offsets, eliminating runtime allocation.

## Stage 5 — Execution

`GraphExecutor::run()` executes the `ExecutablePlan` step by step, reading and writing arena-based tensor storage. The executor allocates arenas per the `MemoryPlan` and runs nodes in topological order. No graph traversal or dynamic dispatch occurs at runtime.

## IR Core Types

Key types from `ir/node.rs`:

### ComputeGraph

The top-level IR container. Holds a flat `Vec<IRNode>` list, ordered graph input and output `NodeId`s, a `HashSet<NodeId>` of required nodes (used by DCE), and an auto-incrementing `next_id` counter.

### IRNode

A single operation in the graph with an `Opcode` variant (91 variants), `Vec<NodeId>` inputs referencing other nodes, an `output_type` (`TensorType` describing shape and dtype), `attrs` (`HashMap<String, String>` for per-op constants), and an optional `name`.

### Opcode Enum

91 variants covering: arithmetic (Add, Sub, Mul, Div, Pow...), neural network (MatMul, Conv1d/2d/3d, ConvTranspose2d, BatchNorm, LayerNorm...), activations (Relu, Gelu, Silu, Sigmoid, Tanh, Mish...), reductions (ReduceSum, ReduceMean, ReduceMax, ArgMax...), shape ops (Reshape, Transpose, Concat, Slice, Gather...), quantization (Quantize, Dequantize, ToF16, ToF32...), and optimizer updates (SgdUpdate, AdamUpdate, AdamWUpdate, MuonUpdate, LionUpdate, RmspropUpdate).

Backend kernel selection (e.g., `matmul_u4`, `matmul_u8`, `conv2d_u4`) happens at runtime based on quantized input dtypes, not via separate opcodes.

### IrDType

| Variant | Description |
|---------|-------------|
| `F32` | 32-bit float |
| `F16` | 16-bit float |
| `BF16` | 16-bit bfloat |
| `I32` | 32-bit signed integer |
| `I64` | 64-bit signed integer |
| `I8` | 8-bit signed integer |
| `Bool` | Boolean |
| `U4{scales,zero_points}` | 4-bit unsigned quantized with per-channel metadata |
| `U8{scales,zero_points}` | 8-bit unsigned quantized with per-channel metadata |
| `I4{scales,zero_points,codebooks}` | 4-bit signed quantized with optional per-block codebook |

`U4` and `U8` carry per-channel `Vec<f32>` for `scales` and `zero_points`. `I4` can additionally carry `codebooks: Vec<[f32; 16]>` for per-block K-means codebook quantization. During dequantization: standard I4 uses `(packed_value - zero_point) * scale`; I4 codebook uses `scales[blk] * codebooks[0][nibble]`.

### PackedTensor Layout

- **U4**: 8 values packed per `u32` word (`I4x8`)
- **U8**: 4 values packed per `u32` word (`I8x4`)
- **I4Codebook**: Same as U4 layout (8×4-bit nibbles per `u32`), but nibbles index a 16-entry f32 codebook instead of quantizing directly. Codebook and per-block scales stored alongside packed data.

The packed layout eliminates wasted padding and enables SIMD-friendly access patterns.

### DimExpr and TensorType

Symbolic and concrete dimension expressions: `Known(u64)` for fully resolved dimensions, `Symbol(String)` for named dimensions resolved at compile time, and `Bounded { sym, max }` for dimensions bounded by an upper limit. `TensorType` combines a shape (`Vec<DimExpr>`) and dtype (`IrDType`) to fully describe a tensor's layout.

## Quantization Architecture

### Weight Reshaping Before Packing

| Weight type | Original shape | Reshaped before packing |
|-------------|---------------|------------------------|
| 2D MatMul | `[K, N]` | Transposed to `[N, K]` to match GEMM convention |
| 4D Conv2d | `[OC, IC, KH, KW]` | Flattened to `[OC, IC*KH*KW]` for im2col+GEMM |

### SIMD_MARGIN

`packed_byte_size` includes a `SIMD_MARGIN` of 16 extra `u32` words per `PackedTensor`. This padding prevents out-of-bounds reads in vectorized kernels.

## v1 vs v2 Comparison

| Aspect | v1.x (removed) | v2.x (current) |
|--------|-----------|---------------|
| Graph representation | Python DAG + Rust layers | ComputeGraph IR |
| Quantization | Layer-level | Compiler pass |
| Memory | Per-tensor allocation | Arena-based memory plan |
| Fusion | None (sequential ops) | MatMul+Add, Conv2d+Add, +ReLU, residual+add+norm |
| Shape inference | Runtime only | Compile-time with DimExpr |
| Backend dispatch | Dynamic at runtime | Compiled into ExecutablePlan |
| ONNX import | Python DAG | Rust OnnxConverter → ComputeGraph |

## See also

- [Development](development.md) — Codebase walkthrough and guide for adding ops, passes, and kernels
- [Performance Roadmap](performance-roadmap.md) — Backend performance priorities and completed milestones
- [ARM NEON Backend](arm-neon.md) — NEON SIMD kernel documentation and build instructions
- [docs/index.md](../index.md) — Documentation home
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — Repository setup, PR process, coding standards
