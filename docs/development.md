# Development Architecture (v2.2.0)

FastNN v2.2.0+ uses an ahead-of-time (AOT) compilation pipeline backed by a graph IR, compiler passes, and backend code generation. The v1 eager-mode path (`PackedLinear`, `PackedConv2d`, `backends/`) has been removed in favor of the unified AOT pipeline. v2.2 adds compiled training (forward+backward+optimizer pipeline), FlashAttention SIMD, WGPU quantized inference, and residual+add+norm fusion.

## Rust Source Layout

```text
src/
  lib.rs                  # Crate exports, module declarations
  error.rs                # Error types
  autograd.rs             # AutogradEngine (v1 compat + build_backward_graph for v2 IR)
  iterator.rs             # TensorIterator
  residual.rs             # Residual connection helper
  storage.rs              # Memory backend and device allocation (DType, Device)
  storage_pool.rs         # Storage pooling for output tensor reuse
  storage_quantized.rs    # QuantizedTensor storage
  packed_tensor.rs        # PackedTensor<T> — packed precision tensor (used by AOT quantized dispatch)
  ir/
    mod.rs                # IR module root
    node.rs               # ComputeGraph, IRNode, Opcode, IrDType, DimExpr, TensorType, ShapeEnv, QuantizedWeightMeta
    builder.rs            # GraphBuilder and GraphTensor — fluent IR construction API
  compiler/
    mod.rs                # Compiler module root
    passes/
      mod.rs              # Pass registry
      shape_inference.rs  # Symbolic shape inference pass (DimExpr evaluation)
      operator_fusion.rs  # MatMul+Add, Conv2d+Add, +ReLU fusion
      quantization.rs     # Weight quantization pass (U4/U8 with per-channel scales)
      memory_planning.rs  # Arena-based memory planning with live-range analysis
  backend/
    mod.rs                # Backend trait + dispatch
    executor.rs           # GraphExecutor, ExecutablePlan, MemoryPlan, compile_with_plan_and_quantize
    cpu/
      mod.rs              # CpuBackend — f32, fused, matmul_u4/u8, conv2d_u4/u8, softmax, reduce, etc.
      blas.rs             # BLAS matmul backend
      im2col.rs            # Im2col for conv2d
      microkernels.rs     # AVX2/AVX-512/SWAR microkernels
      reductions_fast.rs  # Optimized reduction kernels
    wgpu/
      mod.rs              # WgpuBackend dispatch
      argmax.rs           # WGPU argmax shader
      context.rs          # GPU context management
      conv.rs             # WGPU conv shaders
      elementwise.rs      # WGPU elementwise ops
      embed.rs            # WGPU embedding
      matmul.rs           # WGPU matmul shaders
      norm.rs             # WGPU normalization
      pipeline.rs         # WGPU pipeline cache
      pool.rs             # WGPU pooling shaders
      reduce.rs           # WGPU reduction
      softmax.rs          # WGPU softmax
      transpose.rs        # WGPU transpose
  backends/               # v1 eagremode packed-layer backend (legacy)
    mod.rs
    cpu.rs                # CPU backend registration
    packed_simd.rs        # SIMD-accelerated packed GEMV kernels
    packed_blas.rs        # BLIS-style tiled packed micro-kernel
    arm_neon.rs           # ARM NEON SIMD kernels for packed GEMV
    wgpu/
      mod.rs
      mod_impl.rs
  dtypes/
    mod.rs                # PackedWord trait
    u4x8.rs               # 4-bit packed type (8 × i4 per u32 word)
    u8x4.rs               # 8-bit packed type (4 × i8 per u32 word)
    f16x2.rs              # 16-bit packed type (2 × f16 per u32 word)
    f32x1.rs              # 32-bit packed type (1 × f32 per u32 word)
  swar/
    mod.rs
    ops_4bit.rs           # SWAR operations for 4-bit packed values
    ops_8bit.rs           # SWAR operations for 8-bit packed values
    ops_16bit.rs          # SWAR operations for 16-bit packed values
    ops_32bit.rs          # SWAR operations for 32-bit packed values
  nn/
    mod.rs                # Module trait, macros
    linear.rs             # Linear layer
    conv.rs               # Conv1d/Conv2d layers
    activations.rs        # Activation functions
    attention.rs          # Multi-head attention
    transformer.rs        # Transformer blocks
    norm.rs               # BatchNorm, LayerNorm, GroupNorm
    dropout.rs            # Dropout
    embedding.rs          # Embedding layer
    pooling.rs            # MaxPool, AvgPool, Adaptive pool
    fused.rs              # Fused Conv+BN+Activation layers
    sequential.rs         # Sequential container
    residual.rs           # Residual connections
    upsample.rs           # Upsample / interpolate
  optim/
    mod.rs
    sgd.rs
    adam.rs
    adamw.rs
    muon.rs
    lion.rs
    rmsprop.rs
  onnx/
    mod.rs                # ONNX module documentation
    converter.rs          # OnnxConverter: ONNX nodes → ComputeGraph IR
  io/
    mod.rs
    serialize.rs          # Model serialization (save/load)
    dlpack.rs             # DLPack interop (Rust only)
  python/
    mod.rs                # PyO3 module registration
    tensor.rs             # PyTensor bindings
    factories.rs          # Tensor creation bindings
    ops.rs                # Tensor op bindings
    nn.rs                 # Neural network + AotExecutor bindings (quantize param)
    optim.rs              # Optimizer class bindings
    io.rs                 # Save/load bindings
    packed_quantized.rs   # Quantized tensor Python bindings
  tensor/
    mod.rs                # Tensor and TensorImpl
    shape.rs              # view/reshape/transpose/permute/squeeze
    factories.rs          # zeros/ones/full/from_vec
    ops.rs                # elementwise, matmul, activations
    reductions.rs         # sum/mean/max/min/softmax
    device.rs             # CPU/GPU movement and dtype conversion
    indexing.rs           # slice/cat/stack/repeat/where/einsum
```

## Architecture Overview

v2.0.0 has two execution paths:

### AOT-compiled inference path (v2, recommended)

```
ONNX model or GraphBuilder
        │
        ▼
  ComputeGraph (ir/node.rs)
        │
        ▼
  Compiler passes:
    shape_inference → operator_fusion → quantization → memory_planning
        │
        ▼
  Backend (CpuBackend or WgpuBackend)
        │
        ▼
  ExecutablePlan (backend/executor.rs)
        │
        ▼
  GraphExecutor::run()
```

### AOT-compiled training path (v2.2)

```
ComputeGraph with loss node
        │
        ▼
  build_backward_graph() — constructs gradient graph
        │
        ▼
  Compiler passes:
    shape_inference → operator_fusion (forward+backward)
    → quantization (opt.) → memory_planning
        │
        ▼
  Training pass — inserts optimizer update nodes
    (SGD, AdamW, Muon, Lion, or RMSprop)
        │
        ▼
  CompiledTrainingModel — single dispatch per step
    train_step(inputs) → loss scalar
```

1. **IR construction** — Build a `ComputeGraph` programmatically via `GraphBuilder`, or import an ONNX model through `OnnxConverter`.
2. **Compiler passes** — The graph is lowered through four mandatory passes:
   - `shape_inference` resolves `DimExpr` symbols into concrete or symbolic shapes.
   - `operator_fusion` fuses eligible patterns (MatMul+Add, Conv2d+Add+ReLU, etc.) into single fused nodes.
   - `quantization` replaces eligible `MatMul`/`Conv2d` weight nodes with quantized variants carrying `QuantizedWeightMeta` (U4 or U8 with per-channel scales and zero-points).
   - `memory_planning` performs live-range analysis and emits an `MemoryPlan` that aliases output buffers to minimize peak memory.
3. **Backend dispatch** — The compiled `ExecutablePlan` is handed to `CpuBackend` or `WgpuBackend`, each of which maps every `Opcode` variant to a concrete implementation.
4. **Execution** — `GraphExecutor::run()` allocates arenas per the `MemoryPlan` and executes nodes in topological order.

## Module Guide

### `ir/` — Graph IR

The IR is the core data structure for v2.0.0. Key types:

| Type | File | Purpose |
|------|------|---------|
| `ComputeGraph` | `ir/node.rs` | A DAG of `IRNode`s with named inputs and outputs |
| `IRNode` | `ir/node.rs` | A single operation: opcode, inputs, outputs, metadata |
| `Opcode` | `ir/node.rs` | Enum of 30+ ops (MatMul, Conv2d, ReLU, Quantize, etc.) |
| `IrDType` | `ir/node.rs` | Tensor dtype including `U4`, `U8` with per-channel `QuantizedWeightMeta` |
| `DimExpr` | `ir/node.rs` | Symbolic dimension expressions for shape inference |
| `TensorType` | `ir/node.rs` | Describes a tensor's shape and dtype |
| `ShapeEnv` | `ir/node.rs` | Maps symbolic dims to concrete values |
| `QuantizedWeightMeta` | `ir/node.rs` | Per-channel scale, zero-point, and axis for quantized weights |
| `GraphBuilder` | `ir/builder.rs` | Fluent API to construct `ComputeGraph` |
| `GraphTensor` | `ir/builder.rs` | Handle to a tensor within a `GraphBuilder` session |

`GraphBuilder` is the recommended entry point for programmatic graph construction. Example:

```rust
let mut b = GraphBuilder::new("my_model");
let x = b.input("x", TensorType::f32(&[1, 3, 224, 224]));
let w = b.constant("weight", weight_data);
let b_bias = b.constant("bias", bias_data);
let t = b.matmul(x, w)?;
let t = b.add(t, b_bias)?;
let t = b.relu(t)?;
let graph = b.finish();
```

### `compiler/` — Compilation passes

All passes implement a common `Pass` trait registered in `passes/mod.rs`. They transform the `ComputeGraph` in-place.

| Pass | File | Description |
|------|------|-------------|
| `ShapeInferencePass` | `passes/shape_inference.rs` | Propagates `DimExpr` through the graph, resolving symbolic dims via `ShapeEnv` |
| `OperatorFusionPass` | `passes/operator_fusion.rs` | Fuses MatMul+Add, Conv2d+Add, +ReLU, and residual+add+norm into single fused op nodes |
| `QuantizationPass` | `passes/quantization.rs` | Replaces weight-bearing ops with U4/U8 quantized variants; attaches `QuantizedWeightMeta` with per-channel scales |
| `MemoryPlanningPass` | `passes/memory_planning.rs` | Arena-based memory planning using live-range analysis; emits a `MemoryPlan` that maximizes buffer reuse |
| `TrainingPass` | `passes/training.rs` | (v2.2) Compiles forward+backward+optimizer pipeline into a single `ExecutablePlan` with persistent arena |

Passes are run in order: shape inference → fusion → quantization → memory planning. The `compile_with_plan_and_quantize` function in `backend/executor.rs` orchestrates the full pipeline.

### `backend/` — AOT backend dispatch

| Type | File | Purpose |
|------|------|---------|
| `Backend` trait | `backend/mod.rs` | Trait with `execute_node(&self, node, &inputs, output_buf)` |
| `CpuBackend` | `backend/cpu/mod.rs` | CPU implementation of all `Opcode` variants — f32, fused ops, `matmul_u4`/`matmul_u8`, `conv2d_u4`/`conv2d_u8`, softmax, reduce |
| `WgpuBackend` | `backend/wgpu/mod.rs` | GPU implementation; dispatches to specialized shaders per op |
| `GraphExecutor` | `backend/executor.rs` | Top-level executor: takes `ExecutablePlan` + `MemoryPlan`, allocates arenas, runs nodes in topological order |
| `ExecutablePlan` | `backend/executor.rs` | A compiled `ComputeGraph` ready for execution (post all passes) |
| `MemoryPlan` | `backend/executor.rs` | Output of `MemoryPlanningPass` — arena offsets and alias maps |

The `Backend::execute_node` method receives a single `IRNode`, its input buffers, and a pre-allocated output buffer. This design ensures that the backend has zero allocation during execution.

`CpuBackend` leverages:
- BLAS via `blas.rs` for f32 matmul
- `im2col.rs` for conv2d im2col expansion
- `microkernels.rs` for AVX2, AVX-512, and SWAR inner loops (including quantized matmul/conv)
- `reductions_fast.rs` for optimized reduction kernels

`WgpuBackend` dispatches through per-op shader modules (`matmul.rs`, `conv.rs`, `softmax.rs`, `quantized.rs`, etc.) and manages a pipeline cache (`pipeline.rs`) and GPU context (`context.rs`). Since v2.2, quantized U4/U8 ops run on GPU via WGSL compute shaders with per-channel dequantization.

### `onnx/` — ONNX import

`OnnxConverter` in `onnx/converter.rs` maps ONNX operator sets into `ComputeGraph` IR. It supports 55+ ops including all standard arithmetic, reduction, convolution, normalization, and activation ops. The output feeds directly into `GraphBuilder` and the compiler pipeline.

### `autograd` — Automatic differentiation

`autograd.rs` provides:
- `build_backward_graph()` which constructs a `ComputeGraph` for the v2 IR training path
- Used by the compiled training pipeline (v2.2) to produce joint forward+backward+optimizer graphs

When using the AOT path, call `build_backward_graph` on the forward `ComputeGraph` to produce a joint forward+backward graph that can be compiled and executed by `GraphExecutor`. The compiled training path (`compile_train`) extends this with optimizer update nodes for SGD, AdamW, Muon, Lion, or RMSprop.

### `dtypes/` and `swar/` — Packed precision types

| Type | File | Layout |
|------|------|--------|
| `U4x8` | `dtypes/u4x8.rs` | 8 × i4 per u32 word |
| `U8x4` | `dtypes/u8x4.rs` | 4 × i8 per u32 word |
| `F16x2` | `dtypes/f16x2.rs` | 2 × f16 per u32 word |
| `F32x1` | `dtypes/f32x1.rs` | 1 × f32 per u32 word |

All implement the `PackedWord` trait (`dtypes/mod.rs`). SWAR operations (SIMD Within A Register) for each width live in `swar/ops_{4,8,16,32}bit.rs`.

### `nn/`, `optim/`, `tensor/`, `io/`, `python/`

These modules continue to serve their v1 roles:

- **`nn/`** — Layer definitions (Module trait, Linear, Conv, Attention, Transformer, etc.)
- **`optim/`** — Optimizers (SGD, Adam, AdamW, Muon, Lion, RMSprop)
- **`tensor/`** — Core `Tensor` and `TensorImpl` with ops, reductions, shape, indexing, device transfers
- **`io/`** — Serialization and DLPack interop
- **`python/`** — PyO3 bindings; `nn.rs` includes `AotExecutor` bindings with a `quantize` parameter for v2 AOT execution

## Adding a New Operation

Adding a new op in v2.0.0 requires touching multiple modules in a specific order. Use an existing op (e.g., `Gelu`) as a reference.

### Step 1: Opcode and IR

1. Add a new `Opcode` variant in `ir/node.rs`.
2. Add a corresponding method to `GraphBuilder` in `ir/builder.rs` that creates an `IRNode` with the new opcode and wires inputs/outputs.

### Step 2: Compiler passes

3. If the op introduces new shape semantics, add a case in `compiler/passes/shape_inference.rs`.
4. If the op can participate in fusion (e.g., it is an activation that follows a MatMul or Conv2d), add a fusion rule in `compiler/passes/operator_fusion.rs`.
5. If the op can be quantized, add handling in `compiler/passes/quantization.rs`. Most non-linear ops act on the dequantized values, so they typically do not need quantization logic.

### Step 3: Backend implementations

6. Add a handler in `CpuBackend` (`backend/cpu/mod.rs`) implementing the op via existing kernels or new microkernels. If the op needs specialized inner loops, add them to `microkernels.rs` or `reductions_fast.rs`.
7. Add a handler in `WgpuBackend` (`backend/wgpu/mod.rs`). Either write a new WGSL shader in the appropriate `backend/wgpu/*.rs` file or return `UnsupportedOp` if GPU execution is not yet available. Register the shader pipeline in `pipeline.rs`.

### Step 4: ONNX converter

8. If the op has an ONNX equivalent, add a case in `onnx/converter.rs` in the `OnnxConverter`.

### Step 5: Python bindings

9. Add a binding in `python/ops.rs` (or `python/nn.rs` if it is a layer-level op). If the op is exposed through `AotExecutor`, add the parameter path in `python/nn.rs`.

### Step 6: Tests

10. Add unit tests for:
    - IR construction (verify `GraphBuilder` produces the expected `ComputeGraph`)
    - Compiler pass behavior (shape inference, fusion, quantization)
    - CpuBackend execution (numeric correctness)
    - WgpuBackend execution (numeric correctness, if implemented)
    - ONNX round-trip (if applicable)
    - Python binding (if exposed)

## Adding a Packed Precision Type

To add a new packed precision type (e.g., `U2x16`):

1. Define the type in `src/dtypes/` implementing `PackedWord` trait.
2. Add SWAR ops in `src/swar/` for the new width.
3. Add quantization support in `compiler/passes/quantization.rs` so that `QuantizationPass` can emit the new type's `QuantizedWeightMeta`.
4. Add microkernels in `src/backend/cpu/microkernels.rs` for the new width.
5. Add CpuBackend handlers for quantized matmul/conv with the new type in `src/backend/cpu/mod.rs`.
6. If GPU support is desired, add WGPU shader variants in `src/backend/wgpu/quantized.rs`.
7. Register the new dtype in `IrDType` (`ir/node.rs`) and in the `QuantizedWeightMeta` dispatch.
8. Register in `src/lib.rs`.

## Quantization Model

In v2.0.0, quantization is a **compiler pass**, not a layer-level operation. The workflow is:

1. Build a `ComputeGraph` via `GraphBuilder` or `OnnxConverter`.
2. Call `compile_with_quantize(bit_width)` (or `compile_with_plan_and_quantize`) which runs the full pipeline: shape inference → fusion → **quantization** → memory planning.
3. The `QuantizationPass` identifies eligible `MatMul` and `Conv2d` nodes, replaces their weight inputs with quantized variants carrying `QuantizedWeightMeta` (per-channel scales, zero-points, and axis), and inserts `Dequantize` nodes where needed.
4. `CpuBackend` and `WgpuBackend` dispatch to specialized `matmul_u4`/`matmul_u8` and `conv2d_u4`/`conv2d_u8` kernels that consume the quantized weights directly.

## GPU Synchronization Policy

GPU execution is asynchronous by default. Kernel launch helpers submit command buffers and return GPU-backed tensors without calling `device.poll(Maintain::Wait)`.

Synchronization is allowed at explicit host boundaries:

- `Tensor::to_cpu`
- Python `Tensor.numpy()`
- scalar extraction such as `.item()`
- DLPack or serialization paths that require host data
- test-only explicit barriers

Avoid host scalar reads inside optimizer or model inner loops. If a GPU algorithm needs a scalar, compute it into a GPU buffer and consume it from the next GPU kernel.

## Adding a Fused GPU Operation

In v2.0.0, fusion is handled by `OperatorFusionPass` at the IR level, not by hand-written fused kernels. The flow is:

1. Define the fusion pattern in `compiler/passes/operator_fusion.rs` (e.g., MatMul + Add → FusedMatMulAdd).
2. Add the fused `Opcode` variant to `ir/node.rs`.
3. Implement the fused op in `CpuBackend` and/or `WgpuBackend` as a single node handler.

Existing fused patterns:
- `MatMul + Add` → `FusedMatMulAdd`
- `Conv2d + Add` → `FusedConv2dAdd`
- `(MatMul | Conv2d) + Add + ReLU` → `FusedMatMulAddReLU` / `FusedConv2dAddReLU`

To add a new fusion, extend `OperatorFusionPass` with the pattern, add the new `Opcode`, and implement it in backends. The compiler pipeline picks it up automatically.

## Key Differences from v1.x

| Aspect | v1.x | v2.0.0 |
|--------|------|--------|
| Kernel dispatch | `kernels/` with runtime dispatch | `backend/` with AOT-compiled `ExecutablePlan` |
| Quantization | Compiler pass (`QuantizationPass`) with per-channel `QuantizedWeightMeta` | Compiler pass (`QuantizationPass`) with per-channel `QuantizedWeightMeta` |
| Graph representation | None (eager mode) | `ComputeGraph` IR with `IRNode`, `Opcode`, `DimExpr` |
| Shape handling | Runtime only | Symbolic `DimExpr` with `ShapeEnv` via `ShapeInferencePass` |
| Memory management | Per-op allocation | Arena-based `MemoryPlan` with live-range aliasing |
| Fusion | Fused layer types in `nn/fused.rs` | `OperatorFusionPass` in compiler pipeline |
| Backend code | `kernels/cpu/`, `kernels/gpu/` | `backend/cpu/`, `backend/wgpu/` (structured per-op) |
| ONNX import | Python-side `dag.py` | Rust `OnnxConverter` → `ComputeGraph` → compiler pipeline |
| Legacy layer runtime | _Removed_ | _Unified AOT pipeline_ |