# Development Architecture

Codebase walkthrough for contributors adding ops, compiler passes, backend kernels, and tests to the fastnn AOT compilation pipeline.

## Rust Source Layout

```text
src/
  lib.rs, error.rs, autograd.rs, iterator.rs, residual.rs
  storage.rs, storage_pool.rs, storage_quantized.rs, packed_tensor.rs
  ir/
    mod.rs, node.rs (ComputeGraph, IRNode, Opcode, IrDType, DimExpr)
    builder.rs (GraphBuilder, GraphTensor)
  compiler/
    mod.rs, passes/
      shape_inference.rs, operator_fusion.rs, fusion/ (mod.rs, op_relu.rs,
      matmul_add_relu.rs, backward.rs, residual_add_norm.rs)
      quantization.rs, auto_cast.rs, type_inference.rs, constant_folding.rs
      dead_code_elimination.rs, activation_quantization.rs
      arithmetic_simplify.rs, memory_planning.rs, training.rs
  backend/
    mod.rs, executor.rs (GraphExecutor, ExecutablePlan, MemoryPlan)
    cpu/ (mod.rs, blas.rs, im2col.rs, microkernels.rs, reductions_fast.rs, flash_attn.rs)
    wgpu/ (mod.rs, argmax.rs, context.rs, conv.rs, elementwise.rs, matmul.rs,
      norm.rs, quantized.rs, reduce.rs, softmax.rs, transpose.rs, pool.rs, embed.rs, pipeline.rs)
  dtypes/ (mod.rs, u4x8.rs, u8x4.rs, f16x2.rs, f32x1.rs)
  swar/ (mod.rs, ops_4bit.rs .. ops_32bit.rs)
  nn/ (Linear, Conv, Transformer, etc.), optim/ (SGD, Adam..RMSprop)
  onnx/ (mod.rs, converter.rs), io/ (serialize.rs, dlpack.rs)
  python/ (mod.rs, tensor.rs, factories.rs, ops.rs, nn.rs, optim.rs, io.rs, packed_quantized.rs)
  tensor/ (mod.rs, shape.rs, factories.rs, ops.rs, reductions.rs, device.rs, indexing.rs)
```

## Architecture Overview

### AOT-compiled inference path (v2, recommended)

```
ONNX model or GraphBuilder
        |
        v
  ComputeGraph (ir/node.rs)
        |
        v
  Compiler passes:
    shape_inference -> auto_cast -> type_inference -> operator_fusion
    -> quantization -> constant_folding -> dead_code_elimination -> memory_planning
        |
        v
  Backend (CpuBackend or WgpuBackend) -> ExecutablePlan -> GraphExecutor::run()
```

### AOT-compiled training path (v2.2+)

```
ComputeGraph with loss node -> build_backward_graph() -> Compiler passes
(with backward fusion) -> Training pass (optimizer injection for SGD, AdamW,
Muon, Lion, RMSprop) -> CompiledTrainingModel -> train_step(inputs) -> loss
```

## Module Guide

### `ir/` -- Graph IR

The IR is the core data structure. Key types in `ir/node.rs`:

| Type | Purpose |
|------|---------|
| `ComputeGraph` | DAG of `IRNode`s with named inputs and outputs |
| `IRNode` | Single operation: opcode, inputs, outputs, metadata |
| `Opcode` | 91 variants (MatMul, Conv2d, ReLU, optimizer updates, etc.) |
| `IrDType` | Tensor dtype including U4/U8/I4 with `QuantizedWeightMeta`; I4 supports optional `codebooks` for per-block K-means quantization |
| `DimExpr` | Symbolic dimension expressions for shape inference |
| `TensorType` | Tensor shape + dtype descriptor |
| `ShapeEnv` | Maps symbolic dims to concrete values |
| `QuantizedWeightMeta` | Per-channel scale, zero-point, axis; optional `codebooks: Vec<[f32; 16]>` for I4Codebook |

`GraphBuilder` in `ir/builder.rs` is the recommended entry point:

```rust
let gb = GraphBuilder::new();
let x = gb.input(&[1, 3, 224, 224], IrDType::F32);
let w = gb.constant(&weight_bytes, weight_tt);
let t = gb.matmul(&x, &w);
let t = gb.bias_add(&t, &b);
let t = gb.relu(&t);
let graph = gb.to_graph();
```

### `compiler/` -- Compilation passes

All passes implement a `Pass` trait and transform the `ComputeGraph` in-place.
Pipeline order: shape inference -> fusion -> quantization -> memory planning.
Orchestrated by `compile_with_plan_and_quantize` in `backend/executor.rs`.

| Pass | Description |
|------|-------------|
| `ShapeInferencePass` | Resolves `DimExpr` symbols via `ShapeEnv` |
| `OperatorFusionPass` | Fuses MatMul+Add, Conv2d+Add+ReLU, residual+add+norm |
| `QuantizationPass` | Replaces weight ops with U4/U8/I4Codebook quantized variants |
| `MemoryPlanningPass` | Arena-based planning with live-range analysis |
| `inject_optimizer()` | Inserts optimizer update nodes (all 6 optimizers) |

Additional passes: `auto_cast` (mixed precision), `type_inference` (dtype propagation),
`constant_folding` (compile-time evaluation), `dead_code_elimination` (unused node
removal), `activation_quantization`, `arithmetic_simplify`.

### `backend/` -- AOT backend dispatch

The `Backend` trait enforces zero-allocation execution: `execute_node(&self, node, &inputs, output_buf)`.
`CpuBackend` uses BLAS (`blas.rs`), im2col (`im2col.rs`), microkernels (`microkernels.rs`),
optimized reductions (`reductions_fast.rs`), and FlashAttention SIMD (`flash_attn.rs`).
`WgpuBackend` dispatches to per-op WGSL shaders with a pipeline cache (`pipeline.rs`)
and GPU context (`context.rs`). Quantized U4/U8 ops run on GPU since v2.2.

### `onnx/`, `autograd/`, `dtypes/`, `swar/`

- **`onnx/converter.rs`**: Maps ~67 ONNX opsets into `ComputeGraph` IR.
- **`autograd.rs`**: `build_backward_graph()` produces joint forward+backward graphs
  for the compiled training pipeline.
- **`dtypes/`**: Packed types -- `I4x8` (8xi4/u32), `I8x4` (4xi8/u32),
  `F16x2` (2xf16/u32), `F32x1`. All implement the `PackedWord` trait.
- **`swar/`**: SIMD Within A Register operations for each packed width.

### `nn/`, `optim/`, `tensor/`, `io/`, `python/`

- **`nn/`**: Layer definitions (Linear, Conv, Transformer, norm, dropout, pooling, residual, fused layers)
- **`optim/`**: Optimizers (SGD, Adam, AdamW, Muon, Lion, RMSprop)
- **`tensor/`**: Core Tensor with ops, reductions, shape manipulation, indexing, device transfers
- **`io/`**: Model serialization (save/load) and DLPack interop
- **`python/`**: PyO3 bindings covering tensor ops, nn modules, optimizers, serialization, and AotExecutor

## How to Add a New IR Op

1. **Opcode + IR**: Add `Opcode` variant in `ir/node.rs`. Add `GraphBuilder` method in `ir/builder.rs` that creates an `IRNode` with the new opcode and wires inputs/outputs.
2. **Compiler passes**: Update `shape_inference.rs` if new shape semantics. Add fusion rule in `fusion/` if the op can be part of a fused pattern. Update `quantization.rs` if quantizable.
3. **Backend**: Implement in `CpuBackend` (`backend/cpu/mod.rs`). Optionally in `WgpuBackend`. For SIMD inner loops, add to `microkernels.rs` with runtime ISA dispatch (AVX-512 -> AVX2 -> NEON -> scalar).
4. **ONNX**: Add a case in `onnx/converter.rs` if an ONNX equivalent exists.
5. **Python**: Add binding in `python/ops.rs` or `python/nn.rs`.
6. **Tests**: Unit tests for IR construction, compiler pass behavior, CpuBackend correctness, WgpuBackend correctness (if implemented), ONNX round-trip (if applicable), Python binding (if exposed).

## How to Add a New Compiler Pass

1. Create a new file in `compiler/passes/` with a struct implementing the `Pass` trait (`run(&self, graph: &mut ComputeGraph)`).
2. Register the pass in `compiler/passes/mod.rs`.
3. Insert at the correct pipeline position in `backend/executor.rs`.
4. Test independently with a known input graph, verifying the expected transformation.

Use `ShapeInferencePass` or `DeadCodeEliminationPass` as reference.

## How to Add a New Backend Kernel

1. **CpuBackend**: Add implementation to `microkernels.rs` (inner loops), `reductions_fast.rs` (reductions), or `flash_attn.rs` (attention). Wire dispatch in `backend/cpu/mod.rs` by matching the `Opcode` variant and calling the new kernel.
2. **WgpuBackend**: Write a WGSL compute shader, register the pipeline in `pipeline.rs`, wire dispatch in `backend/wgpu/mod.rs`.

## How to Add a Packed Precision Type

1. Define the type in `src/dtypes/` implementing the `PackedWord` trait.
2. Add SWAR ops in `src/swar/` for the new width.
3. Add quantization support in `compiler/passes/quantization.rs`.
4. Add microkernels in `src/backend/cpu/microkernels.rs`.
5. Add CpuBackend handlers for quantized matmul/conv in `src/backend/cpu/mod.rs`.
6. Optionally add WGPU shader variants in `src/backend/wgpu/quantized.rs`.
7. Register the new dtype in `IrDType` (`ir/node.rs`) and `QuantizedWeightMeta` dispatch.
8. Register in `src/lib.rs`.

## Quantization Model

Quantization is a **compiler pass**, not a layer-level operation:

1. Build a `ComputeGraph` via `GraphBuilder` or `OnnxConverter`.
2. Call `compile_with_quantize(bit_width)` which runs the full pipeline including `QuantizationPass`. Use `"i4cb"` for codebook quantization.
3. The pass identifies eligible `MatMul`/`Conv2d` nodes, replaces weight inputs with quantized variants carrying `QuantizedWeightMeta` (per-channel scales, zero-points, axis; optionally `codebooks` for I4Codebook), and inserts `Dequantize` nodes where needed.
4. Backends dispatch to specialized `matmul_u4`/`matmul_u8` and `conv2d_u4`/`conv2d_u8` kernels that consume the quantized weights directly. I4Codebook quantizes via `from_f32_per_block_codebook` and dequantizes in the GEMM fallback path (dequant to f32, then plain matmul).

## GPU Synchronization Policy

GPU execution is asynchronous by default. Kernel launch helpers submit command buffers and return GPU-backed tensors without calling `device.poll(Maintain::Wait)`.

Synchronization is allowed only at explicit host boundaries:

- `Tensor::to_cpu`, Python `Tensor.numpy()`, scalar extraction (`.item()`)
- DLPack or serialization paths that require host data
- Test-only explicit barriers

Avoid host scalar reads inside optimizer or model inner loops. If a GPU algorithm needs a scalar, compute it into a GPU buffer and consume it from the next GPU kernel.

## See also

- [Architecture](architecture.md) -- AOT compiler pipeline: IR, compiler passes, backends
- [Release Process](release-process.md) -- Release workflow and checklist
- [Performance Roadmap](performance-roadmap.md) -- Backend performance priorities
- [ARM NEON Backend](arm-neon.md) -- NEON SIMD kernel documentation and benchmarks
- [docs/index.md](../index.md) -- Documentation home
- [CONTRIBUTING.md](../../CONTRIBUTING.md) -- Repository setup, PR process, coding standards
