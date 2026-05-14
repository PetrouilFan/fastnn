# FastNN v2.1 — Final Implementation Plan

**Goal**: A universal compiled training + inference engine that loads any model, trains/transforms in any precision (U4/U8/INT8/F16/F32), and outperforms PyTorch on CPU.

**CPU only for v2.1** — GPU (WGPU/CUDA) deferred to v2.2.

**Execution order**: Sequential. Each phase depends on the previous.

---

## Phase 1: Training Engine (~11.5h)

Make `loss.backward()` work by wiring the Tensor op graph → ComputeGraph → `build_backward_graph()`. Without this, no training is possible at all.

### 1a. Fix backward stubs to store edges + inputs (~2h)

**File**: `src/autograd.rs:114-146`

Replace the unit-struct `stub_backward!` macro with one that stores `edges: Vec<Edge>` and `inputs: Vec<Tensor>`.

```rust
// Before:
pub struct AddBackward;
impl AddBackward { pub fn new() -> Self { AddBackward } }
impl Node for AddBackward {
    fn apply(&self, ...) -> Vec<Option<Tensor>> { vec![None; 2] }
    fn next_edges(&self) -> &[Edge] { &[] }
    fn inputs(&self) -> &[Tensor] { &[] }
}

// After:
pub struct AddBackward {
    edges: Vec<Edge>,
    inputs: Vec<Tensor>,
}
impl AddBackward {
    pub fn new(edges: Vec<Edge>, inputs: Vec<Tensor>) -> Self {
        AddBackward { edges, inputs }
    }
}
impl Node for AddBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        // Actual gradient formula implemented in 1b
        let grad = match grad_outputs.into_iter().next().flatten() {
            Some(g) => g,
            None => return vec![None; 2],
        };
        vec![Some(grad.clone()), Some(grad)]  // dx=grad, dy=grad
    }
    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
}
```

Also modify the first macro form (no-arg, line 115) to match.

### 1b. Implement backward formulas for 12 core ops (~3h)

**File**: `src/autograd.rs` — replace stub `apply()` bodies in backward nodes

| Op | Formula | Uses |
|----|---------|------|
| AddBackward | `dx=grad, dy=grad` | Copy grad to both inputs |
| SubBackward | `dx=grad, dy=-grad` | Tensor::neg() |
| MulBackward | `dx=grad*y, dy=grad*x` | Tensor::mul() |
| DivBackward | `dx=grad/y, dy=-grad*x/y²` | Tensor::div(), mul(), neg() |
| MatMulBackward | `dx=grad@y.T, dy=x.T@grad` | Tensor::matmul(), transpose() |
| NegBackward | `dx=-grad` | Tensor::neg() |
| ReluBackward | `dx=grad*(x>0)` | Tensor::gt_scalar(), where() |
| ExpBackward | `dx=grad*output` | Stored output from forward |
| LogBackward | `dx=grad/x` | Tensor::div() |
| SigmoidBackward | `dx=grad*output*(1-output)` | Tensor::mul(), sub() |
| TanhBackward | `dx=grad*(1-output²)` | Tensor::mul(), sub() |
| SumBackward | `dx=grad` (broadcast to input shape) | Expand grad to input shape |
| MeanBackward | `dx=grad*(1/n)` (broadcast) | Scale grad by 1/numel |

**Deferred formula ops** (still return pass-through `vec![None; n]`):
Gelu, Silu, LeakyRelu, Elu, Softplus, Hardswish, Mish, Clamp, Pow, Sqrt, Abs, Softmax, LogSoftmax, Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Concat, Slice, Pad, Gather, ScatterNd, Where, Embedding, Dropout, MaxPool, AvgPool, BatchNorm, LayerNorm, RMSNorm, Conv2d, ConvTranspose2d

These will be wired to the IR backward system in Phase 1c (graph reconstruction → `build_backward_graph` already has formulas for all of them).

### 1c. Fix each op in ops.rs to pass edges + inputs (~1.5h)

**File**: `src/tensor/ops.rs`

For each of the 12 core ops, change:
```rust
// Before (add, line 80-91):
let _edges = { let mut _edges = make_edge(self); _edges.extend(make_edge(other)); _edges };
let backward = AddBackward::new();

// After:
let edges = { let mut edges = make_edge(self); edges.extend(make_edge(other)); edges };
let inputs = vec![self.clone(), other.clone()];
let backward = AddBackward::new(edges, inputs);
```

Same pattern for all 12 ops. The `_edges` variable is already computed but discarded — we just need to pass it through.

### 1d. Graph reconstruction engine (~4h)

**New function in `src/autograd.rs`** — replaces the no-op `backward()`:

```rust
pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    // 1. Walk grad_fn chain from root via BFS to discover all backward nodes
    let all_nodes = collect_backward_nodes(root);
    if all_nodes.is_empty() { return; }
    
    // 2. Collect unique leaf input tensors (requires_grad, no grad_fn)
    let leaf_inputs = collect_leaf_tensors(&all_nodes);
    
    // 3. Build fresh ComputeGraph from the chain
    let mut builder = GraphBuilder::new();
    let mut tensor_map: HashMap<usize, GraphTensor> = HashMap::new();
    
    // 4. Register leaf inputs (by Arc pointer identity)
    for tensor in &leaf_inputs {
        let shape: Vec<DimExpr> = tensor.shape().iter()
            .map(|&s| DimExpr::Known(s as u64)).collect();
        let gt = builder.input(&shape, ir_dtype_from_dtype(tensor.dtype()));
        tensor_map.insert(ptr_id(tensor), gt);
    }
    
    // 5. Replay ops in topological order
    for node in topological_order(&all_nodes) {
        let input_gts: Vec<GraphTensor> = node.inputs().iter()
            .map(|t| tensor_map[&ptr_id(t)].clone()).collect();
        let output_gt = match node.name() {
            "AddBackward" => builder.add(&input_gts[0], &input_gts[1]),
            "MulBackward" => builder.mul(&input_gts[0], &input_gts[1]),
            // ... map all 12 op names to builder methods
        };
        tensor_map.insert(ptr_id(&forward_tensor_of(node)), output_gt);
    }
    
    // 6. Build backward graph via existing IR infrastructure
    let loss_id = ptr_id(root);
    let loss_gt = &tensor_map[&loss_id];
    let grad_tensors = builder.backward(loss_gt).unwrap();
    
    // 7. Prepare input data, compile, execute
    let input_data: Vec<Vec<u8>> = leaf_inputs.iter()
        .map(|t| t.as_bytes().to_vec()).collect();
    let input_refs: Vec<&[u8]> = input_data.iter().map(|d| d.as_slice()).collect();
    let results = builder.compile_and_execute(
        &grad_tensors.iter().collect::<Vec<_>>(),
        CpuBackend, &input_refs,
    ).unwrap();
    
    // 8. Store gradients in leaf tensor .grad fields
    for (i, tensor) in leaf_inputs.iter().enumerate() {
        if tensor.requires_grad() {
            let grad = Tensor::from_f32_data(&results[i], tensor.shape());
            tensor.set_grad(grad);
        }
    }
}
```

**Helper functions needed** (each ~0.5h):
- `collect_backward_nodes(root)` — BFS from root via `grad_fn()`, `next_edges()`
- `collect_leaf_tensors(nodes)` — find input tensors with `requires_grad && grad_fn.is_none()`
- `topological_order(nodes)` — reverse topological sort of the node DAG
- `ptr_id(tensor)` — `Arc::as_ptr(&tensor.inner) as usize`
- `forward_tensor_of(node)` — get the output tensor that this node was attached to

### 1e. Handle constants, scalars, loss functions (~1h)

**Challenge**: Binary ops like `AddScalar`, `MulScalar` take a scalar value as the second input. These need to be represented as `Constant` nodes in the graph.

**Solution**: When reconstructing the graph, detect scalar values and create `builder.constant(value, dtype)` nodes.

Loss functions (`cross_entropy`, `mse_loss`) create custom backward nodes. Add entries for `CrossEntropyBackward` and `MSELossBackward` in the graph reconstruction matcher. Their backward will go through `build_backward_graph` which already handles them.

### 1f. Tests (~2h)

| Test | Verifies |
|------|----------|
| `test_simple_add_backward` | add backward produces correct grad |
| `test_matmul_backward` | matmul backward, check shapes |
| `test_relu_backward` | relu backward, check non-zero only for x>0 |
| `test_mlp_forward_backward` | 2-layer MLP: forward → backward → grads non-zero |
| `test_training_step` | forward → backward → sgd step → weights change |
| `test_no_grad` | no_grad() context prevents graph building |
| `test_multiple_backward` | backward twice with retain_graph, gradients accumulate |
| `test_custom_grad_output` | backward(non-one grad_output) |

---

## Phase 2: Universal Type System (~12h)

Every tensor in the graph can be U4, U8, INT8 (activations), F16, or F32. Conversion between types is handled by compiler passes.

### 2a. Quantize IR op — F32 → U4/U8 (~1.5h)

**Files**: `src/ir/node.rs`, `src/backend/cpu/mod.rs`, `src/compiler/passes/shape_inference.rs`, `src/autograd.rs`

Add `Opcode::Quantize`:
- Input: f32 tensor + bit_width attribute (4 or 8)  
- Output: U4/U8 tensor with per-channel scales/zero_points
- Backend dispatch: `quantize_f32_kernel` — calls `PackedTensor::from_f32_per_channel()`
- Shape inference: pass-through (output shape = input shape)
- Backward: STE — `d_input = d_output` (gradient passes through quantization)

### 2b. Dequantize IR op — U4/U8 → F32 (~1h)

**Files**: Same as 2a

Add `Opcode::Dequantize`:
- Input: U4/U8 tensor with scales/zero_points
- Output: f32 tensor  
- Backend dispatch: `dequantize_kernel` — applies `val * scale + zero_point` per element
- Shape inference: pass-through
- Backward: STE — `d_input = d_output` (gradient passes through dequantization)

### 2c. ToF16 / ToF32 conversion ops (~1.5h)

**Files**: Same as 2a

Add `Opcode::ToF16` and `Opcode::ToF32`:
- `ToF16`: F32 → F16 (half-precision)
- `ToF32`: F16 → F32
- Backend dispatch: uses `half::f16::from_f32()` / `f16::to_f32()`
- Shape inference: pass-through
- Backward: `ToF16` backward = `ToF32` (gradient is f32, convert to match), `ToF32` backward = `ToF16`

### 2d. INT8 activation quantization (~2.5h)

**Files**: `src/ir/node.rs`, `src/backend/cpu/mod.rs`, `src/compiler/passes/quantization.rs`

Add `Opcode::QuantizeActivations` (INT8) and `Opcode::DequantizeActivations`:

INT8 activation quantization is different from weight quantization:
- Weights are quantized per-channel (static, computed once)
- Activations are quantized per-tensor (dynamic, computed per forward pass)
- Scale = `max(abs(activation)) / 127` (symmetric INT8)

Add an activation quantization compiler pass:
- After shape inference, insert `QuantizeActivations` → `MatMul` → `DequantizeActivations` for MatMul nodes
- This keeps weights in U4/U8 and activations in INT8 during the MatMul
- Reduces activation memory bandwidth by 4x during training (INT8 = 1 byte vs F32 = 4 bytes)

Backward of activation quantization:
- `d_input = d_output` (STE)
- The scale is computed from the forward activation values (which are stored)

### 2e. Type inference pass (~2h)

**File**: `src/compiler/passes/type_inference.rs` (new)

A pass that determines the optimal dtype for each node:
- Weights: use the user-requested dtype (U4/U8/F16)
- Activations: F32 by default, INT8 if activation quantization is enabled
- Insert conversion ops where types don't match

Algorithm:
```
for node in topological_order(graph):
    for input in node.inputs:
        if input.dtype != node.expected_input_dtype(input_index):
            insert_conversion(input, node.expected_input_dtype(input_index))
```

### 2f. Auto-cast insertion pass (~1.5h)

**File**: `src/compiler/passes/auto_cast.rs` (new)

When a user specifies `model.to("u4")`:
1. For each weight parameter, replace f32 constant with `Quantize(f32_weight, bit_width=4)`
2. Insert `Dequantize` before ops that expect f32 (loss computation, softmax, etc.)
3. Activate the INT8 activation quantization pass if requested

This pass makes the type system transparent — users just declare the target precision and the compiler handles the rest.

---

## Phase 3: Quantized Optimizer (~6h)

Optimizer steps work directly on quantized weights. No full-precision copy kept.

### 3a. Dequantize-then-update optimizer (~2h)

**Files**: `src/backend/cpu/mod.rs`, `src/ir/builder.rs`

The optimizer step for a quantized weight becomes a subgraph:

```
packed_weight → Dequantize → f32_weight
f32_weight + f32_grad → OptimizerStep (SGD/Adam/AdamW) → f32_updated
f32_updated → Quantize → new_packed_weight
```

This is transparent to the user — the `GraphBuilder` handles it automatically when the weight parameter has a packed dtype.

Implementation:
- When building the training graph, detect packed weight parameters
- Wrap the optimizer step with Dequantize/Quantize
- The backward gradient is already computed against the packed representation (STE)

### 3b. F16 optimizer state (~1.5h)

**Files**: `src/backend/cpu/mod.rs` (optimizer dispatch kernels)

For Adam optimizer, the state tensors (m, v) are full-precision (f32). To save memory:
- Store m and v as F16 (half the memory)
- Convert to f32 before the optimizer step, convert back after

New kernels: `adam_update_f16_state`, `adamw_update_f16_state`
- Read m, v as F16x2 packed values
- Convert to f32 internally
- Apply Adam update to f32 weight
- Write updated m, v back as F16

### 3c. Scale adjustment during training (~1.5h)

**Files**: `src/compiler/passes/quantization.rs`, `src/packed_tensor.rs`

During training, weight magnitudes change. The initial quantization scales may become stale.

Solution:
- After each optimizer step, recompute scales from the updated dequantized weight
- The `Quantize` op already recomputes scales from the f32 data (Phase 2a)
- Optional: use EMA smoothing: `scale = 0.99 * old_scale + 0.01 * new_scale`

### 3d. Gradient scaling (~1h)

**Files**: `src/autograd.rs`, `src/backend/cpu/mod.rs`

Low-bit training can have vanishing gradients. Gradient scaling multiplies gradients by a scale factor before the optimizer step, preventing underflow.

Implementation:
- Add a `GradientScale(scale: f32)` node that can be inserted before the optimizer
- Backward: passes through (no effect on forward)
- Loss scaling: multiply loss by scale before backward
- Unscaling: divide gradients by scale after backward (or fold into optimizer step)

---

## Phase 4: Performance Optimization (~9h)

### 4a. Forward+backward fusion in compile (~3h)

**Files**: `src/compiler/passes/operator_fusion.rs`

Currently, operator fusion only fuses forward ops (MatMul+Add+ReLU → one kernel). Extend it to fuse across the backward graph:

A fused forward op like `FusedMatMulAddRelu` has a backward that is:
```
grad_input = grad @ W.T  (for MatMul)
grad_weight = x.T @ grad (for MatMul)
grad_bias = sum(grad)    (for Add/BiasAdd)
grad_activation = grad * (x > 0)  (for ReLU)
```

Instead of emitting 4 separate backward kernels, fuse them into one `fused_matmul_add_relu_backward` kernel that computes all 4 gradients in a single pass over the data (better cache utilization, less memory bandwidth).

### 4b. Multi-threaded dispatch (~3h)

**Files**: `src/backend/executor.rs`, `src/backend/cpu/mod.rs`

The current dispatch is single-threaded. Add Rayon-based parallel dispatch for independent subgraphs.

Approach:
- Analyze the instruction sequence at compile time
- Identify groups of instructions with no data dependency
- Execute independent groups in parallel via `rayon::scope()`
- Use `rayon::join()` for fork-join parallelism

Key optimizations:
- Batch normalization: compute mean+var in parallel across channels
- ResNet-style branches: execute both branches in parallel
- Independent backward ops: weight gradient and activation gradient can run in parallel

### 4c. Kernel auto-tuning (~2h)

**File**: `src/backend/cpu/microkernels.rs`

Different GEMM shapes benefit from different kernel variants. Add a compile-time auto-tuner:

```
For each MatMul shape (M, K, N):
    benchmark(blas_sgemm, m, k, n)
    benchmark(avx512_u4_gemm, m, k, n)
    benchmark(avx2_u8_gemm, m, k, n)
    benchmark(scalar_fallback, m, k, n)
    pick_fastest() → store kernel choice in plan
```

The auto-tuner runs once at compile time (not at every execution). Results can be cached.

### 4d. Shape specialization (~1h)

**Files**: `src/compiler/passes/memory_planning.rs`

When input shapes are known at compile time (not dynamic), compute exact memory sizes instead of using `SYMBOL_DIM_MAX` worst-case estimates.

This reduces arena size by up to 90% for models with symbolic dimensions (e.g., batch size = 1 vs SYMBOL_DIM_MAX = 8192).

---

## Phase 5: Model Portability (~7h)

### 5a. ExecutablePlan serialization (~2h)

**File**: `src/backend/mod.rs`, `src/backend/executor.rs`

Save a compiled plan to disk so it can be loaded without recompilation.

```rust
impl ExecutablePlan {
    pub fn save(&self, path: &str) -> Result<()>;
    pub fn load(path: &str) -> Result<ExecutablePlan>;
}
```

Format: Binary with version header, instruction list, slot assignments, arena size, kernel parameters.

### 5b. ComputeGraph serialization (~1.5h)

**File**: `src/ir/node.rs`

Save the full IR graph (including quantized weights) to a `.fnn` file format.

```rust
impl ComputeGraph {
    pub fn save_fnn(&self, path: &str) -> Result<()>;
    pub fn load_fnn(path: &str) -> Result<ComputeGraph>;
}
```

Format: Custom binary with magic bytes `FNN\x02`, version, node count, opcodes, weights (including packed with scales/zero_points), shapes.

### 5c. ONNX export (~2h)

**File**: `src/onnx/converter.rs`, `fastnn/io/export.py`

Export a ComputeGraph back to ONNX format. Required for interoperability with other frameworks.

- Map each Opcode to its ONNX equivalent
- Handle packed weights: dequantize to f32 before export
- Export training checkpoints (weights only) to ONNX or safetensors

### 5d. Standalone runtime (~1.5h)

**File**: `src/backend/runtime.rs` (new)

A minimal runtime that can load and execute compiled plans without the full compiler stack.

- No shape inference, no memory planning, no operator fusion
- Just load plan → map arena → dispatch
- Suitable for embedded deployment
- Optional: `#![no_std]` compatible (with alloc)

---

## Phase 6: Polish & Release (~3h)

### 6a. Error handling: convert 33 reachable panics to PyErr (~1.5h)

**Files**: Multiple — see the panic audit

Convert `panic!()` → `return Err(FastnnError::...)` for all 33 reachable-from-Python panics.

Categories:
- Shape/validation panics (22): `src/tensor/shape.rs`, `src/tensor/indexing.rs`
- GPU tensor panics (8): `src/tensor/mod.rs` — add `.to_cpu()` guards
- Einsum validation (3): `src/tensor/mod.rs`
- DType conversion (2): `src/tensor/mod.rs`
- NN validation (4): `src/nn/transformer.rs`, `src/nn/attention.rs`
- GPU context init (1): `src/backend/wgpu/context.rs`

### 6b. Docstrings & Quick Start (~0.5h)

- Add Rust doc comments to `AotExecutor` struct + methods in `src/python/nn.rs`
- Fix README Quick Start AOT example (define `nodes`, `params` variables)
- Update `docs/getting-started.md` to cover v2.1 training pipeline

### 6c. Build system fixes (~0.5h)

- Fix `pyproject.toml:7` version from `1.3.0` to `2.1.0`
- Remove `pip>=26.0.1` from runtime dependencies (`pyproject.toml:11`)
- Move `torch>=2.11.0` from hard dep to optional extra (`[project.optional-dependencies]`)

### 6d. CI/CD improvements (~0.5h)

- Add cross-platform wheel building to release workflow (ubuntu + windows + macos)
- Add Rust test runs on Windows/macOS CI

---

## Summary Table

| Phase | Description | Effort | Key Deliverable |
|-------|-------------|--------|-----------------|
| **1a** | Fix backward stubs to store edges+inputs | 2h | Grad_fn chain becomes real DAG |
| **1b** | Backward formulas for 12 core ops | 3h | Real gradient math in stubs |
| **1c** | Fix ops to pass edges+inputs | 1.5h | Recording works |
| **1d** | Graph reconstruction engine | 4h | backward() builds ComputeGraph |
| **1e** | Constants, scalars, losses | 1h | Works with real models |
| **1f** | Training tests | 2h | Verified correctness |
| **2a** | Quantize IR op (F32→U4/U8) | 1.5h | Type conversion in graph |
| **2b** | Dequantize IR op (U4/U8→F32) | 1h | Type conversion in graph |
| **2c** | ToF16/ToF32 ops | 1.5h | Half-precision support |
| **2d** | INT8 activation quantization | 2.5h | 4x activation memory savings |
| **2e** | Type inference pass | 2h | Automatic dtype selection |
| **2f** | Auto-cast insertion pass | 1.5h | Transparent precision control |
| **3a** | Dequantize-then-update optimizer | 2h | Quantized weight training |
| **3b** | F16 optimizer state kernels | 1.5h | Half optim state memory |
| **3c** | Scale adjustment | 1.5h | Stable quantized training |
| **3d** | Gradient scaling | 1h | Prevents vanishing gradients |
| **4a** | Forward+backward fusion | 3h | Fused backward kernels |
| **4b** | Multi-threaded dispatch | 3h | Parallel subgraph execution |
| **4c** | Kernel auto-tuning | 2h | Optimal GEMM per shape |
| **4d** | Shape specialization | 1h | Exact memory allocation |
| **5a** | ExecutablePlan serialization | 2h | Save compiled models |
| **5b** | ComputeGraph serialization (.fnn) | 1.5h | Portable model format |
| **5c** | ONNX export | 2h | Model interoperability |
| **5d** | Standalone runtime | 1.5h | No-compiler deployment |
| **6a** | Panic → PyErr conversion (33 panics) | 1.5h | Clean error handling |
| **6b** | Docstrings & Quick Start | 0.5h | Good developer experience |
| **6c** | Build system fixes | 0.5h | Correct packaging |
| **6d** | CI/CD improvements | 0.5h | Cross-platform releases |
| **Total** | | **~48h** | |

## FastNN vs PyTorch: Where the Speed Comes From

| Factor | PyTorch | FastNN v2.1 | Why FastNN wins on CPU |
|--------|---------|-------------|----------------------|
| **Weight precision** | F32 (4 bytes) | U4 (0.5 bytes) | 8x more weights fit in L2 cache |
| **Activation precision** | F32 (4 bytes) | INT8 (1 byte) | 4x less memory bandwidth |
| **Kernel launches per op** | 1 | 0 (fused) | No Python→C++→kernel overhead |
| **Backward kernel count** | ~200 for ResNet-50 | ~20 (fused) | 10x fewer dispatches |
| **Memory allocation** | malloc+free per tensor | Arena reuse | Zero alloc overhead during execution |
| **Optimizer state** | F32 (8 bytes for Adam) | F16 (4 bytes) | Half the memory for optimizer |
| **Operator fusion** | Manual (torch.jit.script) | Automatic | Fused across forward+backward |
| **Quantized training** | Custom QAT wrapper needed | Native in optimizer | No extra tooling |

**The compound effect**: 8x less weight memory + 4x less activation memory + fused kernels + single dispatch = train models 2-3x larger on the same hardware, with faster per-iteration time due to better cache utilization and less overhead.

---

## Recommended Priority for Implementation

```
Phase 1 (Training) ───── Must go first — nothing works without it
        │
Phase 2 (Types) ──────── Needed for quantized training
        │
Phase 3 (Optimizer) ──── Needed for actual quantized training loops
        │
Phase 4 (Performance) ── Optimization layer (can parallelize with 5+6)
        │
Phase 5 (Portability) ── Can parallelize with 4+6
        │
Phase 6 (Polish) ─────── Can parallelize with 4+5
```

Shall I proceed to implement Phase 1?
