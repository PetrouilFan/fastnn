# PreparedPlan and Packed Inference Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn fastnn’s AOT path into a real inference compiler: optimize the model at compile/load time, prepare static weights and kernels, then run inference through low-overhead prepared instructions that enable packed precision and large memory/performance wins without accuracy loss in default mode.

**Architecture:** Add a `PreparedExecutablePlan` layer after existing IR optimization and memory planning. The prepared plan resolves runtime-generic instructions into backend-ready prepared instructions with static shapes, activation metadata, chosen kernel kinds, constant/packed-weight handles, and scratch requirements. Start with semantics-preserving fp32 prepared conv/matmul, then add static weight prepacking, exact graph canonicalization/fusion, packed weight-only inference, activation/layout planning, and finally profile-guided kernel selection.

**Tech Stack:** Rust fastnn core, ONNX→IR converter, compiler passes, CPU backend, Python `AotExecutor`, existing YOLO harnesses (`scripts/yolo_compare_fastnn_pytorch.py`, `scripts/yolo_conv_shape_profile.py`, `scripts/conv_shape_microbench.py`), cargo tests, maturin release builds.

---

## Non-Negotiable Requirements

- Default mode must be exact fp32-preserving unless an explicit precision mode is selected.
- No lossy transform may be enabled silently.
- Every backend optimization must preserve existing tests and compare against PyTorch/ONNX Runtime where applicable.
- Packed/quantized paths must be opt-in at first.
- Keep fallback to existing generic execution until a prepared op is verified.
- Do not YOLO-specialize the compiler. Use YOLO as a benchmark, not as hardcoded logic.
- Commit after each verified task or small task group.

## Success Metrics

Stage 1 success:
- Existing tests pass.
- YOLO output unchanged.
- PreparedPlan path can execute at least Conv2d/Conv2d+Silu through current kernels.
- No meaningful runtime regression.

Stage 2 success:
- Static Conv/MatMul weights are detected and represented as prepared handles.
- Packed/prepared fp32 weight path improves at least one `conv_shape_microbench.py` shape without full-YOLO regression.

Stage 3 success:
- Exact graph canonicalization/folding reduces runtime ops on YOLO and other ONNX graphs without numerical changes.

Stage 4 success:
- Weight-only packed precision path exists for Conv/MatMul behind explicit mode.
- Memory usage decreases measurably.
- Accuracy deltas are reported and gated.

Stage 5 success:
- Compile/load-time kernel selection chooses between safe prepared kernels using shape metadata and conservative thresholds.
- Full YOLO improves over current fastnn baseline.

---

## Phase 0: Baseline and Safety Harness

### Task 0.1: Capture Baseline Commands and Expected Metrics

**Objective:** Establish repeatable baseline verification before architectural work.

**Files:**
- Read/use: `scripts/yolo_compare_fastnn_pytorch.py`
- Read/use: `scripts/yolo_conv_shape_profile.py`
- Read/use: `scripts/conv_shape_microbench.py`
- Create: `docs/plans/prepared-plan-baseline.md`

**Steps:**
1. Run:
   ```bash
   cargo test --release --lib
   .venv/bin/python -m maturin develop --release
   .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 12 --warmup 3 --iters 8
   .venv/bin/python scripts/conv_shape_microbench.py --warmup 10 --iters 40 --threads 1
   ```
2. Save the summary numbers to `docs/plans/prepared-plan-baseline.md`.
3. Include:
   - PyTorch mean/median
   - ONNX Runtime mean/median
   - fastnn mean/median
   - max_abs/mean_abs
   - top profile kernels
   - top microbench shape ratios
4. Commit:
   ```bash
   git add docs/plans/prepared-plan-baseline.md
   git commit -m "docs: capture prepared plan baseline"
   ```

**Acceptance:** Baseline file exists and commands completed successfully.

---

### Task 0.2: Add a PreparedPlan Feature Flag

**Objective:** Allow PreparedPlan work without destabilizing default execution.

**Files:**
- Modify: `Cargo.toml`

**Steps:**
1. Add feature:
   ```toml
   prepared-plan = []
   ```
2. Do not add it to `default` yet.
3. Run:
   ```bash
   cargo test --release --lib
   cargo test --release --lib --features prepared-plan
   ```
4. Commit:
   ```bash
   git add Cargo.toml
   git commit -m "feat: add prepared plan feature flag"
   ```

**Acceptance:** Both default and `prepared-plan` test runs pass.

---

## Phase 1: PreparedPlan Skeleton

### Task 1.1: Create PreparedPlan Module

**Objective:** Add data structures for a prepared executable plan without wiring execution yet.

**Files:**
- Create: `src/backend/prepared.rs`
- Modify: `src/backend/mod.rs`

**Implementation Sketch:**

```rust
// src/backend/prepared.rs
use crate::backend::BufferSlice;
use crate::ir::node::NodeId;

#[derive(Clone, Debug)]
pub struct PreparedExecutablePlan {
    pub instructions: Vec<PreparedInstruction>,
    pub arena_size: usize,
    pub scratch_size: usize,
}

#[derive(Clone, Debug)]
pub enum PreparedInstruction {
    Generic {
        instruction_index: usize,
    },
    Conv2d(PreparedConv2d),
    MatMul(PreparedMatMul),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedActivation {
    None,
    Relu,
    Gelu,
    Silu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedConvKernelKind {
    CurrentIm2colGemm,
    CurrentOneByOneGemm,
    FuturePackedFp32,
    FuturePackedI8,
    FuturePackedU4,
}

#[derive(Clone, Debug)]
pub struct PreparedConv2d {
    pub node_id: Option<NodeId>,
    pub input: BufferSlice,
    pub weight: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub f: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub activation: PreparedActivation,
    pub kernel: PreparedConvKernelKind,
    pub packed_weight: Option<PackedWeightId>,
    pub scratch_offset: usize,
    pub scratch_len: usize,
}

#[derive(Clone, Debug)]
pub struct PreparedMatMul {
    pub node_id: Option<NodeId>,
    pub a: BufferSlice,
    pub b: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub activation: PreparedActivation,
    pub packed_weight: Option<PackedWeightId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedWeightId(pub usize);
```

2. Export from `src/backend/mod.rs`:
   ```rust
   pub mod prepared;
   ```
3. Run:
   ```bash
   cargo test --release --lib --features prepared-plan
   ```
4. Commit:
   ```bash
   git add src/backend/prepared.rs src/backend/mod.rs
   git commit -m "feat(backend): add prepared plan data structures"
   ```

**Acceptance:** New module compiles; no behavior changes.

---

### Task 1.2: Add PreparedPlan Builder Stub

**Objective:** Build a prepared plan that initially wraps every existing instruction as `Generic`.

**Files:**
- Modify: `src/backend/prepared.rs`
- Modify if needed: `src/backend/executor.rs`

**Implementation Sketch:**

```rust
pub fn prepare_executable_plan(
    plan: &crate::backend::ExecutablePlan,
) -> PreparedExecutablePlan {
    PreparedExecutablePlan {
        instructions: (0..plan.instructions.len())
            .map(|instruction_index| PreparedInstruction::Generic { instruction_index })
            .collect(),
        arena_size: plan.arena_size,
        scratch_size: 0,
    }
}
```

**Tests:**
- Add a unit test that constructs/uses a tiny executable plan if existing constructors allow it.
- If constructors are awkward, test through a small AOT model in Python later.

**Commands:**
```bash
cargo test --release --lib --features prepared-plan
```

**Commit:**
```bash
git add src/backend/prepared.rs src/backend/executor.rs
git commit -m "feat(backend): build generic prepared plans"
```

**Acceptance:** A prepared plan can be created for an existing executable plan.

---

### Task 1.3: Store PreparedPlan in Python AotExecutor Behind Feature Flag

**Objective:** Make `AotExecutor` hold a prepared plan without using it yet.

**Files:**
- Modify: `src/python/nn.rs`

**Implementation Sketch:**

```rust
pub struct AotExecutor {
    plan: crate::backend::ExecutablePlan,
    #[cfg(feature = "prepared-plan")]
    prepared_plan: crate::backend::prepared::PreparedExecutablePlan,
    memory_plan: crate::compiler::passes::memory_planning::MemoryPlan,
    graph: crate::ir::node::ComputeGraph,
    executor: crate::backend::GraphExecutor,
    input_names: Vec<String>,
    output_names: Vec<String>,
    output_map: Vec<(String, usize)>,
}
```

At construction:
```rust
#[cfg(feature = "prepared-plan")]
let prepared_plan = crate::backend::prepared::prepare_executable_plan(&plan);
```

**Commands:**
```bash
cargo test --release --lib
cargo test --release --lib --features prepared-plan
.venv/bin/python -m maturin develop --release
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --warmup 1 --iters 1
```

**Commit:**
```bash
git add src/python/nn.rs
git commit -m "feat(python): store prepared plan in AOT executor"
```

**Acceptance:** Default and feature builds pass; YOLO still runs using old path.

---

## Phase 2: Prepared Execution Fallback Path

### Task 2.1: Add Prepared Execution Entry Point

**Objective:** Add `GraphExecutor::execute_prepared` that currently delegates every `Generic` instruction to the old executor logic.

**Files:**
- Modify: `src/backend/executor.rs`
- Modify: `src/backend/prepared.rs`

**Approach:**
- Do not duplicate the full executor yet.
- Add a method that iterates prepared instructions.
- For `PreparedInstruction::Generic { instruction_index }`, execute the corresponding original instruction.
- If old executor internals are not easily callable per instruction, first add a private helper `execute_instruction(...)` extracted from the existing loop.

**Steps:**
1. Extract the existing per-instruction dispatch body into:
   ```rust
   fn execute_instruction_at(
       &mut self,
       graph: &ComputeGraph,
       plan: &mut ExecutablePlan,
       memory_plan: &MemoryPlan,
       input_refs: &[&[u8]],
       instruction_index: usize,
   ) -> Result<(), BackendError>
   ```
   Adjust exact signature to match existing code.
2. Keep existing `execute` using the helper.
3. Add `execute_prepared` using the helper for `Generic`.
4. Add `execute_profile_prepared` only after non-profile version is stable.

**Commands:**
```bash
cargo test --release --lib --features prepared-plan
```

**Acceptance:** Old tests pass; no behavior change.

---

### Task 2.2: Route AotExecutor Through Prepared Execution Behind Feature Flag

**Objective:** Make `AotExecutor.forward()` use prepared execution when `prepared-plan` is enabled.

**Files:**
- Modify: `src/python/nn.rs`
- Modify: `src/backend/executor.rs`

**Steps:**
1. In `AotExecutor.forward`, under `#[cfg(feature = "prepared-plan")]`, call `execute_prepared`.
2. Under default build, keep existing `execute`.
3. Rebuild with feature:
   ```bash
   .venv/bin/python -m maturin develop --release --features prepared-plan
   ```
4. Verify:
   ```bash
   .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --warmup 3 --iters 8
   ```
5. Compare max_abs and mean_abs against baseline.

**Commit:**
```bash
git add src/python/nn.rs src/backend/executor.rs
git commit -m "feat(backend): execute AOT models through prepared fallback path"
```

**Acceptance:** Prepared fallback path produces identical output.

---

## Phase 3: Prepared Conv Metadata

### Task 3.1: Detect Conv2d Instructions During Preparation

**Objective:** Convert eligible conv instructions from `Generic` into `PreparedInstruction::Conv2d`, but still execute them through old kernel helper.

**Files:**
- Modify: `src/backend/prepared.rs`
- Read/modify as needed: `src/backend/executor.rs`
- Read: `src/backend/cpu/mod.rs`

**Steps:**
1. Inspect `ExecutablePlan` instruction representation.
2. Identify Conv2d opcodes and input/output `BufferSlice`s.
3. Extract static conv params:
   - n/c/h/w/f/kh/kw
   - stride/padding/dilation/groups
   - fused activation from `attrs["fused_op"]`
4. Select kernel kind:
   ```rust
   if kh == 1 && kw == 1 && stride == 1 && padding == 0 && dilation == 1 && groups == 1 {
       CurrentOneByOneGemm
   } else {
       CurrentIm2colGemm
   }
   ```
5. If anything is dynamic/unsupported, leave `Generic`.

**Tests:**
- Add unit test with a tiny Conv graph where preparation yields one `PreparedInstruction::Conv2d`.
- Add negative test for unsupported/dynamic case if possible.

**Commands:**
```bash
cargo test --release --lib prepared --features prepared-plan
cargo test --release --lib --features prepared-plan
```

**Commit:**
```bash
git add src/backend/prepared.rs
 git commit -m "feat(backend): prepare conv2d instruction metadata"
```

**Acceptance:** Conv metadata appears in prepared plan; execution still unchanged.

---

### Task 3.2: Execute Prepared Conv Through Existing CPU Kernel

**Objective:** Bypass generic dispatch for prepared Conv2d and call the existing CPU conv kernel with precomputed metadata.

**Files:**
- Modify: `src/backend/executor.rs`
- Modify: `src/backend/cpu/mod.rs` or create a small callable helper
- Possibly modify: `src/backend/cpu/microkernels.rs`

**Approach:**
- Extract the existing Conv2d dispatch body from `src/backend/cpu/mod.rs` into a reusable helper:
  ```rust
  pub fn execute_prepared_conv2d_f32(
      arena: &mut Arena,
      conv: &PreparedConv2d,
      scratch: &mut [f32],
  ) -> Result<(), BackendError>
  ```
- Use `arena::with_nary_f32_slices` like the existing optimized path.
- Keep exact current kernels and activation behavior.

**Tests:**
- Existing conv tests.
- Add Python micro model test if existing test framework supports Python integration.

**Commands:**
```bash
cargo test --release --lib --features prepared-plan
.venv/bin/python -m maturin develop --release --features prepared-plan
.venv/bin/python scripts/conv_shape_microbench.py --warmup 10 --iters 20 --threads 1
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 12 --warmup 3 --iters 8
```

**Acceptance:** Accuracy unchanged; no runtime regression beyond noise.

---

## Phase 4: Constant Arena and Static Weight Handles

### Task 4.1: Add Constant and Packed Weight Store Types

**Objective:** Introduce immutable prepared constant storage separate from runtime activation arena.

**Files:**
- Modify: `src/backend/prepared.rs`
- Possibly modify: `src/backend/mod.rs`

**Implementation Sketch:**

```rust
#[derive(Clone, Debug)]
pub struct PreparedConstants {
    pub raw_f32: Vec<PreparedRawF32>,
    pub packed_weights: Vec<PackedWeight>,
}

#[derive(Clone, Debug)]
pub struct PreparedRawF32 {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum PackedWeight {
    F32ConvIm2col(PackedF32ConvWeight),
    F32MatMul(PackedF32MatMulWeight),
    FutureI8,
    FutureU4,
}

#[derive(Clone, Debug)]
pub struct PackedF32ConvWeight {
    pub original_shape: [usize; 4],
    pub packed: Vec<f32>,
    pub layout: PackedF32ConvLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedF32ConvLayout {
    RawOihw,
    RowMajorFk,
    FutureBlocked,
}
```

**Acceptance:** Types compile; not wired yet.

---

### Task 4.2: Detect Static Weights in Prepared Conv

**Objective:** Mark Conv weights/biases as static when they come from constants.

**Files:**
- Modify: `src/backend/prepared.rs`
- Inspect: `src/backend/executor.rs`
- Inspect: ONNX parameter handling in `src/python/nn.rs`

**Steps:**
1. Determine how constants/params are represented in the executable plan and memory plan.
2. Add helper:
   ```rust
   fn is_static_buffer(slice: BufferSlice, plan: &ExecutablePlan, graph: &ComputeGraph) -> bool
   ```
3. For static Conv weight, create a `PackedWeightId` but initially store `RawOihw` copy.
4. Prepared Conv references that ID.
5. Leave non-static weights as raw `BufferSlice`.

**Tests:**
- Prepare a one-layer Conv ONNX model and assert weight is static.
- Dynamic weight graph should not be marked static.

**Acceptance:** Static Conv weights are detected correctly.

---

### Task 4.3: Stop Rewriting Immutable Constants Every Forward

**Objective:** Move constant materialization out of every inference run safely.

**Files:**
- Modify: `src/backend/executor.rs`
- Modify: `src/backend/prepared.rs`
- Modify: `src/python/nn.rs`

**Important:** This must be semantics-safe. Earlier naive `write_const` skipping was noisy and added complexity. This task must formalize immutable constant storage rather than ad-hoc skipping.

**Approach:**
- Constants/packed weights live outside activation arena.
- Prepared ops read static weights from `PreparedConstants` directly.
- Leave old `WriteConst` instructions for non-prepared/generic ops.
- Do not globally skip all `WriteConst` yet.

**Acceptance:** Prepared Conv no longer needs its weight `WriteConst` at runtime. Generic instructions still work.

---

## Phase 5: FP32 Weight Packing

### Task 5.1: Pack 1x1 Conv Weights as RowMajor F×K

**Objective:** Add first real prepared weight path for static 1x1 Conv.

**Files:**
- Modify: `src/backend/prepared.rs`
- Modify: `src/backend/cpu/microkernels.rs` or new `src/backend/cpu/prepared_conv.rs`

**Approach:**
- For 1x1 Conv weights `[F,C,1,1]`, pack into contiguous row-major `[F,K]`.
- Existing raw OIHW may already be equivalent, but the prepared path formalizes this and allows future blocked layouts.
- Prepared kernel consumes `PackedF32ConvWeight`.

**Commands:**
```bash
cargo test --release --lib --features prepared-plan
.venv/bin/python -m maturin develop --release --features prepared-plan
.venv/bin/python scripts/conv_shape_microbench.py --warmup 10 --iters 40 --threads 1
```

**Acceptance:** 1x1 microbench unchanged or improved; no accuracy loss.

---

### Task 5.2: Add Blocked FP32 Weight Layout Experiment

**Objective:** Try a blocked layout for selected Conv/MatMul shapes without affecting defaults.

**Files:**
- Modify: `src/backend/prepared.rs`
- Modify/create: `src/backend/cpu/prepared_conv.rs`

**Approach:**
- Add layout:
  ```rust
  BlockedOcK { oc_block: usize, k_block: usize }
  ```
- Start with one shape class from microbench.
- Add a feature/heuristic gate so fallback remains easy.

**Acceptance:** Keep only if microbench and full YOLO improve. Otherwise revert but record lesson.

---

## Phase 6: Exact Graph Optimization Passes

### Task 6.1: Add Constant Folding Framework

**Objective:** Evaluate constant-only subgraphs at compile time.

**Files:**
- Create: `src/compiler/passes/constant_folding.rs`
- Modify: `src/compiler/passes/mod.rs`
- Tests: same file or `tests/`

**Supported initially:**
- `Shape`
- `Gather`
- `Unsqueeze`
- `Concat`
- `Reshape` when input constant
- simple scalar arithmetic on constants

**Acceptance:** Existing YOLO shape subgraphs fold where safe. No numerical model outputs change.

---

### Task 6.2: Add No-Op Canonicalization Pass

**Objective:** Remove exact no-op graph operations.

**Files:**
- Create: `src/compiler/passes/canonicalize.rs`
- Modify: `src/compiler/passes/mod.rs`

**Rules:**
- identity `Reshape` when shape unchanged and layout unchanged
- identity `Transpose`
- adjacent inverse transposes
- `Add(x, 0)`
- `Mul(x, 1)`
- `Div(x, 1)`
- `Concat([x]) -> x`

**Tests:**
- One test per rewrite.
- Negative tests for unsafe cases.

**Acceptance:** Node count reduces; outputs unchanged.

---

### Task 6.3: Add Conv+BatchNorm Folding

**Objective:** Fold constant BatchNorm into preceding Conv weights/bias exactly under inference semantics.

**Files:**
- Create: `src/compiler/passes/fusion/conv_batchnorm.rs`
- Modify: `src/compiler/passes/fusion/mod.rs`
- Modify: `Cargo.toml` feature list if fusion passes are feature-gated

**Formula:**
For Conv output `y = conv(x, W) + b`, BN:
```text
scale = gamma / sqrt(var + eps)
W' = W * scale[:, None, None, None]
b' = (b - mean) * scale + beta
```
If Conv has no bias, use zero bias.

**Acceptance:** Exact test vs unfused graph within fp32 tolerance. No change for training-mode/dynamic BN.

---

### Task 6.4: Expand MatMul/Linear Fusions

**Objective:** Prepare transformer/MLP workloads, not just Conv models.

**Files:**
- Existing/new files under `src/compiler/passes/fusion/`
- `src/backend/cpu/mod.rs`

**Patterns:**
- MatMul + Add -> Linear
- MatMul + Add + Relu/Gelu/Silu -> fused Linear activation
- Gemm ONNX canonicalization to MatMul+Bias or Linear

**Acceptance:** Unit tests and at least one synthetic ONNX model benchmark.

---

## Phase 7: Weight-Only Packed Precision

### Task 7.1: Define Explicit Precision Mode

**Objective:** Add a user-visible opt-in for packed precision.

**Files:**
- Modify: `src/python/nn.rs`
- Modify: `src/backend/prepared.rs`

**API Sketch:**
```python
executor = fastnn.AotExecutor.from_onnx(
    path,
    params,
    precision="fp32",        # default
)
executor = fastnn.AotExecutor.from_onnx(
    path,
    params,
    precision="weight_i8",   # opt-in
)
```

If current constructor does not support keyword args, add a new constructor or config object.

**Acceptance:** Default behavior unchanged. Invalid precision mode errors clearly.

---

### Task 7.2: Implement I8 Weight-Only Packing for MatMul

**Objective:** Start packed precision on MatMul first because it is simpler than Conv.

**Files:**
- Modify: `src/backend/prepared.rs`
- Create: `src/backend/cpu/packed_matmul.rs`

**Quantization:**
- symmetric per-output-channel or per-row scales
- store int8 weights
- fp32 activations
- accumulate to i32/f32 depending kernel design
- dequantize to fp32 output

**Acceptance:** Synthetic MatMul tests show bounded error and memory reduction.

---

### Task 7.3: Implement I8 Weight-Only Packing for 1x1 Conv

**Objective:** Extend MatMul packing to 1x1 Conv.

**Files:**
- Create/modify: `src/backend/cpu/packed_conv.rs`

**Acceptance:** `conv_shape_microbench.py` includes packed mode and reports speed/memory/error.

---

### Task 7.4: Implement I8 Weight-Only Packing for Im2col Conv

**Objective:** Support 3x3 Conv in packed weight-only mode.

**Files:**
- Modify: `src/backend/cpu/packed_conv.rs`
- Modify: `src/backend/prepared.rs`

**Acceptance:** Improves or at least reduces memory on top YOLO 3x3 shapes with bounded error.

---

### Task 7.5: Add U4/NF4 Packed Weight Format

**Objective:** Add the memory-efficiency path fastnn is aiming for.

**Files:**
- Existing packed dtype code, likely under `src/dtypes/`
- Modify: `src/backend/prepared.rs`
- Modify: `src/backend/cpu/packed_matmul.rs`
- Modify: `src/backend/cpu/packed_conv.rs`

**Approach:**
- Start with weight-only U4.
- Use per-channel/group scales.
- Keep activations fp32.
- Add explicit mode only: `precision="weight_u4"`.

**Acceptance:** Memory drop is measured; speed/error reported honestly.

---

## Phase 8: Layout and Activation Planning

### Task 8.1: Add Internal Layout Metadata

**Objective:** Let prepared ops express layouts beyond plain NCHW.

**Files:**
- Modify: `src/backend/prepared.rs`

**Types:**
```rust
pub enum TensorLayout {
    Nchw,
    Nhwc,
    BlockedNchw { block_c: usize },
    PackedWeight(PackedWeightId),
}
```

**Acceptance:** Metadata only; no behavior change.

---

### Task 8.2: Add Layout Conversion Planning

**Objective:** Insert/avoid layout conversions at compile time instead of ad-hoc runtime transforms.

**Approach:**
- Start metadata-only.
- Then support one real conversion if needed.
- Never enable a layout if it causes extra conversions that outweigh gains.

**Acceptance:** No default behavior change until explicitly enabled.

---

### Task 8.3: Add Activation Precision Planning

**Objective:** Allow future internal BF16/F16/I8 activations while keeping default fp32.

**Approach:**
- Add dtype/layout propagation metadata.
- Insert quantize/dequantize nodes only in explicit packed modes.
- Add no-op elimination for redundant dequant/quant pairs.

**Acceptance:** Metadata passes and explicit packed mode tests.

---

## Phase 9: Compile-Time Kernel Selection and Auto-Tuning

### Task 9.1: Add Kernel Selection Table

**Objective:** Centralize shape-based kernel choice.

**Files:**
- Create: `src/backend/cpu/kernel_selection.rs`
- Modify: `src/backend/prepared.rs`

**Function:**
```rust
pub fn choose_conv_kernel(shape: &PreparedConvShape, precision: PrecisionMode) -> PreparedConvKernelKind
```

**Rules Initially:**
- conservative current paths only
- no BLAS unless a shape is verified
- no direct tiled fallback for YOLO low-channel 3x3; known regression

**Acceptance:** Same kernel choices as today.

---

### Task 9.2: Add Offline Kernel Benchmark Cache Format

**Objective:** Prepare for profile-guided selection without forcing auto-tuning every load.

**Files:**
- Create: `src/backend/cpu/kernel_cache.rs`

**Format:**
- CPU model / feature flags
- fastnn version
- shape key
- precision mode
- winning kernel kind
- timestamp

**Acceptance:** Can read/write cache. Not used by default yet.

---

### Task 9.3: Add Optional Load-Time Auto-Tuning

**Objective:** Benchmark candidate kernels for static shapes at model load when requested.

**API Sketch:**
```python
executor = fastnn.AotExecutor.from_onnx(path, params, autotune=True)
```

**Rules:**
- Opt-in only.
- Cache results.
- Validate candidate output for one random input if feasible.

**Acceptance:** Autotune chooses a kernel for at least one microbench shape.

---

## Phase 10: Verification and Reporting

### Task 10.1: Add PreparedPlan Debug Dump

**Objective:** Make prepared optimization visible and debuggable.

**Files:**
- Modify: `src/python/nn.rs`
- Modify: `src/backend/prepared.rs`

**API:**
```python
executor.debug_prepared_plan()
```

Output fields:
- op count by prepared instruction kind
- conv kernel kind counts
- static packed weight count
- packed precision mode
- scratch bytes
- constant bytes
- activation arena bytes

**Acceptance:** CLI/Python can print summary for YOLO.

---

### Task 10.2: Add Regression Benchmark Script

**Objective:** One command to compare baseline vs prepared/packed modes.

**Files:**
- Create: `scripts/prepared_plan_regression.py`

**Runs:**
- default fp32
- prepared fp32
- optional packed mode
- YOLO compare
- conv microbench

**Acceptance:** Emits JSON and plain text summary.

---

### Task 10.3: Documentation

**Objective:** Explain how to use and validate prepared/packed inference.

**Files:**
- Create/modify: `docs/prepared-inference.md`

**Include:**
- what is exact by default
- what is opt-in/lossy
- how to benchmark
- how to read debug dump
- known limitations

**Acceptance:** Docs match current API and commands run.

---

## Recommended Execution Order

1. Phase 0: baseline + feature flag
2. Phase 1: PreparedPlan skeleton
3. Phase 2: prepared fallback execution
4. Phase 3: prepared Conv metadata + execution through existing kernel
5. Phase 4: static constant/weight handles
6. Phase 5: fp32 prepared weight packing
7. Phase 6: exact graph optimization passes
8. Phase 7: weight-only packed precision
9. Phase 8: layout/activation planning
10. Phase 9: auto-tuning
11. Phase 10: reporting/docs

## First Implementation Slice to Start Immediately

The first PR should be deliberately boring:

- add `prepared-plan` feature
- add `src/backend/prepared.rs`
- define `PreparedExecutablePlan` and `PreparedInstruction`
- build a generic prepared plan wrapping old instructions
- store it in `AotExecutor` under feature flag
- do not change runtime behavior

This de-risks the architecture. After that, we can migrate Conv2d one piece at a time.

## Verification Gates for Every PR

Run before each commit that touches Rust runtime code:

```bash
cargo fmt --check
cargo test --release --lib
.venv/bin/python -m maturin develop --release
```

If the PR affects prepared-plan feature:

```bash
cargo test --release --lib --features prepared-plan
.venv/bin/python -m maturin develop --release --features prepared-plan
```

If the PR affects Conv/MatMul performance:

```bash
.venv/bin/python scripts/conv_shape_microbench.py --warmup 10 --iters 40 --threads 1
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 12 --warmup 3 --iters 8
```

Accuracy gate for default fp32:
- no meaningful drift from baseline
- YOLO max_abs should remain around current accepted range (~5e-4 vs PyTorch export path)
- mean_abs should remain around ~1e-6

Packed precision gate:
- explicit precision mode only
- report max_abs/mean_abs
- report model memory bytes
- report speed
- do not compare as “lossless”

## Known Rejected Paths to Avoid Repeating

- Global OpenBLAS conv routing: regressed YOLO shapes.
- Broad low-F alternative GEMM orientation (`C[spatial,f]` + scatter): regressed early 3x3 shapes.
- Direct tiled fallback for YOLO stem/early 3x3: severe regression.
- AVX row-SiLU called from conv rows: regressed speed/numerics in this setup.
- Ad-hoc skipping all `write_const`: profile looked cleaner but end-to-end was noisy/regressed and semantics were not formalized.

## Strategic Rationale

The plan is not “add optimizations randomly.” It is to make fastnn behave like an inference compiler:

- spend more time at compile/load
- exploit static graph knowledge
- prepack constant weights
- select shape-specific kernels
- enable packed precision cleanly
- keep default fp32 exact

This is the path that can close enough of the PyTorch fp32 gap while setting up the real fastnn advantage: packed precision memory and performance efficiency.
