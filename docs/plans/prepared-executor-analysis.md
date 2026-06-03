# Prepared Executor Fallback Analysis

> Read-only design analysis for Phase 2 of the prepared-plan implementation.
> No source code changes — only extraction point identification and API design.

---

## 1. Current Executor Control Flow Summary

The execution pipeline has four distinct stages:

```
Python AotExecutor.forward()
  → GraphExecutor::execute()
    → tighten + arena management
    → CpuBackend::dispatch()
      → per-instruction kernel dispatch loop
```

### Stage 1: Python Entry (`src/python/nn.rs:1218-1356`)

`AotExecutor::forward()`:
1. Collects input byte slices from a `HashMap<String, PyTensor>` in `input_names` order.
2. Calls `self.executor.execute(&self.graph, &mut self.plan, &self.memory_plan, &input_refs)`.
3. Converts output `Vec<Vec<u8>>` back to `PyTensor` using graph output node dtype/shape metadata.

### Stage 2: GraphExecutor::execute (`src/backend/executor.rs:139-258`)

1. **Shape environment**: `ShapeEnv::from_graph_inputs(graph, inputs)` resolves symbolic dims.
2. **Shape validation**: `validate_shapes(graph, &shape_env)` checks dim compatibility for all ops.
3. **Memory tightening**: `memory_plan.tighten(graph, &shape_env)` shrinks slots from `SYMBOL_DIM_MAX` worst-case to actual sizes.
4. **Slice tightening**: `tighten_slices(plan, memory_plan, &tightened_memory_plan, graph)` updates every `BufferSlice` offset/size and `CallKernel` params in-place.
5. **Safety check**: Verifies each node's resolved output fits in its tightened slot.
6. **Arena management**: Reuses cached arena if capacity sufficient, otherwise allocates new.
7. **Input writing**: Copies each input bytes into the arena at its tightened slot.
8. **Backend dispatch**: `self.backend.dispatch(plan, arena, &shape_env)`.
9. **Output reading**: Reads output bytes from arena slots using resolved sizes.

### Stage 3: CpuBackend::dispatch (`src/backend/cpu/mod.rs:3372-end`)

Iterates `plan.instructions` sequentially. For each instruction:
- `CallKernel`: Matches on `kernel_name` string → dispatches to inline kernel logic.
- `MemCopy`: `data.copy_within(src.., dst)`.
- `Fill`: Casts slice to `&mut [f32]` → `.fill(value)`.
- `WriteConst`: Copies `data` bytes into arena at `dst.offset`.

Each `CallKernel` arm typically: (a) pre-copies inputs from arena, (b) calls the kernel function, (c) writes output.

### Stage 4: Kernel dispatch patterns

Two dispatch patterns exist:
- **Arena-based** (in `dispatch()`): Direct `arena.data_mut()` access per-instruction.
- **Pre-copied** (in `execute_parallel_level()` and `run_kernel_precopied()`): Inputs copied to `Vec<Vec<u8>>` first, then dispatched. This is the parallel-safe path.

---

## 2. Exact Functions/Structs Involved

### Structs

| Struct | Location | Role |
|--------|----------|------|
| `GraphExecutor<B: Backend>` | `executor.rs:33-36` | Holds backend + cached arena. Owns `execute()` and `compile()`. |
| `ExecutablePlan` | `mod.rs:96-103` | `instructions: Vec<Instruction>`, `arena_size`, `levels: Vec<usize>`. |
| `Instruction` (enum) | `mod.rs:57-93` | `CallKernel{...}`, `MemCopy`, `Fill`, `WriteConst`. |
| `MemoryPlan` | (re-exported from `memory_planning`) | `slots: HashMap<NodeId, Slot>`, `tightened_params`, `total_size`. |
| `CpuBackend` | `cpu/mod.rs:1366` | Zero-field unit struct implementing `Backend`. |
| `CpuBuffer` | `cpu/mod.rs:1337` | `UnsafeCell<Vec<u8>>` arena. |
| `AotExecutor` | `python/nn.rs:1111-1118` | Python-facing wrapper. Holds `plan`, `memory_plan`, `graph`, `executor`, `input_names`, `output_map`. |

### Key Functions

| Function | Location | Signature (simplified) |
|----------|----------|----------------------|
| `GraphExecutor::execute` | `executor.rs:139` | `(&mut self, graph, plan, memory_plan, inputs) -> Result<Vec<Vec<u8>>>` |
| `tighten_slices` | `executor.rs:568` | `(plan, original_mp, tightened_mp, graph) -> Result<()>` |
| `CpuBackend::dispatch` | `cpu/mod.rs:3372` | `(&self, plan, arena, shape_env) -> Result<()>` |
| `execute_parallel_level` | `cpu/mod.rs:1158` | `(group, plan, arena, shape_env) -> Result<()>` |
| `run_kernel_precopied` | `cpu/mod.rs:362` | `(kernel_name, inputs, output, secondary, params, param_dims, weight_meta, shape_env) -> Result<()>` |
| `AotExecutor::forward` | `python/nn.rs:1218` | `(&mut self, inputs: HashMap) -> Result<HashMap>` |

---

## 3. Recommended Extraction Points for `execute_instruction_at`

### Primary extraction: single-instruction dispatch in `dispatch()`

The `dispatch()` method at `cpu/mod.rs:3407` contains a `for` loop over `plan.instructions`. The loop body is ~1000 lines of match arms. This is the natural extraction boundary.

**Proposed helper:**

```rust
/// Execute a single instruction against the arena.
/// Extracted from the dispatch loop body to enable per-instruction
/// dispatch for prepared execution and profiling.
fn execute_instruction(
    instr_idx: usize,
    instr: &Instruction,
    plan: &ExecutablePlan,
    arena: &CpuBuffer,
    shape_env: &ShapeEnv,
) -> Result<(), BackendError> {
    match instr {
        Instruction::CallKernel { kernel_name, input_slices, output_slice, ... } => {
            // existing match-on-kernel_name dispatch logic
        }
        Instruction::MemCopy { dst, src } => { ... }
        Instruction::Fill { dst, value } => { ... }
        Instruction::WriteConst { dst, data } => { ... }
    }
}
```

**Why this boundary:**
- The loop body is already self-contained per-instruction.
- The `arena` is borrowed mutably per-instruction (no aliasing between instructions at the same level).
- `shape_env` is read-only.
- Extracting this function lets both `dispatch()` and a future `execute_prepared()` share the same kernel dispatch logic.

**What NOT to extract:**
- Do not try to extract the parallel level grouping logic (`build_level_groups`, `execute_parallel_level`). The prepared execution path should initially run sequentially; parallelism can be added later by reusing `execute_parallel_level`.
- Do not try to extract the input pre-copy loop from `execute_parallel_level` into the shared helper — that's specific to the parallel path.

### Secondary extraction: per-instruction metadata resolution

For prepared execution, we need a variant that dispatches based on `PreparedInstruction` kind rather than `Instruction` kind. The cleanest approach is a **match on prepared instruction kind** that delegates to the same `execute_instruction` helper for `Generic` variants, and to dedicated helpers for `Conv2d`/`MatMul` variants.

---

## 4. Minimal API Sketch

### `execute_prepared`

```rust
impl<B: Backend> GraphExecutor<B> {
    /// Execute a compiled plan through the prepared instruction dispatch path.
    ///
    /// For `PreparedInstruction::Generic { instruction_index }`, delegates
    /// to the existing instruction dispatch. For specialized prepared
    /// instructions, dispatches through prepared-specific kernel helpers.
    pub fn execute_prepared(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        prepared_plan: &PreparedExecutablePlan,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        // 1. Shape env, validation, tightening — identical to execute()
        // 2. Arena allocation — identical
        // 3. Input writing — identical
        // 4. NEW: iterate prepared_plan.instructions
        //    - Generic { instruction_index } → execute_instruction(plan.instructions[instruction_index], ...)
        //    - Conv2d(c) → execute_prepared_conv2d(c, arena, ...)
        //    - MatMul(m) → execute_prepared_matmul(m, arena, ...)
        // 5. Output reading — identical to execute()
    }
}
```

**Key design decision:** The prepared execution path reuses all of the existing `execute()` preamble (shape env, tightening, arena, I/O). Only the dispatch loop differs. This means the method body is ~80% identical to `execute()` with the dispatch loop replaced. To avoid duplication, extract the preamble into a shared helper:

```rust
/// Shared setup for execute/execute_prepared: shape env, tightening, arena, input write.
/// Returns (tightened_memory_plan, arena) for the caller to dispatch and read outputs.
fn execute_setup(
    &mut self,
    graph: &ComputeGraph,
    plan: &mut ExecutablePlan,
    memory_plan: &MemoryPlan,
    inputs: &[&[u8]],
) -> Result<(MemoryPlan, /* arena ref handled internally */), BackendError>
```

However, since the arena is `&self.cached_arena` (borrowed from `&mut self`), the cleanest approach is to have `execute_prepared` duplicate the setup code and only diverge at the dispatch loop. This is acceptable for Phase 2; the setup is ~40 lines and unlikely to change.

### `execute_profile_prepared`

```rust
/// Profile entry for a single instruction.
pub struct ProfileEntry {
    pub instruction_index: usize,
    pub node_id: Option<NodeId>,
    pub kernel_name: String,
    pub elapsed_ns: u64,
}

impl<B: Backend> GraphExecutor<B> {
    /// Like execute_prepared, but times each instruction and returns profiling data.
    pub fn execute_profile_prepared(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        prepared_plan: &PreparedExecutablePlan,
        inputs: &[&[u8]],
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        // Same setup as execute_prepared
        // For each instruction:
        //   let t0 = Instant::now();
        //   execute_instruction(...)  // or prepared dispatch
        //   let elapsed = t0.elapsed().as_nanos() as u64;
        //   profile_entries.push(ProfileEntry { ... });
        // Read outputs, return (outputs, profile_entries)
    }
}
```

**Note:** The `execute_profile` method referenced in `python/nn.rs:1374` does **not yet exist** in `executor.rs`. This must be implemented as part of Phase 2 or as a prerequisite. The `ProfileEntry` struct and `execute_profile` are needed regardless of the prepared path.

---

## 5. Risk List

### 5.1 Profiling Parity

**Risk:** `execute_profile` is called in `python/nn.rs:1374` but does not exist in `executor.rs`. The code does not currently compile.

**Mitigation:** Phase 2 must implement `execute_profile` (the non-prepared version) as a prerequisite. Alternatively, this may have been intended to be added by the prepared-core branch. Verify with the prepared-core implementer.

**Impact:** High — blocks compilation of the Python module.

### 5.2 Memory Arena Mutation

**Risk:** The arena uses `UnsafeCell<Vec<u8>>` for interior mutability. The current `dispatch()` processes instructions sequentially, and `execute_parallel_level` uses pre-copied inputs + exclusive output slices for parallel safety. A prepared execution path must maintain the same invariant: no two concurrent writers to the same arena region.

**Mitigation:** Phase 2 prepared execution runs sequentially (no parallelism). Prepared `Conv2d`/`MatMul` kernels receive `&mut [u8]` slices from the arena, same as existing kernels. No new aliasing is introduced.

**Impact:** Low — sequential prepared execution is safe by construction.

### 5.3 Input/Output Handling

**Risk:** The prepared plan may eventually reference packed weights from `PreparedConstants` rather than from the arena. In Phase 2, all `Generic` instructions still use arena-resolved slices. But prepared `Conv2d`/`MatMul` instructions may reference `BufferSlice`s that point to constant data in the arena (from `WriteConst` instructions). If we later skip `WriteConst` for prepared ops (Phase 4), we must ensure the packed weight store is separate.

**Mitigation:** Phase 2 does not skip any `WriteConst`. All prepared ops still read from the arena. Phase 4 introduces `PreparedConstants` separately. No conflict in Phase 2.

**Impact:** Low in Phase 2; medium in Phase 4.

### 5.4 Dynamic Shapes

**Risk:** The prepared plan is built at compile time with potentially symbolic dimensions. `PreparedConv2d` stores concrete `n/c/h/w/f/kh/kw` etc. If the graph has dynamic spatial dims (e.g., variable H/W), the prepared metadata would be stale.

**Mitigation:** Phase 2 `PreparedInstruction::Generic` carries no shape metadata — it delegates to the original instruction which already handles param_dims/shape_env resolution. Only future `PreparedInstruction::Conv2d` with all-static shapes would be promoted out of `Generic`. The builder must leave any dynamic-dim conv as `Generic`.

**Impact:** Medium — the prepared builder must be conservative about which instructions to promote.

### 5.5 Tighten_slices Interaction

**Risk:** `tighten_slices()` modifies `plan.instructions` in-place (updating `BufferSlice` offsets/sizes and `CallKernel` params). The prepared plan's `instruction_index` must reference the **post-tightened** instruction indices. Since `tighten_slices` does not reorder or remove instructions (only updates fields in-place), index mapping is stable.

**Mitigation:** `prepared_plan.instructions[i].Generic.instruction_index` maps directly to `plan.instructions[index]` after tightening. No remapping needed.

**Impact:** Low — index stability is guaranteed by `tighten_slices`.

### 5.6 ProfileEntry struct naming

**Risk:** The `python/nn.rs:1380-1388` code expects fields `instruction_index`, `node_id`, `kernel_name`, `elapsed_ns`, `node_name` on the profile entry. The `ProfileEntry` struct must match these exactly.

**Mitigation:** Define `ProfileEntry` with these fields from the start.

**Impact:** Low — straightforward struct definition.

---

## 6. Bite-Sized Implementation Sequence for Phase 2

### Step 2.0: Implement `execute_profile` (non-prepared) — prerequisite

The `AotExecutor::profile()` in `python/nn.rs:1358-1395` already calls `execute_profile`, which does not exist. This must be implemented first.

**Files:** `src/backend/executor.rs`

**Actions:**
1. Define `ProfileEntry` struct:
   ```rust
   pub struct ProfileEntry {
       pub instruction_index: usize,
       pub node_id: Option<NodeId>,
       pub kernel_name: String,
       pub elapsed_ns: u64,
   }
   ```
2. Add `execute_profile()` to `GraphExecutor` that wraps the `execute()` logic but times each instruction in the dispatch loop.
3. The cleanest approach: add a `dispatch_profiled()` method to `CpuBackend` (or a standalone function) that returns `Vec<ProfileEntry>`.

**Test:** `cargo test --release --lib` passes; `python -c "import fastnn"` works.

### Step 2.1: Extract `execute_instruction` helper from `CpuBackend::dispatch()`

**Files:** `src/backend/cpu/mod.rs`

**Actions:**
1. Extract the loop body of `dispatch()` (the `match instr { ... }` block at `cpu/mod.rs:3408`) into a standalone function:
   ```rust
   fn execute_single_instruction(
       instr: &Instruction,
       plan: &ExecutablePlan,
       arena: &CpuBuffer,
       shape_env: &ShapeEnv,
   ) -> Result<(), BackendError>
   ```
2. Refactor `dispatch()` to call this helper in its loop.
3. Verify `dispatch()` behavior is unchanged.

**Test:** `cargo test --release --lib` passes with no behavior change.

### Step 2.2: Add `execute_prepared` entry point (Generic-only)

**Files:** `src/backend/executor.rs`, `src/backend/prepared.rs`

**Actions:**
1. Add `execute_prepared()` to `GraphExecutor` that:
   - Runs the same setup as `execute()` (shape env, tightening, arena, input writing).
   - Iterates `prepared_plan.instructions`.
   - For `Generic { instruction_index }`: calls `execute_single_instruction(plan.instructions[instruction_index], ...)`.
   - For `Conv2d`/`MatMul`: panics or returns error (not yet implemented).
2. Add `execute_profile_prepared()` similarly, wrapping each instruction dispatch in timing.
3. Feature-gate behind `#[cfg(feature = "prepared-plan")]`.

**Test:** `cargo test --release --lib --features prepared-plan` passes.

### Step 2.3: Route `AotExecutor` through prepared execution

**Files:** `src/python/nn.rs`

**Actions:**
1. In `AotExecutor::forward()`, under `#[cfg(feature = "prepared-plan")]`, call `execute_prepared()` instead of `execute()`.
2. In `AotExecutor::profile()`, under `#[cfg(feature = "prepared-plan")]`, call `execute_profile_prepared()`.
3. Under default build, keep existing `execute()` path.

**Test:**
```bash
cargo test --release --lib
cargo test --release --lib --features prepared-plan
python -m maturin develop --release
python scripts/yolo_compare_fastnn_pytorch.py --warmup 3 --iters 8
```

**Acceptance:** Prepared fallback path produces identical output.

---

## 7. Expected Conflicts with prepared-core

### 7.1 `src/backend/prepared.rs` — Module creation

**Conflict:** prepared-core Phase 1 (Task 1.1) creates `src/backend/prepared.rs` with `PreparedExecutablePlan` and `PreparedInstruction`. Phase 2 modifies this file to add execution helpers.

**Resolution:** Phase 2 should **not** add execution helpers to `prepared.rs`. Instead, execution helpers belong in `executor.rs` (the `execute_prepared` method) and `cpu/mod.rs` (the `execute_single_instruction` function). The `prepared.rs` module remains data-structure-only.

### 7.2 `src/python/nn.rs` — AotExecutor struct

**Conflict:** prepared-core Phase 1 (Task 1.3) adds `#[cfg(feature = "prepared-plan")] prepared_plan: PreparedExecutablePlan` to `AotExecutor`. Phase 2 routes `forward()` through the prepared path.

**Resolution:** No conflict if Phase 2 is implemented after Phase 1 lands. The `prepared_plan` field is already present; Phase 2 just uses it in `forward()`.

### 7.3 `src/backend/mod.rs` — Module exports

**Conflict:** prepared-core adds `pub mod prepared;` to `mod.rs`. Phase 2 doesn't change `mod.rs`.

**Resolution:** No conflict.

### 7.4 `src/backend/executor.rs` — execute_profile

**Conflict:** The `execute_profile` method is called in `nn.rs:1374` but does not exist. This may be something prepared-core intends to add, or it may be a bug.

**Resolution:** Phase 2 Step 2.0 must implement `execute_profile` regardless of prepared-core. If prepared-core adds it first, Phase 2 can build on top. If not, Phase 2 adds it.

### 7.5 Feature flag gating

**Conflict:** prepared-core adds `prepared-plan` feature to `Cargo.toml`. Phase 2 uses the same feature flag.

**Resolution:** No conflict — both phases use the same flag. Phase 2 should not modify `Cargo.toml`.

---

## Summary of Key Findings

1. **`execute_profile` does not exist** — it's called in `python/nn.rs:1374` but not defined. This is either a bug or an intended future addition. Phase 2 must implement it.

2. **The cleanest extraction boundary** is the per-instruction dispatch body inside `CpuBackend::dispatch()` at `cpu/mod.rs:3408`. Extracting this into `execute_single_instruction()` enables both prepared and profiled execution.

3. **Phase 2 should NOT duplicate the execute() preamble** — setup code (shape env, tightening, arena, I/O) is shared. The only divergence is the dispatch loop.

4. **Index stability** is guaranteed — `tighten_slices` does not reorder instructions, so `instruction_index` in `PreparedInstruction::Generic` remains valid after tightening.

5. **No parallelism in Phase 2** — prepared execution starts sequential. Parallelism can be added later by reusing `execute_parallel_level`.

6. **Dynamic shapes are safe** — the builder promotes only all-static-shape ops out of `Generic`; dynamic ops stay as `Generic` and use the existing param_dims/shape_env resolution.
