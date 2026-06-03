# Prepared Packed Weight Store Design

> Design document for Phase 4 of the prepared-plan implementation.
> Defines how fastnn represents constant weights, packed handles, and prepared Conv/MatMul weight metadata.

---

## 1. Current State Summary

### `PreparedConv2d.packed_weight`

`PreparedConv2d` (`src/backend/prepared.rs:35-58`) carries `packed_weight: Option<PackedWeightId>`. It is **always `None`** today — populated at `try_prepare_conv2d` line 188. `PackedWeightId(pub usize)` (line 74-75) is a newtype index intended as a handle into a future weight store. `PreparedMatMul` (line 60-72) has the same `Option<PackedWeightId>` field, also always `None`.

### Current Constants / Arena Behavior

The runtime arena (`CpuBuffer`, an `UnsafeCell<Vec<u8>>`) is a flat byte buffer shared by all instructions. Constants are materialized every forward pass:

1. **Compile time** (`src/backend/cpu/mod.rs:1470-1489`): Each `Opcode::Constant` node becomes an `Instruction::WriteConst { dst, data }` (for `TensorValue::Data`) or `Instruction::Fill { dst, value }` (for scalar `Float`/`Int`).
2. **Runtime** (`src/backend/cpu/mod.rs:1224-1228`): `WriteConst` copies the instruction's embedded `data: Vec<u8>` into the arena at `dst.offset`. This happens on **every** forward pass — the same weight bytes are copied from the instruction into the arena every time.

The `data` payload is heap-allocated inside each `Instruction::WriteConst` variant and serialized with the `ExecutablePlan` via bincode. This means:
- Constant bytes are stored twice: once in the instruction, once in the arena at dispatch.
- No sharing or deduplication across instructions that reference the same constant.
- Weights cannot be packed or transposed without mutating the instruction's `data` field.

### How Conv2d Instructions Reference Input/Weight/Bias Slices

`Instruction::CallKernel` for conv2d (`src/backend/cpu/mod.rs:1759-1869`):
- `input_slices[0]` → input tensor `BufferSlice` in the arena
- `input_slices[1]` → weight tensor `BufferSlice` in the arena (pointing into the arena slot populated by a preceding `WriteConst`)
- `input_slices.get(2)` → optional bias `BufferSlice`
- `params` → `[stride, padding, dilation, groups, c, h, w, kh, kw]` (9 static values baked at compile time)
- `weight_meta: Option<QuantizedWeightMeta>` → carries scales/zero-points for quantized kernels

`try_prepare_conv2d` (`src/backend/prepared.rs:122-192`) extracts these slices and params into `PreparedConv2d` fields. The weight `BufferSlice` still points into the arena — the prepared struct has no direct reference to the raw constant bytes.

---

## 2. Proposed Data Structures

All types live in `src/backend/prepared.rs` behind `#[cfg(feature = "prepared-plan")]`.

### `PreparedConstantArena`

An immutable, reference-counted container holding all static constant data for a prepared plan. Lives alongside `PreparedExecutablePlan` but is never mutated after construction.

```rust
use std::sync::Arc;

/// Immutable arena for static constant data. Constructed once at prepare
/// time and shared across all executions. Never mutated after creation.
#[derive(Clone, Debug)]
pub struct PreparedConstantArena {
    /// Raw f32 constant data (bias tensors, shape constants, scalars).
    /// Each entry is a contiguous f32 slice with its byte offset in the
    /// activation arena — at dispatch, the executor copies from here
    /// instead of from WriteConst instructions.
    pub raw_f32: Vec<PreparedRawF32>,

    /// Packed weight handles for static Conv/MatMul weights.
    /// Indexed by `PackedWeightId.0`.
    pub packed_weights: Vec<PackedWeight>,
}

#[derive(Clone, Debug)]
pub struct PreparedRawF32 {
    /// Byte offset in the activation arena where this constant belongs.
    pub dst_offset: usize,
    /// The constant data as f32 slice.
    pub data: Vec<f32>,
}
```

### `PackedWeightStore`

A builder that accumulates packed weights during preparation. Consumed into `PreparedConstantArena` when preparation is complete.

```rust
/// Mutable builder for packed weights. Accumulated during
/// `prepare_executable_plan` and finalized into PreparedConstantArena.
pub struct PackedWeightStore {
    weights: Vec<PackedWeight>,
}

impl PackedWeightStore {
    pub fn new() -> Self {
        Self { weights: Vec::new() }
    }

    /// Register a packed weight and return its handle.
    pub fn insert(&mut self, weight: PackedWeight) -> PackedWeightId {
        let id = PackedWeightId(self.weights.len());
        self.weights.push(weight);
        id
    }

    /// Look up a packed weight by handle.
    pub fn get(&self, id: PackedWeightId) -> Option<&PackedWeight> {
        self.weights.get(id.0)
    }

    /// Finalize into an immutable arena.
    pub fn into_arena(self, raw_f32: Vec<PreparedRawF32>) -> PreparedConstantArena {
        PreparedConstantArena {
            raw_f32,
            packed_weights: self.weights,
        }
    }
}
```

### `PackedWeightId`

Already exists (`src/backend/prepared.rs:74-75`). No changes needed.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedWeightId(pub usize);
```

### `PackedWeightKind`

Enum describing what kind of packed representation a weight has been transformed into. Discriminates dispatch paths.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedWeightKind {
    /// Raw OIHW f32 data, no transformation.
    RawF32,
    /// f32 conv weight rearranged for row-major GEMM.
    F32ConvIm2col,
    /// f32 matmul weight in row-major [K, N] layout.
    F32MatMul,
    /// Future: symmetric I8 per-channel weight-only.
    FutureI8,
    /// Future: U4/NF4 per-group weight-only.
    FutureU4,
}
```

### `PackedWeight`

Enum carrying the actual weight data and metadata. Variants hold different packed layouts.

```rust
#[derive(Clone, Debug)]
pub enum PackedWeight {
    /// Raw f32 conv weight (OIHW). Used as fallback before packing.
    RawF32Conv {
        /// Original logical shape [F, C, KH, KW].
        original_shape: [usize; 4],
        /// Packed byte data (little-endian f32).
        data: Vec<u8>,
        /// Byte offset in activation arena where the original WriteConst
        /// would have placed this weight. Used for fallback dispatch.
        arena_dst: usize,
    },

    /// f32 conv weight packed for im2col GEMM.
    F32ConvIm2col(PackedF32ConvWeight),

    /// f32 matmul weight packed as row-major [K, N].
    F32MatMul(PackedF32MatMulWeight),

    /// Placeholder for future I8 weight-only format.
    FutureI8 {
        original_shape: Vec<usize>,
    },

    /// Placeholder for future U4/NF4 weight-only format.
    FutureU4 {
        original_shape: Vec<usize>,
    },
}

impl PackedWeight {
    pub fn kind(&self) -> PackedWeightKind {
        match self {
            PackedWeight::RawF32Conv { .. } => PackedWeightKind::RawF32,
            PackedWeight::F32ConvIm2col(_) => PackedWeightKind::F32ConvIm2col,
            PackedWeight::F32MatMul(_) => PackedWeightKind::F32MatMul,
            PackedWeight::FutureI8 { .. } => PackedWeightKind::FutureI8,
            PackedWeight::FutureU4 { .. } => PackedWeightKind::FutureU4,
        }
    }
}
```

### FP32 Packed Conv Variant

```rust
#[derive(Clone, Debug)]
pub struct PackedF32ConvWeight {
    /// Original logical shape [F, C, KH, KW].
    pub original_shape: [usize; 4],
    /// Packed data in the chosen layout.
    pub packed: Vec<f32>,
    /// Which layout `packed` uses.
    pub layout: PackedF32ConvLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedF32ConvLayout {
    /// Unmodified OIHW (identity, used as first fallback).
    RawOihw,
    /// Row-major F×(C*KH*KW) — the natural GEMM orientation.
    RowMajorFk,
    /// Future: blocked layout for SIMD register tiling.
    FutureBlocked,
}
```

### FP32 Packed MatMul Variant

```rust
#[derive(Clone, Debug)]
pub struct PackedF32MatMulWeight {
    /// Original logical shape [K, N].
    pub original_shape: [usize; 2],
    /// Row-major [K, N] f32 data.
    pub packed: Vec<f32>,
}
```

### Future I8/U4 Variants

These are placeholder variants that carry only shape metadata. The actual quantized data and packing logic live in `PackedTensor<T>` (`src/packed_tensor.rs`) and are consumed by CPU microkernels at dispatch time.

```rust
// Inside PackedWeight enum:
FutureI8 { original_shape: Vec<usize> },
FutureU4 { original_shape: Vec<usize> },
```

When these are eventually populated, the packed byte data will come from `PackedTensor<U8x4>` / `PackedTensor<U4x8>` plus `QuantizedWeightMeta` (scales, zero_points). The design intentionally defers this to Phase 7.

---

## 3. First Implementation Slice

### Exact Files to Edit

| File | Action | Description |
|------|--------|-------------|
| `src/backend/prepared.rs` | Modify | Add `PackedWeightStore`, `PreparedConstantArena`, `PackedWeightKind`, `PackedWeight`, `PackedF32ConvWeight`, `PackedF32ConvLayout`, `PackedF32MatMulWeight`. Extend `PreparedExecutablePlan` with `constants: Option<PreparedConstantArena>`. |
| `src/backend/prepared.rs` | Modify | Add `detect_static_weight()` helper. Extend `prepare_executable_plan()` to populate `PackedWeightStore` with `RawF32Conv` entries for detected static conv weights. |
| `src/backend/mod.rs` | Modify | Add `pub mod prepared;` re-export if needed (already done). Add `PreparedConstantArena` to public API surface. |

### API Changes

1. `PreparedExecutablePlan` gains:
   ```rust
   pub struct PreparedExecutablePlan {
       pub instructions: Vec<PreparedInstruction>,
       pub arena_size: usize,
       pub scratch_size: usize,
       /// Immutable constant data. `None` when no static weights detected.
       pub constants: Option<PreparedConstantArena>,
   }
   ```

2. `prepare_executable_plan()` signature remains `fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan` but internally:
   - Creates a `PackedWeightStore`.
   - Scans instructions for `WriteConst` that feed conv/matmul weight inputs.
   - Registers detected static weights as `PackedWeight::RawF32Conv`.
   - Assigns `PackedWeightId` to matching `PreparedConv2d.packed_weight`.
   - Finalizes the store into `PreparedConstantArena`.

3. New helper:
   ```rust
   /// Determine if a WriteConst instruction's destination feeds a
   /// recognized conv/matmul weight slot. Returns the consumer
   /// instruction index if identified.
   fn detect_static_weight(
       write_const_index: usize,
       write_const_dst: BufferSlice,
       plan: &ExecutablePlan,
   ) -> Option<StaticWeightInfo>
   ```

### Tests

Add to `src/backend/prepared.rs` `#[cfg(test)] mod tests`:

1. **`packed_weight_store_insert_and_get`** — Insert weights, retrieve by ID, verify round-trip.
2. **`packed_weight_kind_discrimination`** — Verify `kind()` returns correct variant for each `PackedWeight`.
3. **`prepare_plan_detects_static_conv_weight`** — Build a plan with `WriteConst` + `CallKernel(conv2d)` where the WriteConst destination overlaps the conv weight slice. Assert `PreparedConv2d.packed_weight.is_some()` and `constants.packed_weights.len() == 1`.
4. **`prepare_plan_no_static_weight_for_dynamic`** — Build a plan where conv weight comes from a non-constant source (e.g., `MemCopy` from an input). Assert `packed_weight.is_none()`.
5. **`prepare_plan_no_static_weight_for_matmul`** — Build a plan with `WriteConst` + `CallKernel(matmul)`. Assert `PreparedMatMul.packed_weight.is_some()`.
6. **`constant_arena_immutability`** — Build a `PreparedConstantArena`, clone it, mutate one, assert the other is unchanged.

### YOLO Validation

Run the YOLO comparison script to verify no behavioral regression from the structural changes:

```bash
cargo test --release --lib --features prepared-plan
.venv/bin/python -m maturin develop --release --features prepared-plan
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --warmup 1 --iters 2
```

The YOLO output must remain identical to baseline (max_abs ~5e-4 vs PyTorch). This slice adds no runtime dispatch changes — only metadata preparation — so any YOLO drift indicates a correctness bug in static weight detection.

---

## 4. Safety Constraints

### No Runtime Behavior Change Until Fallback Verified

The first implementation slice **only** populates `PackedWeightId` and `PreparedConstantArena` metadata. It does **not**:
- Skip any `WriteConst` instructions at dispatch.
- Change how the CPU backend reads weight data.
- Alter `CpuBackend::dispatch()` in any way.

The existing dispatch path (`WriteConst → arena copy → CallKernel reads from arena`) remains the sole execution path. Packed weight metadata is informational only until Phase 5.

### No Packed Precision Without Accuracy Gates

No `PackedWeight::FutureI8` or `PackedWeight::FutureU4` variant may carry real quantized data until:
- An explicit `precision=` mode is exposed in the Python API.
- `max_abs` and `mean_abs` deltas are measured and bounded.
- YOLO comparison shows no regression beyond the declared precision mode.

### No Global BLAS/OpenBLAS Switch

The design does not introduce any automatic BLAS dispatch. Kernel selection remains explicit per-shape. Known rejected paths from the plan history (global OpenBLAS conv routing, broad low-F GEMM orientation, direct tiled fallback for YOLO stem) must not be re-introduced without a dedicated design review.

---

## 5. How to Detect Static Weights from `ExecutablePlan` / `WriteConst` / Call Inputs

### Detection Algorithm

A weight buffer is "static" when:

1. The `CallKernel` instruction's `input_slices[1]` (conv weight) or `input_slices[1]` (matmul weight B) points to an arena slot `[dst.offset, dst.offset + dst.size)`.
2. There exists an earlier `Instruction::WriteConst { dst, data }` in the plan whose `dst` fully contains that slot (i.e., `write_dst.offset <= weight_slice.offset` and `write_dst.offset + write_dst.data.len() >= weight_slice.offset + weight_slice.size`).
3. The `WriteConst` data is non-empty and the slot is not subsequently overwritten by another instruction before the `CallKernel`.

Condition (3) is conservatively approximated by requiring that no intervening `MemCopy` or `WriteConst` targets the same arena range between the `WriteConst` and the `CallKernel`. In practice, weights are typically written before any consumer and never overwritten, so a simple "earlier WriteConst covers this range" check suffices for the first implementation.

### Helper Signature

```rust
struct StaticWeightInfo {
    /// Index of the WriteConst instruction in the plan.
    write_const_index: usize,
    /// The WriteConst's destination slice.
    write_const_dst: BufferSlice,
    /// The WriteConst's raw data bytes.
    data: Vec<u8>,
}

/// Scan the instruction list for a WriteConst that populates the given
/// arena range. Returns info about the matching WriteConst, or None
/// if the weight is dynamic.
fn find_write_const_for_slice(
    weight_slice: BufferSlice,
    plan: &ExecutablePlan,
    before_index: usize,
) -> Option<StaticWeightInfo>
```

### Handling Bias

Bias tensors are typically small (`[F]`, F * 4 bytes) and may come from `WriteConst` or from `Fill` (for zero bias). For the first slice, bias is **not** packed — only the weight tensor (input_slices[1]) is registered in `PackedWeightStore`. Bias can be added to the packed store in a later slice.

---

## 6. Open Questions / Blockers

| # | Question | Status | Notes |
|---|----------|--------|-------|
| 1 | Should `PreparedConstantArena` be `Arc`-wrapped for cheap cloning across executor calls? | Open | `PreparedExecutablePlan` is already `Clone`. `Arc<PreparedConstantArena>` avoids deep copies when the plan is cloned for parallel dispatch. Recommend `Arc` in final design. |
| 2 | How to handle `WriteConst` for non-weight constants (shape tensors, scalars)? | Open | These should remain in the arena via existing `WriteConst` path. Only weight-like constants (conv/matmul weight slices) should be registered in `PackedWeightStore`. The detection heuristic must exclude small scalars and shape constants. |
| 3 | Should `PackedF32ConvLayout::RowMajorFk` be the first real packing, or should we start with `RawOihw` identity? | Open | Identity (`RawOihw`) is trivially correct and validates the plumbing. `RowMajorFk` provides actual GEMM benefit. Recommend starting with `RawOihw` in the first slice, then adding `RowMajorFk` in Phase 5 Task 5.1. |
| 4 | How does memory tightening interact with packed weight offsets? | Open | `tighten_slices()` mutates `BufferSlice` offsets. If packed weights reference arena offsets, tightening must update those too. For the first slice this is moot (metadata only), but must be addressed before dispatch integration. |
| 5 | Serialization: should `PreparedConstantArena` be included in `ExecutablePlan` bincode format? | Open | It could be serialized as part of a new `PreparedExecutablePlan` binary format. Recommend a separate `.prepared` file extension to avoid breaking existing plan loading. |

---

## 7. Acceptance Criteria

- [ ] `PackedWeightStore`, `PreparedConstantArena`, `PackedWeightKind`, `PackedWeight`, `PackedF32ConvWeight`, `PackedF32ConvLayout`, `PackedF32MatMulWeight` types compile and have unit tests.
- [ ] `prepare_executable_plan()` correctly identifies static conv/matmul weights from `WriteConst` instructions and populates `PackedWeightId` on matching `PreparedConv2d` / `PreparedMatMul`.
- [ ] `PreparedExecutablePlan.constants` is `Some(...)` when static weights exist, `None` otherwise.
- [ ] All existing tests pass: `cargo test --release --lib --features prepared-plan`.
- [ ] YOLO comparison shows no regression: `max_abs` and `mean_abs` within baseline noise.
- [ ] No runtime dispatch changes — `WriteConst` instructions still execute identically.
- [ ] No packed precision enabled without explicit opt-in.
- [ ] Design doc verification passes:
  ```bash
  python - <<'PY'
  from pathlib import Path
  p = Path('docs/plans/prepared-packed-weight-design.md')
  text = p.read_text()
  assert 'PackedWeightStore' in text
  assert 'Acceptance criteria' in text
  assert 'YOLO' in text
  print('design doc ok')
  PY
  ```
