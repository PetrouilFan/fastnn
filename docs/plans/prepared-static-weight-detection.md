# Prepared static-weight detection (lane B addendum)

Status: metadata-only detection pass, landed alongside
`PreparedConstantArena` (lane A). No runtime dispatch change.

This addendum documents how `prepare_executable_plan` discovers static
weight and bias tensors in an `ExecutablePlan`, what it records about
them, and what is left for the first prepacked Conv2d execution lane
(wave 4+).

## 1. Scope

- New types: `StaticWeightBinding`, `StaticWeightMap`,
  `WriteConstEntry` (private), plus a `try_prepare_matmul` promotion
  helper.
- New function: `detect_static_weights(plan, arena) -> StaticWeightMap`.
- `prepare_executable_plan` now runs the detection pass, registers the
  resulting bindings on `PreparedConv2d::packed_weight` /
  `PreparedMatMul::packed_weight`, and attaches the
  `PreparedConstantArena` to the produced `PreparedExecutablePlan`
  using `register_constant_arena`.
- Out of scope: any change to `CpuBackend::dispatch`,
  `GraphExecutor::execute`, weight tensor layouts, or the bincode
  format of `ExecutablePlan`. The `WriteConst` instructions that
  materialise the static tensors are intentionally preserved — the
  current dispatch path is byte-identical.

## 2. The constant-output opcode

The executor lowers `Opcode::Constant(...)` to
`Instruction::WriteConst { dst, data }` (see
`src/backend/mod.rs` and `src/backend/cpu/mod.rs`). The detection
pass therefore matches consumer input slots against `WriteConst`
destinations — there is no separate `Output::Constant` variant in the
prepared-plan layer.

## 3. Detection walk

`detect_static_weights` performs two passes over the
[`ExecutablePlan`]:

1. **Producer sweep.** Every `Instruction::WriteConst { dst, data }`
   with non-empty `data` is recorded as a `WriteConstEntry { writer_idx,
   dst, data }`. The list preserves plan order, so the first writer of
   a given range wins the tie-break in step 2.

2. **Consumer sweep.** Every `Instruction::CallKernel` whose
   `kernel_name` starts with `"conv2d"` or matches a matmul-family
   name (see [`is_matmul_kernel_name`]) is inspected:
   - `input_slices[1]` is the **weight** slot. It is matched against
     the producer list with a "covers" check
     (`wc.offset <= slot.offset` and
     `wc.offset + wc.data.len() >= slot.offset + slot.size`).
   - `input_slices[2]` is the **bias** slot (present on conv2d when
     the plan supplied a bias, and on `fused_matmul_add_*` matmuls).
     Same covers check; bias-less matmuls are skipped naturally.

   When a producer covers a slot, the producer's bytes are decoded as
   little-endian f32 (`bytes_to_f32_vec`) and inserted into the
   supplied `PreparedConstantArena` under a stable name. The arena
   handles dedup — the same name always returns the same
   `PackedWeightId`, so two consumers reading the same `WriteConst`
   share the slot.

A binding is only recorded when a `WriteConst` is found. Dynamic
inputs (e.g. a `MemCopy` writing into the same region, or an upstream
kernel feeding the consumer) produce no binding and the
`packed_weight` field stays `None`.

## 4. Naming scheme

Arena entries are keyed by the producer, not the consumer, so dedup
is exact regardless of how many consumers reference the same
`WriteConst`:

```
{role}_{writer_idx}_i{input_index}
```

- `role` is `"conv"` for `conv2d` kernels, `"matmul"` for the
  matmul family.
- `writer_idx` is the `WriteConst`'s instruction index in the source
  plan.
- `input_index` is the consumer's input slot (`1` = weight, `2` =
  bias).

Examples for a plan `[WriteConst(0), WriteConst(1), Conv2d(2),
MatMul(3)]` where conv reads weight from writer 0 and bias from
writer 1, and matmul reads weight from writer 0:

- `conv_0_i1`  → conv weight (writer 0)
- `conv_1_i2`  → conv bias   (writer 1)
- `matmul_0_i1` → matmul weight (writer 0, **same slot** as conv_0_i1)

## 5. Binding shape

```rust
pub struct StaticWeightBinding {
    pub instruction_index: usize,  // consumer's plan-wide index
    pub input_index: usize,         // 1 = weight, 2 = bias
    pub weight_id: PackedWeightId,  // handle into the arena
    pub kind: PackedWeightKind,     // always Fp32 today
}

pub type StaticWeightMap = Vec<StaticWeightBinding>;
```

`detect_static_weights` returns the full map; `prepare_executable_plan`
keeps it in a `HashMap<instruction_index, Vec<&StaticWeightBinding>>`
and consults it while promoting instructions, so the per-instruction
lookup is O(1) and the bindings are attached to the right
`PreparedConv2d` / `PreparedMatMul` via `apply_static_weight_bindings`.

## 6. Wiring in `prepare_executable_plan`

```rust
pub fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan {
    let mut arena = PreparedConstantArena::new();
    let bindings = detect_static_weights(plan, &mut arena);
    /* bucket bindings by consumer instruction_index, promote each
       instruction with try_prepare_conv2d || try_prepare_matmul,
       apply the matching packed_weight via apply_static_weight_bindings */
    let mut prepared = PreparedExecutablePlan { /* … */, constant_arena: None };
    prepared.register_constant_arena(arena);
    prepared
}
```

A non-trivial consequence: every plan now carries an attached arena,
even when it has no Conv2d / MatMul. The arena is empty in that case
(`is_empty() == true`) and reports `len() == 0`. This keeps the
introspection surface uniform.

## 7. Tests

Required coverage added to `src/backend/prepared.rs`:

- `detects_conv_weight_from_write_const`
- `detects_matmul_weight_from_write_const`
- `skips_dynamic_input`
- `arena_reused_for_duplicate_weight`

Plus supporting tests for the matmul promotion path
(`try_prepare_matmul_basic`, `try_prepare_matmul_with_activation`,
`try_prepare_matmul_fused_with_bias`,
`try_prepare_matmul_quantized`, the param/insufficient-input
negatives, and `is_matmul_kernel_name_recognises_family`) and for the
detection internals (`detects_fused_matmul_weight_and_bias`,
`detects_conv_weight_and_bias`,
`detect_static_weights_empty_plan`,
`bytes_to_f32_vec_round_trip`).

## 8. What's next: first prepacked Conv2d path

The detection pass is the metadata foundation; the first execution
lane that *consumes* the arena will live in a follow-up wave. That
work needs to:

1. **Add a `packed_bias` field** on `PreparedConv2d` /
   `PreparedMatMul` (the `StaticWeightBinding` already records bias
   bindings; the consumer structs just do not expose them yet).
2. **Decide a packing layout.** `PackedWeightStore::Unpacked` is the
   f32 reference payload the current CPU backend already consumes;
   the first real packing lane is expected to start with
   `RawOihw`/`RawKhwCrs` identity (trivially correct) before moving
   to `RowMajorFk` / `Im2colGemm` packed variants.
3. **Branch in the dispatch path.** `CpuBackend::dispatch` should
   route to the prepacked kernel only when the consumer's
   `kernel == FuturePackedFp32`, the arena has a matching slot, and
   the original `WriteConst` byte length equals the prepared weight
   byte length. Otherwise it falls back to the current
   im2col/gemm path. The YOLO comparison stays as the accuracy gate.
4. **Memory planner integration.** If the packed layout lives at a
   different arena offset than the original `WriteConst`, the
   `MemoryPlan` will need a tightening pass that re-points
   `input_slices[1]` at the packed slot. The current detection does
   not touch offsets — that is left for the integration lane.
5. **YOLO validation.** Run `cargo test --release --lib --features
   prepared-plan` and the YOLO comparison script to confirm no
   regression. Any drift indicates a correctness bug in the
   detection or packing code; the structural change in this slice
   must remain byte-identical.
