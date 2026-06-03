# Prepared Constant Arena — wave 4 lane A notes

> Implementation addendum for the
> [Prepared Packed Weight Store Design](prepared-packed-weight-design.md).
> Captures the actual surface introduced by wave 4 lane A and what the
> next lane needs to plug into.

---

## 1. Chosen layout

The skeleton lives entirely in `src/backend/prepared.rs` and introduces
four cooperating types:

| Type | Role |
|------|------|
| `PreparedConstantArena` | Owns the `Vec<f32>` payloads + per-tensor metadata for a prepared plan. |
| `PreparedConstantEntry` | One row in the arena: `id`, `name`, `kind`, `numel`, `byte_len`, `store`. |
| `PackedWeightStore` (enum) | Backing storage variant: `Unpacked(Vec<f32>)` today, `Reserved` placeholder discriminant for future packed precision lanes. |
| `PackedWeightKind` (`#[repr(u8)]` enum) | Dtype tag: `Fp32` is the only live variant; `I8`, `U4`, `Nf4` are reserved. |

`PackedWeightId` is **kept as a copyable handle** but switched from the
old single-field tuple struct to a named-field struct:

```rust
pub struct PackedWeightId {
    pub index: usize,
    pub kind: PackedWeightKind,
    pub arena_slot: Option<u32>,
}
```

`PackedWeightId::new(index)`, `PackedWeightId::with_kind(...)`,
`PackedWeightId::default()` and `From<usize>` cover the existing
construction shape and let later lanes annotate slots without a breaking
change.

### Arena semantics

* `insert(name, data)` allocates a new dense slot the first time
  `name` is seen and returns the assigned `PackedWeightId`. Repeat
  inserts with the same name return the **existing** id and drop the
  new payload — duplicate names never duplicate storage.
* `id_for(name)` / `get(id) -> Option<&[f32]>` / `entry(id)` provide
  read-only lookups. `entry()` exposes the full metadata row.
* `len()`, `is_empty()`, `total_bytes()` are O(1) / O(n) summary
  helpers (`total_bytes` sums `byte_len` across all entries).
* `entries()` iterates in insertion order, which is also id order.

### Plan attachment

`PreparedExecutablePlan` gains a private
`constant_arena: Option<PreparedConstantArena>` field with two methods:

* `register_constant_arena(arena)` stores the arena on the plan.
* `constant_arena() -> Option<&PreparedConstantArena>` returns a
  borrow. The field is `None` for every plan produced by
  `prepare_executable_plan(&plan)` today.

The plan struct is now `Clone + Debug + Default`. Construction still
goes through the existing `prepare_executable_plan` entry point and
struct literals (test fixtures) explicitly set
`constant_arena: None`.

## 2. Why this is metadata-only

This mission is the storage skeleton — **no runtime code path consults
the arena yet**. Specifically:

* `prepare_executable_plan` builds a plan with `constant_arena: None`.
* `AotExecutor`'s `prepared_plan` field (added in wave 3) is still
  unused at dispatch time, so attaching an arena affects nothing.
* The CPU backend's `WriteConst → CallKernel` path is untouched —
  every static weight is still materialised into the activation arena
  on each forward pass exactly as before.
* No new public Python methods, no new opcode, no new feature flag.

The whole surface is therefore safe to land without a kernel- or
accuracy regression: there is no behavioural delta to measure.

## 3. What the next lane should plug into

The next wave-4 lane (lane B / static-weight detection) should:

1. **Detect `WriteConst` producers feeding conv/matmul weight slots.**
   The existing `ExecutablePlan` already encodes everything needed:

   * `Instruction::WriteConst { dst, data }` (`src/backend/mod.rs:89`)
     declares the source bytes.
   * The downstream `Instruction::CallKernel`
     (`src/backend/cpu/mod.rs:1759` for conv2d, matmul kernels alike)
     references that arena range via `input_slices[1]` (weight) /
     `input_slices[2]` (bias).

   The detection helper sketched in
   [`prepared-packed-weight-design.md` §5](prepared-packed-weight-design.md#5-how-to-detect-static-weights-from-executableplan-writeconst-call-inputs)
   (`find_write_const_for_slice`) is the recommended starting point.

2. **Materialise detected weights into a `PreparedConstantArena`.**
   For each statically resolved weight slice, call
   `arena.insert(role_name, fp32_payload)` where `role_name` follows
   the convention documented on `PreparedConstantEntry::name`
   (`"conv_weight"`, `"conv_bias"`, `"matmul_weight"`,
   `"matmul_bias"`). Reuse the returned `PackedWeightId` to populate
   `PreparedConv2d::packed_weight` / `PreparedMatMul::packed_weight`
   (both already `Option<PackedWeightId>` and currently always
   `None`).

3. **Attach the arena via the new `register_constant_arena` method.**
   `PreparedExecutablePlan::register_constant_arena(arena)` is the
   only way to do this; the field is private to enforce the
   register-once flow.

4. **Leave dispatch untouched.** Lane B is still metadata only — no
   `WriteConst` may be skipped, no kernel may switch to reading from
   the arena, until a later lane lands the runtime route plus
   accuracy gates (`max_abs` / `mean_abs` deltas, YOLO comparison)
   per [`prepared-packed-weight-design.md` §4](prepared-packed-weight-design.md#4-safety-constraints).

## 4. Tests landing in this lane

All in `src/backend/prepared.rs`:

* `arena_insert_lookup` — round-trips a payload via
  `insert` / `id_for` / `get` / `entry` and checks ordering.
* `arena_duplicate_name_reuses_slot` — second insert with the same
  name returns the original id and keeps the original payload.
* `arena_len_is_empty_and_total_bytes` — covers the size accessors,
  including the duplicate-name no-op.
* `arena_get_returns_none_for_unknown_id` — out-of-range lookups.
* `packed_weight_unpacked_view` — `as_f32_slice()` round-trip and
  `Reserved` arm.
* `packed_weight_kind_name` / `packed_weight_kind_default_is_fp32` —
  variant tags and dtype defaults.
* `packed_weight_id_default_and_from` — `Default` /
  `From<usize>` / `with_kind` constructors.
* `plan_holds_constant_arena` —
  `register_constant_arena` + `constant_arena()` round-trip,
  including a replacement-arena swap.
* `prepare_executable_plan_starts_without_constant_arena` — confirms
  the default-constructed plan has no arena attached.

The existing wave-3 test `packed_weight_id_equality` is updated to use
`PackedWeightId::new(...)` after the struct shape change.

## 5. Out of scope

* No `WriteConst` skipping at dispatch.
* No bias packing, no precision tags beyond `Fp32`.
* No new Python bindings, no new opcode, no new compiler-pass changes.
* No serialization of `PreparedConstantArena` — the activation-arena
  bincode format is untouched.
* No `Arc<PreparedConstantArena>` sharing yet; the arena is cloned
  along with its owning plan (cheap because no real data is
  populated today).
