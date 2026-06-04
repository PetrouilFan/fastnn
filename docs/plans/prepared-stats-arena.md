# Prepared arena stats — mission 015 notes

This addendum documents the three new introspection keys added to
`AotExecutor.prepared_stats()` in mission 015, on top of the baseline
keys from the earlier introspection lane:

- `total`
- `generic`
- `conv2d`
- `matmul`

The new keys surface the static-weight metadata created in wave 4
(`PreparedConstantArena`, `StaticWeightBinding`,
`detect_static_weights`).

## 1. Scope

- New helper methods on `PreparedExecutablePlan` in
  `src/backend/prepared.rs`:
  - `static_weight_binding_count(&self) -> usize`
  - `constant_arena_entry_count(&self) -> usize`
  - `constant_arena_total_bytes(&self) -> usize`
- `AotExecutor.prepared_stats()` in `src/python/nn.rs` extended with
  three corresponding `HashMap<String, usize>` entries.
- New Python-binding test in
  `tests/test_aot_executor_prepared_stats.py`.
- No runtime change: dispatch path, `WriteConst` placement, and
  `GraphExecutor::execute` are byte-identical. The work is pure
  introspection.

## 2. New keys

| key | type | meaning |
|------|------|---------|
| `static_weight_bindings` | `int` | Number of `PreparedConv2d.packed_weight` / `PreparedMatMul.packed_weight` slots that are `Some(_)`. Equivalent to the number of weight consumers that the detection pass resolved against a `WriteConst` producer. |
| `constant_arena_entries` | `int` | Number of entries in the attached `PreparedConstantArena` (`0` when no arena is attached). |
| `constant_arena_bytes`   | `int` | `arena.total_bytes()` — sum of `PreparedConstantEntry::byte_len` across all entries (`0` when no arena is attached). |

## 3. Counting rules

The wave-4 representation stores **at most one** packed-weight slot
per prepared consumer instruction — the weight slot on
`PreparedConv2d::packed_weight` / `PreparedMatMul::packed_weight`.
Bias bindings produced by `detect_static_weights` are still
metadata-only and are not stored in the prepared structs (the
detection helper returns them, but the prepared plan keeps no
`packed_bias` field), so they are **not** counted in
`static_weight_bindings`. The count is honest with respect to the
actual stored data structures; the bias count is left for the
follow-up lane that plumbs bias through the dispatch path.

When two consumers share the same `WriteConst` producer,
`detect_static_weights` deduplicates the arena slot and both
consumers point at the same `PackedWeightId`. In that case:

- `constant_arena_entries` counts the **deduplicated** slot once.
- `static_weight_bindings` counts **both** consumers' `packed_weight`
  slots (they both hold `Some(_)`).

## 4. Helper semantics

All three helpers are pure metadata accessors — no execution
behavior:

- `static_weight_binding_count` walks the plan's `instructions`
  vector and counts `Conv2d` / `MatMul` entries whose
  `packed_weight` is `Some(_)`. `Generic` instructions never
  contribute.
- `constant_arena_entry_count` returns
  `self.constant_arena.as_ref().map(|a| a.len()).unwrap_or(0)`.
- `constant_arena_total_bytes` returns
  `self.constant_arena.as_ref().map(|a| a.total_bytes()).unwrap_or(0)`.

All three return `0` for a hand-built `PreparedExecutablePlan` with
`constant_arena: None`, matching the contract that no arena → zero
counts.

## 5. Tests

### Rust unit tests (`src/backend/prepared.rs`)

- `static_weight_binding_count_empty_plan` — empty plan → 0.
- `static_weight_binding_count_no_static_weights` — Conv2d fed by a
  `MemCopy` (dynamic) → 0.
- `static_weight_binding_count_counts_conv_and_matmul` — plan with
  a `WriteConst`-fed conv weight and a `WriteConst`-fed matmul
  weight (and a dynamic bias) → exactly 2.
- `static_weight_binding_count_hand_construction` — toggling
  `packed_weight` on hand-built `PreparedConv2d` /
  `PreparedMatMul` drives the count from 0 → 1 → 2 → 0.
- `constant_arena_entry_count_empty_plan_starts_at_zero` — auto
  arena stays empty when no static weight is detected.
- `constant_arena_entry_count_no_arena_returns_zero` — hand-built
  plan with `constant_arena: None` → 0.
- `constant_arena_entry_count_reflects_arena_size` — a plan with
  two `WriteConst`-fed consumers reports `2`.
- `constant_arena_total_bytes_empty_plan_starts_at_zero` —
  same as the entry-count empty-plan variant.
- `constant_arena_total_bytes_no_arena_returns_zero` —
  hand-built plan with no arena.
- `constant_arena_total_bytes_sums_entry_sizes` — 8 + 6 f32 →
  32 + 24 bytes.
- `constant_arena_total_bytes_matches_arena_helper` — cross-checks
  the helper against reading the arena directly.

### Python-binding test

`tests/test_aot_executor_prepared_stats.py::test_prepared_stats_has_arena_keys`
builds a tiny `AotExecutor` from a hand-written `Relu` node graph
and asserts that the returned dict carries both the baseline keys
and the three new ones, with sane types. Because the plan contains
no Conv2d / MatMul, the new keys all report `0`.

## 6. Example

Building the new `AotExecutor` against the yolov8n.onnx graph (via
`scripts/yolo_prepared_stats.py`, once the model is present)
returns a dict with the seven keys documented above. The
`static_weight_bindings` count is expected to track the number of
detected static conv weight tensors (typically bounded above by
`conv2d`) and is the signal future lanes can use to confirm the
detection pass is finding weights it should.

## 7. What's next

The bias count is still not exposed. A future lane that plumbs bias
through the dispatch path will:

1. Add a `packed_bias: Option<PackedWeightId>` field on
   `PreparedConv2d` / `PreparedMatMul` (the `StaticWeightBinding`
   already records bias bindings).
2. Extend `apply_static_weight_bindings` to populate
   `packed_bias` for the binding with `input_index == 2`.
3. Update `static_weight_binding_count` to count weight + bias
   bindings (or expose them as a separate `static_bias_bindings`
   key) — the exact split will be decided when the dispatch
   surface is defined.
