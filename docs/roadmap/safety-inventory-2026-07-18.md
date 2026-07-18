# Safety and Assurance Inventory — 2026-07-18

## Purpose

This is the narrow pre-redesign assurance checkpoint between whole-plan
verification and the canonical dtype/storage redesign. It records the current
unsafe, concurrency, determinism, and fuzzing boundaries without freezing the
current representation as a permanent contract.

The inventory is lexical evidence, not a claim that every occurrence is a bug.
Counts include test code located in production modules and should only be used
to prioritize ownership review.

## Baseline

- Branch: `refactor/navigation-foundation`
- Phase 2 checkpoint: `6453f48 feat: verify complete execution plans`
- Rust toolchain has `rust-src`, Clippy, and rustfmt installed.
- Miri is not installed for the active toolchain.
- `cargo-fuzz` is available, but the repository has no fuzz targets or corpus.

A source scan found 288 lexical `unsafe { ... }` blocks and 23 `unsafe impl`
items across Rust source files. The same scan found 168 `SAFETY:` comments.
Comment count is not proof of adequate justification: one comment may cover
multiple blocks, and some comments do not cover bounds, alignment, aliasing,
and lifetime together.

## Priority boundaries

### P0 — representation and allocation foundations

| Boundary | Current owner | Required Phase 4 decision |
|---|---|---|
| Packed-word initialization and transmutation | `src/packed_tensor.rs` | Replace generic `T` transmutation with a representation whose word type and lane geometry are explicit. |
| Aligned raw allocation and slice construction | `src/storage.rs` | Keep allocation capacity distinct from encoded payload length; make allocation failure and layout ownership explicit. |
| Tensor pointer and typed-slice views | `src/tensor/mod.rs` | Derive byte ranges and alignment from canonical scalar/storage descriptors, not overloaded dtype labels. |
| Arena typed views and dispatch aliasing | `src/backend/cpu/arena.rs`, `src/backend/cpu/mod.rs` | Require validated typed operand descriptors before constructing shared/mutable views. |
| DLPack ownership and foreign pointers | `src/io/dlpack.rs` | Map canonical scalar/storage/layout into one checked FFI descriptor and retain explicit ownership transfer rules. |

These boundaries must be traced before deleting legacy dtype or packed-tensor
representations. They are the first targets for later Miri and malformed-input
fuzzing once the canonical schema is stable.

### P1 — optimized CPU leaves

The largest unsafe concentration is under `src/backend/cpu/`, especially:

- backend dispatch and typed arena views;
- GEMM and convolution microkernels;
- dispatch helpers and elementwise operations;
- packed convolution/GEMM;
- im2col and parallel reductions.

These leaves may remain unsafe for SIMD or FFI reasons, but each safe caller must
establish complete bounds, alignment, aliasing, representation, and lifetime
invariants through typed descriptors. Kernel-local assertions are not substitutes
for validation at the safe boundary.

### P2 — tensor operations and generated macros

`src/tensor/ops.rs`, `src/tensor/factories.rs`, and
`src/macros/tensor_ops.rs` perform pointer arithmetic and parallel writes. Their
proofs currently depend on dtype-derived widths, contiguity, non-overlap, and
chunk partitioning. Phase 4 must route these assumptions through canonical
storage sizing and plain-scalar stride APIs.

## Determinism inventory

- `src/lib.rs` owns an optional global seeded `StdRng`.
- `src/nn/dropout.rs` directly constructs `thread_rng()`, bypassing that seeded
  owner.
- Calibration tests also use `thread_rng()`, but test randomness is a separate
  reproducibility concern.
- Rayon is used in tensor operations and CPU kernels.
- Python thread configuration initializes Rayon's global thread pool.

No final determinism contract is established yet. The later assurance phase
must define seeded RNG ownership, deterministic versus fastest modes,
thread-count parity, reduction ordering, and oversubscription policy after the
canonical representation and compiler ownership are stable.

## Concurrency inventory

- CPU arenas are `Send` but deliberately not `Sync`; one mutable arena cannot be
  safely shared between concurrent dispatches.
- Python executor owners are unsendable to preserve that rule.
- Independent model execution, global caches, persistent weights, lazy
  initialization, and Rayon pool interaction still need explicit concurrency
  tests.
- Unsafe `Send`/`Sync` implementations for aligned storage and packed word types
  must be re-audited against their actual ownership and mutation APIs.

The redesign must not weaken these type-level ownership constraints merely to
make sharing convenient.

## Assurance sequencing

Before or during Phase 4:

1. Preserve the P0 boundary list while canonical types replace legacy labels.
2. Require local `SAFETY:` explanations for newly touched unsafe code.
3. Add focused tests when a representation migration changes byte sizing,
   alignment, or pointer construction.
4. Do not build broad fuzz targets against legacy serialized schemas.

After canonical representation, typed IR, and serialization ownership stabilize:

1. Install/run Miri for compatible safe abstractions.
2. Add fuzz targets for canonical representations, serialized artifacts, plan
   validation, shape tightening, typed attributes, and import boundaries.
3. Add concurrency tests for independent prepared execution and shared immutable
   persistent state.
4. Define and test deterministic execution modes.
5. Narrow remaining unsafe leaves behind validated typed descriptors.

## Exit decision

The pre-redesign Phase 3 inventory is complete when this document is checked in
and Phase 4 work uses the P0 boundaries as migration constraints. Full assurance
is intentionally deferred until the schemas under test are stable.
