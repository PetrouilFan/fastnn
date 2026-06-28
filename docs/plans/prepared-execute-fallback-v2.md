# Prepared-Execute Fallback v2 — Mission 017

## Goal

Add an opt-in prepared-execution fallback that runs the *original*
`ExecutablePlan` through the existing dispatch path, but exposed
under a clearly named "prepared" API. The fallback is a
**behaviour-identical scaffold** for future prepared-instruction
specialisation: it does not consult `PreparedConstantArena`, it does
not skip or remove `WriteConst`, and it does not change any default
runtime behaviour.

## Public surface

| Layer | Symbol | Notes |
| --- | --- | --- |
| Rust helper | `crate::backend::prepared::validate_prepared_against_plan` | O(N) sanity check; refuses to dispatch when the prepared plan disagrees with the live plan. |
| Rust method | `GraphExecutor::execute_prepared_fallback` (gated on `prepared-plan`) | Validates then delegates to `GraphExecutor::execute`. |
| Python method | `AotExecutor.forward_prepared_fallback` (gated on `prepared-plan`) | Same input dict shape as `forward`; returns a `dict[str, Tensor]` of outputs. |
| Python guard | `AotExecutor.forward_prepared_fallback` (default build) | Returns `RuntimeError` if the feature is not enabled. |

## Behaviour

1. The new Rust method `execute_prepared_fallback` is added to
   `GraphExecutor<B>` in `src/backend/executor.rs` and gated on
   `#[cfg(feature = "prepared-plan")]`.
2. It first calls
   `crate::backend::prepared::validate_prepared_against_plan` to make
   sure the prepared plan is the identity permutation of the live
   plan (`original_instruction_order() == [0, 1, ..., N-1]`). When the
   check fails it returns `BackendError::Dispatch(_)` and the executor
   is **not** invoked.
3. On success it delegates to the existing `GraphExecutor::execute`
   path. The same memory-plan tightening, shape-env construction, and
   `Backend::dispatch` loop run, so on-the-wire behaviour is
   byte-identical to `AotExecutor.forward`.
4. `AotExecutor.forward_prepared_fallback` exposes the same path to
   Python. It accepts the same `dict[str, Tensor]` input shape as
   `forward` and decodes outputs through the same private
   `decode_outputs` helper that `forward` uses, so the two methods
   cannot drift.

## Why this is a scaffold, not an optimisation

- No `PreparedConstantArena` data is read at runtime. Static-weight
  bindings are still metadata-only.
- No prepared instruction is replaced with a specialised kernel. The
  executor dispatches the live `ExecutablePlan` instructions in their
  original order.
- `WriteConst` instructions remain in the plan and are executed as
  before. No producer is bypassed.
- The arena allocation, write/read, and memory-plan reuse rules are
  identical to the regular `execute` path.

Future lanes can layer kernel specialisation on top of this scaffold
without changing the contract: the prepared plan will be consulted
*before* the dispatch loop, individual prepared instructions will be
substituted with specialised kernels, and the unchanged
`WriteConst` + generic-fallback instructions will keep flowing through
the original dispatch path.

## Changed files

- `src/backend/prepared.rs` — added `validate_prepared_against_plan`
  helper + 4 unit tests.
- `src/backend/executor.rs` — added `execute_prepared_fallback` method
  on `GraphExecutor<B>` (gated on `prepared-plan`) + 5 end-to-end
  tests covering a `WriteConst`+`CallKernel` plan, a pure-elementwise
  plan, a symbolic-shape plan, repeatability, and validation refusal.
- `src/python/nn.rs` — added `forward_prepared_fallback` method on
  `AotExecutor` (gated on `prepared-plan`) + refactored the
  output-decoding loop into a private `decode_outputs` helper that
  `forward` and `forward_prepared_fallback` both call.
- `tests/test_aot_executor_prepared_execute.py` — new test module
  covering Relu, Relu+Neg, Add+Relu with a constant bias, repeatability,
  and method existence.
- `docs/plans/prepared-execute-fallback-v2.md` — this file.

## Verification

- `cargo fmt --check` — clean
- `cargo test --release --lib --features prepared-plan prepared::` — 74
  pass (70 existing + 4 new validation tests).
- `cargo test --release --lib --features prepared-plan` — 243 pass
  (existing 238 + 5 new executor-fallback tests).
- `cargo test --release --lib` (default features) — 238 pass,
  no behaviour change.
- `maturin develop --release --features prepared-plan` then
  `pytest tests/test_aot_executor_prepared_stats.py
  tests/test_aot_executor_prepared_execute.py` — both green.

## Risk assessment

- **None identified for the default `forward()` path.** The new method
  lives in a separate, gated entry point. The original `forward` call
  site is byte-for-byte identical: the only change inside
  `forward` is the call to the extracted `decode_outputs` helper, which
  performs exactly the same operations in the same order.
- **The validation check is cheap and conservative.** It only inspects
  `original_instruction_order()` (an already-cached `Vec<usize>`); a
  failure produces a `Dispatch` error rather than a panic.
- **The prepared-plan struct gains no new field or invariant.** Future
  lanes that need richer prepared-plan metadata can extend the struct
  without breaking this contract.
