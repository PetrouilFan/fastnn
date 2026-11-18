# 3.0 inference core guide

`src/fn3/` is the FastNN 3.0 inference-only core. It is a new tree built beside
2.x so parallel work never edits 2.x paths; FN3-19 (#65) promotes and renames it
when the 2.x modules are deleted. The contract authority is
`docs/roadmap/fn3-architecture.md`.

## Ownership

- `dtype.rs`, `ops/mod.rs`, `device.rs`: leaf contracts (no imports from other
  `fn3` modules).
- `error.rs`: the single error type. Depends on `dtype`, `device`, `ops` only.
- `shape.rs`, `storage.rs`: shape/stride/bound algebra and the aligned buffer.
- `tensor.rs`: metadata, views and owned tensors.
- `backend/mod.rs`: `Backend`, `Capabilities`, `Support`, `OpParams`.
- `backend/cpu/`: the alpha reference backend.
- `model/mod.rs`: `ExecutionPlan`, `ModelWeights`, `PreparedModel`.
- `model/kv.rs`: `KvCacheConfig`, `KvCacheLayout`, `KvCacheState` — the bounded
  per-layer key/value buffers, the live extent and the two-phase append
  protocol (FN3-12 / #58).
- `model/session.rs`: `Session`, `SessionPhase` — `prefill`/`decode`, the
  per-session plan cache and slot pool, and the step commit ordering (FN3-12 /
  #58).
- `model/plan.rs`: bounded `Plan`, slot table, folded constants, `PlanId`,
  `PlanBound`, `PlanCache` (FN3-10 / #56).
- `model/specialize.rs`: shape resolution, integer constant folding, typed
  `OpParams` mapping and the `Planner` that builds plans (FN3-10 / #56).
- `mod.rs`: facade and re-exports.

## Invariants

- Dependency direction is a DAG (`dtype/device/ops → error → shape/storage →
  tensor → backend → model`); do not import upward.
- `src/fn3/` imports no 2.x module and 2.x imports nothing from `fn3`.
- All element-count, byte-count and stride arithmetic is checked; overflow is
  `Fn3Error::ShapeOverflow`, never a wrap.
- Strides are elements and signed. Writable views require positive,
  non-overlapping strides; zero strides require an explicitly broadcast
  (read-only) view.
- Explicit `DeviceRequest` values never resolve to another device. Only `Auto`
  chooses. Explicit device failures name the request and list available devices.
- A backend that cannot run an op returns `UnsupportedOp`; it never returns an
  input, a zero, or an approximation. `Support::Partial` names its subset and the
  op still rejects everything outside it.
- Plan preflight asks `Backend::admits` for the plan's concrete operand shapes
  and dtypes; the default body is `supports`, so a backend with no concrete
  rule keeps its op-level verdicts. An `admits` override may confirm a
  `Partial` subset for these operands, never widen `Unsupported`.
- `PreparedModel` is immutable and `Send + Sync`; all mutable inference state
  lives in `Session`, which is `Send` but deliberately `!Sync`. A failed step
  changes neither the KV live extent nor the step counter.
- A `Session` executes every instruction through the one backend it was opened
  with; a backend whose device differs from the model's is rejected before any
  work. KV buffers and value slots for a Vulkan device are host-staged
  (`host_staged = true` in the backend's residency record) — that is not a CPU
  compute path, and no session code owns a second backend.
- Unsafe code is confined to `storage.rs` and every block carries a `SAFETY:`
  comment covering alignment, bounds, aliasing and lifetime.

## Validation

```bash
cargo fmt --check
cargo build --no-default-features
cargo build
cargo test --test fn3_contracts
cargo test --lib fn3
```

`cargo test --lib` must also keep passing: it proves the frozen 2.x tree still
compiles while `src/fn3/` coexists. Contract changes require a new or updated
case in `tests/fn3_contracts.rs`.
