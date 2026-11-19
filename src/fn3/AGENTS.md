# FastNN inference core guide

`src/fn3/` is the inference-only core and the crate's primary API. The contract
authority is `docs/roadmap/fn3-architecture.md`.

## Ownership

- `dtype.rs`, `ops/mod.rs`, `device.rs`: leaf contracts (no imports from other
  `fn3` modules).
- `error.rs`: the single error type. Depends on `dtype`, `device`, `ops` only.
- `shape.rs`, `storage.rs`: shape/stride/bound algebra and the aligned buffer.
- `tensor.rs`: metadata, views and owned tensors.
- `backend/mod.rs`: `Backend`, `Capabilities`, `Support`, `OpParams`.
- `backend/cpu/`: the portable reference backend.
- `backend/vulkan/`: the Vulkan platform and kernel families.
  `resident.rs` owns device-resident tensors, the persistent-tensor table and
  the residency counters (FN3-26 / #72).
- `import/`: strict ONNX reader and graph construction.
- `model/mod.rs`: `ExecutionPlan`, `ModelWeights`, `PreparedModel`.
- `model/kv.rs`: `KvCacheConfig`, `KvCacheLayout`, `KvCacheState` — bounded
  per-layer buffers, the live extent and the two-phase append protocol.
- `model/session.rs`: `Session`, `SessionPhase` — `prefill`/`decode`, the
  per-session plan cache and slot pool, and the step commit ordering.
- `model/plan.rs`: bounded `Plan`, slot table, folded constants, `PlanId`,
  `PlanBound`, `PlanCache`.
- `model/specialize.rs`: shape resolution, integer constant folding, typed
  `OpParams` mapping and the `Planner` that builds plans.
- `model/format/`: the versioned prepared-model artifact.
- `python/`: the PyO3 binding, declared by path from `src/python/mod.rs`.
- `cli/`: the `fn3-infer` implementation.
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
  input, a zero or an approximation. `Support::Partial` names its subset and the
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
  work. Value slots and the KV cache are device-resident when the backend can
  allocate there (`SessionOptions::default()`, `host_staged = false` for a
  fully resident op) and host-staged otherwise (`SessionOptions::staged()` or
  `FN3_VK_STAGED=1`, `host_staged = true`). Neither is a CPU compute path.
- A `Storage` is either host-backed or device-resident
  (`Storage::from_resident`). A resident storage has **no host bytes**: every
  host byte accessor returns `Fn3Error::NotImplemented` and
  `Tensor::to_host`/`Storage::download` is the only materialisation path, so
  stale bytes cannot be observed. CPU tensors are never resident.
- Immutable model data is uploaded once (`Backend::prepare_persistent`) and
  bound by handle afterwards. A prepared tensor that reaches a host upload is a
  hard error (`VulkanBackend::check_staged_input`), never a silent re-upload.
  The host tensors stay valid for the CPU/fallback path.
- Unsafe code is confined to `storage.rs` and every block carries a `SAFETY:`
  comment covering alignment, bounds, aliasing and lifetime.

## Validation

```bash
cargo fmt --check
cargo build --no-default-features
cargo build
cargo test --lib fn3
cargo test --test fn3_contracts
cargo test --test fn3_cpu_reference
```

`cargo test --lib` must also keep passing: it proves the secondary 2.x tree
still compiles beside `src/fn3/`. Contract changes require a new or updated case
in `tests/fn3_contracts.rs`.
