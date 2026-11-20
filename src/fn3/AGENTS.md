# FastNN inference core guide

`src/fn3/` is the inference-only core and the crate's primary API. The contract
authority is `docs/roadmap/fn3-architecture.md`.

## Ownership

- `dtype.rs`, `ops/mod.rs`, `device.rs`, `error.rs`: leaf contracts; the error
  type depends on the first three only and none imports another core module.
- `shape.rs`, `storage.rs`, `tensor.rs`: shape/stride/bound algebra, the aligned
  buffer, and tensor metadata/views.
- `backend/mod.rs`: `Backend`, `Capabilities`, `Support`, `OpParams`.
  `backend/cpu/` is the portable reference backend; `backend/vulkan/` holds the
  platform and kernel families, where `resident.rs` owns device-resident
  storage, the persistent table and the residency counters (FN3-26 / #72).
- `import/`: strict ONNX reader and graph construction.
- `model/mod.rs`: `ExecutionPlan`, `ModelWeights`, `PreparedModel`.
- `model/kv.rs`: bounded per-layer KV buffers, the live extent, two-phase
  append.
- `model/session.rs`: `Session`, `SessionPhase` — prefill/decode, the plan
  cache, the slot pool and the step commit ordering.
- `model/plan.rs`: bounded `Plan`, slot table, folds, `PlanId` and cache.
- `model/specialize.rs`: shape resolution, constant folding, typed `OpParams`
  and the `Planner`.
- `model/format/`: the versioned prepared-model artifact.
- `python/`: the PyO3 binding, by path from `src/python/mod.rs`; `cli/`: the
  `fastnn-infer` implementation; `mod.rs`: facade and re-exports.

## Invariants

- Dependency direction is a DAG (`dtype/device/ops → error → shape/storage →
  tensor → backend → model`); nothing imports upward. `src/fn3/` imports no 2.x
  module and 2.x imports nothing from `inference`.
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
  and dtypes; the default body is `supports`. An `admits` override may confirm
  a `Partial` subset, never widen `Unsupported`.
- `PreparedModel` is immutable and `Send + Sync`; all mutable inference state
  lives in `Session`, which is `Send` but deliberately `!Sync`. A failed step
  changes neither the KV live extent nor the step counter.
- A `Session` executes every instruction through the one backend it was opened
  with; a backend whose device differs from the model's is rejected before any
  work. Value slots and the KV cache are device-resident when the backend can
  allocate there (`SessionOptions::default()`) and host-staged otherwise
  (`SessionOptions::staged()`, `FN3_VK_STAGED=1`); neither is a CPU path.
- A `Storage` is host-backed or device-resident (`Storage::from_resident`). A
  resident storage has **no host bytes**: every host accessor returns
  `Fn3Error::NotImplemented` and `to_host`/`download` is the only
  materialisation path, so stale bytes cannot be observed. CPU storage is
  never resident.
- A resident step is **one submission**: `begin_step(label, batch)` opens one
  command buffer, every instruction records into it, `end_step` submits and
  waits once. The driver infers no inter-dispatch ordering, so each kernel
  records an explicit compute→compute barrier over its bindings. A batched op
  may not read a host output and returns a typed error instead of stale bytes.
  The staged path does not batch. `queue_submissions` is the authoritative
  `vkQueueSubmit` counter.
- A failure while recording a step batch marks it partial; `end_step` discards
  it without submitting a prefix of the plan and reports
  `Fn3Error::StepDiscarded`, never `Ok(0)`. A failed step releases its value
  slots and leaves the KV extent, `step`, `tokens_consumed`, phase and plan
  cache unchanged. The multi-layer KV commit rolls every layer back to its
  pre-step extent if a device copy fails partway; bytes past the live extent
  are unobservable, so a retry cannot double-append.
- A device-level failure (device lost at submit, or a fence whose completion
  is unknown) is **terminal**: the device is poisoned, later operations return
  `Fn3Error::DeviceUnavailable` naming the adapter, and `Session::recover`
  never retries. Every other step failure is recoverable in place; `recover`
  drops pooled outputs and returns `Ok`.
- `vulkan-fault-injection` is a non-default, test-only feature: one `take()`
  per allocation, submit, fence, readback, recording and resident-copy seam,
  compiled out of every shipped configuration. Its output is injected
  evidence, never physical-device evidence.
- Immutable model data is uploaded once (`Backend::prepare_persistent`) and
  bound by handle; a prepared tensor reaching a host upload is a hard error
  (`check_staged_input`). Persistent entries are keyed by `Storage::generation`,
  never a host address, and released with the last handle. Unsafe code lives
  only in `storage.rs`, each block with a `SAFETY:` comment covering alignment,
  bounds, aliasing and lifetime.

## Validation

```bash
cargo fmt --check
cargo build --no-default-features
cargo test --lib inference
cargo test --test fn3_contracts
cargo test --test fn3_cpu_reference
```

`cargo test --lib` must also pass: it proves the 2.x tree still compiles beside
`src/fn3/`. Contract changes need a new or updated `tests/fn3_contracts.rs` case.
