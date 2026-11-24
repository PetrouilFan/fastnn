# FastNN inference core guide

`src/fn3/` is the inference-only core and the crate's primary API. The contract
authority is `docs/archive/roadmap/fn3-architecture.md`.

## Ownership

- `dtype.rs`, `ops/mod.rs`, `device.rs`, `error.rs`: leaf contracts; the error
  type depends on the first three only and none imports another core module.
- `shape.rs`, `storage.rs`, `tensor.rs`: shape/stride/bound algebra, the aligned
  buffer, and tensor metadata/views.
- `ir/`: the typed v3 graph (`DimExpr`, `TensorType`, `SourceLoc`, `Graph`) and
  `infer.rs`, the one shape pass. It owns the reading of `OpParams` for shape
  rules; a rule that needs a constant reads it from `OpParams`, never from a
  tensor payload. `import/typed.rs` is the only converter into it.
- `backend/mod.rs`: `Backend`, `Capabilities`, `Support`, `OpParams`, and the
  plan-time provider contract (`SegmentEngine`, `SegmentOperand`,
  `SegmentProvider`, `ScalarOnly`). `backend/window.rs` owns the pad/dilation/
  output-extent rules shared by the scalar kernels and the oneDNN engine, so a
  provider cannot disagree with `ir::infer` about a window. `backend/cpu/` is
  the portable scalar backend (`dispatch_kernels` is the one table the runtime
  and the Vulkan backend call); `backend/cpu/onednn/` is the opt-in oneDNN engine
  (`cpu-onednn`), split into `ffi.rs` (hand-written C API + RAII handles),
  `geometry.rs` (pure claim/prepare rules) and `mod.rs` (engine, prepack,
  blocks, the fused-epilogue rules and the channels-last layout policy).
  `backend/vulkan/` holds the platform and kernel families (`resident.rs` owns
  device-resident storage, the persistent table and the residency counters).
  It is compiled behind `vulkan`, but no session reaches it until phase 6
  (#108): this tree has no Vulkan test and no v3 execution path that selects
  it.
- `import/`: strict ONNX reader and graph construction. `import_onnx_model` adds
  the decoder geometry profile; `import_onnx_graph` stops at the typed graph, so
  a CNN imports without being forced through that profile.
- `model/mod.rs`: the artifact module tree and the weight store; session state
  is not here.
- `runtime/`: the v3 segment executor. `plan.rs` builds one slot per value
  (weights are read-only `SlotSource::Weight` slots backed by the importer's
  bytes or by an artifact mapping, never copied), a liveness-based offset table, and one `Segment` per
  node or per chain: `fuse.rs` folds `conv+activation`, the ONNX SiLU and
  `add(conv, other)` (with an optional activation after the sum) into
  `SegmentOp::Fused`, only when the chain removes at least one segment and every
  declared consumer of each intermediate value is inside it. `Plan::fusions` is
  the report and `Plan::unfused_segment_count` accounts for the difference;
  every segment records the engine the plan chose (`Scalar` or `OneDnn`) and
  `Plan::fallbacks` records what the offered provider declined. `Plan::layout`
  is a *policy* (`Plain` / `Chained`), never a descriptor: the oneDNN engine
  records the channels-last conversions it runs in `Plan::reorders` with their
  bytes. `arena.rs` owns the one 64-byte-aligned allocation and splits it into
  disjoint windows per segment; `threads.rs` builds the session's own
  `fn3-worker` Rayon pool (never the global one, never at import time) and its
  `run_blocks` drives oneDNN blocks; `exec.rs` validates named inputs, walks the
  segments, dispatches per segment engine and reads outputs back;
  `runtime/kernels/` holds the scalar NCHW kernels for the twelve CNN op ids,
  and `kernels/fused.rs` is the scalar reference for every fused form (same
  arithmetic order as the unfused chain, so a fused scalar plan is bit-identical
  to an unfused one).
- `import/graph.rs` also owns the decoder geometry profile, including
  `KvCacheConfig`/`KvCacheLayout`; allocation belongs to phase 6 (#108).
- `model/format/`: the versioned prepared-model artifact, format version 2
  (`FNN3ART2`). `artifact.rs` owns the fixed header, directory, hashes and the
  mmap reader; `graph.rs` owns the typed-IR codec and the full `OpParams`;
  `plan.rs` owns one stored plan per target; `weights.rs` owns the contiguous
  64-byte-aligned payload. There is no version-1 reader.
- `python/`: the PyO3 binding, by path from `src/python/mod.rs`; `cli/`: the
  `fastnn-infer` implementation; `mod.rs`: facade and re-exports.

## Invariants

- Dependency direction is a DAG (`dtype/device/ops → error → shape/storage →
  tensor → backend → model`), with `runtime` on top of `tensor`, `storage` and
  `backend`. It has exactly one upward edge:
  `backend::cpu::dispatch_kernels` calls `runtime::kernels::dispatch`, so
  `Support::Full` and the kernel behind it cannot drift apart. Anything else
  that imports upward is a bug. `src/fn3/` imports no 2.x module and 2.x
  imports nothing from `inference`.
- The v3 executor never allocates from its arena during a forward: the plan's
  offsets are fixed and `Arena::allocation_count` is constant after
  `Session::new`. Segment operands are bound as disjoint arena windows (a
  `split_at_mut` walk in `Arena::frame`); no plan bug can become aliasing UB.
  A session's pool is its own, sized exactly, and `Threading::Auto` means
  physical cores.
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
- The planner has no feature test: a session passes one `SegmentProvider`, the
  provider's claim is a pure function of `(op, operand shapes, dtypes,
  constant-ness, params)`, and a declined segment runs on the scalar kernels
  with its reason in `Plan::fallbacks`. A plan naming `SegmentEngine::OneDnn` is
  rejected by a build without `cpu-onednn`, never silently run scalar. Fusion is
  plan structure, not a provider decision: every fused form has a scalar
  implementation, and a provider that cannot take one gets a scalar segment
  (or a recorded fallback) rather than a re-expanded chain.
- A fused segment evaluates `anchor`, then the residual sum, then the
  activation, in that order; the recogniser only accepts graph shapes that mean
  that order. The scalar fused kernel and the engine's scalar epilogue use
  `backend::Activation::apply`, so a fused scalar plan and an unfused one agree
  bit for bit. The engine keeps `sum` as a oneDNN post-op (it reads the
  destination back, which the executor pre-fills with the residual) and applies
  the activation itself: measured, oneDNN's eltwise post-ops run on a reference
  path on this build and are slower than the scalar loop.
- The oneDNN engine requires a **sequential** oneDNN
  (`ONEDNN_CPU_RUNTIME=SEQ`, pinned by `tools/cmake/onednn-seq.cmake` through
  `.cargo/config.toml`); the session's pool is the only parallelism. The engine
  verifies `dnnl_version()->cpu_runtime` and falls back to serial block
  execution if a foreign build reports a threaded runtime. Every weight and bias
  it claims must be read-only model data and is prepacked at session open; a
  forward never reorders a weight, installs a data handle only for the duration
  of one `dnnl_primitive_execute`, and never touches a process-global oneDNN
  setting or cache. The user scratchpad is a session pool slot leased for one
  block call, not a buffer per block: there are 391-672 blocks on the YOLO
  graphs, and per-block buffers cost 27-29 MB. Blocks divide a convolution
  along output channels (whole groups for a grouped convolution, one channel
  per group for depthwise) or,
  when the channel split is limited by the output-channel count and re-reads
  the whole input per block, along output rows with a clamped halo: each row
  block's source is a row submemory, its destination a row submemory, and the
  rows outside the real input are supplied by that block's oneDNN padding. The
  dimension is a deterministic function of `(shape, params, worker count)`, is
  recorded in the prepared segment, and is reported by
  `Session::one_dnn_splits`. A segment never mixes layout families: the chained
  policy keeps an internal conv-to-conv edge channels-last only when the
  producer emits contiguous row ranges (one dense block counts, so the policy
  is live at one worker too) and every consumer is a oneDNN convolution, and it inserts one counted conversion
  at each family boundary; a fused residual forces the plain destination, and a
  channels-last destination forces a row split. Each block owns its primitive,
  stream and one leased scratchpad slot; the row blocks of one convolution
  share one read-only prepacked weight buffer.
- Plan preflight asks `Backend::admits` for the plan's concrete operand shapes
  and dtypes; the default body is `supports`. An `admits` override may confirm
  a `Partial` subset, never widen `Unsupported`.
- The prepared artifact is immutable and `Send + Sync`; all mutable inference
  state lives in `runtime::Session`, which is `Send` but deliberately `!Sync`.
  A failed forward leaves the arena and every output untouched.
- The Vulkan invariants below describe `backend/vulkan/`, which is compiled but
  unreachable from a session in this tree (#108). They are the contract for
  that backend's own work, not a statement about the CPU runtime.
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
cargo test --lib
cargo test --test fn3_cnn_import
cargo test --test fn3_cnn_run
cargo test --test fn3_constant_folding
cargo test --test fn3_shape_binding
cargo test --test fn3_artifact_v2
cargo test --no-default-features --features cpu-onednn --test fn3_onednn
cargo build --no-default-features --features cpu-onednn
cargo doc --no-deps --no-default-features
```

`tests/fn3_cnn_run.rs` needs `/home/petrouil/.hermes/yolov8n.onnx` and an ONNX
Runtime dump; generate the latter with
`python tools/parity/ort_reference.py <model> target/cnn-parity`. The test
prints that command when the dump is absent and never `#[ignore]`s itself.

The oneDNN engine is pinned to oneDNN's sequential CPU runtime by
`tools/cmake/onednn-seq.cmake` through `.cargo/config.toml`; a tree that built
oneDNN before that pin needs `cargo clean -p onednn-src`. The engine checks the
runtime it linked and reports it, so a foreign threaded build still runs, with
serial blocks. `tests/fn3_cnn_perf.rs` is opt-in (`FN3_CNN_PERF=scalar|onednn`,
one process per configuration) and is the only place latency claims are made.
With `--features segment-timing` it also prints a per-op table at one and eight
threads; that feature is instrumentation only, adds no clock read or session
field to a shipped build, and keeps its counters in the session rather than in
process-global state.

Contract changes need a new or updated test beside the code; there is no
separate contract file after the 2.x tree was deleted.
