# fastnn contributor guide

## Scope

fastnn is an inference-only neural-network runtime with a Rust core and PyO3
bindings. There is no autograd, optimizer, training loop or wgpu backend.

- `src/fn3/` is the whole runtime, published as `fastnn::inference`: dtype and
  shape, storage and tensors, backends (CPU reference and Vulkan), strict ONNX
  import, the versioned prepared-model artifact, planning, session/KV state,
  the CLI and the Python binding.
- The 2.x AOT tree (graph IR, compiler passes, eager tensor/nn API, `.fnn` I/O,
  the quantized CPU pipeline, the Rust benchmark harness) was removed. It is
  recoverable at its archive tag; see `docs/archive/README.md`.
- The v3 redesign replaces the previous runtime with one compiler, one runtime and
  one artifact: `docs/v3/adr-0001-one-runtime.md` is the authority for where
  this tree is going. New work follows it rather than extending the previous shape.

## Entry points

- Rust public surface: `src/lib.rs`.
- Inference core: `src/fn3/`.
- Inference CLI: `src/bin/fastnn-infer.rs`.
- Python extension registration: `src/python/mod.rs`.
- Python public facade: `fastnn/inference/__init__.py`.
- Performance reports: `tools/perf/`; CI lanes: `tools/ci/`.
- Rust integration tests and Python tests: `tests/`.

## Change rules

- fastnn has no external users yet. Public paths, `fastnn._core`
  registrations, Python facade names, CLI behaviour, feature combinations and
  serialized formats may change when a cleaner architecture warrants it.
- Do not reintroduce a training, autograd, optimizer or wgpu path, and do not
  reintroduce the removed 2.x surface behind a new name.
- Keep mechanical moves separate from incidental numerical or performance
  changes. Do not retain compatibility re-exports for hypothetical users.
- The plan's unit is a fused segment; layout and precision are plan decisions,
  not runtime tags. A change that adds a per-op kernel per dtype per ISA is
  going the wrong way.
- A session owns its threads, arena and KV state. No process-global thread
  pool, no import-time pinning.
- Import and planning belong in Rust; Python is a facade, not a frontend.
- Rust unsafe code requires a local `SAFETY:` justification covering aliasing,
  bounds, alignment and lifetime assumptions.
- Quantized execution must preserve integer-space behaviour. Do not introduce
  unnecessary FP32 scratch buffers or Q/DQ pairs.
- Every performance or accuracy claim needs a measurement with the machine,
  thread count, median/p95 and memory recorded. A negative result is documented
  and the work closes; it is not retold as a win.
- Update only the closest relevant `AGENTS.md` when its ownership or invariants
  change. Do not add hand-maintained counts, versions or source line ranges.

## Validation

Run the narrowest relevant checks first, then the applicable gates:

```bash
cargo fmt --check
cargo build --no-default-features
cargo build
cargo build --no-default-features --features vulkan
cargo test --lib
cargo test --test fn3_contracts
cargo test --test fn3_ci_smoke
uv run pytest tests/ -m "not slow" -v
```

Physical-GPU work also requires `cargo test --features vulkan --test
fn3_vulkan_smoke` on the named device with zero validation errors.
