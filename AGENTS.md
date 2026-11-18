# fastnn contributor guide

## Scope

fastnn is an inference-only neural-network runtime with a Rust core and PyO3
bindings. There is no autograd, optimizer, training loop or wgpu backend.

- `src/fn3/` is the primary API: contracts, storage, the CPU reference, the
  Vulkan platform and kernels, strict ONNX import, the versioned prepared-model
  artifact, the planner, session/KV state, the CLI and the Python binding.
- The remaining 2.x modules (`ir`, `compiler`, `backend`, `tensor`, `nn`,
  `onnx`, `io`) are a secondary inference surface kept for the legacy AOT path
  and its quantized CPU pipeline.

## Entry points

- Rust public surface: `src/lib.rs`.
- Inference core: `src/fn3/`.
- AOT compilation pipeline: `src/backend/executor.rs`.
- IR and graph construction: `src/ir/`.
- CPU execution: `src/backend/cpu/` and `src/fn3/backend/cpu/`.
- Inference CLI: `src/bin/fn3-infer.rs`.
- Python extension registration: `src/python/mod.rs`.
- Python public facade: `fastnn/fn3/__init__.py`.
- Rust integration tests and Python tests: `tests/`.

## Change rules

- fastnn has no external users yet. Public paths, `fastnn._core`
  registrations, Python facade names, CLI behavior, feature combinations and
  serialized formats may change when a cleaner architecture warrants it.
- Do not reintroduce a training, autograd, optimizer or wgpu path. A feature
  that needs one is a new product decision, not a refactor.
- Keep mechanical moves separate from incidental numerical or performance
  changes. Do not retain compatibility re-exports solely for hypothetical users.
- Do not resolve a dependency cycle by broadly changing items to `pub`.
- `src/fn3/` imports no 2.x module and 2.x imports nothing from `fn3`.
- Rust unsafe code requires a local `SAFETY:` justification covering aliasing,
  bounds, alignment and lifetime assumptions.
- Quantized execution must preserve integer-space behavior. Do not introduce
  unnecessary FP32 scratch buffers or Q/DQ pairs.
- Update only the closest relevant AGENTS.md when its ownership or invariants
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
cargo test --test quantized_pipeline -- --test-threads=1
uv run pytest tests/ -m "not slow" -v
```

Physical-GPU work also requires `cargo test --features vulkan --test
fn3_vulkan_smoke` on the named device with zero validation errors.
