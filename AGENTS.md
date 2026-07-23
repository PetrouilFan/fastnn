# fastnn contributor guide

## Scope

fastnn is a Rust neural-network library with PyO3 bindings. It supports eager
tensor/autograd execution and AOT graph execution:

```text
GraphBuilder → ComputeGraph → compiler passes → Backend → CPU/WGPU dispatch
```

Read `docs/roadmap/codebase-reorganization.md` before structural work. It is
the source of truth for ownership, compatibility, and migration order.

## Entry points

- Rust public surface: `src/lib.rs`.
- AOT compilation pipeline: `src/backend/executor.rs`.
- IR and graph construction: `src/ir/`.
- CPU execution: `src/backend/cpu/`.
- Python extension registration: `src/python/mod.rs`.
- Python public facade: `fastnn/__init__.py`.
- Rust integration tests and Python tests: `tests/`.

## Change rules

- fastnn currently has no external users. Public paths, `fastnn._core`
  registrations, Python facade names, CLI behavior, feature combinations, and
  serialized formats may change when a cleaner architecture warrants it.
- Keep mechanical moves separate from incidental numerical or performance
  changes. Do not retain compatibility re-exports solely for hypothetical users.
- Do not resolve a dependency cycle by broadly changing items to `pub`.
- Rust unsafe code requires a local `SAFETY:` justification covering aliasing,
  bounds, alignment, and lifetime assumptions.
- Quantized execution must preserve integer-space behavior. Do not introduce
  unnecessary FP32 scratch buffers or Q/DQ pairs.
- Update only the closest relevant AGENTS.md when its ownership or invariants
  change. Do not add hand-maintained counts, versions, or source line ranges.

## Validation

Run the narrowest relevant checks first. Structural changes must use the
verification gates in `docs/roadmap/codebase-reorganization.md`; CPU changes
also require allocator/performance comparison and serialization changes require
legacy fixture coverage.

Useful commands:

```bash
cargo fmt --check
cargo test --lib
cargo test --test quantized_pipeline -- --test-threads=1
cargo test --test optim_test -- --test-threads=1
cargo test --test autograd_gradient_checks -- --test-threads=1
cargo test --test cpu_arena_telemetry -- --test-threads=1
uv run pytest tests/ -m "not slow" -v
```