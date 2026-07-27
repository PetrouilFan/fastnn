# Test-suite guide

`tests/` contains Rust integration tests and Python pytest tests. Prefer tests
that validate public behavior, serialization compatibility, and numerical
results over assertions about private file layout.

## Test selection

- IR/compiler changes: shape inference, compiler edge cases, graph execution,
  and quantized pipeline tests.
- Autograd/training changes: gradient checks, compiled-training, and optimizer
  tests.
- CPU backend changes: CPU reference/oracle, quantized pipeline, prepared-plan,
  and arena telemetry tests; compare allocator/performance behavior where
  required.
- Python facade/binding changes: modular API, tensor/nn/optimizer, I/O, and
  AOT executor tests.
- ONNX/I/O changes: ONNX import/execute, graph optimizer, shape inference, and
  serialization round trips.

## Conventions

- Keep Rust integration test entry points compatible with Cargo discovery.
- Use existing numerical helpers and appropriate tolerances; do not weaken a
  tolerance to conceal a regression.
- Update or replace fixtures with the intended format when serialization-bearing
  types change; legacy-load coverage is unnecessary without users.
- Mark genuinely expensive pytest cases with existing markers rather than
  making unrelated tests slow.

Run the narrowest affected test first, then the structural verification gates
in `docs/roadmap/codebase-reorganization.md` before declaring a move complete.