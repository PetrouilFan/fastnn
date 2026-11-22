# Test-suite guide

`tests/` contains Rust integration tests and Python pytest tests. Prefer tests
that validate public behavior, serialization compatibility and numerical
results over assertions about private file layout.

## Test selection

- Inference core changes: `fn3_contracts`, `fn3_tensor_storage`,
  `fn3_cpu_reference`, `fn3_import`, `fn3_model_format`, `fn3_planner`,
  `fn3_session`, `fn3_e2e`, `fn3_cli`, `fn3_ci_smoke`.
- Vulkan changes: `fn3_vulkan_smoke` plus the kernel-family tests with
  `--features vulkan`; physical-device results must name the device UUID.
- Secondary 2.x AOT changes: shape inference, compiler edge cases, graph
  execution, quantized pipeline and arena telemetry.
- Python binding changes: modular API, tensor/nn forward, I/O and the
  `test_fn3_python.py` inference API.

## Conventions

- Keep Rust integration test entry points compatible with Cargo discovery.
- Use existing numerical helpers and appropriate tolerances; do not weaken a
  tolerance to conceal a regression.
- The pinned model file is not committed. A test that needs it must print a
  skip reason and must not claim a pass.
- Update or replace fixtures with the intended format when serialization-bearing
  types change; legacy-load coverage is unnecessary without users.
- Mark genuinely expensive pytest cases with existing markers rather than
  making unrelated tests slow.

Run the narrowest affected test first, then the structural verification gates
in `docs/archive/roadmap/codebase-reorganization.md` before declaring a move complete.
