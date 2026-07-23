# CPU backend guide

This directory implements `Backend` for CPU execution: instruction lowering,
arena dispatch, SIMD/scalar kernels, packed quantized paths, and allocation
telemetry.

## Ownership boundaries

- Keep `CpuBuffer` byte views, interior mutability, and arena safety invariants
  isolated from opcode lowering and kernel selection.
- Keep lowering distinct from normal and persistent-view dispatch.
- Keep operation-family routing distinct from leaf microkernels.
- Keep packed U4/U8 and other quantized kernels on their integer execution
  paths. Do not introduce FP32 scratch or redundant Q/DQ work as a refactor.

## Safety and performance

- Every unsafe block needs a `SAFETY:` explanation covering bounds, alignment,
  aliasing, and dispatch/lifetime assumptions.
- Preserve overlap handling for arena slices and existing thread-local scratch
  reuse behavior.
- Do not rename instruction kernel strings or change selection criteria in a
  move-only commit.
- Benchmark and allocator telemetry are required evidence for CPU control-plane
  changes; treat noise separately from numerical regressions.

## Change flow

For a new operation, trace the full path before editing: IR opcode, shape/type
rules, backend lowering, dispatch, kernel, autograd rule if trainable, and ONNX
mapping if imported. Keep such behavior additions separate from file moves.

Run targeted CPU, quantized-pipeline, prepared-plan, and arena telemetry tests
defined by `docs/roadmap/codebase-reorganization.md`.