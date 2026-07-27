# fastnn Roadmap

Current development priorities and historical planning documents.

---

## Current Focus

fastnn v2.6 is in release stabilization after the canonical tensor-contract,
compiler/runtime safety, signed quantization, and prepared grouped W4A8 MatMul
migration. New feature work is frozen until the release gates pass.

**Active areas:**

- **Release correctness** — package/version consistency, wheel smoke tests, and
  Linux/macOS/Windows/AArch64 validation
- **CPU numerical contracts** — signed I8/I4 endpoints, affine correction,
  malformed-metadata rejection, and deterministic model evidence
- **Prepared grouped W4A8 MatMul** — G32/G64/G128 integration, scalar fallback,
  AVX2 execution, durable compensation, and shared activation reuse

The active source-layout and ownership plan is
[Codebase Reorganization Roadmap](codebase-reorganization.md). It defines
contract-impact constraints, dependency direction, and the Rust/Python compiler
ownership decisions required before mechanical module moves.

The broader engineering backlog is in
[Architecture Improvement Backlog](architecture-improvements.md). The dtype and
quantization redesign is separately specified in
[Dtype and Quantization Redesign](dtype-redesign.md). CPU low-bit storage,
compute-family, and benchmark policy is evaluated in
[CPU Low-Bit Engine Direction](cpu-low-bit-engine.md).

The dated [Safety and Assurance Inventory](safety-inventory-2026-07-18.md)
records the pre-redesign unsafe, determinism, concurrency, and fuzzing
boundaries that constrain the dtype/storage migration.

**Post-v2.6 candidates:**

- Transformer-level grouped W4A8 quality and perplexity validation
- Profile-guided removal of remaining CPU hot-path allocations and copies
- Separate calibrated Conv2d quantization work after compiler-ordering repair

Grouped W4A8 Conv2d/YOLO, batched or dynamic-shape grouped MatMul, AVX-512,
distributed execution, and WGPU expansion are outside the v2.6 release scope.

For detailed performance work across all backends, see
[Performance Roadmap](../internals/performance-roadmap.md).

---

## Historical Roadmaps

Planning documents from prior development cycles. Preserved for context.

- [v2.3 Roadmap](v2.3-roadmap.md) — CPU benchmark expansion, telemetry, arena copy
  reduction, and backend module split planning

---

## See also

- [Documentation Home](../index.md) — full documentation index
- [Performance Roadmap](../internals/performance-roadmap.md) — GPU, fusion, and
  optimizer roadmap
