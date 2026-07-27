# fastnn Roadmap

Current development priorities and historical planning documents.

---

## Current Focus

fastnn v2.5 is stabilizing CPU performance work, improving maintainability,
and expanding toward GPU-resident execution and multi-device training.

**Active areas:**

- **CPU backend maturity** — benchmark expansion, copy-reduction, fused epilogues
- **Telemetry and observability** — arena allocation counters, dispatch profiling
- **Module maintainability** — splitting the large CPU backend module after perf work

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

**Upcoming:**

- WGPU-resident arena execution
- Multi-GPU training with real device-resident gradient synchronization
- Maintained WGPU benchmark baseline

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
