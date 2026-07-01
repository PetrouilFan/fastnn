# fastnn Roadmap

Current development priorities and historical planning documents.

---

## Current Focus

fastnn v2.4 is stabilizing the v2.3 CPU performance work and expanding toward
GPU-resident execution and multi-device training.

**Active areas:**

- **CPU backend maturity** — benchmark expansion, copy-reduction, fused epilogues
- **Telemetry and observability** — arena allocation counters, dispatch profiling
- **Module maintainability** — splitting the large CPU backend module after perf work

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
