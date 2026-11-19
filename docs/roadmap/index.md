# fastnn Roadmap

This page is the historical 2.x planning record, kept for context. The shipped
product is the 3.0 inference runtime specified in
[FastNN 3.0 Architecture](fn3-architecture.md); the 2.x surfaces that remain
and their callers are listed in the
[2.x disposition](../internals/legacy-2x.md).

---

## Historical 2.x Focus

fastnn v2.6.0 was released from the 2.x line. That work established an
architecture-neutral transformer and LLM runtime rather than adding a
model-specific GPT-2 path.

**Areas that were active in 2.x:**

- **Typed ONNX correctness** — exact mixed-dtype initializer round-trips,
  external-data validation, strict unsupported-operation errors, and faithful
  symbolic dimensions
- **Runtime shape semantics** — tensor-driven Reshape, Slice, Split, Expand,
  Range, and bounded semantic extents
- **Transformer execution foundations** — reusable prefill/decode, persistent
  session state, KV-cache ownership, and MHA/GQA geometry

The dependency order, GPT-2/TinyLlama/Qwen audit, and acceptance gates of that
period are in [Architecture-neutral LLM Runtime Roadmap](llm-runtime.md).

The 2.x source-layout and ownership record is
[Codebase Reorganization Roadmap](codebase-reorganization.md). It defines
contract-impact constraints, dependency direction, and the Rust/Python compiler
ownership decisions required before mechanical module moves.

The 3.0 inference-only rewrite is specified in
[FastNN 3.0 Architecture](fn3-architecture.md). It supersedes the training,
autograd, quantization and wgpu gates; the 2.x tree is retained only where the
[disposition](../internals/legacy-2x.md) names a caller.

The broader 2.x engineering backlog is in
[Architecture Improvement Backlog](architecture-improvements.md). The dtype and
quantization redesign was specified in
[Dtype and Quantization Redesign](dtype-redesign.md). CPU low-bit storage,
compute-family, and benchmark policy was evaluated in
[CPU Low-Bit Engine Direction](cpu-low-bit-engine.md).

The dated [Safety and Assurance Inventory](safety-inventory-2026-07-18.md)
records the pre-redesign unsafe, determinism, concurrency, and fuzzing
boundaries that constrain the dtype/storage migration.

**Candidates recorded after v2.6 (not the 3.0 plan):**

- Transformer-level grouped W4A8 quality and perplexity validation
- Profile-guided removal of remaining CPU hot-path allocations and copies
- Separate calibrated Conv2d quantization work after compiler-ordering repair

Grouped W4A8 Conv2d/YOLO, batched or dynamic-shape grouped MatMul, AVX-512,
distributed execution, and WGPU expansion are outside the v2.6 release scope.

For the 2.x performance work, see
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
