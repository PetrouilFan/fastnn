# Performance Roadmap

Performance priorities for the inference product.

## Current (3.0)

- **Vulkan kernel coverage.** Transform, embedding, pointwise, matmul, norm,
  reduce and fused attention run on the physical GTX 1080 Ti. Weights, value
  slots and the KV cache are device-resident by default (FN3-26), so kernels
  chain device-side and a step's operators share one batched submission; the
  KV appends and the logits download are still submitted separately.
  `FN3_VK_STAGED=1` keeps the old per-op staging path as a diagnostic control,
  not a supported execution mode.
- **Quantized weights.** Weight-only INT8 with FP32 activations ships on CPU
  and resident Vulkan. Packing happens at session setup from the FP32 weights
  in the unchanged `.fnn3` artifact, and the default path stays FP32; see
  [artifact INT8](../quantization/artifact-int8.md).
- **CPU reference speed.** The CPU backend stays correctness-first; the
  measured baseline is the `fn3-bench` harness, and the kernel and thread
  budget results are in
  [the release report](../reports/fn3-3.0.0-release.md#8-performance).
- **Plan reuse.** `PlanCache`, the slot pool and the KV append protocol are
  measured through `fn3_session`; regressions must show up as plan misses or
  extra allocations, not as tolerance changes.

## Secondary 2.x AOT tree

- Reduce avoidable arena copies in scalar and elementwise CPU dispatch.
- Benchmark and optimize CPU matmul epilogues, reductions and normalization.
- Keep `benches/cpu_baselines.rs` as the 2.x CPU baseline, including the
  copy/allocation telemetry counters.

## Deferred past 3.0

- The post-3.0 work list is the measured triage in
  [the 3.0.0 release report, section 8](../reports/fn3-3.0.0-release.md#8-performance):
  coalesce the KV copy rows, avoid the tied-head transpose in decode, add a
  `ReduceMean` fast path, then slot-pool eviction. Each needs a fresh parity
  sweep; none of them changes a result.
- A tiled/flash-style attention kernel to lift the current `keys <= 4096`
  bound (`MAX_KEYS` in `src/fn3/backend/vulkan/kernels/attention.rs`).

## Archive

Training, autograd, optimizer kernels, wgpu and WGSL performance work were
deleted in FN3-19 (#65). The measurements in the pinned 2.x revision `a2c0759`
are historical evidence and are not targets for this tree.

## See also

- [Architecture](architecture.md) -- inference core and the secondary AOT pipeline
- [Development](development.md) -- codebase walkthrough and test layout
- [ARM NEON Backend](arm-neon.md) -- NEON SIMD kernel documentation
- [docs/index.md](../index.md) -- documentation home
