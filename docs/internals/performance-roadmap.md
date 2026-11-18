# Performance Roadmap

Performance priorities for the inference product.

## Current (3.0 alpha)

- **Vulkan kernel coverage.** Transform, embedding, pointwise, matmul, norm,
  reduce and fused attention run on the physical GTX 1080 Ti; the next step is
  removing per-op staging so kernels chain device-side.
- **CPU reference speed.** The 3.0 CPU backend is deliberately small and
  correctness-first; the measured baseline is the `fn3-bench` harness.
- **Plan reuse.** `PlanCache`, the slot pool and the KV append protocol are
  measured through `fn3_session`; regressions must show up as plan misses or
  extra allocations, not as tolerance changes.
- **Prepared-artifact load time.** Import and load each read the ONNX source;
  `GraphIr::from_parts` should make `load` artifact-only.

## Secondary 2.x AOT tree

- Reduce avoidable arena copies in scalar and elementwise CPU dispatch.
- Benchmark and optimize CPU matmul epilogues, reductions and normalization.
- Keep `benches/cpu_baselines.rs` as the 2.x CPU baseline, including the
  copy/allocation telemetry counters.

## Deferred past the alpha

- Quantized execution: one tested format with CPU and Vulkan kernels. The op-id
  and dtype-code ranges are reserved in `src/fn3/ops/mod.rs` and
  `src/fn3/dtype.rs`; no quantized kernel exists yet.
- A tiled/flash-style attention kernel to lift the current `keys <= 4096`
  bound.
- Device-resident `Storage` so model weights and the KV cache stay on the GPU
  between steps.

## Archive

Training, autograd, optimizer kernels, wgpu and WGSL performance work were
deleted in FN3-19 (#65). The measurements in the pinned 2.x revision `a2c0759`
are historical evidence and are not targets for this tree.

## See also

- [Architecture](architecture.md) -- inference core and the secondary AOT pipeline
- [Development](development.md) -- codebase walkthrough and test layout
- [ARM NEON Backend](arm-neon.md) -- NEON SIMD kernel documentation
- [docs/index.md](../index.md) -- documentation home
