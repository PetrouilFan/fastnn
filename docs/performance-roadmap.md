# Performance Roadmap

This roadmap tracks backend work that should be implemented alongside the modular reorganization.

## P0

- ~~Remove unconditional per-op GPU waits from unary, binary, matmul, scalar, and reduction launch paths.~~ ✅ Done (v1.1.0, 2024)
- ~~Expand fusion beyond `fused_add_relu` to common epilogues:
  - `matmul + bias`
  - `matmul + bias + GELU`
  - `matmul + bias + ReLU`
  - `conv + bias + activation`
  - `residual + add + norm`~~ ✅ Mostly done (v1.2.0, 2024). `residual + add + norm` still pending.
- ~~Add fused GPU optimizer kernels for SGD, Adam, AdamW, RMSprop, Lion, and Muon update paths.~~ ✅ Partially done — AdamW/Adam fused update (v1.2.0), others pending.
- Implement real GPU reductions for N-D tensors. CPU fallback should be reserved for unsupported device or dtype combinations.

## P1

- Remove scalar host synchronization inside optimizer loops, especially Muon normalization and Newton-Schulz steps.
- Add WGPU-side pooling for transient output, staging, scratch, and parameter/state buffers.
- Cache specialized matmul and reduction bind groups and shader variants by operation, dtype, shape class, and layout class.

## P2

- Add a GPU embedding/gather kernel so embedding-heavy models do not bounce through CPU execution.
- Integrate packed precision into full model execution paths, not only standalone GEMV and SWAR benchmarks.
- Explore packed activation storage and optimizer state for numerically tolerant paths.

## Current Status

- CPU storage pooling exists.
- GPU context code includes buffer and bind-group caches.
- WGPU unary, binary, scalar, matmul, and supported 2D reduction launch paths no longer poll immediately after submit (since v1.1.0).
- Explicit readback paths still synchronize when mapping staging buffers.
- Fused CPU kernels: Conv+BN+SiLU (14-25× speedup), Conv+BN+ReLU, Conv+BN+GELU, LayerNorm+GELU, RMSNorm+GELU (since v1.2.0).
- AdamW/Adam fused update step eliminates redundant loops (since v1.2.0).

