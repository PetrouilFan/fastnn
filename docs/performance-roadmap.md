# Performance Roadmap

This roadmap tracks backend work that should be implemented alongside the modular reorganization.

## P0

- ~~Remove unconditional per-op GPU waits from unary, binary, matmul, scalar, and reduction launch paths.~~ ✅ Done (v1.1.0)
- ~~Expand fusion beyond `fused_add_relu` to common epilogues:
  - `matmul + bias`
  - `matmul + bias + GELU`
  - `matmul + bias + ReLU`
  - `conv + bias + activation`
  - `residual + add + norm`~~ ✅ Done (v1.2.0+). `residual + add + norm` completed in v2.2.
- ~~Add fused GPU optimizer kernels for SGD, Adam, AdamW, RMSprop, Lion, and Muon update paths.~~ ✅ Partially done — AdamW/Adam fused update (v1.2.0), compiled training with all 6 optimizers (v2.2).
- ~~Implement real GPU reductions for N-D tensors.~~ ✅ Done (v1.2.0). Dynamic shader generation handles any dimensionality for sum/mean/max/min.
- ~~Integrate packed precision into full model execution paths.~~ ✅ Done (v1.3.0). Quantized dispatch for MatMul, ConvTranspose, Embedding; Q/DQ folding; batch GEMM for packed MatMul; ARM NEON SIMD kernels; WGPU packed conv shader.

## P1

- Remove scalar host synchronization inside optimizer loops, especially Muon normalization and Newton-Schulz steps.
- Add WGPU-side pooling for transient output, staging, scratch, and parameter/state buffers.
- Cache specialized matmul and reduction bind groups and shader variants by operation, dtype, shape class, and layout class.

## P2

- ~~Add a GPU embedding/gather kernel so embedding-heavy models do not bounce through CPU execution.~~ ✅ Done (v1.2.0). WGSL shader + dispatch + ops.rs registration.
- ~~FlashAttention SIMD optimization — tiled online-softmax with AVX-512/AVX2 tile matmul~~ ✅ Done (v2.2). 2-4× speedup over baseline.
- ~~Raspberry Pi benchmark suite — ARM NEON validation on physical hardware~~ ✅ Done (v2.2). CI cross-compilation, NEON kernel tests, cross-architecture consistency tests.
- ~~Full fused GPU optimizer kernels — all optimizers have GPU-native step functions~~ ✅ Done (v2.2). Compiled training with SGD, Adam, AdamW, Muon, Lion, RMSprop.
- ~~`residual + add + norm` fusion — single-pass skip connection + add + layer norm~~ ✅ Done (v2.2).
- Explore packed activation storage and optimizer state for numerically tolerant paths.
- ONNX training export support — enable autograd through the DAG and export .fnn graphs that preserve gradient flow.

## Beyond (future)

- Multi-GPU training — distribute tensor-parallel across wgpu devices
- Process-based multiprocessing for DataLoader — bypass GIL for true parallel data loading
- 2-bit quantization and sparse computation

## Current Status

- CPU storage pooling exists.
- GPU context code includes buffer and bind-group caches.
- WGPU unary, binary, scalar, matmul, and N-D reduction launch paths no longer poll immediately after submit (since v1.1.0).
- N-D GPU reductions (sum, mean, max, min) for any dimensionality (since v1.2.0+).
- GPU embedding/gather kernel with WGSL shader (since v1.2.0+).
- Explicit readback paths still synchronize when mapping staging buffers.
- Fused CPU kernels: Conv+BN+SiLU (14-25× speedup), Conv+BN+ReLU, Conv+BN+GELU, LayerNorm+GELU, RMSNorm+GELU (since v1.2.0).
- Fused linear+activation: `fused_linear_relu`, `fused_linear_gelu` (since v1.2.0).
- AdamW/Adam fused update step eliminates redundant loops (since v1.2.0).
- **Packed precision expansion**: Per-row packing, batch GEMM, ARM NEON `gemv_u4x8_neon`/`gemv_u8x4_neon`, WGPU packed conv shader (since v1.3.0).
- **Quantized dispatch**: Q/DQ folding in ONNX importer, `QLinearConv`, `MatMulInteger`, packed dispatch for MatMul/ConvTranspose/Embedding (since v1.3.0).
- **Calibration & profiling**: `ActivationCalibrator` with KL-divergence scale refinement, `PrecisionProfiler` for per-layer sensitivity analysis and automatic mixed-precision config (since v1.3.0).
- **Packed fused layers**: Conv+ReLU, Linear+GELU fusions with BN folding (since v1.3.0).
- **Compiled training**: Forward+backward+optimizer pipeline compiled to `ExecutablePlan` with 6 optimizers (SGD, Adam, AdamW, Muon, Lion, RMSprop) (since v2.2).
- **FlashAttention SIMD**: Tiled online-softmax with AVX-512/AVX2 tile matmul, 2-4× speedup (since v2.2).
- **WGPU packed shaders**: U4/U8 quantized inference entirely on GPU via WGSL compute shaders (since v2.2).
- **Residual+add+norm fusion**: Single-pass skip connection + add + layer norm (since v2.2).
- **ARM NEON validation suite**: CI cross-compilation, NEON kernel tests, cross-architecture consistency tests, Raspberry Pi benchmark runner (since v2.2).
- **ONNX quantized op exporters**: QLinearMatMul, QLinearConv with per-channel scale/zp (since v2.2).
- **Compiler test suite**: 52 new pass tests (type_inference, auto_cast, activation_quantization, shape_inference, memory_planning, edge cases) (since v2.2).

_Note: Legacy packed layer classes (`PackedLinear`, `PackedConv2d`, etc.) have been removed in v2.1. All quantization now goes through the AOT compiler's `QuantizationPass`._
