# Performance Roadmap

Backend performance priorities, completed milestones, and future work.

## P0 -- Completed

- Remove unconditional per-op GPU waits from unary, binary, matmul, scalar, and reduction launch paths. (v1.1.0)
- Expand fusion beyond fused_add_relu to matmul+bias, matmul+bias+GELU, matmul+bias+ReLU, conv+bias+activation, residual+add+norm. (v1.2.0+, residual+add+norm v2.2)
- Fused GPU optimizer kernels -- AdamW/Adam fused update (v1.2.0), compiled training with all 6 optimizers (v2.2).
- Real GPU reductions for N-D tensors with dynamic shader generation handling any dimensionality. (v1.2.0)
- Packed precision integration -- quantized dispatch for MatMul, ConvTranspose, Embedding; batch GEMM; ARM NEON SIMD kernels; WGPU packed conv shader. (v1.3.0)
- Compiler test suite -- 52 new pass tests for type_inference, auto_cast, activation_quantization, shape_inference, memory_planning, and edge cases. (v2.2)
- FlashAttention SIMD -- tiled online-softmax with AVX-512/AVX2 tile matmul, 2-4x speedup. (v2.2)
- ARM NEON validation suite -- CI cross-compilation, NEON kernel tests, cross-architecture consistency, Raspberry Pi benchmark runner. (v2.2)
- ONNX quantized op exporters -- QLinearMatMul, QLinearConv with per-channel scale/zp. (v2.2)
- Multi-architecture CI benchmarks -- cross-architecture consistency testing across x86, aarch64, and ARM NEON. (v2.3)
- CPU benchmark suite expansion -- elementwise, reductions, fusion, and optimizer loop coverage in cpu_baselines. (v2.3)

## P1 -- Current (v2.4)

- Reduce avoidable arena copies in scalar and elementwise CPU dispatch.
- Benchmark and optimize CPU matmul epilogues where internal kernels are used.
- Benchmark and optimize CPU reductions and normalization paths.
- Add CPU copy/allocation telemetry: arena temporary copies, copied bytes, TLS vector-pool allocation/reuse.
- Expand cpu_baselines with attention, norm, and quantized GEMM workloads.

## P2 -- Next

- Split src/backend/cpu/mod.rs into focused CPU modules after performance changes stabilize.
- Explore packed activation storage and optimizer state for numerically tolerant paths.
- ONNX/fastnn-native training graph export metadata.

## Beyond (future)

- WGPU-resident arena execution with maintained WGPU benchmark baseline.
- Multi-GPU training -- distribute tensor-parallel across wgpu devices.
- Process-based multiprocessing for DataLoader -- bypass GIL for true parallel data loading.
- 2-bit quantization and sparse computation.

## Current Status

- CPU storage pooling and arena allocation (zero-initialized for deterministic correctness on Linux and macOS aarch64) are in place.
- CPU benchmark suite cpu_baselines is the maintained performance baseline, being expanded for v2.4.
- WGPU context has buffer and bind-group caches; launch paths no longer poll after submit (since v1.1.0). N-D GPU reductions handle any dimensionality (v1.2.0). GPU embedding/gather kernel with WGSL shader (v1.2.0).
- Quantized dispatch includes Q/DQ folding in ONNX importer, QLinearConv, MatMulInteger, packed dispatch for MatMul/ConvTranspose/Embedding.
- Calibration and profiling: ActivationCalibrator with KL-divergence scale refinement, PrecisionProfiler for per-layer sensitivity analysis and automatic mixed-precision config.
- Fused CPU kernels: Conv+BN+SiLU (14-25x speedup), Conv+BN+ReLU, Conv+BN+GELU, LayerNorm+GELU, RMSNorm+GELU (since v1.2.0). Fused linear+activation: fused_linear_relu, fused_linear_gelu. Packed fused layers: Conv+ReLU, Linear+GELU with BN folding (v1.3.0).
- Compiled training: forward+backward+optimizer pipeline compiled to ExecutablePlan with 6 optimizers (SGD, Adam, AdamW, Muon, Lion, RMSprop) since v2.2.
- Residual+add+norm single-pass fusion (v2.2). WGPU packed shaders for U4/U8 quantized inference entirely on GPU via WGSL compute shaders (v2.2).
- ARM NEON validation runs in CI with cross-architecture consistency tests.
- Legacy packed layer classes (PackedLinear, PackedConv2d) removed in v2.1. All quantization goes through the AOT compiler's QuantizationPass.
- Explicit GPU readback paths still synchronize when mapping staging buffers.

Per-layer sensitivity analysis via PrecisionProfiler helps identify which ops benefit from higher precision, enabling targeted mixed-precision configurations.

## See also

- [Architecture](architecture.md) -- AOT compiler pipeline: IR, compiler passes, backends
- [Development](development.md) -- Codebase walkthrough and how-to guides
- [ARM NEON Backend](arm-neon.md) -- NEON SIMD kernel documentation and benchmarks
- [docs/index.md](../index.md) -- Documentation home
