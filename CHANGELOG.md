# Changelog

## v0.6.0 — Native Multi-Precision Support

### New Features

- **Native Packed Precision Types**: `U4x8` (4-bit, 8× memory savings), `U8x4` (8-bit, 4× savings), `F16x2` (16-bit, 2× savings), `F32x1` (32-bit baseline)
- **PackedWord Trait**: Uniform API for packing/unpacking values into u32 words
- **PackedTensor**: Packed storage with scale/zero dequantization, cache-line aligned allocation
- **PackedLinear**: Linear layer with auto backend selection (CPU/GPU) and memory savings reporting
- **MasterWeightOptimizer**: Adam optimizer with f32 master weights for stable low-bit training
- **SWAR Operations**: Element-wise add/sub/relu/max operating directly on raw u32 words — no unpacking
- **SIMD SWAR ReLU**: AVX2 sign-bit spreading, processes 8 u32 words at once
- **SIMD GEMV Kernels**:
  - U8x4: AVX2 int8→f32 widening via `_mm_cvtepi8_epi32`
  - U4x8: Branchless SIMD sign-extend via `_mm_sub_epi8`, 2× ILP throughput
  - F16x2: F16C hardware conversion via `_mm_set_epi32` + `_mm256_cvtph_ps` (zero allocation)
  - F32x1: Direct f32 reinterpret path
- **Batched GEMM**: Unpack each weight row once, process all N input vectors
- **wgpu Backend**: Runtime WGSL shader generation per packed type
- **Type-dispatched SIMD**: `TypeId`-based zero-cost dispatch to optimized kernels

### Performance (GEMV 4096×4096, AMD Ryzen 7 3700X, 8 threads)

| Type | Time | GFLOP/s | vs PyTorch f32 | Memory |
|------|------|---------|----------------|--------|
| PyTorch f32 (MKL) | 4.04ms | 8.3 | 1.0× | 64 MB |
| fastnn F16x2 | 1.80ms | 18.6 | 2.2× | 32 MB |
| fastnn U8x4 | 0.76ms | 44.4 | 5.3× | 16 MB |
| fastnn U4x8 | 0.55ms | 61.1 | 7.4× | 8 MB |

### Testing

- 57 unit tests (dtypes, SWAR ops, tensors, layers, SIMD kernels, training)
- 6 integration tests (2-layer forward+backward+optimizer per type, loss-decrease verification)
- PyTorch head-to-head benchmark suite (`bench_compare.py`)

### Dependencies Added

- `half = "2"` — f16/bf16 support for F16x2 type
- `bytemuck = "1"` — Safe transmute for packed buffer uploads

### Files Added

- `src/dtypes/` — PackedWord trait + 4 type implementations
- `src/swar/` — SWAR operations for 4-bit, 8-bit, 16-bit, 32-bit
- `src/packed_tensor.rs` — PackedTensor with aligned allocation
- `src/packed_layer.rs` — PackedLinear layer
- `src/packed_train.rs` — MasterWeightOptimizer
- `src/backends/packed_simd.rs` — SIMD-accelerated GEMV kernels
- `src/backends/packed_blas.rs` — BLIS-style tiled micro-kernel
- `benches/packed_bench.rs` — GEMV/ReLU benchmarks
- `tests/packed_integration.rs` — End-to-end integration tests

---

## v0.5.0 — Stability and Performance Fixes

Previous release. See git history for details.
