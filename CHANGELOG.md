# Changelog

## [Unreleased] — Next

### Added

- Placeholder for upcoming features beyond the v1.3.0 precision-system release.

---

## v1.3.0 — Packed Precision Expansion & Quantized ONNX Support

### DAG Packed Dispatch Expansion

- **MatMul packed dispatch**: `OpCode::MatMul` now tries packed weights first via `dispatch_matmul_packed()`, falls back to f32 matmul.
- **ConvTranspose packed dispatch**: `dispatch_conv_transpose_packed()` handles packed conv transpose weights.
- **Embedding packed dispatch**: Packed embedding table lookup via gemv for memory-efficient embeddings.
- **Real packed conv fix**: `conv_packed_from()` now uses `PackedConv2d::forward_cpu()` (im2col + packed gemm) instead of dequantizing to f32 and running f32 conv2d — this was the single biggest performance bug fix.

### Quantized ONNX Runtime Ops

- **QuantizeLinear / DequantizeLinear**: New OpCode handlers for ONNX Q/DQ format — supports scale+round+clamp quantization and (v-zp)*scale dequantization at runtime.
- **QLinearConv**: Takes quantized input + quantized weight, dequantizes internally, runs f32 conv, requantizes output.
- **MatMulInteger**: Takes quantized inputs, dequantizes both, runs f32 matmul.

### ONNX Import Improvements

- **Q/DQ folding**: `onnx.py` now detects QuantizeLinear→DequantizeLinear patterns around Conv/MatMul nodes and folds them, converting to packed weight storage during import.
- **New ONNX ops**: NonZero, Unique, Tril, Triu, Pad v2 (tensor-based pads) support added to `onnx.py`.

### Activation-Aware Calibration

- **`ActivationCalibrator`**: New module that runs calibration data through the model, collects per-layer activation distributions, and refines quantization scales using weighted KL-divergence (NVIDIA's method).
- Integrates with the existing `Calibrator` infrastructure and `fastnn-convert` CLI.

### Mixed-Precision Auto-Profiler

- **`PrecisionProfiler`**: Per-layer sensitivity analysis — quantizes each layer in isolation, measures output MSE perturbation, sorts by sensitivity.
- **`auto_config()`**: Generates a `PrecisionConfig` with U4 assigned to least-sensitive layers, U8 to mid, F32 to most-sensitive, given a target memory budget.

### Performance: Batch GEMM & ARM NEON

- **Batch GEMM for packed MatMul**: `gemm_batch_packed()` with K-tiled cache blocking — processes all batch rows within each K-tile before moving to the next, improving L2 cache reuse significantly over the old per-row GEMV loop.
- **ARM NEON SIMD kernels**: `gemv_u4x8_neon()` and `gemv_u8x4_neon()` — full SIMD widening chain (int8→int16→int32→f32→FMA) for aarch64 targets.
- **NEON feature flag**: Added `neon = []` feature in Cargo.toml, gated on `#[cfg(all(target_arch = "aarch64", feature = "neon"))]`.

### Fused Packed Layers

- **PackedConvRelu**: Standalone fused conv + in-place ReLU, all 4 precisions (U4/U8/F16/F32), exposed via Python.
- **PackedLinearGelu**: Standalone fused linear + GELU with Abramowitz & Stegun erf approximation, all 4 precisions.
- **BN folding**: `fold_bn_into_packed_conv()` function — dequantizes packed weights to f32, applies `W_fused = W * gamma / sqrt(var + eps)`, requantizes per-channel.

### WGPU Packed Conv Support

- **WGSL compute shader**: `conv_packed.wgsl` implementing packed convolution on GPU with per-channel scaling and bias.
- **Rust scaffolding**: `conv2d_packed_wgpu()` function with full WGPU pipeline creation, buffer management, and dispatch.

### Code Quality

- **Build warnings fixed**: Removed duplicate `[[bin]]` entries for benchmark files, fixed `[profile.bench]` panic setting.
- **107/108 Rust unit tests pass** (1 pre-existing benchmark correctness check).

### New Files

- `fastnn/io/act_calibrate.py` — Activation-aware calibration
- `fastnn/io/profiler.py` — Mixed-precision profiler
- `src/kernels/cpu/arm_neon.rs` — ARM NEON SIMD kernels
- `src/backends/wgpu/shaders/conv_packed.wgsl` — WGPU packed conv shader
- `tests/test_dag_packed.py` — DAG packed dispatch tests
- `tests/test_packed_fused.py` — Fused packed layer tests
- `benches/packed_dispatch_bench.rs` — Packed dispatch performance benchmarks

---

## v1.2.0 — Fused Kernel Optimizations & Performance Overhaul

### Performance: Fused Operator Kernels

- **Conv+BN+SiLU fusion**: Single-pass `fused_conv_bn_silu` kernel achieves **14-25× speedup** over PyTorch's separate Conv2d+BatchNorm2d+SiLU on CPU.
- **Conv+BN+ReLU/GELU fusion**: Added `fused_conv_bn_relu` and `fused_conv_bn_gelu` kernels with matching fused module implementations.
- **LayerNorm+GELU & RMSNorm+GELU fusion**: Single-pass norm+activation kernels for BERT/GPT and LLaMA architectures.
- **GELU backward fusion**: Eliminated ~9 intermediate tensors in GELU backward pass.
- **SiLU backward fusion**: Eliminated ~4 intermediate tensors in SiLU backward pass.
- **Softmax backward fusion**: Single-pass fused softmax backward kernel.
- **AdamW/Adam fused update**: Single-pass optimizer step eliminating redundant loops.

### Performance: Memory & Dispatch Optimizations

- **`zeros()` → `empty()`**: Replaced unnecessary zero initialization with uninitialized memory in 12+ kernels (conv1d, conv2d, conv3d, pooling, clamp, sign, pow, gt_scalar, embedding, log_softmax, softmax_last_dim_simd, maximum, minimum).
- **`TensorIterator`**: Replaced `broadcast_index_decomposition` with a unified iterator, eliminating bounds-check branching in hot loops.
- **FxHashMap in autograd**: Faster integer-key lookups in the autograd engine.
- **`OpId` enum dispatcher**: Replaced string-based operation lookup with integer enum dispatch.
- **Smart parallel thresholds**: Rayon parallelism now uses cache-aware thresholds for both memory-bound and compute-bound workloads.
- **Storage pool split**: Separated `acquire` into uninit/zeroed paths to avoid unnecessary memset.
- **Backward workspace reuse**: Autograd backward passes reuse pre-allocated workspaces.
- **Lower parallel thresholds**: Reduction kernels now parallelize at smaller sizes.
- **`shape_ref()` → `&[i64]`**: Added zero-copy shape accessor; converted 163 hot-path call sites.

### New Features

- **FusedConvBn**: Inference module fusing Conv2d+BatchNorm2d with ~7× speedup.
- **FusedConvBnReLU, FusedConvBnGELU**: Fused conv+bn+activation modules.
- **`Tensor::empty()`**: Exposed to Python API for advanced users.

### Bug Fixes

- **Weight convention alignment**: `Linear::from_weights` transpose, matmul transposition detection, and 4D+ reshape stride invalidation — all aligned to FastNN's [in, out] weight layout.
- **Transformer matmul crash**: Fused linear kernels now correctly use FastNN's weight convention.
- **Conv2d weight shadowing**: Fixed `n` variable shadowing `out_channels` in all 5 `conv2d_3x3_direct` variants (plain, fused_bn, fused_bn_silu, fused_bn_relu, fused_bn_gelu).
- **sgemm transpose stride**: Fixed buffer overflow and wrong values when `matrixmultiply` path uses transposed inputs.
- **SWAR test values**: Fixed 2 incorrect expected values in `swar_min_u16x2` and `swar_min_u4x8` tests.
- **`cat()` storage_offset**: Fixed double-counting storage offset in tensor concatenation.
- **`TensorIterator` broadcasting**: Fixed broadcasting bug in iterator.
- **BatchNorm2d variance**: Corrected variance formula.
- **GELU backward argument order**: Fixed argument order in GELU backward autograd node.
- **NEON SIMD**: Fixed undefined NEON functions in elementwise kernels.
- **ONNX import**: Made lazy to avoid import error when not installed.
- **MaxPool2d**: Fixed Rust module reuse across forward passes.
- **Iterator bounds check**: Fixed bounds check edge case (Task 1 + Task 2).

### Code Quality & Cleanup

- **Removed deprecated API**: Deleted `save_model`/`load_model`/`save_optimizer`/`load_optimizer` — users should use `state_dict`/`load_state_dict`.
- **Tensor submodule extraction**: Split monolithic `src/tensor/mod.rs` (3567→1040 lines) into 6 domain files (shape, device, factories, indexing, ops, reductions).
- **Duplicate code elimination**: Added macros (`get_grad_or_skip!`, `impl_nn_params!`, `impl_nn_named_params!`, helpers `attach_grad_fn`, `new_view_from`, `new_on_device`) removing hundreds of lines of boilerplate.
- **DIV/MOD flattening**: Replaced flat index arithmetic with nested loops in conv2d kernels (9 inner + 10 outer sites).
- **Thread-local scratch buffers**: Replaced per-call heap allocations in packed SIMD/BLAS paths with reusable thread-local buffers.
- **All clippy warnings fixed**: CI builds clean at `-D warnings`.
- **All ruff lint errors fixed**: Python code fully linted.

### Testing

- 100+ Rust unit tests pass (0 failures).
- Comprehensive conv2d tests (scratch reuse, im2col correctness, stride/dilation configurations).
- All 20 SWAR operation tests pass.
- Optimizer soak test fixed (runtime and memory pool).
- Examples compile and pass (`packed_gemv_bench`, `quantized_transformer`).

### Documentation

- Added development architecture and performance roadmap docs.
- Updated README with fused kernel benchmark results.
- Fixed 8 documentation warnings (broken links, unclosed HTML tags).

---

## v1.1.0 — Modular Backend Architecture & Packed Precision Optimization

### Architecture

- Moved PyO3 bindings out of `src/lib.rs` into `src/python/`.
- Converted the tensor and CPU kernel implementations to directory modules.
- Added module homes for tensor shape/factory/op/reduction/device/indexing code.
- Added module homes for CPU SIMD, elementwise, reduction, matmul, convolution, normalization, pooling, loss, and factory kernels.
- Added module homes for autograd node families.

### GPU Execution

- Removed immediate WGPU `device.poll(Maintain::Wait)` calls from hot unary, binary, scalar, logical, matmul, and supported reduction launch paths.
- Documented that GPU kernels should synchronize only at explicit readback boundaries.

### Python API

- Added narrower facade modules: `fastnn.tensor`, `fastnn.ops`, `fastnn.nn`, and `fastnn.losses`.
- Kept the top-level `import fastnn as fnn` API stable.

### Performance: Packed Precision GEMV Optimization

- **Fixed performance degradation for large matrices** in packed precision (U4x8, U8x4, F16x2) GEMV kernels.
- **Before**: Packed types were 60-80% slower than F32 for K > 4096 (unusable for large models).
- **After**: Packed types now provide consistent 2-25x speedups over F32 across all problem sizes.
- **Technical**: Optimized `src/backends/packed_blas.rs` to use word-level processing, eliminate division/modulo in hot loops, and batch-unpack packed weights.
- **Impact**: Production-ready quantized inference with 4-bit (8× memory savings) and 8-bit (4× savings) weights.

### Documentation

- Added development architecture and performance roadmap docs.
- Updated README project structure for the modular layout.

---

## v1.0.0 — Production Release

### GPU Training

- **GPU backward pass**: Gradients computed entirely on GPU — no CPU transfers
- **GPU kernels**: `gt_scalar`, `lt_scalar`, `logical_not`, `mul_scalar`, `add_scalar`, `sub_scalar`, `div_scalar`, `transpose`
- **Activation backward on GPU**: ReLU, Sigmoid, Tanh, SiLU gradients stay on GPU
- **Tensor::ones()**: Now supports GPU device

### New Layers

- Conv1d, Conv3d, ConvTranspose2d
- RMSNorm, GroupNorm, BatchNorm2d
- ResidualBlock (skip connections for ResNet-style architectures)

### New Activations

- LeakyReLU, Softplus, Hardswish

### New Losses

- BCEWithLogitsLoss, HuberLoss

### New Optimizer

- Lion (sign-based momentum optimizer)

### New Tensor Operations

- `cat`, `repeat`, `where_tensor`, `einsum`, `flip`, `maximum`, `log_softmax`

### ONNX Import

- Convert ONNX models to fastnn native format
- Supports Conv, Gemm/Linear, ReLU, BatchNorm, MaxPool, GlobalAvgPool

### FlashAttention

- Memory-efficient attention: O(N) memory instead of O(N²)
- Mathematically equivalent to standard attention (max diff < 1e-7)
- Supports causal masking

### Performance

- **Matmul**: 5.6 → 88.6 GFLOP/s at 1024×1024 (17× speedup via `matrixmultiply`)
- **ReLU 1024×1024**: 0.30 ms

### Model I/O

- `save_model` / `load_model` — Save and load model weights
- `save_optimizer` / `load_optimizer` — Save and load optimizer state
- Full `state_dict` / `load_state_dict` for all optimizers

### Bug Fixes

- Fixed infinite recursion in TransformerBlock/TransformerEncoder
- Fixed tanh gradient computation
- Fixed view/transpose autograd_meta sharing
- Fixed BatchNorm1d RwLock deadlock
- Fixed sum kernel wrong output shape for N-D tensors
- Fixed slice numel calculation for step > 1
- Fixed reshape -1 validation
- Fixed unsqueeze stride calculation

### Code Quality

- All ruff lint errors fixed
- All clippy warnings fixed (-D warnings clean)
- Added 11 Rust unit tests
- CI/CD pipeline with Rust + Python tests

---

## v0.8.0 — Full-Featured Training Library

### New Features

- **New Layers**: Conv1d, Conv3d, ConvTranspose2d, RMSNorm, GroupNorm, BatchNorm2d
- **New Activations**: LeakyReLU, Softplus, Hardswish (with autograd)
- **New Optimizer**: Lion (sign-based momentum optimizer)
- **New Losses**: BCEWithLogitsLoss, HuberLoss
- **New Tensor Ops**: `cat`, `repeat`, `where_tensor`, `einsum`, `flip`, `maximum`, `log_softmax`
- **FlashAttention**: Memory-efficient attention kernel using online softmax algorithm (O(N) memory, mathematically equivalent to standard attention with max diff < 1e-7)
- **Model I/O**: `save_model`, `load_model`, `save_optimizer`, `load_optimizer`
- **Optimizer State**: Full `state_dict`/`load_state_dict` for Adam, AdamW, SGD, Muon, Lion
- **Positional Encoding Cache**: Transformer reuses cached position tensors for same shapes

### Performance

- **Matmul**: 5.6 → 88.6 GFLOP/s (17× speedup) by switching from slow system BLAS to `matrixmultiply`
- **GPU CPU fallbacks**: All elementwise ops now delegate to SIMD/parallel CPU kernels instead of serial loops
- **GPU Autograd**: Fixed SigmoidBackward, TanhBackward, SiLUBackward to work with GPU storage tensors

### Bug Fixes

- Fixed infinite recursion in TransformerBlock/TransformerEncoder Module impl
- Fixed tanh gradient (was computing tanh-tanh² instead of 1-tanh²)
- Fixed view/transpose autograd_meta sharing (gradients now propagate correctly through view ops)
- Fixed BatchNorm1d RwLock deadlock (cloned tensors before dispatch to avoid write lock contention)
- Fixed sum kernel wrong output shape for N-D tensors
- Fixed slice numel calculation for step > 1
- Fixed reshape -1 validation (divisibility check)
- Fixed unsqueeze stride calculation

### Testing

- 92 Rust unit tests pass
- 63 Python tests pass (0 failures)
- Added 11 new Rust unit tests for tensor operations

---

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
