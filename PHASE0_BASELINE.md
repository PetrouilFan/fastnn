# Phase 0 - Baseline Assessment
Date: 2026-03-13

## Current State Summary

### Build Status
- Rust compilation: ✅ Success (with warnings)
- Python bindings: ✅ Functional
- Tests: 57 collected, 37 passed, 20 failed

### Performance Baseline (Median times in microseconds)

| Operation | Size | Fastnn (μs) | PyTorch (μs) | Slowdown | Status |
|-----------|------|-------------|--------------|----------|--------|
| Add | 1000x1000 | 1894.4 | 280.9 | 6.7x | ⚠️ Slower |
| Mul | 1000x1000 | 2076.9 | 208.1 | 10.0x | ⚠️ Slower |
| ReLU | 1000x1000 | 1839.1 | 188.0 | 9.8x | ⚠️ Slower |
| FusedAddReLU | 1000x1000 | 2117.1 | 3386.3 | 0.6x | ✅ Faster |
| Sigmoid | 1000x1000 | 1384.0 | 753.1 | 1.8x | ⚠️ Slower |
| Tanh | 1000x1000 | 2270.3 | 1808.4 | 1.3x | ⚠️ Slower |
| GELU | 1000x1000 | 6674.0 | 1261.3 | 5.3x | ⚠️ Slower |
| MatMul | 256x512x256 | 2321.7 | 683.7 | 3.4x | ⚠️ Slower |
| Linear | 128x256x512 | 4856.9 | 559.1 | 8.7x | ⚠️ Slower |
| Max | 1000x1000 | 2085.2 | 535.5 | 3.9x | ⚠️ Slower |
| Sum | 1000x1000 | 975.0 | 74.6 | 13.1x | ⚠️ Slower |
| Mean | 1000x1000 | 1096.4 | 113.2 | 9.7x | ⚠️ Slower |

### Test Failures Summary

**Gradient Test Failures (20 total):**
1. `test_abs_gradient_sign` - Abs gradient returns None
2. `test_gelu_gradient` - Numerical mismatch
3. `test_sigmoid_gradient` - Numerical mismatch
4. `test_tanh_gradient` - Numerical mismatch
5. `test_add_grad` - Gradient accumulation failure
6. `test_mul_grad` - Gradient accumulation failure
7. `test_div_grad` - Gradient accumulation failure
8. `test_sub_grad` - Gradient accumulation failure
9. `test_exp_grad` - Gradient accumulation failure
10. `test_log_grad` - Gradient accumulation failure
11. `test_sqrt_grad` - Gradient accumulation failure
12. `test_neg_grad` - Gradient accumulation failure
13. `test_matmul_grad` - Gradient accumulation failure
14. `test_softmax_grad` - Gradient accumulation failure
15. `test_layer_norm_grad` - Gradient accumulation failure
16. `test_embedding_grad` - Gradient accumulation failure

**Passing Tests (37 total):**
- All autograd context tests pass
- All tensor creation tests pass
- All basic tensor ops pass (add, sub, mul, matmul, view, reshape, transpose)
- All I/O tests pass
- All NN forward pass tests pass
- All trainer tests pass

## Audit Report

### Critical Correctness Issues (Phase 1 Priority)

1. **Autograd Engine Issues:**
   - Gradient accumulation returns None for many operations
   - Backward pass not properly propagating gradients
   - `to_numpy()` round-trips in engine causing performance/memory issues

2. **Tensor Constructor Invariants:**
   - `TensorImpl::new` hardcodes DType::F32
   - Device parameter not honored in factory methods
   - `to_dtype()` and `to_device()` are no-ops (TODO)

3. **DLPack Interop:**
   - `from_dlpack_capsule` returns placeholder `Tensor::from_scalar(0.0)`

4. **Slice/Semantics:**
   - Incorrect slice end clamping (uses `size - 1` instead of exclusive end)
   - Storage offset not applied in `as_f32_slice`

### Performance Hotspots

1. **Elementwise Operations:**
   - ReLU: 9.8x slower than PyTorch
   - Sigmoid: 1.8x slower (AVX2 approximation vs scalar)
   - Tanh: 1.3x slower (approximation issues)

2. **Matrix Operations:**
   - MatMul: 3.4x slower (BLAS integration issue?)
   - Linear: 8.7x slower (likely due to bias+activation overhead)

3. **Reductions:**
   - Sum: 13.1x slower
   - Mean: 9.7x slower
   - Max: 3.9x slower

### Code Quality Issues

1. **Dead Code:**
   - `abs_parallel_avx2` and `abs_parallel_scalar` unused
   - `SimdLevel::Scalar` and `SimdLevel::Avx512` variants never constructed

2. **Unused Variables:**
   - `four` and `four_neg` in CPU kernels
   - `grad_gamma` in autograd (unused variable `g`)

## Phase 1 Plan

### Immediate Fixes (Correctness First)

1. **Fix Tensor Constructor Invariants** (`src/tensor.rs`)
   - Ensure dtype, device, storage, sizes, strides, storage_offset always agree
   - Honor device parameter in factory methods (zeros, ones, full)
   - Implement proper `to_dtype()` and `to_device()`

2. **Fix Autograd Engine** (`src/autograd/engine.rs`)
   - Replace queue-based traversal with proper topological backward pass
   - Remove `to_numpy()` round-trips
   - Implement tensor-native gradient accumulation

3. **Fix Broadcast-Aware Backward Rules**
   - Implement `sum_to_shape` utility
   - Fix AddBackward, SubBackward, MulBackward, DivBackward for broadcast
   - Fix MatmulBackward for batch-broadcast cases

4. **Fix Slice Semantics**
   - Correct exclusive-end slicing
   - Apply storage_offset in view operations

### Testing Strategy

1. Add finite-difference gradient tests for all failing operations
2. Add shape/stride/storage-offset property tests
3. Add cross-backend parity tests

## Next Steps

1. Start with tensor constructor fixes (Phase 1)
2. Then fix autograd engine (Phase 1)
3. Then fix backward rules (Phase 1)
4. Run tests after each change to ensure no regressions
5. Update performance baselines after correctness is fixed
