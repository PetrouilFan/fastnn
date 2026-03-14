# Phase 1 Progress Report
Date: 2026-03-13

## Completed Fixes

### 1. Tensor Constructor Invariants ✅

**Fixed Issues:**
- `TensorImpl::new()` now accepts `dtype` parameter instead of hardcoding `DType::F32`
- `TensorImpl::new_with_device()` now accepts `dtype` parameter
- Factory methods (`zeros`, `ones`, `full`) now honor the `device` parameter
- All calls to `TensorImpl::new()` in kernel files updated with correct dtype

**Files Modified:**
- `src/tensor.rs`: Lines 24-42, 44-66, 658-671, 673-710
- `src/kernels/cpu.rs`: Line 825 (create_output function)
- `src/kernels/gpu/ops.rs`: 16 locations (all TensorImpl::new calls)
- `src/kernels/gpu/mod.rs`: 4 locations (all TensorImpl::new calls)

**Test Results:**
- All tensor factory tests pass (15/15 in test_tensor.py)
- Device parameter is now properly honored
- Dtype parameter is now properly tracked in TensorImpl

### 2. Compilation Status ✅
- Rust compilation: SUCCESS (only warnings, no errors)
- All 6 warnings are pre-existing dead code issues, not related to our changes

## Pending Fixes in Phase 1

### 1. Autograd Engine Issues ⚠️ (Critical)
- Gradient accumulation still returns `None` for many operations
- Backward pass not properly propagating gradients
- `to_numpy()` round-trips in engine causing performance/memory issues

**Test Failures:**
- `test_abs_gradient_sign` - x.grad is None
- `test_gelu_gradient` - numerical mismatch
- `test_sigmoid_gradient` - numerical mismatch
- `test_tanh_gradient` - numerical mismatch
- `test_add_grad`, `test_mul_grad`, `test_div_grad`, `test_sub_grad` - gradient accumulation failure
- `test_exp_grad`, `test_log_grad`, `test_sqrt_grad`, `test_neg_grad` - gradient accumulation failure
- `test_matmul_grad`, `test_softmax_grad`, `test_layer_norm_grad`, `test_embedding_grad` - gradient accumulation failure

### 2. Tensor Factory Methods ⚠️ (Partial)
- `from_scalar()` and `from_vec()` still hardcode `DType::F32`
- These need to accept dtype parameter for full correctness

### 3. to_dtype and to_device ⚠️ (Partial)
- `to_dtype()` is currently a no-op (returns clone)
- `to_device()` is currently a no-op (returns clone)
- Need proper implementation for dtype/device conversion

## Next Steps for Phase 1

### Priority 1: Fix Autograd Engine
1. Replace queue-based backward traversal with proper topological sort
2. Remove `to_numpy()` round-trips in gradient accumulation
3. Implement tensor-native gradient accumulation
4. Fix broadcast-aware backward rules (AddBackward, SubBackward, etc.)

### Priority 2: Complete Tensor Factory Methods
1. Fix `from_scalar()` to accept dtype parameter
2. Fix `from_vec()` to accept dtype parameter

### Priority 3: Implement to_dtype and to_device
1. Implement proper dtype conversion
2. Implement proper device transfer

## Verification

### Compilation
```bash
cargo build
# Result: SUCCESS (with pre-existing warnings)
```

### Tests
```bash
uv run pytest tests/test_tensor.py -v
# Result: 15/15 PASSED

uv run pytest tests/test_gradients.py -v
# Result: 2/19 PASSED (17 failures due to autograd issues)
```

## Impact Assessment

### Positive Impact
- ✅ Tensor constructors now maintain proper invariants (dtype, device, storage agreement)
- ✅ Factory methods honor device parameter
- ✅ Code is more maintainable and correct
- ✅ No regression in tensor operation tests

### Neutral Impact
- ⚠️ Autograd issues still present (Phase 1B task)
- ⚠️ Performance unchanged (correctness fixes first)

### Next Benchmark
After fixing autograd, run comparison benchmarks to see if correctness fixes affect performance.
