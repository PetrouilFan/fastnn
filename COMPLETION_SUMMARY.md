# GPU Kernel Dispatch Fix - Completion Summary

## Objective
Fix the GPU kernel dispatch system so that binary operations (add, sub, mul, div, matmul, etc.) correctly route to GPU kernels when either tensor is on GPU, rather than incorrectly using CPU kernels.

## Root Cause Found
The dispatch system was only checking the first tensor's device (`self.device()`) to determine which kernel to call, not both tensors' devices. This caused operations like `cpu_tensor + gpu_tensor` to use the CPU kernel, which would then panic when trying to access GPU memory.

## Changes Made

### 1. Fixed Binary Operation Dispatch Logic (`src/tensor.rs`)
Updated the following methods to check both tensor devices:
- `add()` - Lines 1296-1321
- `sub()` - Lines 1378-1402
- `mul()` - Lines 1404-1428
- `div()` - Lines 1430-1450
- `matmul()` - Lines 1452-1472
- `fused_linear_gelu()` - Lines 1658-1676

The new dispatch logic:
```rust
let dispatch_key = match (self.device(), other.device()) {
    (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
    (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
    _ => device_to_dispatch_key(Device::Cpu),
};
```

### 2. Added Slicing Support (`src/lib.rs`)
Updated `PyTensor::__getitem__` to support both integer indices and slice objects:
- Accepts `&Bound<'_, PyAny>` instead of `usize`
- Checks if input is a `PySlice` and handles it appropriately
- Falls back to integer indexing for backward compatibility

### 3. Added `to_gpu` Methods
Added `to_gpu` methods to:
- `PyTensor` - For moving individual tensors to GPU
- `Linear` - For moving linear layer parameters to GPU
- `Conv2d` - For moving conv layer parameters to GPU
- `PySequential` (Python) - For moving all layers in a sequence to GPU

### 4. Implemented `bucket_allreduce` Function
Added a basic implementation of `bucket_allreduce` in `src/lib.rs`:
- Averages gradients across GPU replicas
- Handles parameter synchronization for DDP

### 5. Fixed Python Module Registration
Fixed the registration of `bucket_allreduce` in the `_core` module:
- Changed `wrap_pyfunction!(bucket_allreduce, m)` to `wrap_pyfunction!(bucket_allreduce, py)`

## Testing Results

### 1 GPU Benchmark
✅ **SUCCESS** - The 1 GPU benchmark runs successfully, completing all 5 epochs.

### 2 GPU DDP Benchmark
⚠️ **PARTIAL SUCCESS** - The benchmark now gets past the initial setup and into actual computation, but encounters a GPU kernel dispatch limit:

```
wgpu error: Validation Error
Caused by:
  In ComputePass::end
    In a dispatch command, indirect:false
      Each current dispatch group size dimension ([65536, 1, 1]) must be less or equal to 65535
```

This error indicates that a GPU kernel is trying to dispatch with a workgroup size of 65536, which exceeds the hardware limit of 65535. This is a separate issue from the dispatch logic and requires modifying the GPU kernels to process data in chunks.

## Files Modified
- `src/tensor.rs` - Fixed dispatch logic for binary operations
- `src/lib.rs` - Added slicing support, `to_gpu` methods, and `bucket_allreduce` function
- `fastnn/__init__.py` - Added `to_gpu` method to `PySequential`
- `fastnn/parallel.py` - Updated to use `to_gpu` methods instead of `.data` attribute

## Conclusion
The GPU kernel dispatch system has been successfully fixed. Binary operations now correctly route to GPU kernels when either tensor is on GPU. The 1 GPU benchmark runs successfully, and the 2 GPU DDP benchmark gets much further than before.

The remaining issue (GPU kernel dispatch limit) is a separate problem related to GPU kernel implementation and would require modifying the kernels to process data in chunks rather than all at once.
