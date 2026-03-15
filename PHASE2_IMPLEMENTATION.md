# Phase 2 Implementation Notes

## Direct GPU-to-GPU Transfers

### Analysis
After reviewing the code, I found that:
1. `to_gpu()` is mainly used for CPU-to-GPU transfers (not GPU-to-GPU)
2. In DDP case, `x` and `y` are created on CPU using `fnn.randn()` and `fnn.randint()`
3. GPU-to-GPU transfers only happen when moving tensors between different GPU devices
4. wgpu's `copy_buffer_to_buffer` only works within the same device

### Decision
Given the complexity of implementing cross-device GPU transfers and the limited impact (since most transfers are CPU-to-GPU), I'll focus on other optimizations in Phase 2.

## Phase 2 Priorities

### 1. Profile Individual GPU Kernels
**Goal**: Identify slowest GPU operations in backward pass

**Implementation**:
- Add timing to GPU kernel execution
- Profile backward pass operations
- Identify top 5 slowest kernels

### 2. Optimize Slowest Kernels
**Goal**: Improve performance of identified bottlenecks

**Approach**:
- Review shader code for inefficiencies
- Optimize memory access patterns
- Consider vectorized operations

### 3. Add Performance Logging
**Goal**: Better visibility into kernel execution times

**Implementation**:
- Add timing macros to GPU kernels
- Log kernel execution times
- Create performance report

## Next Steps

1. Add timing to GPU kernels in `src/kernels/gpu/ops.rs`
2. Run benchmark with detailed timing
3. Identify slowest kernels
4. Optimize identified kernels
5. Validate improvements

Let me proceed with adding timing to GPU kernels.