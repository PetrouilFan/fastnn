# FASTNN UPGRADE ROADMAP

## Phase 1: Arena Memory Pool (Allocation Optimization)
- [x] Create `src/storage_pool.rs`.
- [x] Implement a `StoragePool` struct using a global `parking_lot::RwLock<HashMap<(usize, Device), Vec<Arc<Storage>>>>`.
- [x] Modify `TensorImpl::new_with_device` and `Tensor::empty`/`Tensor::zeros` in `src/tensor.rs` to acquire buffers from the pool.
- [x] Implement `Drop` on `TensorImpl` to return storage to the pool when `Arc::strong_count` drops to 1.
- [x] Add tests in `tests/test_tensor.py` to verify memory reuse.

## Phase 2: Muon Optimizer (SOTA Optimizer)
- [x] Create `src/optim/muon.rs` following the pattern of `adam.rs`.
- [x] Implement the Newton-Schulz iteration (5 matmuls/adds) for 2D weight matrices.
- [x] Add Nesterov momentum accumulation.
- [x] Expose `Muon` via PyO3 in `src/lib.rs`.
- [x] Create Python wrapper in `fastnn/optimizers.py` with fallback to AdamW for 1D parameters (biases/LayerNorm).
- [x] Add tests in `tests/test_nn.py`.
- [x] Fix Arc::make_mut usage in optimizers to use data_ptr_f32_mut() for in-place updates.
- [x] Fix Newton-Schulz numerical stability (add normalization and zero-norm handling).

## Phase 3: BF16 Support (Mixed Precision)
- [x] Add `BF16` to the `DType` enum in `src/storage.rs`.
- [x] Wire the `half` crate (already in `Cargo.toml`) for CPU storage and serialization.
- [x] Update `src/kernels/cpu.rs` and `src/tensor.rs` to handle `DType::BF16` operations via upcasting to F32 for compute.
- [x] Add PyO3 conversion logic for BF16.
- [x] Add F16 support alongside BF16.
- [x] Test BF16/F16 tensor creation, operations, and conversion.

## Phase 4: WGSL Shader Compilation Cache
- [x] Modify `src/kernels/gpu/mod.rs` (specifically `GpuContext`).
- [x] Add a caching mechanism that serializes compiled WGPU compute pipelines to `~/.cache/fastnn/shaders/`.
- [x] Generate cache keys based on `(op_name, dtype, hash)`.
- [x] Bypass `create_compute_pipeline` if the cache hits.
- [x] Updated all pipeline creation calls to pass dtype for cache key generation.

## Phase 5: Gradient Checkpointing
- [x] Update `src/autograd/engine.rs` to introduce a `CheckpointNode`.
- [x] Implement a `checkpoint(fn, inputs)` API that disables gradient storage during the forward pass.
- [x] Modify `apply_grad` for `CheckpointNode` to re-run the forward pass with gradients enabled to materialize activations just-in-time.
- [x] Expose the checkpoint function to Python in `fastnn/core.py` and `fastnn/__init__.py`.
- [x] Add PyO3 wrapper for checkpoint function in `src/lib.rs`.
- [x] Test checkpoint function API.

## Summary - All Phases Complete ✓
- **Phase 1**: Arena Memory Pool - Completed
- **Phase 2**: Muon Optimizer - Completed
- **Phase 3**: BF16/F16 Support - Completed
- **Phase 4**: WGSL Shader Compilation Cache - Completed
- **Phase 5**: Gradient Checkpointing - Completed
