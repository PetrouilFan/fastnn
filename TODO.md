# FASTNN UPGRADE ROADMAP

## Phase 1: Arena Memory Pool (Allocation Optimization)
- [x] Create `src/storage_pool.rs`.
- [x] Implement a `StoragePool` struct using a global `parking_lot::RwLock<HashMap<(usize, Device), Vec<Arc<Storage>>>>`.
- [x] Modify `TensorImpl::new_with_device` and `Tensor::empty`/`Tensor::zeros` in `src/tensor.rs` to acquire buffers from the pool.
- [x] Implement `Drop` on `TensorImpl` to return storage to the pool when `Arc::strong_count` drops to 1.
- [x] Add tests in `tests/test_tensor.py` to verify memory reuse.

## Phase 2: Muon Optimizer (SOTA Optimizer)
- [ ] Create `src/optim/muon.rs` following the pattern of `adam.rs`.
- [ ] Implement the Newton-Schulz iteration (5 matmuls/adds) for 2D weight matrices.
- [ ] Add Nesterov momentum accumulation.
- [ ] Expose `Muon` via PyO3 in `src/lib.rs`.
- [ ] Create Python wrapper in `fastnn/optimizers.py` with fallback to AdamW for 1D parameters (biases/LayerNorm).
- [ ] Add tests in `tests/test_nn.py`.

## Phase 3: BF16 Support (Mixed Precision)
- [ ] Add `BF16` to the `DType` enum in `src/storage.rs`.
- [ ] Wire the `half` crate (already in `Cargo.toml`) for CPU storage and serialization.
- [ ] Update `src/kernels/cpu.rs` and `src/tensor.rs` to handle `DType::BF16` operations via upcasting to F32 for compute.
- [ ] Add PyO3 conversion logic for BF16.

## Phase 4: WGSL Shader Compilation Cache
- [ ] Modify `src/kernels/gpu/mod.rs` (specifically `GpuContext`).
- [ ] Add a caching mechanism that serializes compiled WGPU compute pipelines to `~/.cache/fastnn/shaders/` using `serde_json`.
- [ ] Generate cache keys based on `(op_name, dtype, naga_version)`.
- [ ] Bypass `create_compute_pipeline` if the cache hits.

## Phase 5: Gradient Checkpointing
- [ ] Update `src/autograd/engine.rs` to introduce a `CheckpointNode`.
- [ ] Implement a `checkpoint(fn, inputs)` API that disables gradient storage during the forward pass.
- [ ] Modify `apply_grad` for `CheckpointNode` to re-run the forward pass with gradients enabled to materialize activations just-in-time.
- [ ] Expose the checkpoint function to Python in `fastnn/autograd.py`.
