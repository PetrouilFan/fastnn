# Development Architecture

FastNN keeps the user-facing Python API stable while the Rust internals are split by responsibility. New functionality should fit into a narrow set of files instead of growing central modules.

## Target Rust Layout

```text
src/
  lib.rs                  # crate exports and module declarations
  python/
    mod.rs                # PyO3 module registration
    tensor.rs             # PyTensor bindings and DLPack hooks
    factories.rs          # tensor creation bindings
    ops.rs                # tensor op bindings
    nn.rs                 # neural network class bindings
    optim.rs              # optimizer class bindings
    io.rs                 # save/load bindings
  tensor/
    mod.rs                # Tensor and TensorImpl
    shape.rs              # view/reshape/transpose/permute/squeeze
    factories.rs          # zeros/ones/full/from_vec
    ops.rs                # elementwise, matmul, activations
    reductions.rs         # sum/mean/max/min/softmax
    device.rs             # CPU/GPU movement and dtype conversion
    indexing.rs           # slice/cat/stack/repeat/where/einsum
    grad.rs               # requires_grad, grad, detach, clipping
  kernels/
    cpu/
      mod.rs              # register_cpu_kernels
      simd.rs
      elementwise.rs
      reductions.rs
      matmul.rs
      conv.rs
      norm.rs
      pooling.rs
      losses.rs
      factories.rs
      compare.rs
    gpu/
      mod.rs
      buffers.rs
      sync.rs
      bind_groups.rs
      elementwise.rs
      reductions.rs
      matmul.rs
      fusion.rs
      embedding.rs
      optim.rs
  autograd/
    mod.rs                # Node trait, metadata, no_grad
    engine.rs
    elementwise.rs
    reductions.rs
    matmul.rs
    conv.rs
    losses.rs
    views.rs
    checkpoint.rs
```

## Adding A Tensor Operation

Adding an op should normally touch these layers:

1. Tensor method in `src/tensor/ops.rs` or `src/tensor/reductions.rs`.
2. Dispatcher registration and backend kernel in `src/kernels/cpu/*` and, when supported, `src/kernels/gpu/*`.
3. Autograd node in `src/autograd/*` if gradients are supported.
4. PyO3 binding in `src/python/ops.rs`.
5. Python re-export in `fastnn/ops.py` and, if stable, `fastnn/__init__.py`.
6. Numeric, gradient, and import tests.
7. User-facing docs when the public API changes.

## GPU Synchronization Policy

GPU execution is asynchronous by default. Kernel launch helpers should submit command buffers and return GPU-backed tensors without calling `device.poll(Maintain::Wait)`.

Synchronization is allowed at explicit host boundaries:

- `Tensor::to_cpu`
- Python `Tensor.numpy()`
- scalar extraction such as `.item()`
- DLPack or serialization paths that require host data
- test-only explicit barriers

Avoid host scalar reads inside optimizer or model inner loops. If a GPU algorithm needs a scalar, compute it into a GPU buffer and consume it from the next GPU kernel.

## Adding A Fused GPU Operation

Prefer adding fused kernels for common epilogues and bandwidth-bound op chains:

- `matmul + bias`
- `matmul + bias + activation`
- `conv + bias + activation`
- `residual + add + norm`
- optimizer updates

Fused kernels should live in `src/kernels/gpu/fusion.rs` or `src/kernels/gpu/optim.rs`, with Python bindings only when the fused op is part of the public API.

