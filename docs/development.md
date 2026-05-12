# Development Architecture

FastNN keeps the user-facing Python API stable while the Rust internals are split by responsibility. New functionality should fit into a narrow set of files instead of growing central modules.

## Target Rust Layout

```text
src/
  lib.rs                  # crate exports and module declarations
  error.rs                # Error types
  iterator.rs             # TensorIterator
  residual.rs             # Residual connection helper
  storage.rs              # Memory backend and device allocation
  storage_pool.rs         # Storage pooling for output tensor reuse
  storage_quantized.rs    # Quantized tensor storage
  dispatcher.rs           # Dynamic kernel dispatch (CPU vs GPU)
  packed_tensor.rs        # PackedTensor<T> — packed precision tensor
  packed_layer.rs         # PackedLinear<T> — packed linear layer
  packed_conv.rs          # PackedConv2d<T> — packed convolutional layer
  packed_train.rs         # MasterWeightOptimizer — training with packed weights
  python/
    mod.rs                # PyO3 module registration
    tensor.rs             # PyTensor bindings and DLPack hooks
    factories.rs          # tensor creation bindings
    ops.rs                # tensor op bindings
    nn.rs                 # neural network class bindings
    optim.rs              # optimizer class bindings
    io.rs                 # save/load bindings
    packed_tensor.rs      # packed tensor Python bindings
    packed_linear.rs      # packed linear Python bindings
    packed_conv.rs        # packed conv Python bindings
    packed_optim.rs       # packed optimizer Python bindings
    packed_quantized.rs   # quantized tensor Python bindings
  tensor/
    mod.rs                # Tensor and TensorImpl
    shape.rs              # view/reshape/transpose/permute/squeeze
    factories.rs          # zeros/ones/full/from_vec
    ops.rs                # elementwise, matmul, activations
    reductions.rs         # sum/mean/max/min/softmax
    device.rs             # CPU/GPU movement and dtype conversion
    indexing.rs           # slice/cat/stack/repeat/where/einsum
  kernels/
    constants.rs          # Kernel tuning constants
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
      arm_neon.rs         # ARM NEON SIMD kernels for packed GEMV
    gpu/
      mod.rs
      ops.rs              # GPU elementwise, matmul, reductions, embedding, fusion, optimizer ops
  backends/
    mod.rs
    cpu.rs                # CPU backend registration
    packed_simd.rs        # SIMD-accelerated packed GEMV kernels
    packed_blas.rs        # BLIS-style tiled packed micro-kernel
    wgpu/
      mod.rs
      mod_impl.rs
      shaders/
        conv_packed.wgsl  # WGPU packed convolution compute shader
  dtypes/
    mod.rs                # PackedWord trait
    u4x8.rs               # 4-bit packed type (8 × i4 per u32 word)
    u8x4.rs               # 8-bit packed type (4 × i8 per u32 word)
    f16x2.rs              # 16-bit packed type (2 × f16 per u32 word)
    f32x1.rs              # 32-bit packed type (1 × f32 per u32 word)
  swar/
    mod.rs
    ops_4bit.rs           # SWAR operations for 4-bit packed values
    ops_8bit.rs           # SWAR operations for 8-bit packed values
    ops_16bit.rs          # SWAR operations for 16-bit packed values
    ops_32bit.rs          # SWAR operations for 32-bit packed values
  autograd/
    mod.rs                # Node trait, metadata, no_grad
    engine.rs
    elementwise.rs
    reductions.rs
    matmul.rs
    conv.rs
    losses.rs
    views.rs
  nn/
    mod.rs                # Module trait, macros
    linear.rs
    conv.rs
    activations.rs
    attention.rs
    transformer.rs
    norm.rs
    dropout.rs
    embedding.rs
    pooling.rs
    fused.rs              # Fused Conv+BN+Activation layers
    sequential.rs
    residual.rs
    upsample.rs
    dag.rs                # Rust DAGExecutor for ONNX graph execution
  optim/
    mod.rs
    sgd.rs
    adam.rs
    adamw.rs
    muon.rs
    lion.rs
    rmsprop.rs
  io/
    mod.rs
    serialize.rs          # Model serialization (save/load)
    dlpack.rs             # DLPack interop (Rust only)
```

## Python Package Layout

```text
fastnn/
  __init__.py             # Public API exports
  _common.py              # Shared Python utilities
  _model_wrapper.py       # ModuleWrapperMixin for Python/Rust bridging
  activations.py          # Activation registry (register_activation, get_activation)
  callbacks.py            # Training callbacks (EarlyStopping, ModelCheckpoint, etc.)
  core.py                 # Core utilities (no_grad, set_seed, checkpoint)
  data.py                 # Dataset, TensorDataset, DataLoader, Samplers
  functional.py           # torch.nn.functional-style API (conv2d, batch_norm, etc.)
  init.py                 # Weight initialization (kaiming, xavier, orthogonal, etc.)
  layers.py               # Python-side layer wrappers (Flatten, PySequential, BasicBlock)
  losses.py               # Loss function wrappers
  module.py               # Module base class
  nn.py                   # NN module wrappers
  ops.py                  # Operation re-exports from _core
  optimizers.py           # Optimizer re-exports from _core
  parallel.py             # DataParallel for multi-GPU training
  precision.py            # Precision type system (Precision, Quantizer, PrecisionConfig)
  schedulers.py           # LR schedulers (StepLR, CosineAnnealingLR, etc.)
  tensor.py               # Tensor type with helpers
  typing.py               # Type aliases
  io/
    __init__.py           # Serialization API (save, load, convert_from_*, constants)
    serialization.py      # _save_model / _load_model implementations
    export.py             # PyTorch model export
    onnx.py               # ONNX importer (50 ops)
    graph_builder.py      # Model building from .fnn (auto-detect provenance)
    dag_model.py          # Python DAG prototype
    shape_inference.py    # Shape inference for ONNX ops
    graph_optimizer.py    # Graph optimization (dead node elimination, fusion)
    calibrate.py          # Calibration infrastructure
    act_calibrate.py      # Activation calibration (KL-divergence scale refinement)
    profiler.py           # Precision profiling & sensitivity analysis
    validate.py           # Model validation
  models/
    __init__.py           # Exports MLP, Transformer, create_mlp, BaseModel, YOLO
    base.py               # BaseModel with save/load
    builder.py            # create_mlp builder
    mlp.py                # MLP model
    transformer.py        # Transformer model
    yolo.py               # YOLO object detection wrapper
  utils/
    __init__.py           # Exports to_numpy, to_tensor, NMS utilities
    tensor_utils.py       # to_numpy, to_tensor conversion helpers
    nms.py                # NMS, yolo_decode, yolo_dfl_decode, xywh2xyxy, scale_boxes
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

## Adding A Packed Precision Type

To add a new packed precision (e.g., `U2x16`):

1. Define the type in `src/dtypes/` implementing `PackedWord` trait.
2. Add SWAR ops in `src/swar/` if needed.
3. Add SIMD kernels in `src/kernels/cpu/arm_neon.rs` and `src/backends/packed_simd.rs`.
4. Create `PackedTensor<T>` specialization in `src/packed_tensor.rs`.
5. Create `PackedLinear<T>` specialization in `src/packed_layer.rs`.
6. Create `PackedConv2d<T>` specialization in `src/packed_conv.rs`.
7. Add Python bindings in `src/python/packed_tensor.rs`, `src/python/packed_linear.rs`, `src/python/packed_conv.rs`.
8. Add `MasterWeightOptimizer<T>` in `src/packed_train.rs` and `src/python/packed_optim.rs`.
9. Register in `src/lib.rs` and `src/python/mod.rs`.

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

Fused GPU kernels currently live in `src/kernels/gpu/mod.rs`. A future refactor may split them into `src/kernels/gpu/fusion.rs` and `src/kernels/gpu/optim.rs`.
