# Breaking Changes in v0.8.0

## Exception Handling

All `RuntimeError` exceptions are now typed. Update all `except RuntimeError` handlers in your agent code:

| Old Pattern | New Pattern |
|-------------|-------------|
| `except RuntimeError` (shape mismatch) | `except fastnn.ShapeError` |
| `except RuntimeError` (OOM/GPU) | `except fastnn.DeviceError` |
| `except RuntimeError` (backward pass) | `except fastnn.AutogradError` |
| `except RuntimeError` (optimizer) | `except fastnn.OptimizerError` |
| `except RuntimeError` (save/load) | `except fastnn.IoError` |
| `except RuntimeError` (CUDA) | `except fastnn.CudaError` |

All exceptions inherit from `fastnn.FastnnError`, which inherits from `RuntimeError`. A broad `except RuntimeError` will still catch them, but `except fastnn.FastnnError` is the recommended pattern.

```python
# Before
try:
    out = model(x)
except RuntimeError as e:
    print(f"Error: {e}")

# After
try:
    out = model(x)
except fastnn.ShapeError as e:
    # Handle shape mismatch specifically
    reset_environment()
except fastnn.FastnnError as e:
    # Catch-all for fastnn errors
    print(f"fastnn error: {e}")
```

## DLPack / NumPy Interop

`np.from_dlpack(tensor)` now returns a **zero-copy view** into Rust memory.

**Critical behavioral change:**
- Mutating the returned NumPy array **mutates the Rust tensor in-place**
- This was not possible before (old `to_numpy()` returned a copy)

```python
t = fastnn.randn([64, 64])
arr = np.from_dlpack(t)
arr[0, 0] = 99.0  # THIS MUTATES THE RUST TENSOR

# Verify:
print(t.numpy()[0, 0])  # 99.0
```

**Safe usage patterns:**
- Read-only observation extraction: `obs = np.from_dlpack(tensor)`
- Logging and visualization
- In-place mutation only when you own the tensor exclusively and have no gradient tracking

**Unsafe usage patterns:**
- Mutating during training when gradients are tracked
- Sharing the view across threads without synchronization

## Optimizer API

`PyAdamW` and `PyAdam` now support `no_decay` per-parameter-group:

```python
opt = fastnn.PyAdamW(params, lr=1e-3, weight_decay=0.01)

# Mark 1D parameters (biases) to skip weight decay
opt.mark_biases_no_decay()

# Or manually specify which parameters skip weight decay
opt.add_no_decay([2, 5])  # indices of params to skip
```

**Default behavior:** All parameters receive weight decay unless explicitly marked.

## Checkpoint Format

Checkpoints now include format versioning and magic bytes. Old `.fnn` files are **not compatible** with v0.8.0.

**Migration:** Re-save all checkpoints using v0.8.0.

```python
# Old checkpoint format cannot be loaded
# Re-train and re-save, or use a migration script
model = load_your_model()
fastnn.save_model(model, "model_v080.fnn")
```

## Autograd

`Node::apply()` signature changed from `&[Option<Tensor>]` to `Vec<Option<Tensor>>`. This is an internal API change that should not affect Python users, but custom Rust extensions implementing autograd nodes will need to be updated.

## Safety

`as_f32_slice()` and `as_f32_slice_mut()` now perform unconditional bounds checking in release builds. Previously, out-of-bounds access in release mode would cause undefined behavior. Now it panics with a descriptive error message.

## GIL Release

Compute-intensive operations (`matmul`, `relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `softmax`, `backward`) now release the GIL during execution. This enables true multi-threaded Python code when using fastnn.

**Note:** If you have code that relied on the GIL for synchronization between fastnn calls and other Python code, you may need to add explicit locking.
