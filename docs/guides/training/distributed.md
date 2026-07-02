# Distributed Training

fastnn provides experimental support for multi-device and distributed training workloads.

> ⚠️ Distributed training APIs are experimental and may change.

## Data Parallel

Work in progress. For current multi-device workloads, use the compiled training pipeline to offload computation to GPU via the WGPU backend.

## WGPU GPU Training

When the `gpu` feature is enabled, fastnn can execute compiled training graphs on GPU (Vulkan, Metal, DX12):

```python
# Compiled training with GPU
model = fnn.compile_train_model(
    ...,
    device="gpu",  # or "cpu" (default)
)
```

### GPU Synchronization Policy

GPU execution is asynchronous by default. Synchronization happens only at explicit host boundaries:
- `Tensor::to_cpu()` / `Tensor.numpy()`
- Scalar extraction (`.item()`)
- DLPack or serialization paths

Avoid host scalar reads inside optimizer or model inner loops.

## See Also

- [Training Basics](training-basics.md)
- [WGPU Backend Internals](../../internals/architecture.md)
