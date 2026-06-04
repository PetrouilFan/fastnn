# Conv backend probe

Roadmap Task C1 adds `scripts/conv_backend_probe.py` to probe candidate optimized Conv backends before any integration work.

The script compares:

- fastnn isolated Conv+SiLU totals from `examples/conv_phase_bench.rs`
- PyTorch CPU `torch.nn.functional.conv2d` + `silu` on the same YOLO shapes
- PyTorch contiguous vs channels-last tensors
- several `torch.set_num_threads(...)` settings

PyTorch CPU uses its configured optimized CPU backend (oneDNN/MKLDNN when enabled). This is a probe only; it does not integrate PyTorch/oneDNN/ORT into fastnn.

## Command

```bash
OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/conv_backend_probe.py \
  --iters 30 \
  --warmup 5 \
  --threads 1,2,4 \
  --fastnn-iters 30 \
  --json /tmp/conv_backend_probe.json
```

## Representative result

```text
best comparisons
  stem_f16_c3_sp25600: fastnn_total=4.4332ms best_torch=0.6599ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=85.1%
  yolo_f32_c16_sp6400: fastnn_total=2.7314ms best_torch=0.3902ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=85.7%
  yolo_f32_c32_sp1600: fastnn_total=0.8787ms best_torch=0.2155ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=75.5%
  yolo_f64_c64_sp400: fastnn_total=0.5498ms best_torch=0.2108ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=61.7%
  yolo_f64_c64_sp1600: fastnn_total=2.3855ms best_torch=0.5420ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=77.3%
  yolo_f80_c80_sp1600: fastnn_total=3.1383ms best_torch=0.8399ms backend=torch_conv2d_silu_contiguous threads=4 torch_beats_by=73.2%
  yolo_f128_c128_sp100: fastnn_total=0.3770ms best_torch=0.2639ms backend=torch_conv2d_silu_channels_last threads=4 torch_beats_by=30.0%
json /tmp/conv_backend_probe.json
```

## Interpretation

The optimized PyTorch CPU path beats fastnn OpenBLAS Conv+SiLU by more than the C1 acceptance threshold on every probed YOLO shape, including repeated heavy shapes:

- `M=64,K=576,N=400`: PyTorch channels-last 4-thread path is ~62% faster than fastnn isolated OpenBLAS Conv+SiLU.
- `M=64,K=576,N=1600`: PyTorch channels-last 4-thread path is ~77% faster.
- `M=80,K=720,N=1600`: PyTorch 4-thread path is ~73% faster.

This supports the roadmap hypothesis: the remaining gap is not just GEMM. A backend/kernel with direct Conv scheduling, better memory format, fused epilogue, and thread scheduling can beat im2col+OpenBLAS by a large margin.

## Caveats

- This compares isolated shape kernels, not full YOLO graph execution.
- PyTorch includes mature backend selection and thread scheduling; copying the API is not equivalent to integrating oneDNN or another backend into fastnn.
- Exactness is not checked here because the probe uses independently generated tensors for PyTorch and fastnn phase bench. This is a backend-screening benchmark, not an integration validation.

## Next step

Task C1 passes its screening criterion. Before custom blocked-kernel work, the next decision point is whether to pursue an optional optimized Conv backend integration path (for example oneDNN/ORT-style direct Conv) or move to C3 and build a pure-Rust blocked kernel for the repeated `M=64,K=576,N=400` family. Any integration must remain feature-gated and must pass full-YOLO accuracy and speed gates.
