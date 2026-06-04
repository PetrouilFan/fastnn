# OpenBLAS Conv GEMM integration

This slice adds a feature-gated OpenBLAS path inside `conv2d_f32_im2col_gemm`.

## Scope

- Enabled only when fastnn is built with `--features openblas`.
- Default builds continue to use `matrixmultiply::sgemm`.
- The OpenBLAS path handles both current Conv GEMM layouts:
  - 1x1 Conv fast path: `B=[K,N]` row-major, CBLAS `NoTrans`.
  - general im2col Conv: im2col is `[N,K]` row-major but viewed as `B=[K,N]`, CBLAS `TransB`.
- Bias and fused activation behavior are unchanged.

## Validation results

Fresh no-OpenBLAS baseline, 10 YOLO iterations:

```text
mean_ms=50.9376
median_ms=50.6213
conv2d_silu total_ms=43.9910
max_abs_vs_pytorch=0.000457763671875
mean_abs_vs_pytorch=1.404639e-06
```

OpenBLAS build with `OPENBLAS_NUM_THREADS=2`, 10 YOLO iterations:

```text
run A:
mean_ms=44.0441
median_ms=44.4927
conv2d_silu total_ms=36.3903
max_abs_vs_pytorch=0.00048828125
mean_abs_vs_pytorch=1.406245e-06

run B after final gates:
mean_ms=46.2378
median_ms=47.6481
conv2d_silu total_ms=38.3627
max_abs_vs_pytorch=0.00048828125
mean_abs_vs_pytorch=1.406245e-06
```

OpenBLAS build with `OPENBLAS_NUM_THREADS=4`, 10 YOLO iterations:

```text
mean_ms=46.6557
median_ms=46.4234
conv2d_silu total_ms=36.9135
max_abs_vs_pytorch=0.000518798828125
mean_abs_vs_pytorch=1.389609e-06
```

OpenBLAS build with `OPENBLAS_NUM_THREADS=8` regressed badly in an earlier sample:

```text
mean_ms≈77.2
```

Default fastnn vs OpenBLAS fastnn on the same deterministic input:

```text
max_abs_vs_default=0.000335693359375
mean_abs_vs_default=1.034235e-06
p99_abs_vs_default=3.051758e-05
```

The differences are expected fp32 accumulation-order differences from a different GEMM backend.

## Verdict

`OPENBLAS_NUM_THREADS=2` is the best measured setting on this machine for full YOLO. The feature-gated path gives a real end-to-end CPU speedup while preserving normal fp32 accuracy.

Do not enable OpenBLAS by default yet. Keep it as an opt-in build feature until dependency/build/threading policy is finalized.

Recommended usage for YOLO CPU inference experiments:

```bash
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
  .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'

OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 8 --warmup 3 --iters 10
```

Runtime escape hatch for A/B testing or machines where OpenBLAS regresses:

```bash
FASTNN_DISABLE_OPENBLAS_CONV_GEMM=1 \
  .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 8 --warmup 3 --iters 10
```

The escape hatch is read once per process. Set it before importing/running fastnn.

Verification of the escape hatch in an OpenBLAS build, `OPENBLAS_NUM_THREADS=2`, 3 YOLO iterations:

```text
OpenBLAS enabled:
mean_ms=38.0030
conv2d_silu total_ms=35.4304
max_abs_vs_pytorch=0.00048828125
mean_abs_vs_pytorch=1.406245e-06

FASTNN_DISABLE_OPENBLAS_CONV_GEMM=1:
mean_ms=50.0798
conv2d_silu total_ms=43.6652
max_abs_vs_pytorch=0.000457763671875
mean_abs_vs_pytorch=1.404639e-06
```
