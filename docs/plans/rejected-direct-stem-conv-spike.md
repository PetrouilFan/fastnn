# Rejected spike: direct small-channel YOLO stem Conv2d

## Context

After PreparedPlan Wave 7, full YOLO profiling showed `conv2d_silu` dominates runtime. The largest relative microbench gap was the YOLO stem-like shape:

- input: `[1, 3, 320, 320]`
- weight: `[16, 3, 3, 3]`
- stride: `2`
- padding: `1`
- activation: SiLU
- GEMM shape estimate: `(M=16, K=27, N=25600)`

A local spike added a narrow direct loop for `N=1`, `C<=4`, `3x3`, `stride=2`, `padding=1`, `dilation=1`, `groups=1` in `conv2d_f32_im2col_gemm`.

## Result

The direct loop was exact but much slower and was reverted before commit.

Observed regression:

```text
stem microbench baseline: ~3.3-3.7 ms fastnn
stem direct-loop spike:  ~22.8 ms fastnn
full YOLO baseline:      ~54 ms mean fastnn sample
full YOLO direct spike:  ~72.6 ms mean fastnn sample
```

Accuracy stayed acceptable, but performance failed the gate.

## Verdict

Do not retry a naive per-output direct loop for this shape. The existing im2col+sgemm path is substantially faster despite im2col overhead.

Future speed work should focus on:

- packed/pretransposed weights for GEMM paths
- improving/fusing im2col or reusing prepared layout buffers
- shape-selected matrixmultiply/packing strategy
- measuring against `scripts/conv_shape_microbench.py` and full YOLO before merge
