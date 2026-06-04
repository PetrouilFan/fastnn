# Conv im2col variant benchmark

`examples/conv_im2col_variant_bench.rs` isolates Conv+SiLU microkernel variants without touching runtime dispatch.

It compares:

1. current `conv2d_f32_im2col_gemm`
2. experimental k-major im2col layout:
   - materializes im2col as `[K, N]` row-major
   - calls `matrixmultiply::sgemm` with B rows contiguous
   - compares output exactly against current fastnn Conv+SiLU

Run:

```bash
cargo run --release --example conv_im2col_variant_bench -- 120
```

Local 120-iteration result:

```text
stem_f16_c3_sp25600              gemm=(16,27,25600) current=3.4308ms kmajor=2.9058ms ratio=0.847 max_abs=0.000e0
yolo_f32_c32_sp1600              gemm=(32,288,1600) current=0.9666ms kmajor=1.1034ms ratio=1.142 max_abs=0.000e0
yolo_f32_c16_sp6400              gemm=(32,144,6400) current=2.6570ms kmajor=2.4713ms ratio=0.930 max_abs=0.000e0
yolo_f16_c16_sp6400              gemm=(16,144,6400) current=1.6865ms kmajor=1.7765ms ratio=1.053 max_abs=0.000e0
yolo_f64_c64_sp400               gemm=(64,576,400) current=0.5222ms kmajor=0.6044ms ratio=1.157 max_abs=0.000e0
yolo_f128_c128_sp100             gemm=(128,1152,100) current=0.5458ms kmajor=0.5228ms ratio=0.958 max_abs=0.000e0
```

Verdict:

- k-major im2col is not globally better.
- It is an isolated, exact win for the C=3 stem family.
- It is a modest isolated win for `M=32,K=144,N=6400` in this sample.
- It loses clearly for `M=32,K=288,N=1600` and `M=64,K=576,N=400`.

Next integration candidate:

Add a shape-thresholded runtime path inside `conv2d_f32_im2col_gemm` for exactly the k-major winners, starting with the stem:

```text
N=1, C=3, F=16, KH=KW=3, stride=2, padding=1, dilation=1, groups=1
```

This is different from the rejected direct-loop stem spike. The rejected path bypassed GEMM and was much slower. The k-major path still uses GEMM but improves im2col/GEMM layout for the stem.
