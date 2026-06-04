# Conv phase benchmark

`examples/conv_phase_bench.rs` separates current fastnn Conv+SiLU into:

1. current total `conv2d_f32_im2col_gemm`
2. im2col construction only
3. `matrixmultiply::sgemm` only on a prebuilt im2col matrix
4. bias+SiLU epilogue only

Run:

```bash
cargo run --release --example conv_phase_bench -- 120
```

Local 120-iteration result:

```text
stem_f16_c3_sp25600            gemm=(16,27,25600) total=3.4193ms phased=2.7600ms im2col=0.5864ms(21.2%) gemm=0.9434ms(34.2%) silu=1.2303ms(44.6%) gf/s=6.5 max_abs=0.000e0
yolo_f32_c16_sp6400            gemm=(32,144,6400) total=2.5155ms phased=2.2263ms im2col=0.5352ms(24.0%) gemm=1.0982ms(49.3%) silu=0.5928ms(26.6%) gf/s=23.4 max_abs=0.000e0
yolo_f16_c16_sp6400            gemm=(16,144,6400) total=1.8689ms phased=1.7669ms im2col=0.5177ms(29.3%) gemm=0.9587ms(54.3%) silu=0.2904ms(16.4%) gf/s=15.8 max_abs=0.000e0
yolo_f32_c32_sp1600            gemm=(32,288,1600) total=0.9957ms phased=0.9054ms im2col=0.2314ms(25.6%) gemm=0.5203ms(57.5%) silu=0.1537ms(17.0%) gf/s=29.6 max_abs=0.000e0
yolo_f64_c64_sp400             gemm=(64,576,400) total=0.7159ms phased=0.7267ms im2col=0.1079ms(14.8%) gemm=0.5223ms(71.9%) silu=0.0965ms(13.3%) gf/s=41.2 max_abs=0.000e0
yolo_f64_c64_sp1600            gemm=(64,576,1600) total=2.4568ms phased=2.3639ms im2col=0.5801ms(24.5%) gemm=1.4794ms(62.6%) silu=0.3043ms(12.9%) gf/s=48.0 max_abs=0.000e0
yolo_f80_c80_sp1600            gemm=(80,720,1600) total=3.3010ms phased=3.0551ms im2col=0.6865ms(22.5%) gemm=1.9863ms(65.0%) silu=0.3823ms(12.5%) gf/s=55.8 max_abs=0.000e0
yolo_f128_c128_sp100           gemm=(128,1152,100) total=0.6025ms phased=0.5907ms im2col=0.0651ms(11.0%) gemm=0.4823ms(81.7%) silu=0.0433ms(7.3%) gf/s=48.9 max_abs=0.000e0
```

## Interpretation

For heavy repeated YOLO Conv+SiLU shapes, GEMM dominates:

- `M=64,K=576,N=400`: GEMM ~72%
- `M=64,K=576,N=1600`: GEMM ~63%
- `M=80,K=720,N=1600`: GEMM ~65%
- `M=128,K=1152,N=100`: GEMM ~82%

So significant speedup for the repeated heavy shapes is unlikely from im2col-only changes. It requires either:

- better GEMM backend/microkernel,
- packed static-weight GEMM layout matched to the microkernel,
- Winograd for 3x3 stride-1,
- or multithreaded/block GEMM.

For the stem, the bottleneck is different:

```text
M=16,K=27,N=25600: SiLU epilogue ~45%, GEMM ~34%, im2col ~21%
```

So a stem-only k-major im2col path may help, but any larger stem improvement should also address the epilogue. A naive direct Conv loop remains rejected.

## Next benchmark target

The next meaningful isolated benchmark should compare better compute kernels for the GEMM-dominant shapes, not only im2col variants:

1. `matrixmultiply::sgemm` current layout
2. external BLAS/oneDNN candidate if dependency policy allows
3. simple blocked pure-Rust microkernel for one shape family
4. Winograd F(2x2,3x3) prototype for stride-1 3x3

Acceptance criterion before integration: beat current isolated Conv total for at least one repeated heavy shape, with exactness checked.
