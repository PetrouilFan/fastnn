# Conv phase benchmark

`examples/conv_phase_bench.rs` separates current fastnn Conv+SiLU into:

1. current total `conv2d_f32_im2col_gemm`
2. im2col construction only
3. GEMM only on a prebuilt im2col matrix
4. bias+SiLU epilogue only

Roadmap Task B1 extends the benchmark so the split uses the same GEMM backend as the Conv kernel:

- default build: `matrixmultiply::sgemm`
- `--features openblas`: OpenBLAS `cblas_sgemm`, unless `FASTNN_DISABLE_OPENBLAS_CONV_GEMM=1`

The benchmark prints `gemm_backend=...` so results clearly identify which path was measured.

## Commands

```bash
cargo run --release --example conv_phase_bench -- 120
OPENBLAS_NUM_THREADS=2 cargo run --release --features openblas --example conv_phase_bench -- 120
```

## Default matrixmultiply result

Local 120-iteration result:

```text
iters=120 warmup=5 gemm_backend=matrixmultiply openblas_disabled=
stem_f16_c3_sp25600            gemm=(16,27,25600) total=6.6145ms phased=5.8319ms im2col=2.1879ms( 37.5%) gemm=1.7482ms( 30.0%) silu=1.8958ms( 32.5%) gf/s=3.3 max_abs=0.000e0
yolo_f32_c16_sp6400            gemm=(32,144,6400) total=4.2055ms phased=4.0185ms im2col=0.9638ms( 24.0%) gemm=2.1619ms( 53.8%) silu=0.8928ms( 22.2%) gf/s=14.0 max_abs=0.000e0
yolo_f16_c16_sp6400            gemm=(16,144,6400) total=3.1554ms phased=2.9209ms im2col=0.9852ms( 33.7%) gemm=1.5091ms( 51.7%) silu=0.4266ms( 14.6%) gf/s=9.3 max_abs=0.000e0
yolo_f32_c32_sp1600            gemm=(32,288,1600) total=1.6371ms phased=1.5711ms im2col=0.4834ms( 30.8%) gemm=0.8581ms( 54.6%) silu=0.2296ms( 14.6%) gf/s=18.0 max_abs=0.000e0
yolo_f64_c64_sp400             gemm=(64,576,400) total=0.8750ms phased=0.9490ms im2col=0.1440ms( 15.2%) gemm=0.6899ms( 72.7%) silu=0.1151ms( 12.1%) gf/s=33.7 max_abs=0.000e0
yolo_f64_c64_sp1600            gemm=(64,576,1600) total=4.4223ms phased=4.0373ms im2col=1.0716ms( 26.5%) gemm=2.5238ms( 62.5%) silu=0.4419ms( 10.9%) gf/s=26.7 max_abs=0.000e0
yolo_f80_c80_sp1600            gemm=(80,720,1600) total=6.2563ms phased=5.9206ms im2col=1.3821ms( 23.3%) gemm=3.9860ms( 67.3%) silu=0.5525ms(  9.3%) gf/s=29.5 max_abs=0.000e0
yolo_f128_c128_sp100           gemm=(128,1152,100) total=1.2573ms phased=1.3834ms im2col=0.0953ms(  6.9%) gemm=1.2248ms( 88.5%) silu=0.0634ms(  4.6%) gf/s=23.5 max_abs=0.000e0
```

## OpenBLAS result

Local 120-iteration result with `OPENBLAS_NUM_THREADS=2`:

```text
iters=120 warmup=5 gemm_backend=openblas openblas_disabled=
stem_f16_c3_sp25600            gemm=(16,27,25600) total=5.5061ms phased=4.6899ms im2col=2.1358ms( 45.5%) gemm=0.6249ms( 13.3%) silu=1.9292ms( 41.1%) gf/s=4.0 max_abs=0.000e0
yolo_f32_c16_sp6400            gemm=(32,144,6400) total=3.1380ms phased=2.7079ms im2col=0.9291ms( 34.3%) gemm=0.8723ms( 32.2%) silu=0.9065ms( 33.5%) gf/s=18.8 max_abs=0.000e0
yolo_f16_c16_sp6400            gemm=(16,144,6400) total=2.3716ms phased=2.2139ms im2col=0.9992ms( 45.1%) gemm=0.7435ms( 33.6%) silu=0.4712ms( 21.3%) gf/s=12.4 max_abs=0.000e0
yolo_f32_c32_sp1600            gemm=(32,288,1600) total=1.2600ms phased=1.1801ms im2col=0.5117ms( 43.4%) gemm=0.4078ms( 34.6%) silu=0.2606ms( 22.1%) gf/s=23.4 max_abs=0.000e0
yolo_f64_c64_sp400             gemm=(64,576,400) total=0.6616ms phased=0.6284ms im2col=0.1901ms( 30.3%) gemm=0.3208ms( 51.0%) silu=0.1175ms( 18.7%) gf/s=44.6 max_abs=0.000e0
yolo_f64_c64_sp1600            gemm=(64,576,1600) total=3.1477ms phased=2.7777ms im2col=1.0757ms( 38.7%) gemm=1.2225ms( 44.0%) silu=0.4796ms( 17.3%) gf/s=37.5 max_abs=0.000e0
yolo_f80_c80_sp1600            gemm=(80,720,1600) total=3.8898ms phased=3.7029ms im2col=1.2927ms( 34.9%) gemm=1.7991ms( 48.6%) silu=0.6111ms( 16.5%) gf/s=47.4 max_abs=0.000e0
yolo_f128_c128_sp100           gemm=(128,1152,100) total=0.5372ms phased=0.5248ms im2col=0.1633ms( 31.1%) gemm=0.3008ms( 57.3%) silu=0.0607ms( 11.6%) gf/s=54.9 max_abs=0.000e0
```

## Interpretation

OpenBLAS substantially reduces the GEMM phase, so non-GEMM work is now the bottleneck for several YOLO shapes:

- Stem `M=16,K=27,N=25600`: GEMM drops to ~13%; im2col+SiLU are ~87%.
- `M=32,K=144,N=6400`: im2col and SiLU are each about as large as GEMM.
- `M=32,K=288,N=1600`: im2col+SiLU are ~65%.
- Heavy repeated shapes still have meaningful GEMM cost, but less dominant than before:
  - `M=64,K=576,N=400`: GEMM ~51%, im2col+SiLU ~49%.
  - `M=64,K=576,N=1600`: GEMM ~44%, im2col+SiLU ~56%.
  - `M=80,K=720,N=1600`: GEMM ~49%, im2col+SiLU ~51%.
  - `M=128,K=1152,N=100`: GEMM ~57%, im2col+SiLU ~43%.

Task B1 verdict: after OpenBLAS, pure GEMM/backend work alone is unlikely to close the YOLO gap. The next low-risk lane should measure and reduce repeated non-GEMM overhead in full YOLO, starting with constant writes/cache cleanup (Task B2) and then epilogue/im2col work if full-profile evidence supports it.

## Next benchmark target

For Task B2, use profile JSON to quantify `write_const` and related non-GEMM overhead in full YOLO before changing executor behavior. Any accepted cleanup must improve full YOLO by at least 1% and keep outputs exact or within the existing PyTorch tolerance.
