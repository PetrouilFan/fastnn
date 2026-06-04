# GEMM layout benchmark for YOLO Conv candidates

`examples/gemm_layout_bench.rs` compares the current row-major static Conv weight layout used by `matrixmultiply::sgemm` against a pretransposed `K x M` layout called with adjusted strides.

Run with:

```bash
cargo run --release --example gemm_layout_bench -- 120
```

Representative local output after `415cf11`:

```text
m=  64 k= 576 n=  400 row_major_a=0.3722ms transposed_a=0.3146ms ratio=0.845
m=  64 k= 576 n= 1600 row_major_a=1.3584ms transposed_a=1.1654ms ratio=0.858
m=  32 k= 288 n= 1600 row_major_a=0.3926ms transposed_a=0.5496ms ratio=1.400
m=  80 k= 720 n= 1600 row_major_a=1.7292ms transposed_a=1.6202ms ratio=0.937
m= 128 k=1152 n=  100 row_major_a=0.4077ms transposed_a=0.3110ms ratio=0.763
```

## Verdict

Pretransposed fp32 weights are promising but shape-dependent. Do not globally switch Conv GEMM to transposed weights.

A safe runtime lane should be opt-in and shape-thresholded, starting with the repeated YOLO family:

```text
M=64 K=576 N=400/1600
```

Keep exactness gates against the default fastnn path and rerun both:

```bash
.venv/bin/python scripts/conv_shape_microbench.py --warmup 5 --iters 20 --threads 1
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 10 --warmup 2 --iters 5
```
