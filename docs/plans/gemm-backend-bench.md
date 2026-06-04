# GEMM backend benchmark

`examples/gemm_backend_bench.rs` compares fastnn's current `matrixmultiply::sgemm` backend against OpenBLAS `cblas_sgemm` for the GEMM shapes produced by YOLO Conv im2col.

It compiles without OpenBLAS and prints only the matrixmultiply baseline. With the existing `openblas` feature, it also calls CBLAS directly:

```bash
cargo run --release --example gemm_backend_bench -- 40
OPENBLAS_NUM_THREADS=1 cargo run --release --features openblas --example gemm_backend_bench -- 40
OPENBLAS_NUM_THREADS=4 cargo run --release --features openblas --example gemm_backend_bench -- 30
OPENBLAS_NUM_THREADS=8 cargo run --release --features openblas --example gemm_backend_bench -- 30
```

## Single-thread OpenBLAS

Single-thread OpenBLAS is mixed. It wins some shapes, but loses others:

```text
stem_m16_k27_n25600      matrixmultiply=0.8665ms openblas=0.4866ms ratio=0.562
yolo_m32_k144_n6400      matrixmultiply=0.8403ms openblas=0.7011ms ratio=0.834
yolo_m16_k144_n6400      matrixmultiply=0.8096ms openblas=0.3760ms ratio=0.464
yolo_m32_k288_n1600      matrixmultiply=0.3204ms openblas=0.3828ms ratio=1.195  # slower
yolo_m64_k576_n400       matrixmultiply=0.3329ms openblas=0.2719ms ratio=0.817
yolo_m64_k576_n1600      matrixmultiply=1.0632ms openblas=1.1925ms ratio=1.122  # slower
yolo_m80_k720_n1600      matrixmultiply=1.9246ms openblas=1.7844ms ratio=0.927
yolo_m128_k1152_n100     matrixmultiply=0.3931ms openblas=0.2822ms ratio=0.718
```

Conclusion: do not switch to single-thread OpenBLAS globally.

## Multithreaded OpenBLAS

With `OPENBLAS_NUM_THREADS=4`, OpenBLAS is a clear isolated win across the measured YOLO GEMM shapes:

```text
stem_m16_k27_n25600      ratio=0.211  openblas=0.1578ms
yolo_m32_k144_n6400      ratio=0.419  openblas=0.3059ms
yolo_m16_k144_n6400      ratio=0.239  openblas=0.1353ms
yolo_m32_k288_n1600      ratio=0.334  openblas=0.1201ms
yolo_m64_k576_n400       ratio=0.333  openblas=0.1104ms
yolo_m64_k576_n1600      ratio=0.417  openblas=0.4792ms
yolo_m80_k720_n1600      ratio=0.340  openblas=0.6160ms
yolo_m128_k1152_n100     ratio=0.336  openblas=0.1271ms
```

With `OPENBLAS_NUM_THREADS=8`, OpenBLAS is even faster for many shapes:

```text
stem_m16_k27_n25600      ratio=0.134
yolo_m32_k144_n6400      ratio=0.233
yolo_m16_k144_n6400      ratio=0.183
yolo_m32_k288_n1600      ratio=0.215
yolo_m64_k576_n400       ratio=0.326
yolo_m64_k576_n1600      ratio=0.286
yolo_m80_k720_n1600      ratio=0.289
yolo_m128_k1152_n100     ratio=0.290
```

Correctness deltas vs `matrixmultiply` were normal fp32 accumulation-order differences, roughly `~1e-5` to `2e-5` max_abs.

## Verdict

The fastest route to significant CPU YOLO speedup is an optional multithreaded BLAS Conv/GEMM path, not more im2col-only work.

Recommended next integration slice:

1. Add a feature-gated `openblas` Conv GEMM path inside `conv2d_f32_im2col_gemm`.
2. Use it only when `feature = "openblas"` is enabled.
3. Keep the current `matrixmultiply` path as default.
4. Shape-gate initially to YOLO-like GEMM sizes where OpenBLAS wins.
5. Test with `OPENBLAS_NUM_THREADS=4` and `8`.
6. Gate full integration with:
   - `scripts/conv_shape_microbench.py`
   - full YOLO profile
   - accuracy vs PyTorch and default fastnn.

Do not enable BLAS by default until build/runtime threading policy is decided.
