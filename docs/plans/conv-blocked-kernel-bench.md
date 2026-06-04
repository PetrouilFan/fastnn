# Conv blocked kernel bench

Roadmap task: C3 custom blocked Conv/GEMM pre-integration probe.

Objective: test whether a narrow shape-specific blocked pure-Rust GEMM path is worth integrating for the repeated YOLO target shape before touching runtime dispatch.

Target shape:

```text
yolo_m64_k576_n400
Conv2d + SiLU
N=1, C=64, H=W=20, F=64, KH=KW=3, stride=1, padding=1, dilation=1, groups=1
GEMM: M=64, K=576, N=400
YOLO count: 9
```

Command:

```bash
OPENBLAS_NUM_THREADS=2 cargo run --release --features openblas --example conv_blocked_kernel_bench -- 40
```

Representative result after adding a real AVX2/FMA KN-layout microkernel:

```text
shape=yolo_m64_k576_n400 gemm=(64, 576, 400) iters=40 warmup=5 openblas_feature=true
fastnn_current_total_ms=0.3547
blocked_total_ms=11.8516
blocked_im2col_ms=0.5250
blocked_gemm_ms=11.2530
blocked_epilogue_ms=0.0736
avx2_total_ms=0.9717
avx2_im2col_kn_ms=0.2157
avx2_gemm_ms=0.6835
avx2_epilogue_ms=0.0725
openblas_gemm_only_ms=0.1844
blocked_vs_fastnn_ratio=33.415
avx2_vs_fastnn_ratio=2.740
max_abs_vs_fastnn=2.289e-5
avx2_max_abs_vs_fastnn=2.480e-5
blocked_gemm_gflops=2.6
avx2_gemm_gflops=43.2
```

Interpretation:

- The first scalar blocked/pretransposed-weight candidate was exact within normal fp32 accumulation-order tolerance, but much slower than the current OpenBLAS-backed Conv path.
- The benchmark now includes a real AVX2/FMA candidate over a `[K,N]` im2col layout and `[K,M]` pretransposed weights.
- The AVX2 candidate is a large isolated improvement over the scalar blocked path (`~43 GF/s` vs `~2.6 GF/s` in this sample), but it is still slower than current fastnn/OpenBLAS for the full Conv+SiLU target shape and slower than OpenBLAS GEMM-only.
- This does not meet the C3 acceptance gate. Do not integrate this AVX2 path into runtime dispatch.
- The likely blockers are still im2col/layout cost plus OpenBLAS' stronger small-GEMM kernel/threading. Future attempts need a better microkernel, larger shape family coverage, Winograd/direct Conv, or external backend integration.

Verdict: rejected as an integration candidate; useful as C3 SIMD scaffold only.
