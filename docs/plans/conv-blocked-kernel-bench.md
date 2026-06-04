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

Representative result:

```text
shape=yolo_m64_k576_n400 gemm=(64, 576, 400) iters=40 warmup=5 openblas_feature=true
fastnn_current_total_ms=0.7315
blocked_total_ms=15.1407
blocked_im2col_ms=0.7128
blocked_gemm_ms=14.3194
blocked_epilogue_ms=0.1084
openblas_gemm_only_ms=0.3478
blocked_vs_fastnn_ratio=20.698
max_abs_vs_fastnn=2.289e-5
blocked_gemm_gflops=2.1
```

Interpretation:

- The simple blocked/pretransposed-weight pure-Rust candidate is exact within normal fp32 accumulation-order tolerance, but it is far slower than the current OpenBLAS-backed Conv path on the target shape.
- The target-feature/AVX2 guard exists in the benchmark, but the implementation intentionally delegates to the exact scalar blocked kernel. This prevents accidental runtime integration before a real SIMD microkernel exists.
- This does not meet the C3 acceptance gate. Do not integrate this scalar blocked path.
- A future C3 attempt must implement real SIMD/FMA math or use an optimized backend. Benchmark-only scaffolding is now available for that comparison.

Verdict: rejected as an integration candidate; useful as C3 baseline scaffolding only.
