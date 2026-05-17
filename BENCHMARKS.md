# FastNN Benchmarks

## How to Run

```bash
# Run all benchmarks
cargo bench

# Specific benchmark groups
cargo bench --bench quantized_vs_pytorch
cargo bench --bench packed_bench
cargo bench --bench wgpu_bench
```

## Historical Benchmarks

Detailed benchmark results (GEMV throughput, fused kernel speedups, etc.) are maintained in the project's [performance tracking](https://github.com/PetrouilFan/fastnn/wiki/Performance) wiki.

### GEMV Performance (AMD Ryzen 7 3700X, 8 threads)

| Implementation | Time | GFLOP/s | vs PyTorch f32 | Memory |
|---------------|------|---------|----------------|--------|
| PyTorch f32 (MKL) | 4.04 ms | 8.3 | 1.0× | 64 MB |
| fastnn F16x2 | 1.80 ms | 18.6 | 2.2× | 32 MB |
| fastnn U8x4 | 0.76 ms | 44.4 | 5.3× | 16 MB |
| fastnn U4x8 | 0.55 ms | 61.1 | 7.4× | 8 MB |

### Fused Conv2d+BN+SiLU

| Configuration | PyTorch (separate) | fastnn (fused) | Speedup |
|---------------|-------------------|----------------|---------|
| Conv2d(32→64) + BN + SiLU (64×64) | 81.81 ms | 3.27 ms | 25.0× |
| Conv2d(64→128) + BN + SiLU (32×32) | 42.55 ms | 2.01 ms | 21.2× |

> Run `cargo bench --bench packed_bench` on your hardware to reproduce the GEMV numbers.
