# FastNN Benchmarks

## Setup

- CPU: [fill in]
- RAM: [fill in]
- OS: [fill in]
- Rust version: [fill in]
- Date: [fill in]

## Compiled Training vs Naive Training

Measured per-step latency for a 3-layer MLP (64→128→32→1) with AdamW.

| Method | Mean Step Time | vs Naive |
|--------|---------------|----------|
| Naive (eager) | TBD | 1.0x |
| Compiled (AOT) | TBD | TBD |

## WGPU Quantized Inference vs CPU

Measured for quantized matmul (U4, 256×512).

| Backend | Mean Time | Speedup |
|---------|-----------|---------|
| CPU (U4 quantized) | TBD | 1.0x |
| WGPU (GPU) | TBD | TBD |

## Quantized vs F32 (CPU)

See `benches/quantized_vs_pytorch.rs` for detailed results on memory efficiency and GEMV performance for F32x1, F16x2, U8x4, and U4x8 types.

## How to Run

```bash
# All benchmarks (runs all [[bench]] targets)
cargo bench

# Specific benchmark groups
cargo bench --bench compiled_training_vs_pytorch
cargo bench --bench wgpu_inference

# Existing benchmarks
cargo bench --bench quantized_vs_pytorch
cargo bench --bench packed_bench
cargo bench --bench wgpu_bench
```
