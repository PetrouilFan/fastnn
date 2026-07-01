# ARM NEON Support

ARM NEON SIMD acceleration for aarch64 targets in fastnn's CPU backend.

## Building and Testing

```bash
# Cross-compile from x86
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu
cargo build --target aarch64-unknown-linux-gnu --features neon

# Build natively on Raspberry Pi
cargo build --features "neon,simd,parallel"

# Run tests
cargo test --features "neon,simd,parallel"
cargo test --test cross_arch_consistency
```

## NEON Kernels

NEON-optimized implementations in `src/backend/cpu/microkernels.rs`:

- GEMV for I4x8 packed weights (`gemv_u4x8_neon`)
- GEMV for I8x4 packed weights (`gemv_u8x4_neon`)
- Element-wise add, mul, relu
- Softmax
- Reduction sum

## Benchmarking

```bash
bash benchmarks/raspberry_pi/run_bench.sh
```

CI verifies NEON kernel outputs match scalar fallback on every commit.

## Known Limitations

- Requires aarch64 (64-bit ARM). 32-bit ARM (armv7) is not supported.
- Unsupported ops fall back to scalar code.
- NEON + parallel may have diminishing returns on lower-end Pi models.
- Runtime ISA dispatch via `std::is_aarch64_feature_detect` enables a single binary for both NEON and non-NEON targets.

## See also

- [Architecture](architecture.md) -- AOT compiler pipeline and backend dispatch
- [Development](development.md) -- Codebase walkthrough and how-to guides
- [Performance Roadmap](performance-roadmap.md) -- Backend performance priorities
- [docs/index.md](../index.md) -- Documentation home
