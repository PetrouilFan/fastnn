# ARM NEON Support

FastNN supports ARM NEON SIMD acceleration for aarch64 targets.

## Building for ARM

```bash
# Cross-compile from x86
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu
cargo build --target aarch64-unknown-linux-gnu --features neon

# Build natively on Raspberry Pi
cargo build --features "neon,simd,parallel"
```

## Running Tests

```bash
# On Raspberry Pi or ARM machine
cargo test --features "neon,simd,parallel"

# Run cross-architecture consistency tests
cargo test --test cross_arch_consistency
```

## NEON Kernels

The following operations have NEON-optimized implementations:
- GEMV for U4x8 packed weights
- GEMV for U8x4 packed weights
- Element-wise add, mul, relu
- Softmax
- Reduction sum

## Benchmarking

```bash
# Run the Raspberry Pi benchmark suite
bash benchmarks/raspberry_pi/run_bench.sh
```

## Known Limitations

- NEON kernels require aarch64 (64-bit ARM). 32-bit ARM (armv7) is not currently supported.
- Not all operations have NEON implementations; unsupported ops fall back to scalar code.
- NEON + parallel may have diminishing returns on lower-end Raspberry Pi models (Pi 3 or earlier).
