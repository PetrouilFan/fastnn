# AArch64 CPU Support

Build and portability status for fastnn's CPU backend on 64-bit ARM.

## Building and Testing

```bash
# Cross-compile from x86
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu
cargo build --target aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu \
  --features "simd,parallel,fusion-forward,fusion-backward,fusion-residual-add-norm"

# Build natively on Raspberry Pi
cargo build --features "simd,parallel,fusion-forward"

# Run tests
cargo test --features "simd,parallel,fusion-forward"
cargo test --test cross_arch_consistency
```

## SIMD status

The `simd` feature is architecture-portable. x86-specific AVX2 kernels are
compiled only on x86_64; AArch64 builds retain scalar implementations for the
same execution contracts. Runtime ISA reporting detects NEON where available,
but fastnn does not currently claim dedicated NEON low-bit microkernels.

CI cross-compiles both the default feature set and all portable CPU fusion
features for `aarch64-unknown-linux-gnu` on every change to `main` or `dev`.

## Benchmarking

The maintained CPU benchmark is `benches/cpu_baselines.rs`:

```bash
cargo bench --bench cpu_baselines
```

Run it natively on the target machine. Cross-compilation verifies portability,
not target-device latency.

## Known Limitations

- Requires aarch64 (64-bit ARM). 32-bit ARM (armv7) is not supported.
- Unsupported ops fall back to scalar code.
- Parallel execution may have diminishing returns on lower-end devices.
- Dedicated AArch64 low-bit SIMD kernels remain future work.

## See also

- [Architecture](architecture.md) -- AOT compiler pipeline and backend dispatch
- [Development](development.md) -- Codebase walkthrough and how-to guides
- [Performance Roadmap](performance-roadmap.md) -- Backend performance priorities
- [docs/index.md](../index.md) -- Documentation home
