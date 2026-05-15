#!/bin/bash
# Raspberry Pi benchmark runner
# Measures inference performance on ARM NEON

set -e

echo "=== FastNN Raspberry Pi Benchmarks ==="
echo "Date: $(date)"
echo "Model: $(cat /proc/device-tree/model 2>/dev/null || uname -a)"
echo "Arch: $(uname -m)"
echo ""

# Build with NEON optimizations
echo "Building with NEON..."
cargo build --release --features "neon,simd,parallel"

# Run the existing benchmarks
echo ""
echo "=== Running Packed Benchmarks ==="
cargo bench --features neon --bench packed_bench 2>&1 || echo "Benchmark failed (may need nightly)"

echo ""
echo "=== Running Quantized vs PyTorch ==="
cargo bench --features neon --bench quantized_vs_pytorch 2>&1 || echo "Benchmark failed"

echo ""
echo "=== Running Unit Tests ==="
cargo test --features "neon,simd,parallel" 2>&1

echo ""
echo "=== Done ==="
