//! Microbenchmarks for fastnn i4 SIMD operations.
//!
//! Measures: scalar vs SIMD dot product, GEMV, GEMM, and fused activation overhead.
//!
//! Run:  cargo run --release --example bench_i4 --features simd

use std::time::Instant;

use fastnn::backend::cpu::microkernels::gemv_cpu;
use fastnn::backend::cpu::packed_conv::gemm_packed_i4x8_fused;
use fastnn::backend::cpu::packed_gemm::gemm_packed_i4x8;
use fastnn::backend::cpu::swar::{i4x8_dot_packed, i4x8_dot_packed_slice as i4x8_dot_packed_simd_slice};
use fastnn::backend::prepared::PreparedActivation;
use fastnn::dtypes::I4x8;
use fastnn::packed_tensor::PackedTensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random u32 — no external crate needed.
#[inline]
fn prng(i: usize) -> u32 {
    let x = i as u32;
    x.wrapping_mul(0x9E3779B9) ^ x.wrapping_mul(0x85EBCA6B)
}

/// Fill a Vec<f32> with deterministic pseudo-random values in [-1.0, 1.0].
fn fill_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Time a closure, returning elapsed as µs (f64).
fn bench_us<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    // Warmup (untimed)
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_micros() as f64 / iters as f64
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== fastnn i4 Microbenchmarks ===\n");

    // -----------------------------------------------------------------------
    // 1. Dot product — scalar vs SIMD
    // -----------------------------------------------------------------------
    let n_pairs: usize = 1_000_000;
    let a_u32: Vec<u32> = (0..n_pairs).map(prng).collect();
    let b_u32: Vec<u32> = (0..n_pairs).map(|i| prng(i + 1_000_000)).collect();

    // Scalar: call i4x8_dot_packed per pair
    let scalar_us = bench_us(5, || {
        let mut acc: i64 = 0;
        for i in 0..n_pairs {
            acc = acc.wrapping_add(i4x8_dot_packed(a_u32[i], b_u32[i]) as i64);
        }
        // Prevent optimising the loop away
        std::hint::black_box(acc);
    });

    // SIMD: single call over full slice
    let simd_us = bench_us(5, || {
        let result = i4x8_dot_packed_simd_slice(&a_u32, &b_u32);
        std::hint::black_box(result);
    });

    let scalar_ns_per_pair = scalar_us * 1000.0 / n_pairs as f64;
    let simd_ns_per_pair = simd_us * 1000.0 / n_pairs as f64;
    let speedup = if simd_us > 0.0 {
        scalar_us / simd_us
    } else {
        0.0
    };

    println!("1. Dot product ({}M pairs):", n_pairs / 1_000_000);
    println!("   Scalar:  {scalar_us:>8.1} µs  ({scalar_ns_per_pair:>5.1} ns/pair)");
    println!("   SIMD:    {simd_us:>8.1} µs  ({simd_ns_per_pair:>5.1} ns/pair)");
    println!("   Speedup: {speedup:.2}×\n");

    // -----------------------------------------------------------------------
    // 2. GEMV  (1024 × 1024)
    // -----------------------------------------------------------------------
    let gemv_m: usize = 1024;
    let gemv_k: usize = 1024;
    let weight_data = fill_f32(gemv_m * gemv_k, 42);
    let activation = fill_f32(gemv_k, 99);
    let mut output = vec![0.0f32; gemv_m];
    let weights = PackedTensor::<I4x8>::from_f32_per_channel(&weight_data, &[gemv_m, gemv_k]);

    let gemv_us = bench_us(100, || {
        gemv_cpu::<I4x8>(&weights, &activation, &mut output);
        std::hint::black_box(&output);
    });

    println!("2. GEMV ({gemv_m}×{gemv_k}, 100 iters):");
    println!("   Per-call: {gemv_us:.1} µs\n");

    // -----------------------------------------------------------------------
    // 3. GEMM  (512×512 and 1024×1024)
    // -----------------------------------------------------------------------
    for &(size, iters, label) in &[(512usize, 10u32, "512×512"), (1024, 10, "1024×1024")] {
        let a_data = fill_f32(size * size, 1);
        let b_data = fill_f32(size * size, 2);
        let a = PackedTensor::<I4x8>::from_f32_per_channel(&a_data, &[size, size]);
        let b = PackedTensor::<I4x8>::from_f32_per_channel(&b_data, &[size, size]);
        let mut c = vec![0.0f32; size * size];

        let gemm_us = bench_us(iters, || {
            gemm_packed_i4x8(&a, &b, &mut c);
            std::hint::black_box(&c);
        });

        println!("3. GEMM ({label}, {iters} iters):");
        println!("   Per-call: {gemm_us:.1} µs\n");
    }

    // -----------------------------------------------------------------------
    // 4. Fused GEMM activation overhead (256×256)
    // -----------------------------------------------------------------------
    let fuse_n: usize = 256;
    let fuse_k: usize = 256;
    let fuse_a_data = fill_f32(fuse_n * fuse_k, 7);
    let fuse_b_data = fill_f32(fuse_n * fuse_k, 13);
    let fuse_a = PackedTensor::<I4x8>::from_f32_per_channel(&fuse_a_data, &[fuse_n, fuse_k]);
    let fuse_b = PackedTensor::<I4x8>::from_f32_per_channel(&fuse_b_data, &[fuse_n, fuse_k]);
    let mut fuse_c = vec![0.0f32; fuse_n * fuse_n];
    let fuse_iters = 100u32;

    let activations: &[(PreparedActivation, &str)] = &[
        (PreparedActivation::None, "None"),
        (PreparedActivation::Relu, "Relu"),
        (PreparedActivation::Silu, "Silu"),
        (PreparedActivation::Gelu, "Gelu"),
    ];

    println!("4. Fused GEMM activation overhead ({fuse_n}×{fuse_n}, {fuse_iters} iters):");
    for &(act, name) in activations {
        let us = bench_us(fuse_iters, || {
            gemm_packed_i4x8_fused(&fuse_a, &fuse_b, None, act, &mut fuse_c);
            std::hint::black_box(&fuse_c);
        });
        println!("   {name:<5}  {us:>6.1} µs/call");
    }
}
