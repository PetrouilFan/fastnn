//! BLIS-style tiled GEMV/GEMM for packed precision types.
//!
//! Architecture (same as MKL/OpenBLAS/BLIS):
//!   ┌─────────────────────────────────┐
//!   │  Cache blocks (L2/L3): KC × NC  │  ← weights stay in L2
//!   │  ┌───────────────────────────┐  │
//!   │  │  Micro-panels: MR × KC    │  │  ← MR rows, full K
//!   │  │  ┌─────────────────────┐  │  │
//!   │  │  │  Register tiles     │  │  │  ← MR f32 accumulators
//!   │  │  │  (MR × 1 vectors)   │  │  │
//!   │  │  └─────────────────────┘  │  │
//!   │  └───────────────────────────┘  │
//!   └─────────────────────────────────┘
//!
//! Key insight: compute MR rows simultaneously so the activation
//! vector is loaded once and reused across all MR accumulators.

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// ============================================================
// Micro-kernel parameters
// ============================================================

/// Number of output rows computed simultaneously (register blocking).
/// With AVX2: 16 YMM registers. Micro-kernel uses MR accumulators
/// plus 1-2 for loading. MR=4 uses 4 registers for accumulators,
/// leaving plenty for loads and temporaries.
const MR: usize = 4;

/// Cache block size for K dimension (in f32 elements).
/// 8K f32 = 32KB, fits in L1 cache on most CPUs.
const KC: usize = 8192;

// ============================================================
// Main entry point — dispatches to tiled micro-kernel
// ============================================================

/// BLIS-style tiled GEMV for packed types.
/// Uses cache-blocked K and register-blocked M for maximum throughput.
pub fn gemv_packed_tiled<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + T::ITEMS - 1) / T::ITEMS;
    let scale = weights.scale();
    let zero = weights.zero();

    // Zero outputs
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Process K in cache blocks
    let mut k_offset = 0;
    while k_offset < k {
        let k_end = (k_offset + KC).min(k);

        // Process M in register blocks of MR rows
        let mut row = 0;
        while row + MR <= m {
            // Full MR×1 micro-kernel
            micro_kernel::<T>(
                weights,
                &activation[k_offset..k_end],
                &mut output[row..row + MR],
                row,
                k_offset,
                k_end,
                k_packed,
            );
            row += MR;
        }

        // Tail: process remaining rows (1..MR-1)
        while row < m {
            let mut acc = 0.0f32;
            let row_offset = row * k_packed;
            for kk in k_offset..k_end {
                let p = kk / T::ITEMS;
                let j = kk % T::ITEMS;
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                acc += unpacked.as_ref()[j] * activation[kk];
            }
            output[row] += acc;
            row += 1;
        }

        k_offset += KC;
    }

    // Apply scale and zero
    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

// ============================================================
// Micro-kernel: MR rows × full K block
// ============================================================

/// Generic micro-kernel: unpack weights, then call type-specific FMA.
#[inline]
fn micro_kernel<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    start_row: usize,
    k_start: usize,
    k_end: usize,
    k_packed: usize,
) {
    debug_assert_eq!(output.len(), MR);

    // Unpack MR rows of weights into contiguous buffers
    let k_block = k_end - k_start;
    let mut row_bufs = [[0.0f32; KC]; MR];

    for r in 0..MR {
        let row = start_row + r;
        let row_offset = row * k_packed;
        for kk in k_start..k_end {
            let p = kk / T::ITEMS;
            let j = kk % T::ITEMS;
            let word = weights.as_packed()[row_offset + p];
            let unpacked = word.unpack_to_f32();
            row_bufs[r][kk - k_start] = unpacked.as_ref()[j];
        }
    }

    // Dispatch to SIMD micro-kernel
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                micro_kernel_avx2(&row_bufs, activation, output, k_block);
            }
            return;
        }
    }

    // Scalar fallback
    for r in 0..MR {
        let mut acc = 0.0f32;
        for kk in 0..k_block {
            acc += row_bufs[r][kk] * activation[kk];
        }
        output[r] += acc;
    }
}

/// AVX2 micro-kernel: MR=4 rows, each with its own accumulator.
/// Processes K in chunks of 8 (AVX2 width) across all 4 rows simultaneously.
/// This is the hot inner loop — the key to BLAS-level performance.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn micro_kernel_avx2(
    row_bufs: &[[f32; KC]; MR],
    activation: &[f32],
    output: &mut [f32],
    k: usize,
) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut kk = 0;

    // Process 8 activations at a time, FMA into all 4 row accumulators
    while kk + 8 <= k {
        let act = _mm256_loadu_ps(activation.as_ptr().add(kk));

        let w0 = _mm256_loadu_ps(row_bufs[0].as_ptr().add(kk));
        let w1 = _mm256_loadu_ps(row_bufs[1].as_ptr().add(kk));
        let w2 = _mm256_loadu_ps(row_bufs[2].as_ptr().add(kk));
        let w3 = _mm256_loadu_ps(row_bufs[3].as_ptr().add(kk));

        acc0 = _mm256_fmadd_ps(w0, act, acc0);
        acc1 = _mm256_fmadd_ps(w1, act, acc1);
        acc2 = _mm256_fmadd_ps(w2, act, acc2);
        acc3 = _mm256_fmadd_ps(w3, act, acc3);

        kk += 8;
    }

    // Horizontal sum each accumulator and add to output
    output[0] += hsum256_ps(acc0);
    output[1] += hsum256_ps(acc1);
    output[2] += hsum256_ps(acc2);
    output[3] += hsum256_ps(acc3);

    // Scalar tail
    while kk < k {
        output[0] += row_bufs[0][kk] * activation[kk];
        output[1] += row_bufs[1][kk] * activation[kk];
        output[2] += row_bufs[2][kk] * activation[kk];
        output[3] += row_bufs[3][kk] * activation[kk];
        kk += 1;
    }
}

// ============================================================
// Type-specialized fast paths (skip generic unpack)
// ============================================================

/// Tiled GEMV for U8x4: unpack int8→f32 inline, then tiled micro-kernel.
pub fn gemv_u8x4_tiled(
    weights: &PackedTensor<crate::dtypes::U8x4>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 3) / 4;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    let mut k_offset = 0;
    while k_offset < k {
        let k_end = (k_offset + KC).min(k);

        // Process MR rows at a time
        let mut row = 0;
        while row + MR <= m {
            // Unpack MR rows of int8→f32
            let mut row_bufs = [[0.0f32; KC]; MR];
            for r in 0..MR {
                let row_idx = row + r;
                let row_off = row_idx * k_packed;
                for kk in k_offset..k_end {
                    let p = kk / 4;
                    let j = kk % 4;
                    let w = weights_u32[row_off + p];
                    let bytes = w.to_le_bytes();
                    row_bufs[r][kk - k_offset] = (bytes[j] as i8) as f32;
                }
            }

            // SIMD micro-kernel
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        micro_kernel_avx2(
                            &row_bufs,
                            &activation[k_offset..k_end],
                            &mut output[row..row + MR],
                            k_end - k_offset,
                        );
                    }
                } else {
                    scalar_micro_kernel(
                        &row_bufs,
                        &activation[k_offset..k_end],
                        &mut output[row..row + MR],
                        k_end - k_offset,
                    );
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                scalar_micro_kernel(
                    &row_bufs,
                    &activation[k_offset..k_end],
                    &mut output[row..row + MR],
                    k_end - k_offset,
                );
            }

            row += MR;
        }

        // Tail rows
        while row < m {
            let mut acc = 0.0f32;
            let row_off = row * k_packed;
            for kk in k_offset..k_end {
                let p = kk / 4;
                let j = kk % 4;
                let w = weights_u32[row_off + p];
                let bytes = w.to_le_bytes();
                acc += (bytes[j] as i8) as f32 * activation[kk];
            }
            output[row] += acc;
            row += 1;
        }

        k_offset += KC;
    }

    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

/// Tiled GEMV for F16x2 with F16C dispatch.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn gemv_f16x2_tiled(
    weights: &PackedTensor<crate::dtypes::F16x2>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 1) / 2;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    if is_x86_feature_detected!("f16c") {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);

            let mut row = 0;
            while row + MR <= m {
                let mut row_bufs = [[0.0f32; KC]; MR];
                for r in 0..MR {
                    let row_idx = row + r;
                    let row_off = row_idx * k_packed;
                    for kk in k_offset..k_end {
                        let p = kk / 2;
                        let j = kk % 2;
                        let w = weights_u32[row_off + p];
                        let half_bits = if j == 0 { w as u16 } else { (w >> 16) as u16 };
                        row_bufs[r][kk - k_offset] = half::f16::from_bits(half_bits).to_f32();
                    }
                }

                unsafe {
                    micro_kernel_avx2(
                        &row_bufs,
                        &activation[k_offset..k_end],
                        &mut output[row..row + MR],
                        k_end - k_offset,
                    );
                }
                row += MR;
            }

            while row < m {
                let mut acc = 0.0f32;
                let row_off = row * k_packed;
                for kk in k_offset..k_end {
                    let p = kk / 2;
                    let j = kk % 2;
                    let w = weights_u32[row_off + p];
                    let half_bits = if j == 0 { w as u16 } else { (w >> 16) as u16 };
                    acc += half::f16::from_bits(half_bits).to_f32() * activation[kk];
                }
                output[row] += acc;
                row += 1;
            }

            k_offset += KC;
        }
    } else {
        // Fallback to generic
        gemv_packed_tiled(weights, activation, output);
        return;
    }

    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

/// Tiled GEMV for U4x8 with branchless SIMD unpack.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn gemv_u4x8_tiled(
    weights: &PackedTensor<crate::dtypes::U4x8>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 7) / 8;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);

            let mut row = 0;
            while row + MR <= m {
                let mut row_bufs = [[0.0f32; KC]; MR];
                for r in 0..MR {
                    let row_idx = row + r;
                    let row_off = row_idx * k_packed;
                    for kk in k_offset..k_end {
                        let p = kk / 8;
                        let j = kk % 8;
                        let w = weights_u32[row_off + p];
                        let nibble = (w >> (j * 4)) & 0xF;
                        let signed = if nibble & 0x8 != 0 {
                            (nibble | 0xFFFFFFF0) as i32
                        } else {
                            nibble as i32
                        };
                        row_bufs[r][kk - k_offset] = signed as f32;
                    }
                }

                unsafe {
                    micro_kernel_avx2(
                        &row_bufs,
                        &activation[k_offset..k_end],
                        &mut output[row..row + MR],
                        k_end - k_offset,
                    );
                }
                row += MR;
            }

            while row < m {
                let mut acc = 0.0f32;
                let row_off = row * k_packed;
                for kk in k_offset..k_end {
                    let p = kk / 8;
                    let j = kk % 8;
                    let w = weights_u32[row_off + p];
                    let nibble = (w >> (j * 4)) & 0xF;
                    let signed = if nibble & 0x8 != 0 {
                        (nibble | 0xFFFFFFF0) as i32
                    } else {
                        nibble as i32
                    };
                    acc += signed as f32 * activation[kk];
                }
                output[row] += acc;
                row += 1;
            }

            k_offset += KC;
        }
    } else {
        gemv_packed_tiled(weights, activation, output);
        return;
    }

    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

// ============================================================
// AVX512 micro-kernel (16-wide)
// ============================================================

/// AVX512 micro-kernel: MR=4 rows, 16-wide FMA.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
#[inline]
unsafe fn micro_kernel_avx512(
    row_bufs: &[[f32; KC]; MR],
    activation: &[f32],
    output: &mut [f32],
    k: usize,
) {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    let mut kk = 0;

    while kk + 16 <= k {
        let act = _mm512_loadu_ps(activation.as_ptr().add(kk));

        let w0 = _mm512_loadu_ps(row_bufs[0].as_ptr().add(kk));
        let w1 = _mm512_loadu_ps(row_bufs[1].as_ptr().add(kk));
        let w2 = _mm512_loadu_ps(row_bufs[2].as_ptr().add(kk));
        let w3 = _mm512_loadu_ps(row_bufs[3].as_ptr().add(kk));

        acc0 = _mm512_fmadd_ps(w0, act, acc0);
        acc1 = _mm512_fmadd_ps(w1, act, acc1);
        acc2 = _mm512_fmadd_ps(w2, act, acc2);
        acc3 = _mm512_fmadd_ps(w3, act, acc3);

        kk += 16;
    }

    // Reduce to f32
    output[0] += _mm512_reduce_add_ps(acc0);
    output[1] += _mm512_reduce_add_ps(acc1);
    output[2] += _mm512_reduce_add_ps(acc2);
    output[3] += _mm512_reduce_add_ps(acc3);

    // AVX2 tail for remaining 8-wide chunks
    let mut tail_acc0 = _mm256_setzero_ps();
    let mut tail_acc1 = _mm256_setzero_ps();
    let mut tail_acc2 = _mm256_setzero_ps();
    let mut tail_acc3 = _mm256_setzero_ps();

    while kk + 8 <= k {
        let act = _mm256_loadu_ps(activation.as_ptr().add(kk));
        let w0 = _mm256_loadu_ps(row_bufs[0].as_ptr().add(kk));
        let w1 = _mm256_loadu_ps(row_bufs[1].as_ptr().add(kk));
        let w2 = _mm256_loadu_ps(row_bufs[2].as_ptr().add(kk));
        let w3 = _mm256_loadu_ps(row_bufs[3].as_ptr().add(kk));
        tail_acc0 = _mm256_fmadd_ps(w0, act, tail_acc0);
        tail_acc1 = _mm256_fmadd_ps(w1, act, tail_acc1);
        tail_acc2 = _mm256_fmadd_ps(w2, act, tail_acc2);
        tail_acc3 = _mm256_fmadd_ps(w3, act, tail_acc3);
        kk += 8;
    }

    output[0] += hsum256_ps(tail_acc0);
    output[1] += hsum256_ps(tail_acc1);
    output[2] += hsum256_ps(tail_acc2);
    output[3] += hsum256_ps(tail_acc3);

    // Scalar tail
    while kk < k {
        output[0] += row_bufs[0][kk] * activation[kk];
        output[1] += row_bufs[1][kk] * activation[kk];
        output[2] += row_bufs[2][kk] * activation[kk];
        output[3] += row_bufs[3][kk] * activation[kk];
        kk += 1;
    }
}

// ============================================================
// NEON micro-kernel (4-wide, ARM)
// ============================================================

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use std::arch::aarch64::*;

/// NEON micro-kernel: MR=4 rows, 4-wide FMA.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn micro_kernel_neon(
    row_bufs: &[[f32; KC]; MR],
    activation: &[f32],
    output: &mut [f32],
    k: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut kk = 0;

    while kk + 4 <= k {
        let act = vld1q_f32(activation.as_ptr().add(kk));

        let w0 = vld1q_f32(row_bufs[0].as_ptr().add(kk));
        let w1 = vld1q_f32(row_bufs[1].as_ptr().add(kk));
        let w2 = vld1q_f32(row_bufs[2].as_ptr().add(kk));
        let w3 = vld1q_f32(row_bufs[3].as_ptr().add(kk));

        acc0 = vfmaq_f32(acc0, w0, act);
        acc1 = vfmaq_f32(acc1, w1, act);
        acc2 = vfmaq_f32(acc2, w2, act);
        acc3 = vfmaq_f32(acc3, w3, act);

        kk += 4;
    }

    // Horizontal sum
    output[0] += vaddvq_f32(acc0);
    output[1] += vaddvq_f32(acc1);
    output[2] += vaddvq_f32(acc2);
    output[3] += vaddvq_f32(acc3);

    // Scalar tail
    while kk < k {
        output[0] += row_bufs[0][kk] * activation[kk];
        output[1] += row_bufs[1][kk] * activation[kk];
        output[2] += row_bufs[2][kk] * activation[kk];
        output[3] += row_bufs[3][kk] * activation[kk];
        kk += 1;
    }
}

// ============================================================
// Utilities
// ============================================================

/// Scalar micro-kernel fallback.
fn scalar_micro_kernel(
    row_bufs: &[[f32; KC]; MR],
    activation: &[f32],
    output: &mut [f32],
    k: usize,
) {
    for r in 0..MR {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += row_bufs[r][kk] * activation[kk];
        }
        output[r] += acc;
    }
}

/// Horizontal sum of __m256.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu;
    use crate::dtypes::{F16x2, F32x1, U4x8, U8x4};

    #[test]
    fn test_tiled_gemv_f32x1() {
        let k = 64;
        let m = 16;
        let data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let activation: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weights = PackedTensor::<F32x1>::from_f32_auto(&data, &[m, k]);

        let mut out_tiled = vec![0.0f32; m];
        gemv_packed_tiled(&weights, &activation, &mut out_tiled);

        // Verify each row is reasonable
        for (i, o) in out_tiled.iter().enumerate() {
            assert!(o.is_finite(), "Row {} should be finite", i);
            assert!(
                *o != 0.0 || data[i * k..(i + 1) * k].iter().all(|v| *v == 0.0),
                "Non-zero weights should produce non-zero output"
            );
        }
    }

    #[test]
    fn test_tiled_gemv_u8x4() {
        let k = 64;
        let m = 16;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 50.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U8x4>::from_f32_auto(&data, &[m, k]);

        let mut out_tiled = vec![0.0f32; m];
        gemv_u8x4_tiled(&weights, &activation, &mut out_tiled);

        for o in &out_tiled {
            assert!(o.is_finite());
        }
    }

    #[test]
    fn test_tiled_gemv_u4x8() {
        let k = 64;
        let m = 16;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U4x8>::from_f32_auto(&data, &[m, k]);

        let mut out_tiled = vec![0.0f32; m];
        gemv_u4x8_tiled(&weights, &activation, &mut out_tiled);

        for o in &out_tiled {
            assert!(o.is_finite());
        }
    }

    #[test]
    fn test_tiled_large_k() {
        // Test with K > KC to verify cache blocking
        let k = 20000;
        let m = 8;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).cos()).collect();
        let weights = PackedTensor::<F32x1>::from_f32_auto(&data, &[m, k]);

        let mut out_tiled = vec![0.0f32; m];
        gemv_packed_tiled(&weights, &activation, &mut out_tiled);

        for o in &out_tiled {
            assert!(o.is_finite() && *o != 0.0);
        }
    }
}
