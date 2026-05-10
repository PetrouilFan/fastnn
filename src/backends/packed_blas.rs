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

// Thread-local scratch buffer, reused across calls.
thread_local! {
    static BLAS_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn with_blas_scratch<R>(f: impl FnOnce(&mut [f32]) -> R) -> R {
    BLAS_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(KC * MR, 0.0);
        f(&mut buf)
    })
}

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
#[inline(always)]
pub fn gemv_packed_tiled<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    // Zero outputs
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Process K in cache blocks
    with_blas_scratch(|row_bufs| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);
            let k_block = k_end - k_offset;

            let mut row = 0;
            while row + MR <= m {
                micro_kernel::<T>(
                    weights,
                    &activation[k_offset..k_end],
                    &mut output[row..row + MR],
                    row,
                    k_offset,
                    k_end,
                    k_packed,
                    k_block,
                    row_bufs,
                );
                row += MR;
            }

            // Tail rows
            while row < m {
                let mut acc = 0.0f32;
                let row_offset = row * k_packed;
                let packed_start = k_offset / T::ITEMS;
                let packed_end = k_end.div_ceil(T::ITEMS);
                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_offset && idx < k_end {
                            acc += unpacked.as_ref()[j] * activation[idx];
                        }
                    }
                }
                output[row] += acc;
                row += 1;
            }

            k_offset += KC;
        }

        for row in 0..m {
            output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
            debug_assert_eq!(weights.zero_for_row(row), 0.0, "Non-zero zero_point not yet supported in GEMV kernels");
        }
    });
}

// ============================================================
// Micro-kernel: MR rows × full K block
// ============================================================

/// Thread-local scratch: MR rows × KC columns of unpacked f32.
#[inline]
#[allow(clippy::too_many_arguments)]
fn micro_kernel<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    start_row: usize,
    k_start: usize,
    k_end: usize,
    k_packed: usize,
    k_block: usize,
    row_bufs: &mut [f32],
) {
    debug_assert_eq!(output.len(), MR);
    debug_assert!(row_bufs.len() >= KC * MR);

    // Unpack MR rows of weights into contiguous buffers
    for r in 0..MR {
        let row_start = r * KC;
        if k_block < KC {
            row_bufs[row_start + k_block..row_start + KC].fill(0.0);
        }
        let row = start_row + r;
        let row_offset = row * k_packed;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        // Process packed words efficiently
        for p in packed_start..packed_end {
            let word = weights.as_packed()[row_offset + p];
            let unpacked = word.unpack_to_f32();
            let base = p * T::ITEMS;
            for j in 0..T::ITEMS {
                let idx = base + j;
                if idx >= k_start && idx < k_end {
                    row_bufs[row_start + (idx - k_start)] = unpacked.as_ref()[j];
                }
            }
        }
    }

    // Dispatch to type-specific SIMD micro-kernel
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                micro_kernel_avx2(row_bufs, activation, output, k_block);
            }
            return;
        }
    }

    // Scalar fallback
    for r in 0..MR {
        let row_start = r * KC;
        let mut acc = 0.0f32;
        for kk in 0..k_block {
            acc += row_bufs[row_start + kk] * activation[kk];
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
unsafe fn micro_kernel_avx2(row_bufs: &[f32], activation: &[f32], output: &mut [f32], k: usize) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut kk = 0;

    // Process 8 activations at a time, FMA into all 4 row accumulators
    while kk + 8 <= k {
        // Prefetch 32 elements ahead (4 iterations)
        if kk + 32 < k {
            _mm_prefetch(activation.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(KC + kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(2 * KC + kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(3 * KC + kk + 32) as *const i8, _MM_HINT_T0);
        }

        let act = _mm256_loadu_ps(activation.as_ptr().add(kk));

        // Row 0
        let row0 = _mm256_loadu_ps(row_bufs.as_ptr().add(kk));
        acc0 = _mm256_fmadd_ps(row0, act, acc0);

        // Row 1
        let row1 = _mm256_loadu_ps(row_bufs.as_ptr().add(KC + kk));
        acc1 = _mm256_fmadd_ps(row1, act, acc1);

        // Row 2
        let row2 = _mm256_loadu_ps(row_bufs.as_ptr().add(2 * KC + kk));
        acc2 = _mm256_fmadd_ps(row2, act, acc2);

        // Row 3
        let row3 = _mm256_loadu_ps(row_bufs.as_ptr().add(3 * KC + kk));
        acc3 = _mm256_fmadd_ps(row3, act, acc3);

        kk += 8;
    }

    // Handle remaining elements with scalar accumulation
    let mut tail0 = 0.0f32;
    let mut tail1 = 0.0f32;
    let mut tail2 = 0.0f32;
    let mut tail3 = 0.0f32;
    while kk < k {
        let act = *activation.as_ptr().add(kk);
        tail0 += *row_bufs.as_ptr().add(kk) * act;
        tail1 += *row_bufs.as_ptr().add(KC + kk) * act;
        tail2 += *row_bufs.as_ptr().add(2 * KC + kk) * act;
        tail3 += *row_bufs.as_ptr().add(3 * KC + kk) * act;
        kk += 1;
    }

    // Horizontal sum of each accumulator + tail
    *output.as_mut_ptr().add(0) += hsum256_ps(acc0) + tail0;
    *output.as_mut_ptr().add(1) += hsum256_ps(acc1) + tail1;
    *output.as_mut_ptr().add(2) += hsum256_ps(acc2) + tail2;
    *output.as_mut_ptr().add(3) += hsum256_ps(acc3) + tail3;
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
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    with_blas_scratch(|row_bufs| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);
            let k_block = k_end - k_offset;

            // Process MR rows at a time
            let mut row = 0;
            while row + MR <= m {
                // Unpack MR rows of int8→f32 efficiently
                for r in 0..MR {
                    let row_start = r * KC;
                    if k_block < KC {
                        row_bufs[row_start + k_block..row_start + KC].fill(0.0);
                    }
                    let row_idx = row + r;
                    let row_off = row_idx * k_packed;
                    let packed_start = k_offset / 4;
                    let packed_end = k_end.div_ceil(4);

                    for p in packed_start..packed_end {
                            let w = weights_u32[row_off + p];
                            let base = p * 4;
                            if base >= k_offset && base + 4 <= k_end {
                                unsafe {
                                    unpack_u8x4_sse4(
                                        w,
                                        row_bufs.as_mut_ptr().add(row_start + (base - k_offset)),
                                    );
                                }
                            } else {
                                let bytes = w.to_le_bytes();
                                for j in 0..4 {
                                    let idx = base + j;
                                    if idx >= k_offset && idx < k_end {
                                        row_bufs[row_start + (idx - k_offset)] = (bytes[j] as i8) as f32;
                                    }
                                }
                            }
                    }
                }

                // SIMD micro-kernel
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        unsafe {
                            micro_kernel_avx2(
                                row_bufs,
                                &activation[k_offset..k_end],
                                &mut output[row..row + MR],
                                k_end - k_offset,
                            );
                        }
                    } else {
                        scalar_micro_kernel(
                            row_bufs,
                            &activation[k_offset..k_end],
                            &mut output[row..row + MR],
                            k_end - k_offset,
                        );
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    scalar_micro_kernel(
                        row_bufs,
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

        for row in 0..m {
            output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    });
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
    let k_packed = k.div_ceil(2);
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    if is_x86_feature_detected!("f16c") {
        with_blas_scratch(|row_bufs| {
            let mut k_offset = 0;
            while k_offset < k {
                let k_end = (k_offset + KC).min(k);
                let k_block = k_end - k_offset;

                let mut row = 0;
                while row + MR <= m {
                    for r in 0..MR {
                        let row_start = r * KC;
                        if k_block < KC {
                            row_bufs[row_start + k_block..row_start + KC].fill(0.0);
                        }
                        let row_idx = row + r;
                        let row_off = row_idx * k_packed;
                        let packed_start = k_offset / 2;
                        let packed_end = k_end.div_ceil(2);

                        for p in packed_start..packed_end {
                            let w = weights_u32[row_off + p];
                            let idx = p * 2;
                            if idx < k_end {
                                row_bufs[row_start + (idx - k_offset)] =
                                    half::f16::from_bits(w as u16).to_f32();
                            }
                            if idx + 1 < k_end {
                                row_bufs[row_start + (idx + 1 - k_offset)] =
                                    half::f16::from_bits((w >> 16) as u16).to_f32();
                            }
                        }
                    }

                    unsafe {
                        micro_kernel_avx2(
                            row_bufs,
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
                    let packed_start = k_offset / 2;
                    let packed_end = k_end.div_ceil(2);

                    for p in packed_start..packed_end {
                        let w = weights_u32[row_off + p];
                        let idx = p * 2;
                        if idx >= k_offset && idx < k_end {
                            acc += half::f16::from_bits(w as u16).to_f32() * activation[idx];
                        }
                        if idx + 1 >= k_offset && idx + 1 < k_end {
                            acc += half::f16::from_bits((w >> 16) as u16).to_f32()
                                * activation[idx + 1];
                        }
                    }
                    output[row] += acc;
                    row += 1;
                }

                k_offset += KC;
            }
        });
    } else {
        // Fallback to generic
        gemv_packed_tiled(weights, activation, output);
        return;
    }

    for row in 0..m {
        output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
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
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();

    for o in output.iter_mut() {
        *o = 0.0;
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        with_blas_scratch(|row_bufs| {
            let mut k_offset = 0;
            while k_offset < k {
                let k_end = (k_offset + KC).min(k);
                let k_block = k_end - k_offset;

                let mut row = 0;
                while row + MR <= m {
                    for r in 0..MR {
                        let row_start = r * KC;
                        if k_block < KC {
                            row_bufs[row_start + k_block..row_start + KC].fill(0.0);
                        }
                        let row_idx = row + r;
                        let row_off = row_idx * k_packed;
                        let packed_start = k_offset / 8;
                        let packed_end = k_end.div_ceil(8);

                        let mut p = packed_start;
                        while p < packed_end {
                            let base = p * 8;
                            if p + 1 < packed_end && base + 16 <= k_end {
                                unsafe {
                                    unpack_u4x8_wordpair_simd(
                                        weights_u32, row_off, p, row_bufs, row_start, k_offset,
                                    );
                                }
                                p += 2;
                            } else {
                                let w = weights_u32[row_off + p];
                                for j in 0..8 {
                                    let idx = base + j;
                                    if idx >= k_offset && idx < k_end {
                                        let nibble = (w >> (j * 4)) & 0xF;
                                        let signed = if nibble & 0x8 != 0 {
                                            (nibble | 0xFFFFFFF0) as i32
                                        } else {
                                            nibble as i32
                                        };
                                        row_bufs[row_start + (idx - k_offset)] = signed as f32;
                                    }
                                }
                                p += 1;
                            }
                        }
                    }

                    unsafe {
                        micro_kernel_avx2(
                            row_bufs,
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
                    let packed_start = k_offset / 8;
                    let packed_end = k_end.div_ceil(8);

                    for p in packed_start..packed_end {
                        let w = weights_u32[row_off + p];
                        let base = p * 8;
                        for j in 0..8 {
                            let idx = base + j;
                            if idx >= k_offset && idx < k_end {
                                let nibble = (w >> (j * 4)) & 0xF;
                                let signed = if nibble & 0x8 != 0 {
                                    (nibble | 0xFFFFFFF0) as i32
                                } else {
                                    nibble as i32
                                };
                                acc += signed as f32 * activation[idx];
                            }
                        }
                    }
                    output[row] += acc;
                    row += 1;
                }

                k_offset += KC;
            }
        });
    } else {
        gemv_packed_tiled(weights, activation, output);
        return;
    }

    for row in 0..m {
        output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
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
#[allow(clippy::too_many_arguments)]
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
fn scalar_micro_kernel(row_bufs: &[f32], activation: &[f32], output: &mut [f32], k: usize) {
    for r in 0..MR {
        let mut acc = 0.0f32;
        let row_start = r * KC;
        for kk in 0..k {
            acc += row_bufs[row_start + kk] * activation[kk];
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
// SIMD unpack helpers
// ============================================================

/// SIMD unpack a single u32 word of 4xint8 into 4xf32 using PMOVSXBD.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn unpack_u8x4_sse4(w: u32, dst: *mut f32) {
    let wvec = _mm_cvtsi32_si128(w as i32);
    let i32_vals = _mm_cvtepi8_epi32(wvec);
    let f32_vals = _mm_cvtepi32_ps(i32_vals);
    _mm_storeu_ps(dst, f32_vals);
}

/// SIMD unpack 2 u32 words of 8xint4 each into 16xf32 using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn unpack_u4x8_wordpair_simd(
    weights_u32: &[u32],
    row_off: usize,
    p: usize,
    row_bufs: &mut [f32],
    row_start: usize,
    k_offset: usize,
) {
    let w0 = weights_u32[row_off + p];
    let w1 = weights_u32[row_off + p + 1];

    let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_bit = _mm256_set1_epi32(0x8);

    let w0v = _mm256_set1_epi32(w0 as i32);
    let w1v = _mm256_set1_epi32(w1 as i32);

    let nib0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
    let nib1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

    // Branchless sign extension: (nib ^ 8) - 8
    let signed0 = _mm256_sub_epi32(_mm256_xor_si256(nib0, sign_bit), sign_bit);
    let signed1 = _mm256_sub_epi32(_mm256_xor_si256(nib1, sign_bit), sign_bit);

    let f0 = _mm256_cvtepi32_ps(signed0);
    let f1 = _mm256_cvtepi32_ps(signed1);

    let base = p * 8;
    _mm256_storeu_ps(row_bufs.as_mut_ptr().add(row_start + (base - k_offset)), f0);
    _mm256_storeu_ps(row_bufs.as_mut_ptr().add(row_start + (base + 8 - k_offset)), f1);
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U4x8, U8x4};

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
