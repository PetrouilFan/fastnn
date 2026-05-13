//! ARM NEON SIMD kernels for packed precision types.
//!
//! Gates on `target_arch = "aarch64"` + `feature = "neon"`.
//! Provides U8x4 and U4x8 GEMV accelerators using NEON intrinsics.

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use std::arch::aarch64::*;

// ============================================================
// U8x4 NEON GEMV
// ============================================================

/// NEON-accelerated GEMV for U8x4 packed weights.
/// Processes 4 packed u32 words (16 int8 values) per SIMD iteration.
///
/// Intrinsic pipeline:
///   u32x4 → int8x16 → int16x8×2 → int32x4×4 → float32x4×4 → FMA×4
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
pub fn gemv_u8x4_neon(
    weights: &PackedTensor<crate::dtypes::U8x4>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();

    for row in 0..m {
        let row_offset = row * k_packed;
        unsafe {
            output[row] = gemv_row_u8x4_neon(weights_u32, activation, row_offset, k, k_packed)
                * weights.scale_for_row(row)
                + weights.zero_for_row(row);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn gemv_row_u8x4_neon(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut p = 0;
    let mut act_idx = 0;

    while p + 4 <= k_packed && act_idx + 16 <= k {
        let ptr = weights_u32.as_ptr().add(row_offset + p);
        let wvec = vld1q_u32(ptr);
        let bytes = vreinterpretq_s8_u32(wvec);

        let low = vmovl_s8(vget_low_s8(bytes));
        let high = vmovl_s8(vget_high_s8(bytes));

        let l0 = vmovl_s16(vget_low_s16(low));
        let l1 = vmovl_s16(vget_high_s16(low));
        let h0 = vmovl_s16(vget_low_s16(high));
        let h1 = vmovl_s16(vget_high_s16(high));

        let f0 = vcvtq_f32_s32(l0);
        let f1 = vcvtq_f32_s32(l1);
        let f2 = vcvtq_f32_s32(h0);
        let f3 = vcvtq_f32_s32(h1);

        let a0 = vld1q_f32(activation.as_ptr().add(act_idx));
        let a1 = vld1q_f32(activation.as_ptr().add(act_idx + 4));
        let a2 = vld1q_f32(activation.as_ptr().add(act_idx + 8));
        let a3 = vld1q_f32(activation.as_ptr().add(act_idx + 12));

        acc0 = vfmaq_f32(acc0, f0, a0);
        acc1 = vfmaq_f32(acc1, f1, a1);
        acc2 = vfmaq_f32(acc2, f2, a2);
        acc3 = vfmaq_f32(acc3, f3, a3);

        p += 4;
        act_idx += 16;
    }

    let sum01 = vaddq_f32(acc0, acc1);
    let sum23 = vaddq_f32(acc2, acc3);
    let mut total = hsum_f32x4(vaddq_f32(sum01, sum23));

    if p + 2 <= k_packed && act_idx + 8 <= k {
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let b0 = w0.to_le_bytes();
        let b1 = w1.to_le_bytes();
        for j in 0..4 {
            total += (b0[j] as i8) as f32 * activation[act_idx + j];
            total += (b1[j] as i8) as f32 * activation[act_idx + 4 + j];
        }
        p += 2;
        act_idx += 8;
    }

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

// ============================================================
// U4x8 NEON GEMV
// ============================================================

/// NEON-accelerated GEMV for U4x8 packed weights.
/// Each u32 word packs 8 × signed 4-bit nibbles.
/// Extracts nibbles via lane insertion, processes 4 at a time.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
pub fn gemv_u4x8_neon(
    weights: &PackedTensor<crate::dtypes::U4x8>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();

    for row in 0..m {
        let row_offset = row * k_packed;
        unsafe {
            output[row] = gemv_row_u4x8_neon(weights_u32, activation, row_offset, k, k_packed)
                * weights.scale_for_row(row)
                + weights.zero_for_row(row);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn gemv_row_u4x8_neon(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let sign_ext = vdupq_n_u32(8);

    let mut p = 0;
    let mut act_idx = 0;

    while p < k_packed && act_idx + 8 <= k {
        let w = weights_u32[row_offset + p];

        let mut nib_lo = vdupq_n_u32(0);
        nib_lo = vsetq_lane_u32(w & 0xF, nib_lo, 0);
        nib_lo = vsetq_lane_u32((w >> 4) & 0xF, nib_lo, 1);
        nib_lo = vsetq_lane_u32((w >> 8) & 0xF, nib_lo, 2);
        nib_lo = vsetq_lane_u32((w >> 12) & 0xF, nib_lo, 3);

        let mut nib_hi = vdupq_n_u32(0);
        nib_hi = vsetq_lane_u32((w >> 16) & 0xF, nib_hi, 0);
        nib_hi = vsetq_lane_u32((w >> 20) & 0xF, nib_hi, 1);
        nib_hi = vsetq_lane_u32((w >> 24) & 0xF, nib_hi, 2);
        nib_hi = vsetq_lane_u32((w >> 28) & 0xF, nib_hi, 3);

        let s_ext = |v: uint32x4_t| -> int32x4_t {
            let xored = veorq_u32(v, sign_ext);
            let subbed = vsubq_u32(xored, sign_ext);
            vreinterpretq_s32_u32(subbed)
        };

        let f_lo = vcvtq_f32_s32(s_ext(nib_lo));
        let f_hi = vcvtq_f32_s32(s_ext(nib_hi));

        let a_lo = vld1q_f32(activation.as_ptr().add(act_idx));
        let a_hi = vld1q_f32(activation.as_ptr().add(act_idx + 4));

        acc0 = vfmaq_f32(acc0, f_lo, a_lo);
        acc1 = vfmaq_f32(acc1, f_hi, a_hi);

        p += 1;
        act_idx += 8;
    }

    let mut total = hsum_f32x4(vaddq_f32(acc0, acc1));

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

// ============================================================
// Shared NEON utility
// ============================================================

/// Horizontal sum of float32x4_t (aarch64).
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    let pair = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    vget_lane_f32(vpadd_f32(pair, pair), 0)
}

// ============================================================
// Non-NEON fallback stubs
// ============================================================

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
pub fn gemv_u8x4_neon(
    _weights: &PackedTensor<crate::dtypes::U8x4>,
    _activation: &[f32],
    _output: &mut [f32],
) {
    unimplemented!("gemv_u8x4_neon requires aarch64 + neon feature");
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
pub fn gemv_u4x8_neon(
    _weights: &PackedTensor<crate::dtypes::U4x8>,
    _activation: &[f32],
    _output: &mut [f32],
) {
    unimplemented!("gemv_u4x8_neon requires aarch64 + neon feature");
}
