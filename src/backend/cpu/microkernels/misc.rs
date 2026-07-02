//! CPU misc microkernels (matmul, dot product, upsampling, concat, transpose) — extracted from microkernels.rs

#![allow(dead_code, unused_imports)]

use crate::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;

use super::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// ============================================================
// SWAR ReLU kernels (AVX2)
// ============================================================

// ============================================================
// Matmul — cache-blocked row-wise microkernel
// ============================================================

pub const MIN_BLAS_SIZE: usize = 64;

pub fn matmul_blas_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_blas_with_transpose_into(a, b, out, m, k, n, false, false)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_blas_with_transpose_into(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) {
    let rsa: isize = if trans_a { 1 } else { k as isize };
    let csa: isize = if trans_a { m as isize } else { 1 };
    let rsb: isize = if trans_b { 1 } else { n as isize };
    let csb: isize = if trans_b { k as isize } else { 1 };

    // SAFETY: sgemm operates on valid slices with correct
    // dimensions and strides.
    unsafe {
        crate::backend::cpu::sgemm::sgemm(
            m,
            k,
            n,
            1.0f32,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            0.0f32,
            out.as_mut_ptr(),
            n as isize,
            1isize,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
// SAFETY: Caller must ensure `a_ptr`, `b_ptr`, and `out_ptr` are valid,
// non-overlapping, and point to allocations of sufficient size for the
// matrix dimensions (m, k, n) and strides.
pub unsafe fn blocked_row_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    row: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch_stride: usize,
    a_stride_0: usize,
    a_stride_1: usize,
    b_batch_stride: usize,
    b_stride_0: usize,
    b_stride_1: usize,
) {
    const TILE_K: usize = 64;
    const TILE_N: usize = 4;

    let bat = row / m;
    let i = row % m;
    let a_off = bat * a_batch_stride + i * a_stride_0;
    let b_off = bat * b_batch_stride;
    let out_off = row * n;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let use_simd = a_stride_1 == 1 && b_stride_1 == 1 && n >= 8;

    for j in 0..n {
        *out_ptr.add(out_off + j) = 0.0;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if use_simd {
        use std::arch::x86_64::*;

        const TILE_N_SIMD: usize = 8;
        let mut ko = 0;
        while ko < k {
            let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

            let mut jo = 0;
            while jo + TILE_N_SIMD <= n {
                // SAFETY: `a_ptr`, `b_ptr`, and `out_ptr` are valid and non-overlapping
                // (guaranteed by blocked_row_matmul's contract); AVX2 loads/stores
                // use unaligned access which is safe for any valid pointer.
                unsafe {
                    let mut acc = _mm256_setzero_ps();

                    let mut kk = ko;
                    while kk + 4 <= kend {
                        // Prefetch next tile's data — 32 elements (128 bytes) ahead
                        if kk + 32 < k {
                            _mm_prefetch(
                                a_ptr.add(a_off + (kk + 32) * a_stride_1) as *const i8,
                                _MM_HINT_T0,
                            );
                            _mm_prefetch(
                                b_ptr.add(b_off + (kk + 32) * b_stride_0 + jo) as *const i8,
                                _MM_HINT_T0,
                            );
                        }

                        let a0 = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let b0 = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a0, b0, acc);

                        let a1 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 1) * a_stride_1));
                        let b1 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 1) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a1, b1, acc);

                        let a2 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 2) * a_stride_1));
                        let b2 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 2) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a2, b2, acc);

                        let a3 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 3) * a_stride_1));
                        let b3 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 3) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a3, b3, acc);

                        kk += 4;
                    }
                    while kk < kend {
                        let av = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let bv = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(av, bv, acc);
                        kk += 1;
                    }

                    let out_v = _mm256_loadu_ps(out_ptr.add(out_off + jo));
                    _mm256_storeu_ps(out_ptr.add(out_off + jo), _mm256_add_ps(out_v, acc));
                }
                jo += TILE_N_SIMD;
            }

            while jo < n {
                let mut sum = 0.0f32;
                for kk in ko..kend {
                    sum += *a_ptr.add(a_off + kk * a_stride_1)
                        * b_ptr.add(b_off + kk * b_stride_0 + jo).read();
                }
                let p = out_ptr.add(out_off + jo);
                *p += sum;
                jo += 1;
            }

            ko += TILE_K;
        }
        return;
    }

    let mut ko = 0;
    while ko < k {
        let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

        let mut jo = 0;
        while jo + TILE_N <= n {
            let mut acc = [0.0f32; TILE_N];

            let mut kk = ko;
            while kk < kend {
                let av = *a_ptr.add(a_off + kk * a_stride_1);
                for t in 0..TILE_N {
                    acc[t] += av * *b_ptr.add(b_off + kk * b_stride_0 + (jo + t) * b_stride_1);
                }
                kk += 1;
            }

            for t in 0..TILE_N {
                let p = out_ptr.add(out_off + jo + t);
                *p += acc[t];
            }
            jo += TILE_N;
        }

        while jo < n {
            let mut sum = 0.0f32;
            for kk in ko..kend {
                sum += *a_ptr.add(a_off + kk * a_stride_1)
                    * b_ptr.add(b_off + kk * b_stride_0 + jo * b_stride_1).read();
            }
            let p = out_ptr.add(out_off + jo);
            *p += sum;
            jo += 1;
        }

        ko += TILE_K;
    }
}

// ============================================================
// SIMD dot product
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
// SAFETY: Caller must ensure `a` and `b` are valid, non-overlapping slices,
// each at least `len` elements long.
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;

    if is_x86_feature_detected!("avx512f") {
        unsafe {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            while i + 64 <= len {
                let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
                let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
                let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
                let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
                let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
                let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
                let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
                let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));

                acc0 = _mm512_fmadd_ps(a0, b0, acc0);
                acc1 = _mm512_fmadd_ps(a1, b1, acc1);
                acc2 = _mm512_fmadd_ps(a2, b2, acc2);
                acc3 = _mm512_fmadd_ps(a3, b3, acc3);

                i += 64;
            }

            let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            sum += _mm512_reduce_add_ps(acc);

            while i + 16 <= len {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                acc0 = _mm512_fmadd_ps(a_vec, b_vec, _mm512_setzero_ps());
                sum += _mm512_reduce_add_ps(acc0);
                i += 16;
            }
        }
    } else if is_x86_feature_detected!("avx2") {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                i += 32;
            }

            let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            sum += hsum256_ps(acc);

            while i + 8 <= len {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                sum += hsum256_ps(_mm256_mul_ps(a_vec, b_vec));
                i += 8;
            }
        }
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
#[inline]
// SAFETY: Same as the SIMD version — `a` and `b` must be valid for `len` elements.
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

// ============================================================
// UpsampleNearest2d — scalar and AVX2
// ============================================================

/// Nearest-neighbor upsampling: each input pixel replicates into a
/// `scale_h × scale_w` block in the output.
///
/// # Layout
/// Input is NCHW: shape `[nc, h_in, w_in]` stored flat (`nc = N * C`).
/// Output is `[nc, h_in * scale_h, w_in * scale_w]`.
#[inline]
pub fn upsample_nearest2d_f32(
    input: &[f32],
    output: &mut [f32],
    nc: usize,
    h_in: usize,
    w_in: usize,
    scale_h: usize,
    scale_w: usize,
) {
    let hw = h_in * w_in;
    let out_row_stride = w_in * scale_w;
    for nci in 0..nc {
        for hi in 0..h_in {
            for wi in 0..w_in {
                let val = input[nci * hw + hi * w_in + wi];
                for sh in 0..scale_h {
                    let out_row = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    for sw in 0..scale_w {
                        output[out_row + sw] = val;
                    }
                }
            }
        }
    }
}

/// AVX2 nearest-neighbor upsampling using broadcast stores.
///
/// For `scale_w ∈ {1, 2, 4}`, processes `8 / scale_w` input columns per
/// vector iteration, broadcasting each value `scale_w` times into an 8-wide
/// store. This reduces store instructions by up to 87.5% vs scalar.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and sized according to the upsampling parameters (nc, h_in, w_in, scale_h, scale_w).
pub unsafe fn upsample_nearest2d_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    nc: usize,
    h_in: usize,
    w_in: usize,
    scale_h: usize,
    scale_w: usize,
) {
    // Fallback to scalar for unsupported scale factors
    if scale_w != 1 && scale_w != 2 && scale_w != 4 {
        return upsample_nearest2d_f32(input, output, nc, h_in, w_in, scale_h, scale_w);
    }
    let hw = h_in * w_in;
    let out_row_stride = w_in * scale_w;
    // Number of input columns processed per AVX2 iteration (8 output floats / scale_w)
    let vec_step = 8 / scale_w;

    for nci in 0..nc {
        for hi in 0..h_in {
            let mut wi = 0;
            while wi + vec_step <= w_in {
                let in_base = nci * hw + hi * w_in + wi;
                // Build a vector of 8 floats with each input value repeated scale_w times.
                // _mm256_set_ps args go from highest lane (index 7) to lowest (index 0).
                // Memory order after store: arg0 at lowest address, arg7 at highest.
                let v = match scale_w {
                    1 => {
                        // Direct copy: 8 input values, no repetition
                        _mm256_loadu_ps(input.as_ptr().add(in_base))
                    }
                    2 => {
                        // 4 input values → each repeated twice
                        _mm256_set_ps(
                            *input.get_unchecked(in_base + 3),
                            *input.get_unchecked(in_base + 3),
                            *input.get_unchecked(in_base + 2),
                            *input.get_unchecked(in_base + 2),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base),
                            *input.get_unchecked(in_base),
                        )
                    }
                    4 => {
                        // 2 input values → each repeated 4 times
                        _mm256_set_ps(
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base),
                            *input.get_unchecked(in_base),
                            *input.get_unchecked(in_base),
                            *input.get_unchecked(in_base),
                        )
                    }
                    _ => unreachable!(),
                };

                // Write the same 8-wide block to each output row (sh = 0..scale_h)
                for sh in 0..scale_h {
                    let out_base = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_base), v);
                }

                wi += vec_step;
            }
            // Handle remaining columns (wi < vec_step from end)
            for wi in wi..w_in {
                let val = input[nci * hw + hi * w_in + wi];
                for sh in 0..scale_h {
                    let out_base = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    for sw in 0..scale_w {
                        *output.get_unchecked_mut(out_base + sw) = val;
                    }
                }
            }
        }
    }
}

// ============================================================
// Concat — scalar and AVX2
// ============================================================

/// Scalar concat: copies `input` into `output` at `output_offset`.
/// Returns the new output offset after copy.
#[inline]
pub fn concat_f32_scalar(input: &[f32], output: &mut [f32], output_offset: usize) -> usize {
    let end = (output_offset + input.len()).min(output.len());
    let len = end - output_offset;
    output[output_offset..end].copy_from_slice(&input[..len]);
    output_offset + len
}

/// AVX2 concat: copies `input` into `output` at `output_offset` using
/// 8-wide vector stores for the bulk of the copy, falling back to scalar
/// for the remainder.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and `output_offset + input.len() <= output.len()`.
pub unsafe fn concat_f32_avx2(input: &[f32], output: &mut [f32], output_offset: usize) -> usize {
    let out_start = output_offset;
    let copy_len = input.len().min(output.len().saturating_sub(out_start));
    let mut i = 0usize;
    // 8-wide vector copy
    while i + 8 <= copy_len {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(out_start + i), v);
        i += 8;
    }
    // Scalar remainder
    for j in i..copy_len {
        *output.as_mut_ptr().add(out_start + j) = *input.as_ptr().add(j);
    }
    out_start + copy_len
}

// ============================================================
// Transpose 2D — scalar and AVX2 tiled 8×8
// ============================================================

/// Scalar 2D transpose: out[j * m + i] = in[i * n + j]
#[inline]
pub fn transpose_f32_scalar(input: &[f32], output: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            output[j * m + i] = input[i * n + j];
        }
    }
}

/// AVX2 tiled 8×8 2D transpose. Processes 8×8 tiles using unpack
/// and permute instructions, falling back to scalar for edge tiles.
///
/// Uses the classic:
///   t0 = _mm256_unpacklo_ps(a0, a1)  // a0[0], a1[0], a0[1], a1[1], ...
///   t1 = _mm256_unpackhi_ps(a0, a1)  // a0[2], a1[2], a0[3], a1[3], ...
///   ... then combine across rows with permute.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and `input` has at least `m * n` elements, `output` has at least `n * m`.
pub unsafe fn transpose_f32_avx2(input: &[f32], output: &mut [f32], m: usize, n: usize) {
    use std::arch::x86_64::*;

    // Process 8×8 tiles
    let mut i = 0;
    while i + 8 <= m {
        let mut j = 0;
        while j + 8 <= n {
            // Load 8 rows of 8 elements each
            let r0 = _mm256_loadu_ps(input.as_ptr().add(i * n + j));
            let r1 = _mm256_loadu_ps(input.as_ptr().add((i + 1) * n + j));
            let r2 = _mm256_loadu_ps(input.as_ptr().add((i + 2) * n + j));
            let r3 = _mm256_loadu_ps(input.as_ptr().add((i + 3) * n + j));
            let r4 = _mm256_loadu_ps(input.as_ptr().add((i + 4) * n + j));
            let r5 = _mm256_loadu_ps(input.as_ptr().add((i + 5) * n + j));
            let r6 = _mm256_loadu_ps(input.as_ptr().add((i + 6) * n + j));
            let r7 = _mm256_loadu_ps(input.as_ptr().add((i + 7) * n + j));

            // Unpack low/high 64-bit halves (4-element chunks)
            let t01a = _mm256_unpacklo_ps(r0, r1); // r0[0], r1[0], r0[1], r1[1]
            let t01b = _mm256_unpackhi_ps(r0, r1); // r0[2], r1[2], r0[3], r1[3]
            let t23a = _mm256_unpacklo_ps(r2, r3);
            let t23b = _mm256_unpackhi_ps(r2, r3);
            let t45a = _mm256_unpacklo_ps(r4, r5);
            let t45b = _mm256_unpackhi_ps(r4, r5);
            let t67a = _mm256_unpacklo_ps(r6, r7);
            let t67b = _mm256_unpackhi_ps(r6, r7);

            // Combine into 8-wide transpose result
            let q0 = _mm256_shuffle_ps(t01a, t23a, 0b_01_00_01_00); // cols 0,1 of rows 0-3
            let q1 = _mm256_shuffle_ps(t01a, t23a, 0b_11_10_11_10); // cols 2,3 of rows 0-3
            let q2 = _mm256_shuffle_ps(t01b, t23b, 0b_01_00_01_00); // cols 4,5 of rows 0-3
            let q3 = _mm256_shuffle_ps(t01b, t23b, 0b_11_10_11_10); // cols 6,7 of rows 0-3
            let q4 = _mm256_shuffle_ps(t45a, t67a, 0b_01_00_01_00); // cols 0,1 of rows 4-7
            let q5 = _mm256_shuffle_ps(t45a, t67a, 0b_11_10_11_10); // cols 2,3 of rows 4-7
            let q6 = _mm256_shuffle_ps(t45b, t67b, 0b_01_00_01_00); // cols 4,5 of rows 4-7
            let q7 = _mm256_shuffle_ps(t45b, t67b, 0b_11_10_11_10); // cols 6,7 of rows 4-7

            // Permute to final layout — now rows are in order 0,4,1,5,2,6,3,7
            // We need them as 0,1,2,3,4,5,6,7
            let out0 = _mm256_permute2f128_ps(q0, q4, 0x20);
            let out4 = _mm256_permute2f128_ps(q0, q4, 0x31);
            let out1 = _mm256_permute2f128_ps(q1, q5, 0x20);
            let out5 = _mm256_permute2f128_ps(q1, q5, 0x31);
            let out2 = _mm256_permute2f128_ps(q2, q6, 0x20);
            let out6 = _mm256_permute2f128_ps(q2, q6, 0x31);
            let out3 = _mm256_permute2f128_ps(q3, q7, 0x20);
            let out7 = _mm256_permute2f128_ps(q3, q7, 0x31);

            // Store 8 cols × 8 rows in transposed position
            _mm256_storeu_ps(output.as_mut_ptr().add(j * m + i), out0);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 1) * m + i), out1);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 2) * m + i), out2);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 3) * m + i), out3);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 4) * m + i), out4);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 5) * m + i), out5);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 6) * m + i), out6);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 7) * m + i), out7);

            j += 8;
        }
        // Scalar remainder columns
        for jj in j..n {
            for ii in i..i + 8 {
                *output.as_mut_ptr().add(jj * m + ii) = *input.as_ptr().add(ii * n + jj);
            }
        }
        i += 8;
    }
    // Scalar remainder rows
    for ii in i..m {
        for jj in 0..n {
            *output.as_mut_ptr().add(jj * m + ii) = *input.as_ptr().add(ii * n + jj);
        }
    }
}
