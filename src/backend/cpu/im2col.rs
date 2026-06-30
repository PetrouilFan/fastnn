//! Im2col kernel for CPU convolution — operates on a single image (no batch).

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

/// Reference scalar im2col for square kernels.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn im2col_kernel(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    let dk = (kernel_size - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dk {
        (h + 2 * padding - dk) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dk {
        (w + 2 * padding - dk) / stride + 1
    } else {
        0
    };
    let col_w = c * kernel_size * kernel_size;

    for oh in 0..h_out {
        for ow in 0..w_out {
            let row = oh * w_out + ow;
            for ic in 0..c {
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                        let iw = (ow * stride + kw * dilation).wrapping_sub(padding);
                        let dst =
                            row * col_w + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        if ih < h && iw < w {
                            let src = (ic * h + ih) * w + iw;
                            col[dst] = data[src];
                        } else {
                            col[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Im2col kernel supporting non-square kernels (kh != kw).
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn im2col_kernel_rect(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    let dh = (kh - 1) * dilation + 1;
    let dw = (kw - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dh {
        (h + 2 * padding - dh) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dw {
        (w + 2 * padding - dw) / stride + 1
    } else {
        0
    };
    let col_w = c * kh * kw;

    for oh in 0..h_out {
        for ow in 0..w_out {
            let row = oh * w_out + ow;
            for ic in 0..c {
                for kkh in 0..kh {
                    for kkw in 0..kw {
                        let ih = oh * stride + kkh * dilation;
                        let iw = ow * stride + kkw * dilation;
                        let dst = row * col_w + ic * kh * kw + kkh * kw + kkw;
                        if ih < h + padding && iw < w + padding && ih >= padding && iw >= padding {
                            let src = (ic * h + (ih - padding)) * w + (iw - padding);
                            col[dst] = data[src];
                        } else {
                            col[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

// ── AVX2‑accelerated im2col: stride=1, padding=1, dilation=1, kh=kw=3 ──

/// Specialized AVX2 im2col for 3×3 stride-1 pad-1 dil-1 convolutions.
///
/// Processes 8 consecutive `ow` positions in a single SIMD loop: loads 8
/// contiguous input values and writes them to strided column‑buffer positions.
/// Boundary columns (where the 8‑wide window would cross padding) are handled
/// with scalar fallback.
///
/// # Safety
/// Caller must guarantee AVX2+FMA are available at runtime.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn im2col_kernel_rect_avx2(data: &[f32], c: usize, h: usize, w: usize, col: &mut [f32]) {
    let h_out = h; // stride=1, pad=1 → h_out = h
    let w_out = w; // stride=1, pad=1 → w_out = w
    let col_w = c * 9;

    for oh in 0..h_out {
        let row_offset = oh * w_out * col_w;
        for ic in 0..c {
            let col_ch_off = row_offset + ic * 9;

            for kh in 0..3 {
                let ih = oh + kh; // stride=1, pad=1 → ih = oh+kh (before padding adjust)
                if ih < 1 || ih > h {
                    // Padding row: MUST write zeros to all kw×ow positions in col
                    // (skipping would leave stale data from the reused buffer)
                    for kw in 0..3 {
                        let col_k_off = kh * 3 + kw;
                        for ow in 0..w_out {
                            let dst = col_ch_off + col_k_off + ow * col_w;
                            *col.get_unchecked_mut(dst) = 0.0;
                        }
                    }
                    continue;
                }
                let inp_row_base = ic * h * w + (ih - 1) * w;

                for kw in 0..3 {
                    let col_k_off = kh * 3 + kw;

                    // ── Left boundary (ow = 0) ──
                    let iw_left = kw;
                    if iw_left >= 1 && iw_left <= w {
                        let ow = 0;
                        let dst = col_ch_off + col_k_off + ow * col_w;
                        let src = inp_row_base + iw_left - 1;
                        *col.get_unchecked_mut(dst) = *data.get_unchecked(src);
                    } else {
                        // Padding column: write zero
                        let ow = 0;
                        let dst = col_ch_off + col_k_off + ow * col_w;
                        *col.get_unchecked_mut(dst) = 0.0;
                    }

                    // ── SIMD main body: process 8 ow at a time ──
                    let mut ow = 1;
                    let ow_simd_end = match kw {
                        0 | 1 if w_out > 8 => w_out - 8,
                        2 if w_out > 9 => w_out - 9,
                        _ => 1,
                    };

                    while ow < ow_simd_end {
                        let dst_base = col_ch_off + col_k_off + ow * col_w;
                        let col_ptr = col.as_mut_ptr().add(dst_base);
                        let src_base = inp_row_base + ow + kw - 1;
                        let src_ptr = data.as_ptr().add(src_base);

                        *col_ptr = *src_ptr;
                        *col_ptr.add(col_w) = *src_ptr.add(1);
                        *col_ptr.add(2 * col_w) = *src_ptr.add(2);
                        *col_ptr.add(3 * col_w) = *src_ptr.add(3);
                        *col_ptr.add(4 * col_w) = *src_ptr.add(4);
                        *col_ptr.add(5 * col_w) = *src_ptr.add(5);
                        *col_ptr.add(6 * col_w) = *src_ptr.add(6);
                        *col_ptr.add(7 * col_w) = *src_ptr.add(7);

                        ow += 8;
                    }

                    // ── Right tail (scalar) ──
                    while ow < w_out {
                        let iw = ow + kw;
                        let dst = col_ch_off + col_k_off + ow * col_w;
                        if iw >= 1 && iw <= w {
                            let src = inp_row_base + iw - 1;
                            *col.get_unchecked_mut(dst) = *data.get_unchecked(src);
                        } else {
                            *col.get_unchecked_mut(dst) = 0.0;
                        }
                        ow += 1;
                    }
                }
            }
        }
    }
}

// ── AVX2 im2col: stride-2, pad-1, dil-1, 3×3 ─────────────────────

/// Specialized AVX2 im2col for 3×3 stride-2 pad-1 dil-1 convolutions.
///
/// Processes 8 consecutive `ow` positions.  Uses two 256-bit loads
/// covering 16 input columns and extracts every other element via
/// `_mm256_shuffle_ps` to handle the stride-2 access pattern.
///
/// # Safety
/// Caller must guarantee AVX2+FMA are available.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn im2col_kernel_rect_avx2_stride2(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    col: &mut [f32],
) {
    let h_out = h / 2;
    let w_out = w / 2;
    let col_w = c * 9;

    for oh in 0..h_out {
        let row_offset = oh * w_out * col_w;
        for ic in 0..c {
            let col_ch_off = row_offset + ic * 9;

            for kh in 0..3 {
                let ih = oh * 2 + kh;
                if ih < 1 || ih > h {
                    for kw in 0..3 {
                        let col_k_off = kh * 3 + kw;
                        for ow in 0..w_out {
                            *col.get_unchecked_mut(col_ch_off + col_k_off + ow * col_w) = 0.0;
                        }
                    }
                    continue;
                }
                let inp_row_base = ic * h * w + (ih - 1) * w;

                for kw in 0..3 {
                    let col_k_off = kh * 3 + kw;

                    // ── Left boundary: ow = 0 ──
                    let iw0 = kw;
                    let dst0 = col_ch_off + col_k_off;
                    if iw0 >= 1 && iw0 <= w {
                        *col.get_unchecked_mut(dst0) = *data.get_unchecked(inp_row_base + iw0 - 1);
                    } else {
                        *col.get_unchecked_mut(dst0) = 0.0;
                    }

                    // ── SIMD main body: process 8 ow at a time ──
                    // For stride-2: 8 consecutive ow values map to 16 input columns
                    // spaced 2 apart.  Load 16 floats and extract every other.
                    let ow_simd_end = if w_out > 9 { w_out - 8 } else { 1 };

                    let mut ow = 1;
                    while ow < ow_simd_end {
                        let iw_base = ow * 2 + kw; // first input column (0-indexed, before pad)
                        let src_ptr = data.as_ptr().add(inp_row_base + iw_base - 1);
                        let v0 = _mm256_loadu_ps(src_ptr);
                        let v1 = _mm256_loadu_ps(src_ptr.add(8));
                        // Extract every other element from a 16-span window:
                        //   v0[0], v0[2], v1[0], v1[2],  v0[4], v0[6], v1[4], v1[6]
                        // The shuffle places these as  [0,2,8,10, 4,6,12,14].
                        // The permute reorders to     [0,2,4,6,  8,10,12,14].
                        let s = _mm256_shuffle_ps(v0, v1, 0b10_00_10_00);
                        let v = _mm256_permutevar8x32_ps(
                            s,
                            _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7),
                        );

                        // Store directly: extract lanes via cast+extract instead of
                        // spill-to-stack to avoid store-forwarding penalty.
                        let dst_base = col_ch_off + col_k_off + ow * col_w;
                        let ptr = col.as_mut_ptr().add(dst_base);
                        // Lane 0..3 → lower 128 bits, lane 4..7 → upper 128 bits
                        let lo = _mm256_castps256_ps128(v);
                        let hi = _mm256_extractf128_ps(v, 1);
                        _mm_store_ss(ptr, lo);
                        _mm_store_ss(ptr.add(col_w), _mm_unpackhi_ps(lo, lo));
                        _mm_store_ss(ptr.add(2 * col_w), _mm_permute_ps(lo, 0b10_10_00_10));
                        _mm_store_ss(ptr.add(3 * col_w), _mm_permute_ps(lo, 0b10_10_10_00));
                        _mm_store_ss(ptr.add(4 * col_w), hi);
                        _mm_store_ss(ptr.add(5 * col_w), _mm_unpackhi_ps(hi, hi));
                        _mm_store_ss(ptr.add(6 * col_w), _mm_permute_ps(hi, 0b10_10_00_10));
                        _mm_store_ss(ptr.add(7 * col_w), _mm_permute_ps(hi, 0b10_10_10_00));

                        ow += 8;
                    }

                    // ── Right tail (scalar) ──
                    while ow < w_out {
                        let iw = ow * 2 + kw;
                        let dst = col_ch_off + col_k_off + ow * col_w;
                        if iw >= 1 && iw <= w {
                            *col.get_unchecked_mut(dst) = *data.get_unchecked(inp_row_base + iw - 1);
                        } else {
                            *col.get_unchecked_mut(dst) = 0.0;
                        }
                        ow += 1;
                    }
                }
            }
        }
    }
}

// ── Multi-threaded im2col (splits across spatial rows) ──

/// Parallel im2col using rayon. Splits output rows across threads.
///
/// # Safety
/// Same as `im2col_kernel_rect`.
#[cfg(feature = "parallel")]
pub unsafe fn im2col_parallel(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    use rayon::prelude::*;

    let dh = (kh - 1) * dilation + 1;
    let dw = (kw - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dh {
        (h + 2 * padding - dh) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dw {
        (w + 2 * padding - dw) / stride + 1
    } else {
        0
    };
    let col_w = c * kh * kw;
    if h_out == 0 || col_w == 0 || w_out == 0 {
        return;
    }

    let row_elems = w_out * col_w;
    let needed = h_out * row_elems;

    // Only process the first `needed` elements (the buffer may be over-sized
    // from `get_conv_buf!`, which reuses a thread-local Vec).
    let (col_prefix, _) = col.split_at_mut(needed);

    // Parallelize over output rows. Each row writes `row_elems` elements
    // at offset `oh * row_elems`. Split the prefix into row-size chunks.
    col_prefix
        .par_chunks_mut(row_elems)
        .enumerate()
        .for_each(|(oh, row_col)| {
            for ow in 0..w_out {
                for ic in 0..c {
                    for kkh in 0..kh {
                        for kkw in 0..kw {
                            let ih = oh * stride + kkh * dilation;
                            let iw = ow * stride + kkw * dilation;
                            let dst = ow * col_w + ic * kh * kw + kkh * kw + kkw;
                            if ih < h + padding
                                && iw < w + padding
                                && ih >= padding
                                && iw >= padding
                            {
                                let src = (ic * h + (ih - padding)) * w + (iw - padding);
                                row_col[dst] = data[src];
                            } else {
                                row_col[dst] = 0.0;
                            }
                        }
                    }
                }
            }
        });
}

/// Dispatch: calls the SIMD, parallel, or scalar im2col as appropriate.
///
/// Priority: AVX2 (stride=1,pad=1,3×3) → parallel (when feature enabled) → scalar.
///
/// # Safety
/// `data` and `col` must be valid and non-overlapping slices of sufficient length.
#[inline]
pub unsafe fn im2col_dispatch(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) -> (f32, f32) {
    let dh = (kh - 1) * dilation + 1;
    let dw = (kw - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dh {
        (h + 2 * padding - dh) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dw {
        (w + 2 * padding - dw) / stride + 1
    } else {
        0
    };
    let col_elems = h_out * w_out * c * kh * kw;
    let col_len = col.len();
    let col_valid = &mut col[..col_elems.min(col_len)];

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    #[cfg(feature = "parallel")]
    {
        // Use parallel path for large layers even when AVX2 is available:
        // parallel scalar on 8 cores beats single-threaded AVX2 once col_elems
        // is large enough that memory-bandwidth-limited scatter dominates.
        if col_elems > 65536 {
            im2col_parallel(data, c, h, w, kh, kw, stride, padding, dilation, col_valid);
            for &v in col_valid.iter() {
                min = min.min(v);
                max = max.max(v);
            }
            return (min, max);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if kh == 3 && kw == 3 && dilation == 1 && crate::backend::cpu::microkernels::has_avx2() {
        if stride == 1 && padding == 1 {
            im2col_kernel_rect_avx2(data, c, h, w, col_valid);
            for &v in col_valid.iter() {
                min = min.min(v);
                max = max.max(v);
            }
            return (min, max);
        }
        if stride == 2 && padding == 1 {
            im2col_kernel_rect_avx2_stride2(data, c, h, w, col_valid);
            for &v in col_valid.iter() {
                min = min.min(v);
                max = max.max(v);
            }
            return (min, max);
        }
    }
    #[allow(unused_variables)]
    let _ = (kh, kw, stride, padding, dilation);
    let _ = (kh, kw, stride, padding, dilation); // suppress unused warnings

    #[cfg(feature = "parallel")]
    {
        if col_elems > 4096 && col_elems <= 65536 {
            im2col_parallel(data, c, h, w, kh, kw, stride, padding, dilation, col_valid);
            for &v in col_valid.iter() {
                min = min.min(v);
                max = max.max(v);
            }
            return (min, max);
        }
    }

    im2col_kernel_rect(data, c, h, w, kh, kw, stride, padding, dilation, col_valid);
    for &v in col_valid.iter() {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_im2col_matches_scalar() {
        // Test various sizes
        for &(c, h, w) in &[(3, 8, 8), (3, 16, 16), (8, 32, 32), (16, 5, 5)] {
            let stride = 1;
            let padding = 1;
            let dilation = 1;
            let kh = 3;
            let kw = 3;
            let col_w = c * kh * kw;
            let h_out = h;
            let w_out = w;

            let mut input = vec![0.0f32; c * h * w];
            for i in 0..input.len() {
                input[i] = (i as f32 * 0.1).sin();
            }

            let mut col_ref = vec![-1.0f32; h_out * w_out * col_w];
            unsafe {
                im2col_kernel_rect(
                    &input,
                    c,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    &mut col_ref,
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            let has_simd = crate::backend::cpu::microkernels::has_avx2();
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let has_simd = false;

            let mut col_simd = vec![-1.0f32; h_out * w_out * col_w];
            if has_simd {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                unsafe {
                    im2col_kernel_rect_avx2(&input, c, h, w, &mut col_simd);
                }
            } else {
                unsafe {
                    im2col_kernel_rect(
                        &input,
                        c,
                        h,
                        w,
                        kh,
                        kw,
                        stride,
                        padding,
                        dilation,
                        &mut col_simd,
                    );
                }
            }

            let mut max_diff = 0.0f32;
            for i in 0..col_ref.len() {
                let d = (col_ref[i] - col_simd[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            assert!(
                max_diff < 1e-6,
                "c={},h={},w={}: max_diff={}",
                c,
                h,
                w,
                max_diff
            );
        }
    }
}
