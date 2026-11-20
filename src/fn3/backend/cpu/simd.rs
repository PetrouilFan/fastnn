//! Runtime CPU feature detection and the AVX2 kernels it selects.
//!
//! # Why detection, not a compile-time feature
//!
//! `fastnn` ships one x86_64 binary. Building the hot kernels with
//! `-C target-feature=+avx2` would make that binary crash with an illegal
//! instruction on a CPU without AVX2 — a build that "works" but cannot run.
//! Instead every wide kernel lives in a `#[target_feature(enable = "avx2")]`
//! function that is only called after [`avx2_available`] said yes, and the
//! portable scalar path is always compiled. A build therefore never fails for
//! lack of a target feature, and a run on such a CPU is slower but identical in
//! its bytes.
//!
//! Detection is `std::is_x86_feature_detected!`, which caches its answer in an
//! atomic after the first call, so the per-call cost of asking is one relaxed
//! load. On a non-x86_64 target [`avx2_available`] is a constant `false` and no
//! intrinsic is compiled at all.
//!
//! # What is vectorised, and what that means for results
//!
//! Only the output-column axis. Each 8-lane accumulator holds eight different
//! output elements, and lane `j` is fed `+ a[k]*b[k,j]` once per `k` in
//! ascending order — the same operations in the same order as the scalar
//! contract. `mul` and `add` are separate instructions (no FMA), so the
//! rounding sequence is unchanged. There is no horizontal reduction anywhere,
//! which is why the vector path is bit-identical rather than merely close.

/// `true` when this CPU can execute AVX2 instructions.
#[inline]
pub(crate) fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    //! AVX2 matrix product. See the module header for the numerical contract.
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    /// Number of output rows kept in flight; four independent accumulator
    /// chains hide the add latency without touching any single sum's order.
    const ROWS: usize = 4;
    /// AVX2 float lanes.
    const LANES: usize = 8;

    /// `out = lhs @ rhs` over one contiguous sub-slab.
    ///
    /// `rhs_stride` is the element distance between consecutive `rhs` rows and
    /// `out_stride` between consecutive output rows. They differ from `columns`
    /// only for a column band (the decode tiling), where a task owns `columns`
    /// columns of a wider row.
    ///
    /// # Safety
    ///
    /// The caller must have proved the window invariants this function asserts
    /// in debug builds:
    ///
    /// - `lhs` covers `rows * inner` elements,
    /// - `rhs` covers `(inner - 1) * rhs_stride + columns`,
    /// - `out` covers `(rows - 1) * out_stride + columns`.
    ///
    /// With those, every pointer this function forms is in bounds:
    /// `rhs.as_ptr().add(k * rhs_stride + column)` reads 8 floats with
    /// `k < inner` and `column + 8 <= columns`, so the last byte read is at most
    /// `((inner - 1) * rhs_stride + columns - 1) * 4`; the same argument holds
    /// for the `out` stores with `out_stride`, and `lhs` is only ever read
    /// through safe indexed access. The slices come from distinct allocations
    /// (the plan rejects read/write slot aliasing) and, when tiled, from
    /// `par_chunks_mut`/row splits that give each task disjoint output elements,
    /// so no store aliases another task's store or either input. No reference
    /// into `out` is created while a load is outstanding, and nothing here
    /// retains a pointer after returning.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn kernel(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        rows: usize,
        inner: usize,
        columns: usize,
        rhs_stride: usize,
        out_stride: usize,
    ) {
        debug_assert!(lhs.len() >= rows * inner);
        debug_assert!(rhs.len() >= (inner - 1) * rhs_stride + columns);
        debug_assert!(out.len() >= (rows - 1) * out_stride + columns);

        let mut row = 0;
        while row + ROWS <= rows {
            let mut column = 0;
            while column + LANES <= columns {
                // SAFETY: the four rows exist (`row + ROWS <= rows`) and the
                // eight columns exist (`column + LANES <= columns`); the window
                // invariants above make every access inside its slice.
                unsafe {
                    block_4x8(lhs, rhs, out, row, column, inner, rhs_stride, out_stride);
                }
                column += LANES;
            }
            if column < columns {
                // SAFETY: rows and columns are inside the same windows.
                unsafe {
                    rows_4_columns_tail(
                        lhs, rhs, out, row, column, inner, columns, rhs_stride, out_stride,
                    );
                }
            }
            row += ROWS;
        }
        while row < rows {
            let mut column = 0;
            while column + LANES <= columns {
                // SAFETY: one row exists and eight columns exist.
                unsafe {
                    block_1x8(lhs, rhs, out, row, column, inner, rhs_stride, out_stride);
                }
                column += LANES;
            }
            if column < columns {
                // SAFETY: one row, and the remaining columns are in range.
                unsafe {
                    rows_1_columns_tail(
                        lhs, rhs, out, row, column, inner, columns, rhs_stride, out_stride,
                    );
                }
            }
            row += 1;
        }
    }

    /// Four rows by eight columns: four accumulators, one per row, each lane a
    /// different output column.
    ///
    /// # Safety
    ///
    /// `row + 4 <= rows`, `column + 8 <= columns`, and the window invariants of
    /// [`kernel`].
    #[target_feature(enable = "avx2")]
    unsafe fn block_4x8(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        row: usize,
        column: usize,
        inner: usize,
        rhs_stride: usize,
        out_stride: usize,
    ) {
        let mut accumulator = [_mm256_setzero_ps(); ROWS];
        let rhs_base = rhs.as_ptr();
        for k in 0..inner {
            // SAFETY: `column + 8 <= columns` and `k < inner`, so this load is
            // inside the rhs window. `loadu` needs no alignment.
            let right = unsafe { _mm256_loadu_ps(rhs_base.add(k * rhs_stride + column)) };
            for (offset, sums) in accumulator.iter_mut().enumerate() {
                // SAFETY: `row + offset < row + 4 <= rows` and `k < inner`.
                let left =
                    unsafe { _mm256_set1_ps(*lhs.get_unchecked((row + offset) * inner + k)) };
                *sums = _mm256_add_ps(*sums, _mm256_mul_ps(left, right));
            }
        }
        for (offset, sums) in accumulator.iter().enumerate() {
            // SAFETY: `row + offset < rows`; the 8-float store ends at or before
            // the output window end.
            unsafe {
                _mm256_storeu_ps(
                    out.as_mut_ptr().add((row + offset) * out_stride + column),
                    *sums,
                );
            }
        }
    }

    /// One row by eight columns.
    ///
    /// # Safety
    ///
    /// `row < rows`, `column + 8 <= columns`, and the window invariants of
    /// [`kernel`].
    #[target_feature(enable = "avx2")]
    unsafe fn block_1x8(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        row: usize,
        column: usize,
        inner: usize,
        rhs_stride: usize,
        out_stride: usize,
    ) {
        let mut sums = _mm256_setzero_ps();
        let rhs_base = rhs.as_ptr();
        for k in 0..inner {
            // SAFETY: as in `block_4x8`, with a single row.
            unsafe {
                let right = _mm256_loadu_ps(rhs_base.add(k * rhs_stride + column));
                let left = _mm256_set1_ps(*lhs.get_unchecked(row * inner + k));
                sums = _mm256_add_ps(sums, _mm256_mul_ps(left, right));
            }
        }
        // SAFETY: the store window is inside the output span.
        unsafe {
            _mm256_storeu_ps(out.as_mut_ptr().add(row * out_stride + column), sums);
        }
    }

    /// Scalar tail for the columns left after the `8`-wide blocks of a
    /// four-row group.
    ///
    /// # Safety
    ///
    /// `row + 4 <= rows`, `column < columns`, and the window invariants.
    #[target_feature(enable = "avx2")]
    unsafe fn rows_4_columns_tail(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        row: usize,
        column: usize,
        inner: usize,
        columns: usize,
        rhs_stride: usize,
        out_stride: usize,
    ) {
        for target in column..columns {
            for offset in 0..ROWS {
                let mut accumulator = 0.0f32;
                for k in 0..inner {
                    accumulator += lhs[(row + offset) * inner + k] * rhs[k * rhs_stride + target];
                }
                out[(row + offset) * out_stride + target] = accumulator;
            }
        }
    }

    /// Scalar tail for the columns left after the `8`-wide block of a
    /// single-row group.
    ///
    /// # Safety
    ///
    /// `row < rows`, `column < columns`, and the window invariants.
    #[target_feature(enable = "avx2")]
    unsafe fn rows_1_columns_tail(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        row: usize,
        column: usize,
        inner: usize,
        columns: usize,
        rhs_stride: usize,
        out_stride: usize,
    ) {
        for target in column..columns {
            let mut accumulator = 0.0f32;
            for k in 0..inner {
                accumulator += lhs[row * inner + k] * rhs[k * rhs_stride + target];
            }
            out[row * out_stride + target] = accumulator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_total_on_this_host() {
        // Either answer is valid; the point is that asking cannot panic and
        // that the portable path stays compiled beside it.
        let _ = avx2_available();
    }
}
