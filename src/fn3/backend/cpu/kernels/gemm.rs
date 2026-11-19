//! Row-major matrix product: the measured fast path behind
//! [`super::matmul`], with output tiling across the runtime's thread budget.
//!
//! # Why this exists
//!
//! `matmul.rs` is the portable reference: it addresses both operands through
//! [`super::Locator`] and checks a range before every output element, which is
//! the right trade for a contract but costs `O(M*N*K)` index arithmetic on the
//! one op family that is already `O(M*N*K)` in arithmetic. The FN3-25 (#71)
//! profile put `MatMul` at 28% of a 10-token prefill and the
//! `[1,10,128] x [128,50257]` `lm_head` product alone at 82 ms.
//!
//! This module keeps the contract and removes the overhead: when both operands
//! and the output have row-major matrix axes, the contraction is a goto-free
//! loop over exact windows computed and bounds-checked once.
//!
//! # Numerical contract (unchanged)
//!
//! Every output element is still `((0 + a[0]*b[0]) + a[1]*b[1]) + ...` in
//! ascending `k`, with one `f32` accumulator and no fused multiply-add. That is
//! not an accident of the scalar path being copied:
//!
//! - the scalar kernel blocks `4x4` outputs, and each of the 16 accumulators
//!   receives exactly one `mul` and one `add` per `k`;
//! - the AVX2 kernel vectorises **across output columns**, so lane `j` of an
//!   accumulator holds one output element and still sees one `add` per `k`.
//!   It uses `_mm256_mul_ps` followed by `_mm256_add_ps`, never
//!   `_mm256_fmadd_ps`, because FMA would round once instead of twice.
//!
//! Reassociating a dot product (multiple partial sums, `f64` accumulation or
//! FMA) would change the last bits. `tests/fn3_cpu_perf.rs` pins the result
//! bit-for-bit against an independent ascending-`k` scalar evaluation, so a
//! future kernel that reassociates fails a test instead of quietly shifting
//! every logit.
//!
//! # Threading
//!
//! Only the *output* is partitioned, never the contraction: a task owns a
//! disjoint set of output elements and computes each of them in the same order
//! the serial kernel would. The result is therefore **identical for every
//! thread budget**, not merely within tolerance, which the harness checks by
//! comparing logits checksums across the sweep.
//!
//! Two partitions cover the decoder shapes:
//!
//! - **output rows** (`rows >= 2`, the prefill case): each task gets whole
//!   rows, so both its `lhs` rows and its `out` rows are contiguous;
//! - **output channels** (`rows == 1`, the one-token projection case): a
//!   prompt-row split would leave a single row and never scale a decode step,
//!   so the columns are split instead. Each task owns a disjoint band of
//!   output columns and reads the matching column band of every `rhs` row.
//!
//! A product whose per-task share would fall below
//! [`super::super::parallel::MIN_MACS_PER_TASK`] stays serial no matter what the
//! budget says, because below that share the split costs more than it saves;
//! the measurement is quoted on the constant.
//!
//! # Portability
//!
//! The `4x4` blocked scalar kernel is the portable fallback and is always
//! compiled. The AVX2 kernel is selected at **runtime** by
//! [`super::super::simd`]; on a CPU without AVX2 the scalar kernel runs and
//! produces the identical bytes. A build never fails because a target feature is
//! missing: nothing here is compiled with `-C target-feature`, only
//! `#[target_feature]` on individual functions that are never called unless
//! detection succeeded.

use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;

use super::super::parallel::{self, MIN_MACS_PER_TASK};
use super::kernel_bounds_error;

/// Block width/height for the portable kernel: four accumulators on each axis
/// keep four independent `add` chains in flight, which is what hides the
/// floating-point add latency without reassociating any sum.
const BLOCK: usize = 4;

/// `out = lhs @ rhs` for `rows x inner` by `inner x columns`, all row-major.
///
/// `lhs_base`, `rhs_base` and `out_base` are element offsets of the slab inside
/// each buffer. The three windows are bounds-checked here, once; the kernels
/// below then work on exact-length slices. `threads` is the runtime budget for
/// this instruction: at most that many output tiles are submitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn row_major_slab(
    op: OpId,
    lhs: &[f32],
    lhs_base: usize,
    rhs: &[f32],
    rhs_base: usize,
    out: &mut [f32],
    out_base: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    threads: usize,
) -> Result<(), Fn3Error> {
    let lhs_span = rows
        .checked_mul(inner)
        .ok_or_else(|| span_overflow(op, "lhs", rows, inner))?;
    let rhs_span = inner
        .checked_mul(columns)
        .ok_or_else(|| span_overflow(op, "rhs", inner, columns))?;
    let out_span = rows
        .checked_mul(columns)
        .ok_or_else(|| span_overflow(op, "output", rows, columns))?;

    let lhs_end = lhs_base
        .checked_add(lhs_span)
        .ok_or_else(|| span_overflow(op, "lhs", lhs_base, lhs_span))?;
    let rhs_end = rhs_base
        .checked_add(rhs_span)
        .ok_or_else(|| span_overflow(op, "rhs", rhs_base, rhs_span))?;
    let out_end = out_base
        .checked_add(out_span)
        .ok_or_else(|| span_overflow(op, "output", out_base, out_span))?;

    let lhs_window = lhs
        .get(lhs_base..lhs_end)
        .ok_or_else(|| kernel_bounds_error(op, lhs_end, lhs.len()))?;
    let rhs_window = rhs
        .get(rhs_base..rhs_end)
        .ok_or_else(|| kernel_bounds_error(op, rhs_end, rhs.len()))?;
    let out_capacity = out.len();
    let out_window = out
        .get_mut(out_base..out_end)
        .ok_or_else(|| kernel_bounds_error(op, out_end, out_capacity))?;

    if inner == 0 {
        // The empty product is +0.0, matching the reference `sum([])`.
        out_window.fill(0.0);
        return Ok(());
    }

    let macs = (rows as u64)
        .saturating_mul(inner as u64)
        .saturating_mul(columns as u64);
    // Measured work-size threshold: a task must own at least
    // `MIN_MACS_PER_TASK` multiply-accumulates, so a small product stays serial
    // however large the budget is.
    let by_work = (macs / MIN_MACS_PER_TASK) as usize;
    let threads = threads.min(by_work).max(1);

    if threads == 1 || rows * columns < 2 {
        slab(
            lhs_window, rhs_window, out_window, rows, inner, columns, columns, columns,
        );
        return Ok(());
    }

    if rows >= 2 {
        // Prefill: whole output rows per task, balanced so every task gets the
        // same number of rows (plus one for the first few). Both the lhs and the
        // out window a task receives stay contiguous.
        let tasks = threads.min(rows);
        let row_counts = parallel::balanced_lengths(rows, tasks);
        let lengths: Vec<usize> = row_counts.iter().map(|count| count * columns).collect();
        let mut first_row = 0usize;
        let mut row_starts = Vec::with_capacity(tasks);
        for &count in &row_counts {
            row_starts.push(first_row);
            first_row += count;
        }
        parallel::par_slices_mut(threads, out_window, &lengths, |index, part| {
            let first = row_starts[index];
            let height = row_counts[index];
            slab(
                &lhs_window[first * inner..],
                rhs_window,
                part,
                height,
                inner,
                columns,
                columns,
                columns,
            );
        });
        return Ok(());
    }

    // Decode: one output row, so split the output channels. A prompt-row split
    // would leave a single row and never scale a one-token step; here each task
    // owns a disjoint band of output columns and reads the matching column band
    // of every `rhs` row.
    let tasks = threads.min(columns);
    let lengths = parallel::balanced_lengths(columns, tasks);
    let mut first_column = 0usize;
    let mut column_starts = Vec::with_capacity(tasks);
    for &length in &lengths {
        column_starts.push(first_column);
        first_column += length;
    }
    parallel::par_slices_mut(threads, out_window, &lengths, |index, part| {
        let first = column_starts[index];
        slab(
            lhs_window,
            &rhs_window[first..],
            part,
            1,
            inner,
            part.len(),
            columns,
            columns,
        );
    });
    Ok(())
}

fn span_overflow(op: OpId, what: &'static str, left: usize, right: usize) -> Fn3Error {
    Fn3Error::ShapeOverflow {
        context: "cpu matmul row-major window",
        axis: None,
        detail: format!("op {op}: {what} span {left} x {right} overflows usize"),
    }
}

/// Compute one contiguous sub-slab.
///
/// `rhs_stride` is the element distance between consecutive `rhs` rows and
/// `out_stride` between consecutive output rows; they differ from `columns` only
/// for a column band (decode), where the operands keep the full row width while
/// the task owns `columns` columns of it. `out` starts at the task's first
/// column, so its row `r` begins at `r * out_stride`. `lhs` always has `inner`
/// elements per row, including a row-tiled sub-slab, which starts at that row.
#[allow(clippy::too_many_arguments)]
fn slab(
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

    #[cfg(target_arch = "x86_64")]
    if super::super::simd::avx2_available() {
        // SAFETY: `avx2_available()` returned true on this CPU. The caller
        // bounds-checked the slab windows once, and the smaller slices a task
        // receives are subsets of them: `rows * lhs_row_stride` and
        // `(inner - 1) * rhs_stride + columns` elements are exactly what the
        // kernel's own loops address, and `out` is a disjoint `rows * columns`
        // window (`rows * out_stride` for a band). Every `_mm256_loadu_ps`
        // there reads 32 bytes inside those spans and every store writes 32
        // bytes inside them. `loadu`/`storeu` impose no alignment requirement,
        // the pointers come from live slices that outlive the call, and the
        // tasks own disjoint output elements so no store aliases another task's
        // store; the inputs are separate allocations from the output (the plan
        // rejects read/write slot aliasing).
        unsafe {
            super::super::simd::avx2::kernel(
                lhs, rhs, out, rows, inner, columns, rhs_stride, out_stride,
            );
        }
        return;
    }

    blocked_scalar(lhs, rhs, out, rows, inner, columns, rhs_stride, out_stride);
}

/// Portable `4x4` blocked kernel. Each output element is one `f32`
/// accumulator fed in ascending `k`; blocking only decides *which* outputs are
/// computed together.
pub(crate) fn blocked_scalar(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    rhs_stride: usize,
    out_stride: usize,
) {
    let mut row = 0;
    while row < rows {
        let row_end = (row + BLOCK).min(rows);
        let mut column = 0;
        while column < columns {
            let column_end = (column + BLOCK).min(columns);
            let mut accumulator = [[0.0f32; BLOCK]; BLOCK];
            for k in 0..inner {
                let products = &rhs[k * rhs_stride + column..k * rhs_stride + column_end];
                for (offset, row_slot) in accumulator.iter_mut().enumerate() {
                    if row + offset >= row_end {
                        break;
                    }
                    let left = lhs[(row + offset) * inner + k];
                    for (slot, &right) in row_slot.iter_mut().zip(products) {
                        *slot += left * right;
                    }
                }
            }
            for (offset, row_slot) in accumulator.iter().enumerate() {
                if row + offset >= row_end {
                    break;
                }
                let base = (row + offset) * out_stride + column;
                let width = column_end - column;
                out[base..base + width].copy_from_slice(&row_slot[..width]);
            }
            column = column_end;
        }
        row = row_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every path must agree with an independent ascending-`k` evaluation.
    fn reference(lhs: &[f32], rhs: &[f32], rows: usize, inner: usize, columns: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = 0.0f32;
                for k in 0..inner {
                    accumulator += lhs[row * inner + k] * rhs[k * columns + column];
                }
                out[row * columns + column] = accumulator;
            }
        }
        out
    }

    fn values(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as f32 + seed) * 0.37).sin())
            .collect()
    }

    /// Shapes chosen for every tail the blocking has: both dimensions below,
    /// equal to and above `BLOCK`, a column count that is not a multiple of the
    /// SIMD width, and the two real `lm_head` shapes (one row and ten rows) that
    /// are above the parallel threshold.
    const SHAPES: [(usize, usize, usize); 15] = [
        (1, 1, 1),
        (1, 128, 50257),
        (1, 3, 7),
        (1, 1, 9),
        (1, 9, 1),
        (2, 5, 9),
        (3, 64, 10),
        (4, 4, 4),
        (5, 7, 3),
        (7, 13, 11),
        (10, 128, 384),
        (10, 128, 50257),
        (10, 512, 128),
        (32, 128, 384),
        (33, 65, 129),
    ];

    /// Run through the real entry point for every thread count the host allows,
    /// so the serial path and each tiling are all compared.
    fn check_all_budgets(rows: usize, inner: usize, columns: usize) {
        let lhs = values(rows * inner, 0.5);
        let rhs = values(inner * columns, 1.5);
        let expected = reference(&lhs, &rhs, rows, inner, columns);
        for threads in [1usize, 2, 3, 4, 8, 16] {
            let mut out = vec![0.0f32; rows * columns];
            row_major_slab(
                OpId::MatMul,
                &lhs,
                0,
                &rhs,
                0,
                &mut out,
                0,
                rows,
                inner,
                columns,
                threads,
            )
            .unwrap_or_else(|error| panic!("{rows}x{inner}x{columns} at {threads}: {error}"));
            for (index, (&got, &want)) in out.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{rows}x{inner}x{columns} at {threads} threads: element {index} is {got}, \
                     reference {want}"
                );
            }
        }
    }

    #[test]
    fn every_shape_matches_the_reference_at_every_thread_count() {
        for (rows, inner, columns) in SHAPES {
            check_all_budgets(rows, inner, columns);
        }
    }

    /// The portable fallback is exercised directly, so the "no AVX2 on this
    /// CPU" path is tested on a host that has AVX2 rather than only compiled.
    #[test]
    fn blocked_scalar_is_bit_identical_to_the_reference() {
        for (rows, inner, columns) in SHAPES {
            let lhs = values(rows * inner, 0.5);
            let rhs = values(inner * columns, 1.5);
            let expected = reference(&lhs, &rhs, rows, inner, columns);
            let mut out = vec![0.0f32; rows * columns];
            blocked_scalar(&lhs, &rhs, &mut out, rows, inner, columns, columns, columns);
            for (index, (&got, &want)) in out.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{rows}x{inner}x{columns}: element {index} is {got}, reference {want}"
                );
            }
        }
    }

    #[test]
    fn a_window_that_does_not_fit_is_refused_before_writing() {
        let lhs = vec![1.0f32; 8];
        let rhs = vec![1.0f32; 8];
        let mut out = vec![7.0f32; 8];
        let error = row_major_slab(OpId::MatMul, &lhs, 0, &rhs, 0, &mut out, 0, 4, 4, 4, 4)
            .expect_err("the rhs window is short");
        assert!(error.to_string().contains("outside"), "{error}");
        assert_eq!(out, vec![7.0f32; 8], "a refused call wrote nothing");
    }

    #[test]
    fn an_empty_contraction_is_zero() {
        let rhs: Vec<f32> = Vec::new();
        let mut out = vec![7.0f32; 6];
        row_major_slab(OpId::MatMul, &[], 0, &rhs, 0, &mut out, 0, 2, 0, 3, 4).expect("empty k");
        assert_eq!(out, vec![0.0f32; 6]);
    }

    /// A column band keeps the full row stride, so a band computed in isolation
    /// must equal the same columns of the full product: this is the property the
    /// decode tiling relies on.
    #[test]
    fn a_column_band_matches_the_full_product() {
        let (rows, inner, columns) = (1, 128, 50257);
        let lhs = values(rows * inner, 0.25);
        let rhs = values(inner * columns, 1.75);
        let expected = reference(&lhs, &rhs, rows, inner, columns);
        let mut out = vec![0.0f32; columns];
        row_major_slab(
            OpId::MatMul,
            &lhs,
            0,
            &rhs,
            0,
            &mut out,
            0,
            rows,
            inner,
            columns,
            8,
        )
        .expect("banded product");
        assert_eq!(out.len(), expected.len());
        for (index, (&got, &want)) in out.iter().zip(&expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "column {index}");
        }
    }
}
