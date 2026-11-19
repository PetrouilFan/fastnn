//! The fused W8A16 projection kernel (FN3-23 / #69).
//!
//! # Arithmetic contract
//!
//! For a packed weight `W[N, K]` and an FP32 activation row `x[K]`, the kernel
//! computes exactly the normative form of
//! `docs/quantization/fn3-quant-format-v1.md` §5:
//!
//! ```text
//! for each group g = [start, end):                    // 64 columns, tail allowed
//!     acc = sum over k in [start, end) of x[k] * (code[row, k] - zero_point[row, g])
//!     y[row] += acc * scale[row, g]
//! ```
//!
//! Every addition is `f32`, the group sum proceeds in ascending `k`, the code
//! and zero-point stay integer for the whole dot, and the binary16 scale is
//! widened to `f32` exactly once per group. There is **one** multiply by the
//! scale per group, not one per element, and there is no dequantized weight
//! buffer anywhere: [`PackedView`] borrows the code, scale and zero-point bytes
//! in the layout the format defines.
//!
//! # Accumulation order
//!
//! For one output element the order is `g = 0, 1, ...` and, inside a group,
//! `k = start .. end`. A block of up to [`ROW_BLOCK`] output rows is computed
//! together so one activation value feeds several rows, but each output
//! element keeps its own accumulator and the order above. The blocking is
//! therefore not a numerical choice; it changes which rows share the load of
//! `x[k]`, not any sum.
//!
//! # Thread-budget seam (FN3-42 / #71)
//!
//! The kernel takes an explicit [`WorkHint`] and visits output rows through
//! `hint.chunks(n_rows)`. It **never spawns a thread and owns no pool**. The
//! chunks are contiguous, disjoint, ascend, cover every row once and are at
//! most `hint.threads()` long. Because the per-element order is independent of
//! the chunk, running the chunks on different threads is bit-identical to
//! running them in order; [`WorkHint::chunks`] is the only thing #71 has to
//! consume. `src/fn3/backend/cpu/parallel.rs` does not exist on this branch and
//! is not edited; when #71 lands, its `ThreadBudget` supplies the `threads` and
//! `min_rows` values and replaces the sequential `for chunk` loop with a pool
//! dispatch over the same [`RowChunks`] iterator.
//!
//! # Bounds
//!
//! Every shape, dtype, layout and length is validated before the first output
//! write. Row slices are then formed once and indexed by `k`, which is bounded
//! by the slice length, so the inner loop contains no caller-controlled
//! indexing and the module needs no `unsafe`.

use std::ops::Range;

use half::f16;

use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::quant::{QuantTensor, GROUP_SIZE};
use crate::fn3::tensor::{TensorMut, TensorView};

use super::super::kernels::{kernel_bounds_error, require_dtype};
use super::super::CPU_BACKEND_NAME;

/// Output rows computed together so one `x[k]` load feeds several rows.
const ROW_BLOCK: usize = 4;

/// A borrowed, validated view of one packed `fn3-quant-format-v1` weight.
///
/// The three byte slices are exactly what the container carries: signed INT8
/// codes (`n_rows * row_stride`), little-endian binary16 scales and
/// little-endian int16 zero-points (`n_rows * groups_per_row * 2` each). The
/// view performs no copy and no conversion; it is the packed representation.
#[derive(Debug, Clone, Copy)]
pub struct PackedView<'a> {
    pub(crate) n_rows: usize,
    pub(crate) k: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) row_stride: usize,
    pub(crate) codes: &'a [u8],
    pub(crate) scales: &'a [u8],
    pub(crate) zeros: &'a [u8],
}

impl<'a> PackedView<'a> {
    /// Validate borrowed packed bytes.
    ///
    /// `row_stride` is the distance between code rows; the format fixes it to
    /// `align_up(k, 16)` in a container, while the in-memory packer uses
    /// `k` (no padding). Both are accepted here because padding is a storage
    /// detail, not arithmetic.
    pub fn new(
        n_rows: usize,
        k: usize,
        row_stride: usize,
        codes: &'a [u8],
        scales: &'a [u8],
        zeros: &'a [u8],
    ) -> Result<PackedView<'a>, Fn3Error> {
        if n_rows == 0 || k == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!("a packed weight needs n_rows > 0 and k > 0, found {n_rows} x {k}"),
            });
        }
        if row_stride < k {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!("row stride {row_stride} is shorter than k = {k}"),
            });
        }
        let groups = groups_per_row(k);
        let code_bytes = n_rows
            .checked_mul(row_stride)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "W8A16 packed view codes",
                axis: Some(0),
                detail: format!("{n_rows} rows of {row_stride} bytes"),
            })?;
        let meta_bytes = n_rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "W8A16 packed view metadata",
                axis: Some(1),
                detail: format!("{n_rows} rows of {groups} groups"),
            })?;
        if codes.len() != code_bytes {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!(
                    "codes are {} bytes, {n_rows} rows x {row_stride} stride need {code_bytes}",
                    codes.len()
                ),
            });
        }
        if scales.len() != meta_bytes {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!(
                    "scales are {} bytes, {n_rows} rows x {groups} groups need {meta_bytes}",
                    scales.len()
                ),
            });
        }
        if zeros.len() != meta_bytes {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!(
                    "zero-points are {} bytes, {n_rows} rows x {groups} groups need {meta_bytes}",
                    zeros.len()
                ),
            });
        }
        Ok(PackedView {
            n_rows,
            k,
            groups_per_row: groups,
            row_stride,
            codes,
            scales,
            zeros,
        })
    }

    /// Borrow a parsed [`QuantTensor`] without copying or dequantizing it.
    ///
    /// The reader has already enforced `qbits == 8`, `group_size == 64` and the
    /// canonical group directory; this view re-checks the group geometry it
    /// depends on so a hand-built `QuantTensor` cannot desynchronize it.
    pub fn from_quant_tensor(tensor: &'a QuantTensor<'a>) -> Result<PackedView<'a>, Fn3Error> {
        if tensor.qbits() != 8 {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(OpId::MatMul),
                dtype: DType::I32,
                context: "W8A16 packed view: the format fixes qbits to 8",
            });
        }
        if usize::from(tensor.group_size()) != GROUP_SIZE {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!(
                    "group size {} is not the format's {GROUP_SIZE}",
                    tensor.group_size()
                ),
            });
        }
        let view = PackedView::new(
            tensor.n_rows(),
            tensor.k(),
            tensor.row_stride_bytes(),
            tensor.codes(),
            tensor.scales(),
            tensor.zero_points(),
        )?;
        if tensor.groups_per_row() != view.groups_per_row {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packed view",
                detail: format!(
                    "the descriptor declares {} groups, k = {} needs {}",
                    tensor.groups_per_row(),
                    tensor.k(),
                    view.groups_per_row
                ),
            });
        }
        for group in 0..view.groups_per_row {
            let (start, end) = tensor.group_range(group);
            let canonical = view.group_range(group);
            if start != canonical.start || end != canonical.end {
                return Err(Fn3Error::InvalidShape {
                    context: "W8A16 packed view",
                    detail: format!(
                        "group {group} is {start}..{end} in the directory, the canonical range \
                         is {}..{}",
                        canonical.start, canonical.end
                    ),
                });
            }
        }
        Ok(view)
    }

    /// Output rows (N).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Reduction length (K).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Quantization groups per row, `ceil(k / 64)`.
    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }

    /// Code bytes between consecutive rows.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Raw signed codes (`u8` two's complement).
    pub fn codes(&self) -> &'a [u8] {
        self.codes
    }

    /// Raw little-endian binary16 scales.
    pub fn scales(&self) -> &'a [u8] {
        self.scales
    }

    /// Raw little-endian int16 zero-points.
    pub fn zero_points(&self) -> &'a [u8] {
        self.zeros
    }

    /// Code bytes in the packed representation.
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Scale + zero-point bytes in the packed representation.
    pub fn metadata_bytes(&self) -> usize {
        self.scales.len() + self.zeros.len()
    }

    /// Total bytes a resident copy of this weight needs.
    pub fn packed_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() + self.zeros.len()
    }

    /// `(start, end)` of a canonical group along K.
    pub fn group_range(&self, group: usize) -> Range<usize> {
        let start = group * GROUP_SIZE;
        start..(start + GROUP_SIZE).min(self.k)
    }
}

/// Consecutive output rows visited together.
///
/// The sequence is deterministic: same `rows` and [`WorkHint`] produce the same
/// ranges in the same order on every run and every target.
#[derive(Debug, Clone)]
pub struct RowChunks {
    rows: usize,
    size: usize,
    next: usize,
}

impl Iterator for RowChunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        if self.next >= self.rows {
            return None;
        }
        let end = (self.next + self.size).min(self.rows);
        let range = self.next..end;
        self.next = end;
        Some(range)
    }
}

impl ExactSizeIterator for RowChunks {
    fn len(&self) -> usize {
        if self.size == 0 || self.next >= self.rows {
            0
        } else {
            (self.rows - self.next).div_ceil(self.size)
        }
    }
}

/// Explicit work-size hint for the fused kernel.
///
/// This is the seam FN3-42 (#71) consumes. It carries no pool and starts no
/// thread: [`WorkHint::chunks`] only decides how the serial loop groups rows.
/// A `ThreadBudget` with `threads` workers and a minimum useful work unit maps
/// directly onto [`WorkHint::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkHint {
    threads: usize,
    min_rows: usize,
}

impl Default for WorkHint {
    fn default() -> WorkHint {
        WorkHint::serial()
    }
}

impl WorkHint {
    /// One serial chunk; the reference configuration.
    pub const fn serial() -> WorkHint {
        WorkHint {
            threads: 1,
            min_rows: 1,
        }
    }

    /// A budget of `threads` workers with at least `min_rows` rows per piece.
    ///
    /// Neither value changes the arithmetic; see the module docs. Rejects zero
    /// so a caller cannot silently disable the loop.
    pub fn new(threads: usize, min_rows: usize) -> Result<WorkHint, Fn3Error> {
        if threads == 0 || min_rows == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 work hint",
                detail: format!(
                    "threads and min_rows must be positive, found threads = {threads}, \
                     min_rows = {min_rows}"
                ),
            });
        }
        Ok(WorkHint { threads, min_rows })
    }

    /// Workers the caller intends to use.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Smallest useful chunk, in output rows.
    pub fn min_rows(&self) -> usize {
        self.min_rows
    }

    /// Rows per chunk for a weight with `rows` output rows.
    ///
    /// At most `threads` chunks are produced. A weight with fewer rows than
    /// `min_rows` is one chunk (the only case a chunk may be smaller than
    /// `min_rows`).
    pub fn chunk_size(&self, rows: usize) -> usize {
        if rows == 0 {
            return 0;
        }
        let workers = self.threads.min(rows);
        rows.div_ceil(workers).max(self.min_rows).min(rows)
    }

    /// The row ranges the kernel will visit, in order.
    pub fn chunks(&self, rows: usize) -> RowChunks {
        RowChunks {
            rows,
            size: self.chunk_size(rows),
            next: 0,
        }
    }
}

/// Fused `y = x @ Wᵀ` for a packed W8A16 weight.
///
/// `lhs` is `[..., M, K]` FP32 and contiguous; `out` is `[..., M, N]` FP32 and
/// contiguous; `weight` is packed `[N, K]`. The last two axes contract, and
/// the remaining (batch) axes must match exactly. Tails are supported: the
/// final group of a row may hold `1..=64` columns and `K < 64` is one group.
///
/// # Rejections
///
/// A layout outside the named subset — a non-contiguous operand, rank below 2,
/// a K that disagrees with the weight, or a wrong output geometry — is reported
/// as [`Fn3Error::UnsupportedOp`] or the matching shape error before the first
/// output write. [`super::backend::QuantCpuBackend`] counts such a rejection as
/// an explicit fallback and delegates to the FP32 `MatMul`.
pub fn matmul_quantized(
    op: OpId,
    lhs: &TensorView<'_>,
    weight: &PackedView<'_>,
    out: &mut TensorMut<'_>,
    hint: WorkHint,
) -> Result<(), Fn3Error> {
    require_dtype(op, lhs.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    lhs.meta().require_device(Some(op), DeviceId::Cpu)?;
    out.meta().require_device(Some(op), DeviceId::Cpu)?;

    let lhs_rank = lhs.shape().rank();
    if lhs_rank < 2 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: CPU_BACKEND_NAME,
            reason: format!(
                "the fused W8A16 kernel serves MatMul operands of rank >= 2; found rank \
                 {lhs_rank} ({}), which stays on the FP32 path",
                lhs.shape()
            ),
        });
    }
    if out.shape().rank() != lhs_rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("rank {} like the lhs {}", lhs_rank, lhs.shape()),
            actual: format!("rank {} ({})", out.shape().rank(), out.shape()),
        });
    }
    for axis in 0..lhs_rank - 2 {
        if lhs.shape().dims()[axis] != out.shape().dims()[axis] {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("batch axis {axis} of {} preserved", lhs.shape()),
                actual: out.shape().to_string(),
            });
        }
    }
    let m_rows = lhs.shape().dims()[lhs_rank - 2];
    let k = lhs.shape().dims()[lhs_rank - 1];
    if k != weight.k() as u64 {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("lhs contraction of {} (packed weight k)", weight.k()),
            actual: format!("lhs {} contracts {k}", lhs.shape()),
        });
    }
    if out.shape().dims()[lhs_rank - 2] != m_rows
        || out.shape().dims()[lhs_rank - 1] != weight.n_rows() as u64
    {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!(
                "... x {m_rows} x {}, the lhs rows by the packed weight rows",
                weight.n_rows()
            ),
            actual: out.shape().to_string(),
        });
    }
    if !lhs.is_contiguous() {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: CPU_BACKEND_NAME,
            reason: format!(
                "the fused W8A16 kernel serves contiguous activations; lhs {} is strided, \
                 so this invocation stays on the FP32 path",
                lhs.shape()
            ),
        });
    }
    if !out.is_contiguous() {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: CPU_BACKEND_NAME,
            reason: format!(
                "the fused W8A16 kernel writes a contiguous output; {} is strided",
                out.shape()
            ),
        });
    }

    let m_rows = usize::try_from(m_rows).map_err(|_| Fn3Error::ShapeOverflow {
        context: "W8A16 kernel rows",
        axis: Some(lhs_rank - 2),
        detail: "row count does not fit in usize".to_string(),
    })?;
    let k = weight.k();
    let n_rows = weight.n_rows();
    let mut batch = 1usize;
    for (axis, &extent) in lhs.shape().dims()[..lhs_rank - 2].iter().enumerate() {
        let extent = usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
            context: "W8A16 kernel batch",
            axis: Some(axis),
            detail: "batch extent does not fit in usize".to_string(),
        })?;
        batch = batch
            .checked_mul(extent)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "W8A16 kernel batch",
                axis: Some(axis),
                detail: "batch product exceeds usize::MAX".to_string(),
            })?;
    }
    if m_rows == 0 || n_rows == 0 || batch == 0 {
        return Ok(());
    }

    let lhs_data = lhs.as_f32_slice()?;
    let out_data = out.as_mut_slice::<f32>()?;
    let matrix_in = m_rows
        .checked_mul(k)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "W8A16 kernel lhs matrix",
            axis: None,
            detail: format!("{m_rows} x {k}"),
        })?;
    let matrix_out = m_rows
        .checked_mul(n_rows)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "W8A16 kernel output matrix",
            axis: None,
            detail: format!("{m_rows} x {n_rows}"),
        })?;

    for batch_index in 0..batch {
        let lhs_start =
            batch_index
                .checked_mul(matrix_in)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "W8A16 kernel lhs matrix base",
                    axis: None,
                    detail: "batch offset exceeds usize::MAX".to_string(),
                })?;
        let out_start =
            batch_index
                .checked_mul(matrix_out)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "W8A16 kernel output matrix base",
                    axis: None,
                    detail: "batch offset exceeds usize::MAX".to_string(),
                })?;
        let lhs_matrix = lhs_data
            .get(lhs_start..lhs_start + matrix_in)
            .ok_or_else(|| kernel_bounds_error(op, lhs_start + matrix_in, lhs_data.len()))?;
        let out_matrix = {
            let len = out_data.len();
            out_data
                .get_mut(out_start..out_start + matrix_out)
                .ok_or_else(|| kernel_bounds_error(op, out_start + matrix_out, len))?
        };
        project_matrix(op, lhs_matrix, weight, m_rows, out_matrix, hint)?;
    }
    Ok(())
}

/// One contiguous `[M, K] @ [N, K]ᵀ` matrix pair.
fn project_matrix(
    op: OpId,
    lhs: &[f32],
    weight: &PackedView<'_>,
    m_rows: usize,
    out: &mut [f32],
    hint: WorkHint,
) -> Result<(), Fn3Error> {
    let k = weight.k();
    let n_rows = weight.n_rows();
    let groups = weight.groups_per_row();
    let codes = weight.codes();
    let scales = weight.scales();
    let zeros = weight.zero_points();

    for chunk in hint.chunks(n_rows) {
        let mut block_start = chunk.start;
        while block_start < chunk.end {
            let width = (chunk.end - block_start).min(ROW_BLOCK);
            let block_end = block_start + width;

            // One code row per output row in the block. Each range is exactly
            // k long, so `code_rows[j][i]` with i < k cannot be out of bounds.
            let mut code_rows: [&[u8]; ROW_BLOCK] = [&[]; ROW_BLOCK];
            for (offset, row) in (block_start..block_end).enumerate() {
                let from = row * weight.row_stride();
                code_rows[offset] = codes
                    .get(from..from + k)
                    .ok_or_else(|| kernel_bounds_error(op, from + k, codes.len()))?;
            }

            for row_index in 0..m_rows {
                let x = lhs
                    .get(row_index * k..(row_index + 1) * k)
                    .ok_or_else(|| kernel_bounds_error(op, row_index * k + k, lhs.len()))?;
                let mut y = [0.0f32; ROW_BLOCK];

                for group in 0..groups {
                    let range = weight.group_range(group);
                    let mut zero = [0.0f32; ROW_BLOCK];
                    let mut scale = [0.0f32; ROW_BLOCK];
                    for offset in 0..width {
                        let row = block_start + offset;
                        let at = row * groups + group;
                        zero[offset] =
                            i16::from_le_bytes([zeros[at * 2], zeros[at * 2 + 1]]) as f32;
                        scale[offset] = f16::from_bits(u16::from_le_bytes([
                            scales[at * 2],
                            scales[at * 2 + 1],
                        ]))
                        .to_f32();
                    }
                    let mut acc = [0.0f32; ROW_BLOCK];
                    for column in range {
                        let value = x[column];
                        for offset in 0..width {
                            // `u8 as i8` is the format's two's-complement code;
                            // `(code - zero_point)` stays exact in f32 because
                            // both are small integers.
                            let code = code_rows[offset][column] as i8;
                            acc[offset] += value * (f32::from(code) - zero[offset]);
                        }
                    }
                    // Exactly one f32 multiply by the group's binary16 scale.
                    for offset in 0..width {
                        y[offset] += acc[offset] * scale[offset];
                    }
                }

                for offset in 0..width {
                    let index = row_index * n_rows + block_start + offset;
                    let len = out.len();
                    *out.get_mut(index)
                        .ok_or_else(|| kernel_bounds_error(op, index, len))? = y[offset];
                }
            }

            block_start = block_end;
        }
    }
    Ok(())
}

/// Groups per row for K: `ceil(k / 64)`.
pub(crate) fn groups_per_row(k: usize) -> usize {
    k.div_ceil(GROUP_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_row_exactly_once() {
        for threads in 1..=8 {
            for rows in 0..40 {
                let hint = WorkHint::new(threads, 2).unwrap();
                let chunks: Vec<Range<usize>> = hint.chunks(rows).collect();
                assert_eq!(
                    chunks.iter().map(|range| range.len()).sum::<usize>(),
                    rows,
                    "threads {threads}, rows {rows}"
                );
                assert!(
                    chunks.len() <= threads.max(1),
                    "threads {threads}, rows {rows}: {} chunks",
                    chunks.len()
                );
                for pair in chunks.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start, "chunks are contiguous");
                }
                for (index, chunk) in chunks.iter().enumerate() {
                    assert!(!chunk.is_empty(), "no empty chunks");
                    if rows >= 2 && index + 1 < chunks.len() {
                        assert!(chunk.len() >= 2, "min_rows honoured: {chunk:?}");
                    }
                }
                if let Some(last) = chunks.last() {
                    assert_eq!(last.end, rows);
                }
            }
        }
    }

    #[test]
    fn a_serial_hint_is_one_chunk() {
        let hint = WorkHint::serial();
        let chunks: Vec<Range<usize>> = hint.chunks(17).collect();
        assert_eq!(chunks, vec![0..17]);
        assert!(WorkHint::new(0, 1).is_err());
        assert!(WorkHint::new(1, 0).is_err());
    }
}
