//! Data-movement and shape kernels: `Gather`, `Transpose`, `Reshape`,
//! `Flatten`, `Squeeze`, `Unsqueeze`, `Concat`, `Slice`, `Split`, `Expand`,
//! `Shape`, `ConstantOfShape` and `Range`.
//!
//! # Why these are byte-level
//!
//! Every kernel in this file moves elements without interpreting them. They
//! therefore share one implementation for `F32`, `I64`, `I32` and `Bool` and
//! cannot convert a dtype by accident: `Cast` in
//! [`super::elementwise`] is the only place a dtype changes.
//!
//! # Determinism
//!
//! Data movement has no arithmetic, so the result is bit-exact by construction.
//! The *order* is nevertheless documented, because a future tiled or threaded
//! implementation must keep the same observable result: outputs are written in
//! ascending row-major order of their own coordinates, and `Concat`/`Split`
//! visit inputs/outputs in index order.
//!
//! One kernel deviates from that write order and not from its result:
//! [`transpose`] takes a cache-tiled path when the permutation swaps only the
//! last two axes, both views are contiguous and elements are 4 or 8 bytes
//! (FN3-25 / #71). It writes a `32 x 32` tile at a time, and each output
//! element is still written exactly once with the same bytes, so the tensor is
//! identical; only the order of the stores changes, which no caller can
//! observe.
//!
//! # Shapes and axes
//!
//! Attributes come from [`OpParams`]; shapes and sizes that ONNX carries as
//! *tensor inputs* (opset 13+: `Reshape`'s shape, `Slice`'s starts/ends/axes/
//! steps, `Split`'s sizes, `Expand`'s shape, `Squeeze`/`Unsqueeze`'s axes) are
//! taken from inputs of dtype `I64`, exactly as the importer sees them. Where
//! `OpParams` has no field for an ONNX attribute (for example
//! `ConstantOfShape`'s `value`), the value is lifted into a scalar input; the
//! deviation is documented on the kernel and reported to #54/#56.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::{checked_usize, Shape};
use crate::inference::tensor::{TensorMut, TensorView};

use super::{
    at, at_mut, checked_product, copy_element, decompose, kernel_bounds_error, normalize_axis,
    out_window_bytes, read_i64_scalar, read_i64_vector, require_dtype, require_output_shape,
    single_axis, view_window_bytes, Locator,
};

/// Copy `input`'s elements, in logical row-major order, into `out`.
///
/// The shapes may differ as long as the element counts match (`Reshape`,
/// `Flatten`, `Squeeze`, `Unsqueeze`, identity `Cast`). No element is
/// interpreted, so the copy is exact for every dtype.
pub(crate) fn copy_row_major(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if input.dtype() != out.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: input.dtype(),
            actual: out.dtype(),
        });
    }
    let elements = input.shape().numel()?;
    if out.shape().numel()? != elements {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{elements} elements to match {}", input.shape()),
            actual: format!("{} has {} elements", out.shape(), out.shape().numel()?),
        });
    }
    if elements == 0 {
        return Ok(());
    }
    let element_bytes = input.dtype().size_bytes();
    if input.is_contiguous() && out.is_contiguous() {
        let source = view_window_bytes(op, input)?;
        let destination = out_window_bytes(op, out)?;
        if source.len() == destination.len() {
            destination.copy_from_slice(source);
            return Ok(());
        }
    }

    let out_locator = Locator::new(op, out.meta(), out.shape())?;
    let in_dims = input.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    let source = input.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let mut in_coordinates = vec![0u64; in_dims.len()];
    let mut out_coordinates = vec![0u64; out_dims.len()];
    for linear in 0..elements {
        decompose(linear, &in_dims, &mut in_coordinates);
        decompose(linear, &out_dims, &mut out_coordinates);
        let from = input
            .meta()
            .checked_index(&in_coordinates)
            .map_err(|error| error.with_op(op))?;
        let to = out_locator.index(op, &out_coordinates)?;
        copy_element(op, source, from, destination, to, element_bytes)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gather (embedding lookup)
// ---------------------------------------------------------------------------

/// ONNX `Gather`: `out = data[..., index[..], ...]` with the indexed slices
/// along `axis` (default 0).
///
/// An embedding is `Gather` with `axis = 0` over an `F32` table and `I64`
/// indices; negative indices count from the end (`-1` is the last row), which is
/// ONNX behaviour and is used by the fixture's duplicate/last-row case.
pub(crate) fn gather(
    op: OpId,
    data: &TensorView<'_>,
    indices: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let axis = single_axis(op, params, data.shape().rank(), 0, "gather axis")?;
    let index_values = index_values(op, indices)?;

    let data_dims = data.shape().dims();
    let mut expected_dims = data_dims[..axis].to_vec();
    expected_dims.extend_from_slice(indices.shape().dims());
    expected_dims.extend_from_slice(&data_dims[axis + 1..]);
    let expected = Shape::from_dims(expected_dims);
    require_output_shape(op, &expected, out)?;

    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let dimension = data_dims[axis];
    let dimension_i64 = i64::try_from(dimension).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu gather axis extent",
        axis: Some(axis),
        detail: format!("extent {dimension} does not fit in i64"),
    })?;

    let element_bytes = data.dtype().size_bytes();
    let rank = expected.rank();
    let indices_rank = indices.shape().rank();
    let source = data.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let mut out_coordinates = vec![0u64; rank];
    let mut data_coordinates = vec![0u64; data.shape().rank()];
    for linear in 0..elements {
        decompose(linear, expected.dims(), &mut out_coordinates);
        // Which index applies to this output element: the indices tensor's own
        // row-major coordinate, which is what packs the repeated index block
        // into the output.
        let mut index_linear = 0u64;
        for position in 0..indices_rank {
            let coordinate = out_coordinates[axis + position];
            let extent = indices.shape().dims()[position];
            index_linear = index_linear * extent + coordinate;
        }
        let raw = at(
            op,
            &index_values,
            usize::try_from(index_linear).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu gather index position",
                axis: None,
                detail: "index position does not fit in usize".to_string(),
            })?,
        )?;
        let resolved = if raw < 0 {
            raw.saturating_add(dimension_i64)
        } else {
            raw
        };
        if resolved < 0 || resolved >= dimension_i64 {
            return Err(Fn3Error::OutOfBounds {
                what: "gather index",
                index: raw as u64,
                len: dimension,
            });
        }
        data_coordinates[..axis].copy_from_slice(&out_coordinates[..axis]);
        data_coordinates[axis] = resolved as u64;
        data_coordinates[axis + 1..].copy_from_slice(&out_coordinates[axis + indices_rank..]);
        let from = data
            .meta()
            .checked_index(&data_coordinates)
            .map_err(|error| error.with_op(op))?;
        copy_element(
            op,
            source,
            from,
            destination,
            linear as usize,
            element_bytes,
        )?;
    }
    Ok(())
}

/// Read an integer tensor (any rank, any strides) as `i64` values.
pub(crate) fn index_values(op: OpId, view: &TensorView<'_>) -> Result<Vec<i64>, Fn3Error> {
    let elements = view.shape().numel()?;
    let count = usize::try_from(elements).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu kernel integer tensor",
        axis: None,
        detail: "element count does not fit in usize".to_string(),
    })?;
    if view.is_contiguous() {
        return match view.dtype() {
            DType::I64 => Ok(view.as_i64_slice()?.to_vec()),
            DType::I32 => Ok(view.as_i32_slice()?.iter().map(|&v| i64::from(v)).collect()),
            other => Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: DType::I64,
                actual: other,
            }),
        };
    }
    let mut values = Vec::with_capacity(count);
    let dims = view.shape().dims().to_vec();
    let mut coordinates = vec![0u64; dims.len()];
    match view.dtype() {
        DType::I64 => {
            let data = view.storage_slice::<i64>()?;
            for linear in 0..elements {
                decompose(linear, &dims, &mut coordinates);
                let index = view
                    .meta()
                    .checked_index(&coordinates)
                    .map_err(|error| error.with_op(op))?;
                values.push(at(op, data, index)?);
            }
        }
        DType::I32 => {
            let data = view.storage_slice::<i32>()?;
            for linear in 0..elements {
                decompose(linear, &dims, &mut coordinates);
                let index = view
                    .meta()
                    .checked_index(&coordinates)
                    .map_err(|error| error.with_op(op))?;
                values.push(i64::from(at(op, data, index)?));
            }
        }
        other => {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: DType::I64,
                actual: other,
            })
        }
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

/// Axis permutation. An empty `axes` reverses the axes, which is the ONNX
/// default.
pub(crate) fn transpose(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let rank = input.shape().rank();
    let reverse: Vec<usize> = (0..rank).rev().collect();
    let permutation: Vec<usize> = if params.axes.is_empty() {
        reverse
    } else {
        let mut resolved = Vec::with_capacity(params.axes.len());
        for &axis in &params.axes {
            resolved.push(normalize_axis(op, axis, rank, "transpose permutation")?);
        }
        resolved
    };
    Shape::validate_permutation(&permutation, rank).map_err(|error| error.with_op(op))?;
    let expected = input.shape().permuted(&permutation)?;
    require_output_shape(op, &expected, out)?;

    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let element_bytes = input.dtype().size_bytes();
    if transpose_tiled(op, input, out, &permutation, elements, element_bytes)? {
        return Ok(());
    }
    let source = input.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let dims = expected.dims().to_vec();
    let mut out_coordinates = vec![0u64; rank];
    let mut in_coordinates = vec![0u64; rank];
    for linear in 0..elements {
        decompose(linear, &dims, &mut out_coordinates);
        for (out_axis, &in_axis) in permutation.iter().enumerate() {
            in_coordinates[in_axis] = out_coordinates[out_axis];
        }
        let from = input
            .meta()
            .checked_index(&in_coordinates)
            .map_err(|error| error.with_op(op))?;
        copy_element(
            op,
            source,
            from,
            destination,
            linear as usize,
            element_bytes,
        )?;
    }
    Ok(())
}

/// Elements per tile edge in the cache-blocked transpose paths.
///
/// 32 `f32` cells is 128 bytes: two cache lines, which is the largest tile whose
/// column fits one L1 write-combining window while the source rows are still
/// prefetched as ascending 128-byte runs.
const TRANSPOSE_TILE: usize = 32;

/// A permutation this module can move as a matrix transpose of contiguous
/// cells.
///
/// Every variant is an identity outer dimension and one matrix transpose whose
/// source and destination columns are contiguous runs of `cell_bytes`. That
/// covers the two transposes in the pinned YOLOv8n graph and the three shapes
/// the v3 contract names, and leaves every other permutation to
/// [`transpose`]'s element loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TilePlan {
    /// `outer` independent `[rows][cols]` slabs, both contiguous:
    /// `dst[o, c, r] = src[o, r, c]`.
    Slabs {
        outer: usize,
        rows: usize,
        cols: usize,
        cell_bytes: usize,
    },
    /// `[batch, rows, fixed, cols] -> [batch, cols, fixed, rows]` with the
    /// middle `fixed` axis held constant: for each `(batch, fixed)`, one
    /// strided `[rows][cols] -> [cols][rows]` transpose. The pinned graph's
    /// DFL `[0, 3, 2, 1]` transpose has this shape.
    FixedOuter {
        batch: usize,
        fixed: usize,
        rows: usize,
        cols: usize,
        cell_bytes: usize,
    },
}

impl TilePlan {
    fn cell_bytes(self) -> usize {
        match self {
            TilePlan::Slabs { cell_bytes, .. } | TilePlan::FixedOuter { cell_bytes, .. } => {
                cell_bytes
            }
        }
    }
}

/// Try the cache-blocked paths; `Ok(false)` means "not one of them".
///
/// The fast paths require both views contiguous (the arena always is) and an
/// element width that permits aligned typed slices; anything else keeps the
/// generic element loop. Every fast path moves each cell unchanged, so the
/// result is byte-exact: only the order of the stores differs, which no caller
/// can observe.
fn transpose_tiled(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    permutation: &[usize],
    elements: u64,
    element_bytes: usize,
) -> Result<bool, Fn3Error> {
    if !input.is_contiguous() || !out.is_contiguous() {
        return Ok(false);
    }
    let dims = checked_usizes(op, input.shape().dims())?;
    let Some(plan) = tile_plan(permutation, &dims, element_bytes)? else {
        return Ok(false);
    };
    let source = view_window_bytes(op, input)?;
    let expected_bytes =
        elements
            .checked_mul(element_bytes as u64)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu transpose byte length",
                axis: None,
                detail: "element count times element size overflows u64".to_string(),
            })?;
    let destination = out_window_bytes(op, out)?;
    if source.len() as u64 != expected_bytes || destination.len() as u64 != expected_bytes {
        return Err(kernel_bounds_error(
            op,
            expected_bytes as usize,
            source.len(),
        ));
    }
    let destination_len = destination.len();
    match plan.cell_bytes() {
        4 => apply_plan(
            &plan,
            bytemuck::try_cast_slice::<u8, u32>(source).map_err(|error| Fn3Error::StorageCast {
                element: "u32 transpose cell",
                byte_len: source.len(),
                detail: format!("{error:?}"),
            })?,
            bytemuck::try_cast_slice_mut::<u8, u32>(destination).map_err(|error| {
                Fn3Error::StorageCast {
                    element: "u32 transpose cell",
                    byte_len: destination_len,
                    detail: format!("{error:?}"),
                }
            })?,
        ),
        8 => apply_plan(
            &plan,
            bytemuck::try_cast_slice::<u8, u64>(source).map_err(|error| Fn3Error::StorageCast {
                element: "u64 transpose cell",
                byte_len: source.len(),
                detail: format!("{error:?}"),
            })?,
            bytemuck::try_cast_slice_mut::<u8, u64>(destination).map_err(|error| {
                Fn3Error::StorageCast {
                    element: "u64 transpose cell",
                    byte_len: destination_len,
                    detail: format!("{error:?}"),
                }
            })?,
        ),
        _ => apply_plan_bytes(&plan, source, destination),
    }
    Ok(true)
}

/// Resolve a permutation to a [`TilePlan`], or `None`.
fn tile_plan(
    permutation: &[usize],
    dims: &[usize],
    element_bytes: usize,
) -> Result<Option<TilePlan>, Fn3Error> {
    let rank = permutation.len();
    let product = |values: &[usize], what: &'static str| -> Result<usize, Fn3Error> {
        let mut product = 1usize;
        for &value in values {
            product = product
                .checked_mul(value)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: what,
                    axis: None,
                    detail: "extents exceed usize::MAX elements".to_string(),
                })?;
        }
        Ok(product)
    };
    // Identity prefix, then a swap of the last two axes. This is the plain
    // matrix transpose for any rank, including the tied-embedding table.
    if rank >= 2
        && permutation[..rank - 2]
            .iter()
            .enumerate()
            .all(|(axis, &moved)| moved == axis)
        && permutation[rank - 2] == rank - 1
        && permutation[rank - 1] == rank - 2
    {
        return Ok(Some(TilePlan::Slabs {
            outer: product(&dims[..rank - 2], "cpu transpose outer")?,
            rows: dims[rank - 2],
            cols: dims[rank - 1],
            cell_bytes: element_bytes,
        }));
    }
    if rank != 4 {
        return Ok(None);
    }
    let plan = match permutation {
        // NCHW -> NHWC: input viewed as `[N, C, H*W]`.
        [0, 2, 3, 1] => TilePlan::Slabs {
            outer: dims[0],
            rows: dims[1],
            cols: product(&dims[2..], "cpu transpose NHWC columns")?,
            cell_bytes: element_bytes,
        },
        // NCHW -> NWCH: input viewed as `[N, C*H, W]`.
        [0, 3, 1, 2] => TilePlan::Slabs {
            outer: dims[0],
            rows: product(&dims[1..3], "cpu transpose NWCH rows")?,
            cols: dims[3],
            cell_bytes: element_bytes,
        },
        // NCHW -> NHCW: a `[C][H]` transpose whose cells are the W runs.
        [0, 2, 1, 3] => TilePlan::Slabs {
            outer: dims[0],
            rows: dims[1],
            cols: dims[2],
            cell_bytes: dims[3].checked_mul(element_bytes).ok_or_else(|| {
                Fn3Error::ShapeOverflow {
                    context: "cpu transpose NHCW cell",
                    axis: Some(3),
                    detail: "one W run exceeds usize::MAX bytes".to_string(),
                }
            })?,
        },
        // N A C B -> N B C A: the DFL transpose. `rows` and `cols` are A and
        // B; `fixed` is C.
        [0, 3, 2, 1] => TilePlan::FixedOuter {
            batch: dims[0],
            fixed: dims[2],
            rows: dims[1],
            cols: dims[3],
            cell_bytes: element_bytes,
        },
        _ => return Ok(None),
    };
    Ok(Some(plan))
}

fn checked_usizes(op: OpId, dims: &[u64]) -> Result<Vec<usize>, Fn3Error> {
    dims.iter()
        .enumerate()
        .map(|(axis, &extent)| {
            usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu transpose extent",
                axis: Some(axis),
                detail: format!("extent {extent} does not fit in usize (op {op})"),
            })
        })
        .collect()
}

/// Apply a plan to typed cells, 32 x 32 cells per tile.
///
/// Each tile is read as `height` contiguous source runs into a stack tile, then
/// written as `width` contiguous destination runs of `height` cells. The
/// destination run is the `copy_from_slice`; the source run is the other.
fn apply_plan<T: Copy + Default>(plan: &TilePlan, source: &[T], destination: &mut [T]) {
    match *plan {
        TilePlan::Slabs {
            outer, rows, cols, ..
        } => {
            let slab = rows * cols;
            debug_assert_eq!(source.len(), outer * slab);
            debug_assert_eq!(destination.len(), outer * slab);
            for slab_index in 0..outer {
                let from = &source[slab_index * slab..(slab_index + 1) * slab];
                let to = &mut destination[slab_index * slab..(slab_index + 1) * slab];
                tiled_transpose(from, to, rows, cols, cols, rows);
            }
        }
        TilePlan::FixedOuter {
            batch,
            fixed,
            rows,
            cols,
            ..
        } => {
            debug_assert_eq!(source.len(), batch * rows * fixed * cols);
            debug_assert_eq!(destination.len(), batch * cols * fixed * rows);
            for batch_index in 0..batch {
                for fixed_index in 0..fixed {
                    let from = (batch_index * rows * fixed + fixed_index) * cols;
                    let to = (batch_index * cols * fixed + fixed_index) * rows;
                    tiled_transpose(
                        &source[from..],
                        &mut destination[to..],
                        rows,
                        cols,
                        fixed * cols,
                        fixed * rows,
                    );
                }
            }
        }
    }
}

/// The byte-cell form of [`apply_plan`], for a cell wider than 8 bytes (the
/// `[0, 2, 1, 3]` W run). The cell itself is the innermost `copy_from_slice`.
fn apply_plan_bytes(plan: &TilePlan, source: &[u8], destination: &mut [u8]) {
    let cell = plan.cell_bytes();
    let mut copy_matrix = |from: usize,
                           to: usize,
                           rows: usize,
                           cols: usize,
                           source_stride: usize,
                           destination_stride: usize| {
        for row_start in (0..rows).step_by(TRANSPOSE_TILE) {
            let row_end = (row_start + TRANSPOSE_TILE).min(rows);
            for col_start in (0..cols).step_by(TRANSPOSE_TILE) {
                let col_end = (col_start + TRANSPOSE_TILE).min(cols);
                for col in col_start..col_end {
                    let destination_base = (to + (col * destination_stride + row_start)) * cell;
                    for row in row_start..row_end {
                        let source_base = (from + (row * source_stride + col)) * cell;
                        destination[destination_base + (row - row_start) * cell
                            ..destination_base + (row - row_start + 1) * cell]
                            .copy_from_slice(&source[source_base..source_base + cell]);
                    }
                }
            }
        }
    };
    match *plan {
        TilePlan::Slabs {
            outer, rows, cols, ..
        } => {
            let slab = rows * cols;
            for slab_index in 0..outer {
                copy_matrix(slab_index * slab, slab_index * slab, rows, cols, cols, rows);
            }
        }
        TilePlan::FixedOuter {
            batch,
            fixed,
            rows,
            cols,
            ..
        } => {
            for batch_index in 0..batch {
                for fixed_index in 0..fixed {
                    copy_matrix(
                        (batch_index * rows * fixed + fixed_index) * cols,
                        (batch_index * cols * fixed + fixed_index) * rows,
                        rows,
                        cols,
                        fixed * cols,
                        fixed * rows,
                    );
                }
            }
        }
    }
}

/// `dst[c, r] = src[r, c]` for a `[rows][cols]` matrix, 32 x 32 cells a tile.
///
/// `source_row_stride` and `destination_row_stride` are in cells and are at
/// least `cols` and `rows`; the columns themselves are contiguous.
fn tiled_transpose<T: Copy + Default>(
    source: &[T],
    destination: &mut [T],
    rows: usize,
    cols: usize,
    source_row_stride: usize,
    destination_row_stride: usize,
) {
    debug_assert!(source_row_stride >= cols);
    debug_assert!(destination_row_stride >= rows);
    let mut tile = [T::default(); TRANSPOSE_TILE * TRANSPOSE_TILE];
    let mut run = [T::default(); TRANSPOSE_TILE];
    for row_start in (0..rows).step_by(TRANSPOSE_TILE) {
        let row_end = (row_start + TRANSPOSE_TILE).min(rows);
        let height = row_end - row_start;
        for col_start in (0..cols).step_by(TRANSPOSE_TILE) {
            let col_end = (col_start + TRANSPOSE_TILE).min(cols);
            let width = col_end - col_start;
            for (tile_row, row) in (row_start..row_end).enumerate() {
                let from = row * source_row_stride + col_start;
                tile[tile_row * TRANSPOSE_TILE..tile_row * TRANSPOSE_TILE + width]
                    .copy_from_slice(&source[from..from + width]);
            }
            for (tile_col, col) in (col_start..col_end).enumerate() {
                for (tile_row, slot) in run.iter_mut().enumerate().take(height) {
                    *slot = tile[tile_row * TRANSPOSE_TILE + tile_col];
                }
                let to = col * destination_row_stride + row_start;
                destination[to..to + height].copy_from_slice(&run[..height]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reshape / Flatten / Squeeze / Unsqueeze
// ---------------------------------------------------------------------------
/// Resolve an ONNX `Reshape` target (`allowzero = 0`):
///
/// - a positive value is that extent;
/// - `0` copies the input extent at the same position;
/// - `-1` infers the remaining extent once.
pub(crate) fn resolve_reshape_shape(
    op: OpId,
    input: &Shape,
    target: &[i64],
) -> Result<Shape, Fn3Error> {
    let mut dims = Vec::with_capacity(target.len());
    let mut inferred: Option<usize> = None;
    let mut known: u64 = 1;
    for (position, &value) in target.iter().enumerate() {
        let extent =
            match value {
                -1 => {
                    if inferred.is_some() {
                        return Err(Fn3Error::InvalidShape {
                            context: "cpu reshape target",
                            detail: format!(
                                "op {op}: {target:?} infers more than one dimension with -1"
                            ),
                        });
                    }
                    inferred = Some(position);
                    dims.push(0);
                    continue;
                }
                0 => {
                    let extent = input.dims().get(position).copied().ok_or_else(|| {
                        Fn3Error::InvalidShape {
                            context: "cpu reshape target",
                            detail: format!(
                                "op {op}: {target:?} copies dimension {position} with 0, but the \
                             input {} has rank {}",
                                input,
                                input.rank()
                            ),
                        }
                    })?;
                    extent
                }
                positive => u64::try_from(positive).map_err(|_| Fn3Error::InvalidShape {
                    context: "cpu reshape target",
                    detail: format!("op {op}: {target:?} has a negative dimension other than -1"),
                })?,
            };
        dims.push(extent);
        known = known
            .checked_mul(extent)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu reshape target",
                axis: Some(position),
                detail: "known extents exceed u64::MAX".to_string(),
            })?;
    }
    if let Some(position) = inferred {
        let elements = input.numel()?;
        if known == 0 {
            if elements != 0 {
                return Err(Fn3Error::InvalidShape {
                    context: "cpu reshape target",
                    detail: format!(
                        "op {op}: {target:?} leaves the product of known extents at 0, so the \
                         -1 dimension cannot be inferred from {elements} elements"
                    ),
                });
            }
            dims[position] = 0;
        } else {
            if elements % known != 0 {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("{elements} elements divisible by {known}"),
                    actual: format!("{target:?}"),
                });
            }
            dims[position] = elements / known;
        }
    }
    Ok(Shape::from_dims(dims))
}

/// `Reshape(data, shape)`; `shape` is an `I64` rank-1 input, as in ONNX opset 14.
pub(crate) fn reshape(
    op: OpId,
    data: &TensorView<'_>,
    shape_input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    let target = read_i64_vector(op, shape_input, "reshape shape input")?;
    let expected = resolve_reshape_shape(op, data.shape(), &target)?;
    require_output_shape(op, &expected, out)?;
    if expected.numel()? != data.shape().numel()? {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{} elements to match {}", expected.numel()?, data.shape()),
            actual: format!("{expected} has {} elements", expected.numel()?),
        });
    }
    copy_row_major(op, data, out)
}

/// `Flatten` to `[prod(dims[..axis]), prod(dims[axis..])]`.
pub(crate) fn flatten(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let rank = input.shape().rank();
    let axis = if params.axes.is_empty() {
        1.min(rank)
    } else {
        match params.axes.as_slice() {
            [axis] => {
                let resolved = if *axis < 0 {
                    (*axis + rank as i64 + 1).max(0)
                } else {
                    *axis
                };
                usize::try_from(resolved)
                    .ok()
                    .filter(|&value| value <= rank)
                    .ok_or_else(|| Fn3Error::InvalidShape {
                        context: "flatten axis",
                        detail: format!("op {op}: axis {axis} is outside 0..={rank}"),
                    })?
            }
            axes => {
                return Err(Fn3Error::InvalidShape {
                    context: "flatten axis",
                    detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
                })
            }
        }
    };
    let leading = checked_product(&input.shape().dims()[..axis], "cpu flatten")?;
    let trailing = checked_product(&input.shape().dims()[axis..], "cpu flatten")?;
    let expected = Shape::from_dims([leading, trailing]);
    require_output_shape(op, &expected, out)?;
    copy_row_major(op, input, out)
}

/// Resolve the axes of a `Squeeze`/`Unsqueeze`, from `axes` or an input tensor.
pub(crate) fn axes_from_params_or_input(
    op: OpId,
    params: &OpParams,
    axes_input: Option<&TensorView<'_>>,
) -> Result<Vec<i64>, Fn3Error> {
    match (params.axes.as_slice(), axes_input) {
        ([], None) => Ok(Vec::new()),
        (axes, None) => Ok(axes.to_vec()),
        ([], Some(view)) => read_i64_vector(op, view, "axes input"),
        (axes, Some(_)) => Err(Fn3Error::InvalidShape {
            context: "cpu axes attribute",
            detail: format!(
                "op {op} was given both axes={axes:?} and an axes input tensor; \
                 supply exactly one"
            ),
        }),
    }
}

/// `Squeeze`: remove extent-1 axes. With neither `axes` nor an axes input,
/// every extent-1 axis is removed (ONNX opset 11 behaviour).
pub(crate) fn squeeze(
    op: OpId,
    input: &TensorView<'_>,
    axes_input: Option<&TensorView<'_>>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let rank = input.shape().rank();
    let axes = axes_from_params_or_input(op, params, axes_input)?;
    let mut removed = vec![false; rank];
    if axes.is_empty() {
        for axis in 0..rank {
            removed[axis] = input.shape().dims()[axis] == 1;
        }
    } else {
        for &axis in &axes {
            let axis = normalize_axis(op, axis, rank, "squeeze axis")?;
            if removed[axis] {
                return Err(Fn3Error::InvalidShape {
                    context: "squeeze axis",
                    detail: format!("op {op}: axis {axis} is listed twice"),
                });
            }
            if input.shape().dims()[axis] != 1 {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("extent 1 on axis {axis} of {}", input.shape()),
                    actual: format!("{}", input.shape().dims()[axis]),
                });
            }
            removed[axis] = true;
        }
    }
    let dims: Vec<u64> = input
        .shape()
        .dims()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !removed[*axis])
        .map(|(_, &extent)| extent)
        .collect();
    let expected = Shape::from_dims(dims);
    require_output_shape(op, &expected, out)?;
    copy_row_major(op, input, out)
}

/// `Unsqueeze`: insert extent-1 axes. `axes` name positions in the *output*.
pub(crate) fn unsqueeze(
    op: OpId,
    input: &TensorView<'_>,
    axes_input: Option<&TensorView<'_>>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let axes = axes_from_params_or_input(op, params, axes_input)?;
    if axes.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "unsqueeze axes",
            detail: format!("op {op} needs at least one axis to insert"),
        });
    }
    let out_rank = input.shape().rank() + axes.len();
    let mut inserted = vec![false; out_rank];
    for &axis in &axes {
        let axis = normalize_axis(op, axis, out_rank, "unsqueeze axis")?;
        if inserted[axis] {
            return Err(Fn3Error::InvalidShape {
                context: "unsqueeze axis",
                detail: format!("op {op}: axis {axis} is listed twice"),
            });
        }
        inserted[axis] = true;
    }
    let mut dims = Vec::with_capacity(out_rank);
    let mut source = input.shape().dims().iter();
    for axis in 0..out_rank {
        if inserted[axis] {
            dims.push(1);
        } else {
            let extent = source
                .next()
                .copied()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "unsqueeze axes",
                    detail: format!(
                        "op {op}: more inserted axes than the output rank allows for {}",
                        input.shape()
                    ),
                })?;
            dims.push(extent);
        }
    }
    let expected = Shape::from_dims(dims);
    require_output_shape(op, &expected, out)?;
    copy_row_major(op, input, out)
}

// ---------------------------------------------------------------------------
// Concat / Split
// ---------------------------------------------------------------------------

/// Concatenate along `axes[0]` (default 0). Inputs are visited in index order,
/// so the result is exact and order-independent of any scheduling.
pub(crate) fn concat(
    op: OpId,
    inputs: &[TensorView<'_>],
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let first = inputs.first().ok_or_else(|| Fn3Error::InvalidShape {
        context: "concat inputs",
        detail: format!("op {op} needs at least one input"),
    })?;
    let rank = first.shape().rank();
    let axis = single_axis(op, params, rank, 0, "concat axis")?;
    let mut expected_dims = first.shape().dims().to_vec();
    expected_dims[axis] = 0;

    for view in inputs {
        if view.shape().rank() != rank {
            return Err(Fn3Error::RankMismatch {
                op: Some(op),
                expected: rank,
                actual: view.shape().rank(),
            });
        }
        if view.dtype() != first.dtype() {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: first.dtype(),
                actual: view.dtype(),
            });
        }
        for (position, (&extent, &reference)) in view
            .shape()
            .dims()
            .iter()
            .zip(first.shape().dims())
            .enumerate()
        {
            if position != axis && extent != reference {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("every axis but {axis} to match {}", first.shape()),
                    actual: format!("{} differs on axis {position}", view.shape()),
                });
            }
        }
        expected_dims[axis] = expected_dims[axis]
            .checked_add(view.shape().dims()[axis])
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu concat extent",
                axis: Some(axis),
                detail: "concatenated extent exceeds u64::MAX".to_string(),
            })?;
    }
    let expected = Shape::from_dims(expected_dims);
    require_output_shape(op, &expected, out)?;

    let element_bytes = first.dtype().size_bytes();
    let rank = expected.rank();
    // The output is `outer` blocks of `axis_total x inner`; an input block is
    // the same shape with its own axis extent. Both are contiguous when the
    // views are, so a contiguous input moves one memcpy per `outer` block
    // instead of one `copy_element` per element.
    let outer = checked_usize(
        checked_product(&expected.dims()[..axis], "cpu concat outer")?,
        "cpu concat outer",
        "outer block count",
    )?;
    let inner = checked_usize(
        checked_product(&expected.dims()[axis + 1..], "cpu concat inner")?,
        "cpu concat inner",
        "inner block extent",
    )?;
    let axis_total = checked_usize(expected.dims()[axis], "cpu concat axis", "axis extent")?;
    let mut out_coordinates = vec![0u64; rank];
    let mut input_coordinates = vec![0u64; rank];
    let mut axis_offset = 0u64;
    for view in inputs {
        let input_axis = checked_usize(
            view.shape().dims()[axis],
            "cpu concat input",
            "input axis extent",
        )?;
        if view.is_contiguous() && out.is_contiguous() {
            let source = view_window_bytes(op, view)?;
            let destination = out_window_bytes(op, out)?;
            let block_bytes = input_axis
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu concat block",
                    axis: Some(axis),
                    detail: "one contiguous input block exceeds usize::MAX bytes".to_string(),
                })?;
            let destination_stride = axis_total
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu concat block",
                    axis: Some(axis),
                    detail: "one output block row exceeds usize::MAX bytes".to_string(),
                })?;
            let axis_start = checked_usize(axis_offset, "cpu concat input", "axis offset")?
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu concat block",
                    axis: Some(axis),
                    detail: "the input's axis offset exceeds usize::MAX bytes".to_string(),
                })?;
            debug_assert_eq!(source.len(), outer * block_bytes);
            debug_assert_eq!(destination.len(), outer * destination_stride);
            for block in 0..outer {
                let from = block * block_bytes;
                let to = block * destination_stride + axis_start;
                destination[to..to + block_bytes]
                    .copy_from_slice(&source[from..from + block_bytes]);
            }
            axis_offset += view.shape().dims()[axis];
            continue;
        }

        let elements = view.shape().numel()?;
        let source = view.storage_bytes();
        let out_locator = Locator::new(op, out.meta(), &expected)?;
        let destination = out.storage_slice_mut::<u8>()?;
        for linear in 0..elements {
            decompose(linear, view.shape().dims(), &mut input_coordinates);
            out_coordinates.copy_from_slice(&input_coordinates);
            out_coordinates[axis] =
                out_coordinates[axis]
                    .checked_add(axis_offset)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "cpu concat offset",
                        axis: Some(axis),
                        detail: "output coordinate exceeds u64::MAX".to_string(),
                    })?;
            let from = view
                .meta()
                .checked_index(&input_coordinates)
                .map_err(|error| error.with_op(op))?;
            let to = out_locator.index(op, &out_coordinates)?;
            copy_element(op, source, from, destination, to, element_bytes)?;
        }
        axis_offset += view.shape().dims()[axis];
    }
    Ok(())
}

/// `Split`: slice along `axes[0]`.
///
/// Split sizes come from an `I64` input when supplied; otherwise the axis is
/// divided equally between the outputs (and must divide exactly).
pub(crate) fn split(
    op: OpId,
    input: &TensorView<'_>,
    sizes_input: Option<&TensorView<'_>>,
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let count = outputs.len();
    if count == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "split outputs",
            detail: format!("op {op} needs at least one output"),
        });
    }
    let rank = input.shape().rank();
    let axis = single_axis(op, params, rank, 0, "split axis")?;
    let dimension = input.shape().dims()[axis];
    let sizes: Vec<u64> = match sizes_input {
        Some(view) => {
            let values = read_i64_vector(op, view, "split sizes input")?;
            if values.len() != count {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("{count} split sizes to match the outputs"),
                    actual: format!("{} sizes", values.len()),
                });
            }
            let mut sizes = Vec::with_capacity(count);
            for value in values {
                sizes.push(u64::try_from(value).map_err(|_| Fn3Error::InvalidShape {
                    context: "split sizes",
                    detail: format!("op {op}: split size {value} is negative"),
                })?);
            }
            sizes
        }
        None => {
            let dimension_usize =
                usize::try_from(dimension).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu split extent",
                    axis: Some(axis),
                    detail: "extent does not fit in usize".to_string(),
                })?;
            if dimension_usize % count != 0 {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("axis {axis} of {} divisible by {count}", input.shape()),
                    actual: format!("extent {dimension}"),
                });
            }
            vec![dimension / count as u64; count]
        }
    };
    let mut total = 0u64;
    for (position, &size) in sizes.iter().enumerate() {
        total = total
            .checked_add(size)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu split sizes",
                axis: Some(position),
                detail: "split sizes exceed u64::MAX".to_string(),
            })?;
    }
    if total != dimension {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("split sizes summing to {dimension} (axis {axis})"),
            actual: format!("{sizes:?} sum to {total}"),
        });
    }

    let element_bytes = input.dtype().size_bytes();
    let mut axis_offset = 0u64;
    for (index, out) in outputs.iter_mut().enumerate() {
        if out.dtype() != input.dtype() {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: input.dtype(),
                actual: out.dtype(),
            });
        }
        let mut expected_dims = input.shape().dims().to_vec();
        expected_dims[axis] = sizes[index];
        let expected = Shape::from_dims(expected_dims);
        require_output_shape(op, &expected, out)?;
    }

    // The arena's views are contiguous, so a split is `outer` contiguous runs
    // of `size * inner` elements per output: one memcpy per run instead of one
    // checked `copy_element` per element. For YOLOv8n's channel splits that is
    // one memcpy per output (`outer` is the batch extent, 1); on the [1, 32,
    // 80, 80] split the per-element path measured 22.9 ms of a 76 ms forward,
    // which is why this path exists. A strided view keeps the element loop.
    if input.is_contiguous() && outputs.iter().all(|out| out.is_contiguous()) {
        return split_contiguous(op, input, outputs, &sizes, axis, element_bytes);
    }

    for (index, out) in outputs.iter_mut().enumerate() {
        let mut expected_dims = input.shape().dims().to_vec();
        expected_dims[axis] = sizes[index];
        let elements = Shape::from(&expected_dims[..]).numel()?;
        if elements == 0 {
            axis_offset += sizes[index];
            continue;
        }
        let source = input.storage_bytes();
        let out_locator = Locator::new(op, out.meta(), out.shape())?;
        let destination = out.storage_slice_mut::<u8>()?;
        let mut coordinates = vec![0u64; rank];
        let mut input_coordinates = vec![0u64; rank];
        for linear in 0..elements {
            decompose(linear, &expected_dims, &mut coordinates);
            input_coordinates.copy_from_slice(&coordinates);
            input_coordinates[axis] =
                coordinates[axis].checked_add(axis_offset).ok_or_else(|| {
                    Fn3Error::ShapeOverflow {
                        context: "cpu split offset",
                        axis: Some(axis),
                        detail: "input coordinate exceeds u64::MAX".to_string(),
                    }
                })?;
            let from = input
                .meta()
                .checked_index(&input_coordinates)
                .map_err(|error| error.with_op(op))?;
            let to = out_locator.index(op, &coordinates)?;
            copy_element(op, source, from, destination, to, element_bytes)?;
        }
        axis_offset += sizes[index];
    }
    Ok(())
}

/// The contiguous `Split`: one `copy_from_slice` per `(output, outer block)`.
///
/// The input is `outer x axis_extent x inner`; output `i` is
/// `outer x sizes[i] x inner`, and its run for one outer block starts at input
/// element `(block * axis_extent + offset) * inner`, so both sides of every copy
/// are contiguous. `outer`, `inner`, the source block stride and each run
/// length are computed once, before the loop; the loop itself does no division
/// and no per-element index arithmetic.
fn split_contiguous(
    op: OpId,
    input: &TensorView<'_>,
    outputs: &mut [TensorMut<'_>],
    sizes: &[u64],
    axis: usize,
    element_bytes: usize,
) -> Result<(), Fn3Error> {
    let dims = input.shape().dims();
    let inner = checked_usize(
        checked_product(&dims[axis + 1..], "cpu split inner")?,
        "cpu split inner",
        "inner block extent",
    )?;
    let outer = checked_usize(
        checked_product(&dims[..axis], "cpu split outer")?,
        "cpu split outer",
        "outer block count",
    )?;
    let axis_extent = checked_usize(dims[axis], "cpu split axis", "axis extent")?;
    let inner_bytes = inner
        .checked_mul(element_bytes)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "cpu split run",
            axis: Some(axis),
            detail: "one inner row exceeds usize::MAX bytes".to_string(),
        })?;
    let source_slab_bytes =
        axis_extent
            .checked_mul(inner_bytes)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu split source block",
                axis: Some(axis),
                detail: "one input outer block exceeds usize::MAX bytes".to_string(),
            })?;
    let source = view_window_bytes(op, input)?;
    let mut axis_offset = 0usize;
    for (index, out) in outputs.iter_mut().enumerate() {
        let size = checked_usize(sizes[index], "cpu split size", "output axis extent")?;
        let run_bytes = size
            .checked_mul(inner_bytes)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu split output block",
                axis: Some(axis),
                detail: "one output outer block exceeds usize::MAX bytes".to_string(),
            })?;
        let destination = out_window_bytes(op, out)?;
        // `run_bytes * outer` is the output's window, which the view validated;
        // the source block plus this output's offset is inside the input window
        // because the split sizes sum to the axis extent.
        debug_assert_eq!(Some(destination.len()), run_bytes.checked_mul(outer));
        if run_bytes != 0 {
            let axis_offset_bytes = axis_offset * inner_bytes;
            for block in 0..outer {
                let from = block * source_slab_bytes + axis_offset_bytes;
                let to = block * run_bytes;
                destination[to..to + run_bytes].copy_from_slice(&source[from..from + run_bytes]);
            }
        }
        axis_offset += size;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

/// Python/ONNX slice bounds for one axis: `start`, `step`, `count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AxisSlice {
    pub(crate) start: i64,
    pub(crate) step: i64,
    pub(crate) count: u64,
}

/// Resolve `start`, `end` and `step` for one axis with Python's clamping rules.
///
/// `i64::MAX`/`i64::MIN` behave as "unbounded", which is what ONNX exporters
/// emit for open ranges (`/transformer/Slice` in the pinned graph uses
/// `i64::MAX`). Negative steps are supported: the kernel copies in reverse
/// order, so no negative *stride* is ever created (the view contract forbids
/// those).
pub(crate) fn resolve_axis_slice(
    op: OpId,
    dimension: u64,
    start: i64,
    end: i64,
    step: i64,
) -> Result<AxisSlice, Fn3Error> {
    if step == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "slice step",
            detail: format!("op {op}: a slice step of 0 is undefined"),
        });
    }
    let dim = i128::from(dimension);
    let step_wide = i128::from(step);
    let (start, count) = if step > 0 {
        let mut resolved_start = i128::from(start);
        let mut resolved_end = i128::from(end);
        if resolved_start < 0 {
            resolved_start += dim;
        }
        if resolved_end < 0 {
            resolved_end += dim;
        }
        let resolved_start = resolved_start.clamp(0, dim);
        let resolved_end = resolved_end.clamp(0, dim);
        let count = if resolved_end > resolved_start {
            (resolved_end - resolved_start + step_wide - 1) / step_wide
        } else {
            0
        };
        (resolved_start, count)
    } else {
        let mut resolved_start = i128::from(start);
        let mut resolved_end = i128::from(end);
        if resolved_start < 0 {
            resolved_start += dim;
        }
        if resolved_end < 0 {
            resolved_end += dim;
        }
        let upper = (dim - 1).max(0);
        let resolved_start = resolved_start.clamp(0, upper);
        let resolved_end = resolved_end.clamp(-1, upper);
        let count = if resolved_start > resolved_end {
            (resolved_start - resolved_end + (-step_wide) - 1) / (-step_wide)
        } else {
            0
        };
        (resolved_start, count)
    };
    let count = u64::try_from(count).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu slice extent",
        axis: None,
        detail: "slice element count does not fit in u64".to_string(),
    })?;
    let start = i64::try_from(start).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu slice start",
        axis: None,
        detail: "resolved start does not fit in i64".to_string(),
    })?;
    Ok(AxisSlice { start, step, count })
}

/// `Slice(data, starts, ends, axes?, steps?)`, ONNX opset 13+ encoding.
pub(crate) fn slice(
    op: OpId,
    input: &TensorView<'_>,
    starts: &TensorView<'_>,
    ends: &TensorView<'_>,
    axes: Option<&TensorView<'_>>,
    steps: Option<&TensorView<'_>>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    let rank = input.shape().rank();
    let starts = read_i64_vector(op, starts, "slice starts")?;
    let ends = read_i64_vector(op, ends, "slice ends")?;
    if starts.len() != ends.len() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{} ends to match the starts", starts.len()),
            actual: format!("{} ends", ends.len()),
        });
    }
    let count = starts.len();
    let axes = match axes {
        Some(view) => read_i64_vector(op, view, "slice axes")?,
        None => (0..count as i64).collect(),
    };
    let steps = match steps {
        Some(view) => read_i64_vector(op, view, "slice steps")?,
        None => vec![1i64; count],
    };
    if axes.len() != count || steps.len() != count {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{count} axes and {count} steps to match the starts"),
            actual: format!("{} axes and {} steps", axes.len(), steps.len()),
        });
    }

    let mut plan: Vec<AxisSlice> = (0..rank)
        .map(|axis| AxisSlice {
            start: 0,
            step: 1,
            count: input.shape().dims()[axis],
        })
        .collect();
    let mut seen = vec![false; rank];
    for position in 0..count {
        let axis = normalize_axis(op, axes[position], rank, "slice axis")?;
        if seen[axis] {
            return Err(Fn3Error::InvalidShape {
                context: "slice axis",
                detail: format!("op {op}: axis {axis} is sliced twice"),
            });
        }
        seen[axis] = true;
        plan[axis] = resolve_axis_slice(
            op,
            input.shape().dims()[axis],
            starts[position],
            ends[position],
            steps[position],
        )?;
    }

    let expected = Shape::from_dims(plan.iter().map(|axis| axis.count));
    require_output_shape(op, &expected, out)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }

    let element_bytes = input.dtype().size_bytes();
    let source = input.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let mut out_coordinates = vec![0u64; rank];
    let mut in_coordinates = vec![0u64; rank];
    for linear in 0..elements {
        decompose(linear, expected.dims(), &mut out_coordinates);
        for axis in 0..rank {
            let slice = plan[axis];
            let offset = i128::from(slice.start)
                + i128::from(slice.step) * i128::from(out_coordinates[axis]);
            in_coordinates[axis] = u64::try_from(offset).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu slice coordinate",
                axis: Some(axis),
                detail: format!("resolved coordinate {offset} is negative or too large"),
            })?;
        }
        let from = input
            .meta()
            .checked_index(&in_coordinates)
            .map_err(|error| error.with_op(op))?;
        copy_element(
            op,
            source,
            from,
            destination,
            linear as usize,
            element_bytes,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

/// Broadcast `input` to the target shape (right-aligned NumPy rules).
pub(crate) fn expand(
    op: OpId,
    input: &TensorView<'_>,
    shape_input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    let target = read_i64_vector(op, shape_input, "expand shape input")?;
    let mut dims = Vec::with_capacity(target.len());
    for (axis, value) in target.into_iter().enumerate() {
        dims.push(u64::try_from(value).map_err(|_| Fn3Error::InvalidShape {
            context: "expand shape",
            detail: format!("op {op}: extent {value} on axis {axis} is negative"),
        })?);
    }
    let expected = input.shape().broadcast_with(&Shape::from_dims(dims), op)?;
    require_output_shape(op, &expected, out)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }

    let element_bytes = input.dtype().size_bytes();
    let locator = Locator::new(op, input.meta(), &expected)?;
    let out_locator = Locator::new(op, out.meta(), &expected)?;
    let source = input.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let mut coordinates = vec![0u64; expected.rank()];
    for linear in 0..elements {
        decompose(linear, expected.dims(), &mut coordinates);
        let from = locator.index(op, &coordinates)?;
        let to = out_locator.index(op, &coordinates)?;
        copy_element(op, source, from, destination, to, element_bytes)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shape / ConstantOfShape / Range
// ---------------------------------------------------------------------------

/// Materialise the runtime shape as an `I64` tensor of rank 1.
pub(crate) fn shape_of(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    require_dtype(op, out.meta(), DType::I64)?;
    let rank = input.shape().rank();
    let expected = Shape::from_dims([rank as u64]);
    require_output_shape(op, &expected, out)?;
    let destination = out.as_mut_slice::<i64>()?;
    for (axis, &extent) in input.shape().dims().iter().enumerate() {
        let value = i64::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
            context: "cpu shape value",
            axis: Some(axis),
            detail: format!("extent {extent} does not fit in i64"),
        })?;
        *at_mut(op, destination, axis)? = value;
    }
    Ok(())
}

/// `ConstantOfShape(shape, value)`.
///
/// **fn3 deviation:** ONNX carries the fill value in a `value` attribute, which
/// [`OpParams`] cannot express. The fn3 contract lifts it into a scalar second
/// input of the output dtype instead of dropping it or inventing a default.
/// The importer (#54) emits that scalar as a constant instruction.
pub(crate) fn constant_of_shape(
    op: OpId,
    shape_input: &TensorView<'_>,
    value_input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if value_input.dtype() != out.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: out.dtype(),
            actual: value_input.dtype(),
        });
    }
    if value_input.shape().numel()? != 1 {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: "a single-element fill value".to_string(),
            actual: value_input.shape().to_string(),
        });
    }
    let target = read_i64_vector(op, shape_input, "constant_of_shape shape input")?;
    let mut dims = Vec::with_capacity(target.len());
    for (axis, value) in target.into_iter().enumerate() {
        dims.push(u64::try_from(value).map_err(|_| Fn3Error::InvalidShape {
            context: "constant_of_shape shape",
            detail: format!("op {op}: extent {value} on axis {axis} is negative"),
        })?);
    }
    let expected = Shape::from_dims(dims);
    require_output_shape(op, &expected, out)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let element_bytes = out.dtype().size_bytes();
    let fill = super::element_bytes_at(
        op,
        value_input.storage_bytes(),
        value_input.meta().offset(),
        element_bytes,
    )?;
    let destination = out.storage_slice_mut::<u8>()?;
    let capacity = destination.len();
    for linear in 0..elements {
        let to = usize::try_from(linear).map_err(|_| Fn3Error::ShapeOverflow {
            context: "cpu constant_of_shape element",
            axis: None,
            detail: "element index does not fit in usize".to_string(),
        })?;
        let offset = super::byte_offset(op, to, element_bytes)?;
        let end = offset
            .checked_add(element_bytes)
            .ok_or_else(|| super::kernel_bounds_error(op, offset, capacity))?;
        destination
            .get_mut(offset..end)
            .ok_or_else(|| super::kernel_bounds_error(op, offset, capacity))?
            .copy_from_slice(&fill);
    }
    Ok(())
}

/// `Range(start, limit, delta)` for `I64` or `F32`.
///
/// Values are computed as `start + i * delta` (a multiplication, not repeated
/// addition), so the result does not drift for large `i`.
pub(crate) fn range(
    op: OpId,
    start: &TensorView<'_>,
    limit: &TensorView<'_>,
    delta: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    for input in [start, limit, delta] {
        if input.dtype() != out.dtype() {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: out.dtype(),
                actual: input.dtype(),
            });
        }
        if input.shape().numel()? != 1 {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: "scalar start/limit/delta".to_string(),
                actual: input.shape().to_string(),
            });
        }
    }
    match out.dtype() {
        DType::I64 => {
            let start = read_i64_scalar(op, start, "range start")?;
            let limit = read_i64_scalar(op, limit, "range limit")?;
            let delta = read_i64_scalar(op, delta, "range delta")?;
            if delta == 0 {
                return Err(Fn3Error::InvalidShape {
                    context: "range delta",
                    detail: format!("op {op}: a range delta of 0 is undefined"),
                });
            }
            let span = i128::from(limit) - i128::from(start);
            let step = i128::from(delta);
            let count = if (step > 0 && span > 0) || (step < 0 && span < 0) {
                let magnitude = (span / step).abs();
                if span % step != 0 {
                    magnitude + 1
                } else {
                    magnitude
                }
            } else {
                0
            };
            let count = u64::try_from(count).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu range extent",
                axis: None,
                detail: "range element count does not fit in u64".to_string(),
            })?;
            require_output_shape(op, &Shape::from_dims([count]), out)?;
            let destination = out.as_mut_slice::<i64>()?;
            for index in 0..count {
                let value = i128::from(start) + i128::from(index) * step;
                let value = i64::try_from(value).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu range value",
                    axis: None,
                    detail: format!("value {value} does not fit in i64"),
                })?;
                *at_mut(op, destination, index as usize)? = value;
            }
        }
        DType::F32 => {
            let start_values = start.as_slice::<f32>()?;
            let limit_values = limit.as_slice::<f32>()?;
            let delta_values = delta.as_slice::<f32>()?;
            let start_value = *start_values.first().ok_or_else(|| Fn3Error::InvalidShape {
                context: "range start",
                detail: format!("op {op}: start has no element"),
            })?;
            let limit_value = *limit_values.first().ok_or_else(|| Fn3Error::InvalidShape {
                context: "range limit",
                detail: format!("op {op}: limit has no element"),
            })?;
            let delta_value = *delta_values.first().ok_or_else(|| Fn3Error::InvalidShape {
                context: "range delta",
                detail: format!("op {op}: delta has no element"),
            })?;
            if delta_value == 0.0 || !delta_value.is_finite() {
                return Err(Fn3Error::InvalidShape {
                    context: "range delta",
                    detail: format!("op {op}: a range delta of {delta_value} is not usable"),
                });
            }
            let span = f64::from(limit_value) - f64::from(start_value);
            let step = f64::from(delta_value);
            let count = if span / step > 0.0 {
                (span / step).ceil()
            } else {
                0.0
            };
            let count = if count.is_finite() && count >= 0.0 {
                count as u64
            } else {
                0
            };
            require_output_shape(op, &Shape::from_dims([count]), out)?;
            let destination = out.as_mut_slice::<f32>()?;
            for index in 0..count {
                *at_mut(op, destination, index as usize)? =
                    start_value + (index as f32) * delta_value;
            }
        }
        other => {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: other,
                context: "cpu range kernel (i64 and f32 only)",
            })
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceId;
    use crate::inference::shape::Strides;
    use crate::inference::tensor::{Tensor, TensorMeta};

    #[test]
    fn reshape_infers_and_copies_dims() {
        let input = Shape::from_dims([2, 3, 4]);
        assert_eq!(
            resolve_reshape_shape(OpId::Reshape, &input, &[-1, 4]).unwrap(),
            Shape::from_dims([6, 4])
        );
        assert_eq!(
            resolve_reshape_shape(OpId::Reshape, &input, &[0, 3, -1]).unwrap(),
            Shape::from_dims([2, 3, 4])
        );
        assert!(resolve_reshape_shape(OpId::Reshape, &input, &[-1, -1]).is_err());
        assert!(resolve_reshape_shape(OpId::Reshape, &input, &[-1, 5]).is_err());
    }

    #[test]
    fn slice_bounds_follow_python() {
        assert_eq!(
            resolve_axis_slice(OpId::Slice, 5, 0, 5, 1).unwrap(),
            AxisSlice {
                start: 0,
                step: 1,
                count: 5
            }
        );
        assert_eq!(
            resolve_axis_slice(OpId::Slice, 5, 1, i64::MAX, 2).unwrap(),
            AxisSlice {
                start: 1,
                step: 2,
                count: 2
            }
        );
        assert_eq!(
            resolve_axis_slice(OpId::Slice, 5, -2, i64::MAX, 1).unwrap(),
            AxisSlice {
                start: 3,
                step: 1,
                count: 2
            }
        );
        // Reverse.
        assert_eq!(
            resolve_axis_slice(OpId::Slice, 5, i64::MAX, i64::MIN, -1).unwrap(),
            AxisSlice {
                start: 4,
                step: -1,
                count: 5
            }
        );
        // Empty.
        assert_eq!(
            resolve_axis_slice(OpId::Slice, 5, 3, 1, 1).unwrap().count,
            0
        );
        assert!(resolve_axis_slice(OpId::Slice, 5, 0, 1, 0).is_err());
    }

    #[test]
    fn transpose_rejects_a_non_permutation() {
        let shape = Shape::from_dims([2, 3]);
        assert!(Shape::validate_permutation(&[0, 0], shape.rank()).is_err());
    }

    /// The old element loop, written independently: read the output in
    /// row-major order and address the input through its own metadata.
    fn naive_transpose(
        input: &TensorView<'_>,
        permutation: &[usize],
        out_dims: &[u64],
    ) -> Vec<f32> {
        let values = input
            .storage_slice::<f32>()
            .expect("the test storage is aligned f32");
        let elements: u64 = out_dims.iter().product();
        let mut out = Vec::with_capacity(elements as usize);
        let mut out_coordinates = vec![0u64; out_dims.len()];
        let mut in_coordinates = vec![0u64; out_dims.len()];
        for linear in 0..elements {
            decompose(linear, out_dims, &mut out_coordinates);
            for (out_axis, &in_axis) in permutation.iter().enumerate() {
                in_coordinates[in_axis] = out_coordinates[out_axis];
            }
            let index = input
                .meta()
                .checked_index(&in_coordinates)
                .expect("a transposed coordinate addresses the input");
            out.push(values[index]);
        }
        out
    }

    fn assert_transpose_matches_reference(dims: &[u64], permutation: &[usize]) {
        let elements: usize = dims.iter().map(|&d| d as usize).product();
        let values: Vec<f32> = (0..elements).map(|value| value as f32).collect();
        let input = Tensor::from_f32(Shape::from_dims(dims.iter().copied()), &values).unwrap();
        let out_dims: Vec<u64> = permutation.iter().map(|&axis| dims[axis]).collect();
        let mut out = Tensor::zeros(
            Shape::from_dims(out_dims.iter().copied()),
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        transpose(
            OpId::Transpose,
            &input.as_view().unwrap(),
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes(permutation.iter().map(|&axis| axis as i64)),
        )
        .unwrap();
        let expected = naive_transpose(&input.as_view().unwrap(), permutation, &out_dims);
        assert_eq!(out.as_f32_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn transpose_tiles_match_the_element_loop_for_every_fast_path() {
        // NCHW -> NHWC.
        assert_transpose_matches_reference(&[2, 3, 4, 5], &[0, 2, 3, 1]);
        // NCHW -> NWCH (the DFL input transpose in the pinned graph).
        assert_transpose_matches_reference(&[2, 3, 4, 5], &[0, 3, 1, 2]);
        // NCHW -> NHCW: the cell is the W run.
        assert_transpose_matches_reference(&[2, 3, 4, 5], &[0, 2, 1, 3]);
        assert_transpose_matches_reference(&[1, 4, 16, 2100], &[0, 2, 1, 3]);
        // N A C B -> N B C A (the DFL output transpose).
        assert_transpose_matches_reference(&[2, 3, 4, 5], &[0, 3, 2, 1]);
        assert_transpose_matches_reference(&[1, 2100, 4, 16], &[0, 3, 2, 1]);
        // The plain last-two swap, at rank 2 and rank 3.
        assert_transpose_matches_reference(&[3, 5], &[1, 0]);
        assert_transpose_matches_reference(&[2, 3, 5], &[0, 2, 1]);
        // A permutation outside every fast path keeps the generic loop.
        assert_transpose_matches_reference(&[2, 3, 2, 4, 3], &[2, 0, 3, 1, 4]);
    }

    #[test]
    fn transpose_of_an_eight_byte_element_uses_the_typed_tile() {
        let input = Tensor::from_i64(Shape::from_dims([2, 3]), &[1, 2, 3, 4, 5, 6]).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([3, 2]), DType::I64, DeviceId::Cpu).unwrap();
        transpose(
            OpId::Transpose,
            &input.as_view().unwrap(),
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes([1, 0]),
        )
        .unwrap();
        assert_eq!(out.as_i64_slice().unwrap(), &[1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn transpose_of_a_strided_view_keeps_the_element_path() {
        let source =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let meta = TensorMeta::strided(
            Shape::from_dims([3, 2]),
            Strides::from_elements(vec![1, 3]),
            0,
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        let transposed = TensorView::new(source.storage(), meta).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, DeviceId::Cpu).unwrap();
        transpose(
            OpId::Transpose,
            &transposed,
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes([1, 0]),
        )
        .unwrap();
        let expected = naive_transpose(&transposed, &[1, 0], &[2, 3]);
        assert_eq!(out.as_f32_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn concat_moves_each_contiguous_block_once() {
        // Axis 1: two outer rows per input, one element of `inner`.
        let left =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Tensor::from_f32(Shape::from_dims([2, 2]), &[7.0, 8.0, 9.0, 10.0]).unwrap();
        let mut out = Tensor::from_f32(Shape::from_dims([2, 5]), &[0.0; 10]).unwrap();
        concat(
            OpId::Concat,
            &[left.as_view().unwrap(), right.as_view().unwrap()],
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes([1]),
        )
        .unwrap();
        assert_eq!(
            out.as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 7.0, 8.0, 4.0, 5.0, 6.0, 9.0, 10.0]
        );

        // Axis 0: `inner` is the whole row, so the block is wider than one
        // element and the destination stride spans a whole output row.
        let top = Tensor::from_f32(Shape::from_dims([1, 3]), &[1.0, 2.0, 3.0]).unwrap();
        let bottom =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let mut out = Tensor::from_f32(Shape::from_dims([3, 3]), &[0.0; 9]).unwrap();
        concat(
            OpId::Concat,
            &[top.as_view().unwrap(), bottom.as_view().unwrap()],
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes([0]),
        )
        .unwrap();
        assert_eq!(
            out.as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn concat_falls_back_to_elements_for_a_strided_input() {
        let source =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let meta = TensorMeta::strided(
            Shape::from_dims([3, 2]),
            Strides::from_elements(vec![1, 3]),
            0,
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        let transposed = TensorView::new(source.storage(), meta).unwrap();
        let tail = Tensor::from_f32(Shape::from_dims([1, 2]), &[7.0, 8.0]).unwrap();
        let mut out = Tensor::from_f32(Shape::from_dims([4, 2]), &[0.0; 8]).unwrap();
        concat(
            OpId::Concat,
            &[transposed, tail.as_view().unwrap()],
            &mut out.as_mut_view().unwrap(),
            &OpParams::new().with_axes([0]),
        )
        .unwrap();
        assert_eq!(
            out.as_f32_slice().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 8.0]
        );
    }

    /// The pre-fast-path `Split` semantics, written independently of the
    /// kernel: output `i` element `coordinate` is input element `coordinate`
    /// with `sizes[..i]` added on `axis`, in the input's *logical* row-major
    /// order.
    fn naive_split(
        input: &TensorView<'_>,
        axis: usize,
        sizes: &[u64],
    ) -> Vec<(Vec<u64>, Vec<f32>)> {
        let dims = input.shape().dims();
        let values = input
            .storage_slice::<f32>()
            .expect("the test storage is aligned f32");
        let mut outputs = Vec::with_capacity(sizes.len());
        let mut offset = 0u64;
        for &size in sizes {
            let mut out_dims = dims.to_vec();
            out_dims[axis] = size;
            let elements: u64 = out_dims.iter().product();
            let mut out = Vec::with_capacity(elements as usize);
            let mut coordinates = vec![0u64; out_dims.len()];
            for linear in 0..elements {
                decompose(linear, &out_dims, &mut coordinates);
                coordinates[axis] += offset;
                let index = input
                    .meta()
                    .checked_index(&coordinates)
                    .expect("a logical output coordinate addresses the input");
                out.push(values[index]);
                coordinates[axis] -= offset;
            }
            outputs.push((out_dims, out));
            offset += size;
        }
        outputs
    }

    fn run_split(input: &TensorView<'_>, axis: usize, sizes: Option<&[u64]>) -> Vec<Tensor> {
        let dims = input.shape().dims();
        let count = sizes.map_or(2, <[u64]>::len);
        let split_sizes: Vec<u64> =
            sizes.map_or_else(|| vec![dims[axis] / count as u64; count], <[u64]>::to_vec);
        let mut outputs: Vec<Tensor> = split_sizes
            .iter()
            .map(|&size| {
                let mut out_dims = dims.to_vec();
                out_dims[axis] = size;
                Tensor::zeros(Shape::from_dims(out_dims), DType::F32, DeviceId::Cpu).unwrap()
            })
            .collect();
        let sizes_tensor = sizes.map(|values| {
            Tensor::from_i64(
                Shape::from_dims([values.len() as u64]),
                &values.iter().map(|&value| value as i64).collect::<Vec<_>>(),
            )
            .unwrap()
        });
        let sizes_view = sizes_tensor
            .as_ref()
            .map(|tensor| tensor.as_view().unwrap());
        let mut views: Vec<_> = outputs
            .iter_mut()
            .map(|out| out.as_mut_view().unwrap())
            .collect();
        split(
            OpId::Split,
            input,
            sizes_view.as_ref(),
            &mut views,
            &OpParams::new().with_axes([axis as i64]),
        )
        .unwrap();
        drop(views);
        outputs
    }

    fn assert_split_matches_reference(input: &Tensor, axis: usize, sizes: Option<&[u64]>) {
        let view = input.as_view().unwrap();
        let got = run_split(&view, axis, sizes);
        let expected_sizes = sizes.map_or_else(
            || {
                let dimension = input.shape().dims()[axis];
                vec![dimension / 2; 2]
            },
            <[u64]>::to_vec,
        );
        let expected = naive_split(&view, axis, &expected_sizes);
        assert_eq!(got.len(), expected.len());
        for (out, (dims, values)) in got.iter().zip(expected) {
            assert_eq!(out.shape().dims(), dims.as_slice());
            assert_eq!(out.as_f32_slice().unwrap(), values.as_slice());
        }
    }

    #[test]
    fn split_matches_the_old_semantics_on_every_axis_and_size_shape() {
        let values: Vec<f32> = (0..60).map(|value| value as f32).collect();
        // Axis 0, uneven sizes, a zero-size head, and a rank whose outer extent
        // is not one.
        let input = Tensor::from_f32(Shape::from_dims([6, 10]), &values[..60]).unwrap();
        assert_split_matches_reference(&input, 0, Some(&[0, 2, 3, 1]));
        assert_split_matches_reference(&input, 0, Some(&[2, 3, 1]));
        // Axis 1, uneven sizes.
        let input = Tensor::from_f32(Shape::from_dims([3, 4, 5]), &values[..60]).unwrap();
        assert_split_matches_reference(&input, 1, Some(&[1, 2, 1]));
        // The last axis.
        assert_split_matches_reference(&input, 2, Some(&[1, 4]));
        // An equal split into two parts on an evenly divisible axis.
        let input = Tensor::from_f32(Shape::from_dims([3, 4, 4]), &values[..48]).unwrap();
        assert_split_matches_reference(&input, 2, None);
        // A rank-1 split and a split into one part (the whole tensor).
        let line = Tensor::from_f32(Shape::from_dims([7]), &values[..7]).unwrap();
        assert_split_matches_reference(&line, 0, Some(&[3, 4]));
        assert_split_matches_reference(&line, 0, Some(&[7]));
    }

    #[test]
    fn split_of_a_strided_view_keeps_the_element_path() {
        let source =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let meta = TensorMeta::strided(
            Shape::from_dims([3, 2]),
            Strides::from_elements(vec![1, 3]),
            0,
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        let transposed = TensorView::new(source.storage(), meta).unwrap();
        // The view's logical order is [1, 4, 2, 5, 3, 6].
        let expected = naive_split(&transposed, 0, &[1, 2]);
        let mut outputs = [
            Tensor::zeros(Shape::from_dims([1, 2]), DType::F32, DeviceId::Cpu).unwrap(),
            Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, DeviceId::Cpu).unwrap(),
        ];
        let sizes = Tensor::from_i64(Shape::from_dims([2]), &[1, 2]).unwrap();
        let mut views: Vec<_> = outputs
            .iter_mut()
            .map(|out| out.as_mut_view().unwrap())
            .collect();
        split(
            OpId::Split,
            &transposed,
            Some(&sizes.as_view().unwrap()),
            &mut views,
            &OpParams::new().with_axes([0]),
        )
        .unwrap();
        drop(views);
        assert_eq!(outputs[0].as_f32_slice().unwrap(), expected[0].1.as_slice());
        assert_eq!(outputs[1].as_f32_slice().unwrap(), expected[1].1.as_slice());
    }
}
