//! Scalar reference kernels for the alpha CPU backend (FN3-07 / #53).
//!
//! # Purpose
//!
//! This tree is the *portable correctness reference*. Every kernel here is a
//! plain scalar loop with an explicitly documented evaluation order, so a
//! faster implementation (SIMD, blocking, threads: FN3-42 / #71) or a portable
//! GPU implementation (Vulkan: #59/#60/#61) has something exact to compare
//! against. Nothing here is tuned; the seams that performance work may replace
//! are named in [`crate::fn3::backend::cpu::kernels::optimization_handoff`]
//! and in each kernel's doc comment.
//!
//! # Indexing model
//!
//! Kernels never assume contiguity. Every operand is addressed through its
//! validated [`TensorMeta`]: `offset + sum(coord[axis] * stride[axis])`. Two
//! consequences follow, and both are deliberate:
//!
//! - A transposed or sliced view needs no materialisation.
//! - Broadcasting is expressed as an *effective* stride of `0` on the promoted
//!   axis ([`Locator`]), which is the only place a zero stride is legal
//!   (broadcast views are read-only by construction).
//!
//! Where an operand is contiguous and already has the result shape, kernels
//! take an iterator path (`iter().zip()`) that cannot index out of bounds at
//! all. Every other path uses [`TensorMeta::checked_index`] or [`Locator`],
//! which report overflow and out-of-range coordinates as errors. There is no
//! `unwrap`, `expect` or unchecked indexing on caller-controlled values in this
//! tree: [`at`] and [`at_mut`] are the only element accessors and they return
//! `Fn3Error` instead of panicking.
//!
//! # Numerical contract
//!
//! - Compute dtype: `F32` only. `I64`/`I32`/`Bool` are data dtypes: kernel
//!   inputs for indices, shapes, masks and comparisons, and outputs for
//!   `Shape`, `ArgMax`, `Range`, `Cast` and the comparison ops. No kernel
//!   computes in an integer dtype and none casts implicitly; `Cast` is the only
//!   sanctioned conversion.
//! - Reductions accumulate in ascending row-major order of the *input*
//!   coordinates, in `f32`. `MatMul` accumulates in ascending `k`.
//! - Transcendental functions name their implementation. `Erf`/`Gelu` are
//!   evaluated in `f64` and rounded once (see [`elementwise::erf_f32`]).
//! - A failed op writes nothing: every kernel validates arity, dtype, shape,
//!   attributes and output geometry before its first output write.

use crate::fn3::backend::OpParams;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMeta, TensorMut, TensorView};

pub(crate) mod elementwise;
pub(crate) mod logic;
pub(crate) mod matmul;
pub(crate) mod norm;
pub(crate) mod reduce;
pub(crate) mod softmax;
pub(crate) mod transform;

/// Documentation-only module holding the performance hand-off list.
///
/// # Optimization hand-off (FN3-42 / #71)
///
/// These are the symbols later performance work is expected to replace. Each
/// row states the correctness reference a replacement must keep matching; the
/// measured margins are from the `fn3_cpu_reference` run recorded in the FN3-07
/// hand-off report (fixture tolerances from `fixtures/parity/README.md`).
///
/// | Symbol | Why it is the hot spot | Correctness reference (measured) |
/// |---|---|---|
/// | `matmul::dot_rows` | the only `O(M*N*K)` loop in the model (`lm_head` is `1x128x50257`) | one `f32` accumulator, ascending `k`; `matmul.*` fixtures max_abs ≤ 4.8e-6 (atol 2e-5) |
/// | `matmul::matmul` | layout/broadcast shell around `dot_rows` | same fixtures plus the real-shape self-checks (decode max_abs 6.2e-6, bound 1.4e-5) |
/// | `elementwise::binary` | residual adds, additive masks, `x*w` scaling on `[1,13,128]` and `[1,1,S,S]` | right-aligned broadcast, one rounding per element; `matmul.gemm_k13_bias` bias add 9.5e-7, `attention` mask add 3.0e-8 |
/// | `elementwise::unary` | GELU / tanh / sqrt chains over the MLP width | `gelu.erf` 1.8e-8, `gelu.tanh` 5.7e-8 (atol 2e-6) |
/// | `elementwise::erf_f64` | degree-22 Clenshaw in `f64`, per element | `gelu.erf.out` plus its own precision table in `elementwise::tests` |
/// | `softmax::softmax` | attention probabilities, `O(S^2)` per head | max-subtract, `f32` `exp`, ascending sum, divide; `softmax.*` max_abs 2.98e-8 (atol 2e-6), all-masked row exactly `0.25` |
/// | `norm::layer_norm` | two passes over the hidden axis | `(x-mean)/sqrt(var+eps)*w+b`; `layernorm.*` max_abs 4.8e-7 (atol 2e-5) |
/// | `reduce::reduce` / `reduce::argmax` | whole-tensor scans | ascending row-major accumulation; first-wins tie rule |
/// | `transform::copy_element` | every data-movement kernel copies element by element | byte-exact against the source view; `embedding.out` exact (atol 0) |
/// | `kernels::Locator::index` | per-element index recomputation in the general path | `TensorMeta::checked_index` semantics with effective broadcast strides |
///
/// A replacement must keep these results inside `fixtures/parity/README.md`'s
/// tolerance, or publish a new measured reference before changing an order.
/// Nothing in this tree is blocked on the optimisation: the scalar reference is
/// the contract, and #71 replaces it behind the same op ids.
pub(crate) mod optimization_handoff {}

/// Require `meta.dtype == dtype`, naming `op`.
pub(crate) fn require_dtype(op: OpId, meta: &TensorMeta, dtype: DType) -> Result<(), Fn3Error> {
    meta.require_dtype(Some(op), dtype)
}

/// Require the caller-provided output shape to match the shape the kernel
/// computes. Plans pre-allocate outputs, so this is the check that keeps a plan
/// from silently writing into the wrong geometry.
pub(crate) fn require_output_shape(
    op: OpId,
    expected: &Shape,
    out: &TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if out.shape() != expected {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: expected.to_string(),
            actual: out.shape().to_string(),
        });
    }
    Ok(())
}

/// A defensive bounds failure inside a kernel.
///
/// Metadata validation (`TensorMeta::required_bytes` at construction, plus the
/// `checked_index`/`Locator` checks that follow) makes this unreachable. It is
/// an error rather than a panic so that a future kernel bug cannot abort a
/// session or corrupt an unrelated allocation.
pub(crate) fn kernel_bounds_error(op: OpId, index: usize, len: usize) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "cpu kernel bounds",
        detail: format!(
            "op {op}: element index {index} is outside the {len}-element storage; \
             metadata validation should make this unreachable"
        ),
    }
}

/// Read element `index` of `data`, naming `op` if the index is out of range.
pub(crate) fn at<T: Copy>(op: OpId, data: &[T], index: usize) -> Result<T, Fn3Error> {
    data.get(index)
        .copied()
        .ok_or_else(|| kernel_bounds_error(op, index, data.len()))
}

/// Borrow element `index` of `data` mutably.
pub(crate) fn at_mut<T>(op: OpId, data: &mut [T], index: usize) -> Result<&mut T, Fn3Error> {
    let len = data.len();
    data.get_mut(index)
        .ok_or_else(|| kernel_bounds_error(op, index, len))
}

/// Row-major coordinates of a linear index.
///
/// Precondition: every extent in `dims` is non-zero. Callers reach this only
/// after `numel > 0`, which implies it.
pub(crate) fn decompose(linear: u64, dims: &[u64], coordinates: &mut [u64]) {
    let mut remainder = linear;
    for axis in (0..dims.len()).rev() {
        // `numel > 0` implies `dims[axis] > 0`, so the modulo cannot divide by
        // zero.
        coordinates[axis] = remainder % dims[axis];
        remainder /= dims[axis];
    }
}

/// `product(dims)` with checked arithmetic.
pub(crate) fn checked_product(dims: &[u64], context: &'static str) -> Result<u64, Fn3Error> {
    let mut product = 1u64;
    for (axis, &extent) in dims.iter().enumerate() {
        product = product
            .checked_mul(extent)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context,
                axis: Some(axis),
                detail: format!("{dims:?} exceeds u64::MAX elements"),
            })?;
    }
    Ok(product)
}

/// An effective element locator for one operand promoted to a result shape.
///
/// Unlike [`TensorMeta::checked_index`], this accepts an operand whose rank is
/// lower than the result and an operand with extent `1` on an axis where the
/// result is larger: those axes get an effective stride of `0`, which is
/// exactly NumPy broadcasting. The coordinate-product and the final addition
/// are checked, and the resulting index is compared against the storage length
/// by the caller through [`at`]/[`at_mut`].
pub(crate) struct Locator {
    offset: usize,
    strides: Vec<usize>,
}

impl Locator {
    /// Build the locator for `meta` read at `result`'s coordinates.
    pub(crate) fn new(op: OpId, meta: &TensorMeta, result: &Shape) -> Result<Locator, Fn3Error> {
        if result.rank() < meta.rank() {
            return Err(Fn3Error::RankMismatch {
                op: Some(op),
                expected: result.rank(),
                actual: meta.rank(),
            });
        }
        let lead = result.rank() - meta.rank();
        let mut strides = vec![0usize; result.rank()];
        for axis in lead..result.rank() {
            let operand_axis = axis - lead;
            let extent = meta.shape().dims()[operand_axis];
            let result_extent = result.dims()[axis];
            let stride = meta.strides().as_slice()[operand_axis];
            let stride = usize::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
                axis: operand_axis,
                stride,
                detail: "negative strides are not addressable in the alpha view contract"
                    .to_string(),
            })?;
            // A size-1 axis may be promoted to any extent; a larger axis must
            // already match. `broadcast_with` has validated the promotion, so
            // the only case left is "operand extent 1, result larger" -> 0.
            strides[axis] = if extent == 1 && result_extent != 1 {
                0
            } else {
                stride
            };
        }
        Ok(Locator {
            offset: meta.offset(),
            strides,
        })
    }

    /// Element index of `coordinates` (length = result rank).
    pub(crate) fn index(&self, op: OpId, coordinates: &[u64]) -> Result<usize, Fn3Error> {
        let mut index = self.offset as u64;
        for (axis, (&coordinate, &stride)) in
            coordinates.iter().zip(self.strides.iter()).enumerate()
        {
            let delta =
                coordinate
                    .checked_mul(stride as u64)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "cpu kernel locator",
                        axis: Some(axis),
                        detail: format!("coordinate {coordinate} times stride {stride} overflows"),
                    })?;
            index = index
                .checked_add(delta)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu kernel locator",
                    axis: Some(axis),
                    detail: "element index exceeds u64::MAX".to_string(),
                })?;
        }
        usize::try_from(index).map_err(|_| Fn3Error::ShapeOverflow {
            context: "cpu kernel locator",
            axis: None,
            detail: format!("element index {index} does not fit in usize (op {op})"),
        })
    }
}

/// Checked byte offset of element `index` for a dtype of `element_bytes`.
pub(crate) fn byte_offset(op: OpId, index: usize, element_bytes: usize) -> Result<usize, Fn3Error> {
    index
        .checked_mul(element_bytes)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "cpu kernel byte offset",
            axis: None,
            detail: format!("op {op}: element index {index} times {element_bytes} bytes overflows"),
        })
}

/// The contiguous byte window of a view: `storage[byte_offset .. + byte_len]`.
///
/// Only meaningful for a contiguous view, where the window is exactly the
/// view's elements in row-major order.
pub(crate) fn view_window_bytes<'a>(
    op: OpId,
    view: &'a TensorView<'_>,
) -> Result<&'a [u8], Fn3Error> {
    view.meta()
        .require_contiguous("cpu kernel view_window_bytes")?;
    let start = view.meta().byte_offset()?;
    let len = view.meta().byte_len()?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| kernel_bounds_error(op, start, view.storage_bytes().len()))?;
    view.storage_bytes()
        .get(start..end)
        .ok_or_else(|| kernel_bounds_error(op, end, view.storage_bytes().len()))
}

/// The mutable byte window of a contiguous writable view.
pub(crate) fn out_window_bytes<'a>(
    op: OpId,
    out: &'a mut TensorMut<'_>,
) -> Result<&'a mut [u8], Fn3Error> {
    let start = out.meta().byte_offset()?;
    let len = out.meta().byte_len()?;
    let buffer = out.storage_slice_mut::<u8>()?;
    let capacity = buffer.len();
    let end = start
        .checked_add(len)
        .ok_or_else(|| kernel_bounds_error(op, start, capacity))?;
    buffer
        .get_mut(start..end)
        .ok_or_else(|| kernel_bounds_error(op, end, capacity))
}

/// One element as a byte slice of `element_bytes`, addressed from `storage`.
pub(crate) fn element_slice<'a>(
    op: OpId,
    storage: &'a [u8],
    element_index: usize,
    element_bytes: usize,
) -> Result<&'a [u8], Fn3Error> {
    let from = byte_offset(op, element_index, element_bytes)?;
    let end = from
        .checked_add(element_bytes)
        .ok_or_else(|| kernel_bounds_error(op, from, storage.len()))?;
    storage
        .get(from..end)
        .ok_or_else(|| kernel_bounds_error(op, from, storage.len()))
}

/// One element as a mutable byte slice of `element_bytes`.
pub(crate) fn element_slice_mut<'a>(
    op: OpId,
    storage: &'a mut [u8],
    element_index: usize,
    element_bytes: usize,
) -> Result<&'a mut [u8], Fn3Error> {
    let from = byte_offset(op, element_index, element_bytes)?;
    let len = storage.len();
    let end = from
        .checked_add(element_bytes)
        .ok_or_else(|| kernel_bounds_error(op, from, len))?;
    storage
        .get_mut(from..end)
        .ok_or_else(|| kernel_bounds_error(op, from, len))
}

/// Copy one element (a fixed-size byte block) from `src` to `dst`.
///
/// Used by the data-movement kernels, which are dtype-generic: they never
/// interpret the bytes they move, so `F32`, `I64`, `I32` and `Bool` share one
/// implementation and no conversion can happen by accident.
pub(crate) fn copy_element(
    op: OpId,
    src: &[u8],
    src_index: usize,
    dst: &mut [u8],
    dst_index: usize,
    element_bytes: usize,
) -> Result<(), Fn3Error> {
    let source = element_slice(op, src, src_index, element_bytes)?;
    let destination = element_slice_mut(op, dst, dst_index, element_bytes)?;
    destination.copy_from_slice(source);
    Ok(())
}

/// Fill one element-size byte block of `dst` with zeros.
pub(crate) fn zero_element(
    op: OpId,
    dst: &mut [u8],
    dst_index: usize,
    element_bytes: usize,
) -> Result<(), Fn3Error> {
    let destination = element_slice_mut(op, dst, dst_index, element_bytes)?;
    destination.fill(0);
    Ok(())
}

/// Copy `element_bytes` from `src[src_index * element_bytes ..]` into an owned
/// buffer; the scalar form of [`copy_element`], used for fill values.
pub(crate) fn element_bytes_at(
    op: OpId,
    src: &[u8],
    index: usize,
    element_bytes: usize,
) -> Result<Vec<u8>, Fn3Error> {
    element_slice(op, src, index, element_bytes).map(|slice| slice.to_vec())
}

/// Normalise a signed axis against `rank`, rejecting out-of-range values.
pub(crate) fn normalize_axis(
    op: OpId,
    axis: i64,
    rank: usize,
    what: &'static str,
) -> Result<usize, Fn3Error> {
    let rank_i64 = rank as i64;
    let resolved = if axis < 0 { axis + rank_i64 } else { axis };
    if resolved < 0 || resolved >= rank_i64 {
        return Err(Fn3Error::InvalidShape {
            context: what,
            detail: format!(
                "op {op}: axis {axis} is outside the rank-{rank} operand; \
                 legal axes are -{rank}..={}",
                rank.saturating_sub(1)
            ),
        });
    }
    Ok(resolved as usize)
}

/// Resolve an op's single axis attribute, or `default` when it supplies none.
pub(crate) fn single_axis(
    op: OpId,
    params: &OpParams,
    rank: usize,
    default: i64,
    what: &'static str,
) -> Result<usize, Fn3Error> {
    match params.axes.as_slice() {
        [] => normalize_axis(op, default, rank, what),
        [axis] => normalize_axis(op, *axis, rank, what),
        axes => Err(Fn3Error::InvalidShape {
            context: what,
            detail: format!("op {op} takes exactly one axis, but the plan supplies {axes:?}"),
        }),
    }
}

/// Reject every attribute, quoting what the plan supplied.
///
/// The message text is pinned by `tests/fn3_contracts.rs`
/// (`arity_and_attribute_errors_are_reported`): keep "accepts no attributes".
pub(crate) fn require_no_attributes(op: OpId, params: &OpParams) -> Result<(), Fn3Error> {
    if params.axes.is_empty() && !params.keep_dims && params.epsilon == 0.0 {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "cpu backend attributes",
            detail: format!(
                "op {op} accepts no attributes, but the plan supplies axes={:?}, \
                 keep_dims={}, epsilon={}",
                params.axes, params.keep_dims, params.epsilon
            ),
        })
    }
}

/// Accept `axes` only; reject `keep_dims` and `epsilon`.
pub(crate) fn require_only_axes(op: OpId, params: &OpParams) -> Result<(), Fn3Error> {
    if !params.keep_dims && params.epsilon == 0.0 {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "cpu backend attributes",
            detail: format!(
                "op {op} accepts only the axes attribute, but the plan supplies \
                 keep_dims={}, epsilon={}",
                params.keep_dims, params.epsilon
            ),
        })
    }
}

/// Accept `axes` and `keep_dims`; reject `epsilon`.
pub(crate) fn require_axes_and_keep_dims(op: OpId, params: &OpParams) -> Result<(), Fn3Error> {
    if params.epsilon == 0.0 {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "cpu backend attributes",
            detail: format!(
                "op {op} accepts only axes and keep_dims, but the plan supplies \
                 epsilon={}",
                params.epsilon
            ),
        })
    }
}

/// Accept `axes` and `epsilon`; reject `keep_dims`.
pub(crate) fn require_axes_and_epsilon(op: OpId, params: &OpParams) -> Result<(), Fn3Error> {
    if !params.keep_dims {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "cpu backend attributes",
            detail: format!(
                "op {op} accepts only axes and epsilon, but the plan supplies keep_dims=true"
            ),
        })
    }
}

/// Read a rank-1 `I64` tensor of shape values, rejecting anything else.
pub(crate) fn read_i64_vector(
    op: OpId,
    view: &TensorView<'_>,
    what: &'static str,
) -> Result<Vec<i64>, Fn3Error> {
    require_dtype(op, view.meta(), DType::I64)?;
    if view.shape().rank() != 1 {
        return Err(Fn3Error::InvalidShape {
            context: what,
            detail: format!(
                "op {op}: expected a rank-1 i64 tensor, found {} with rank {}",
                view.shape(),
                view.shape().rank()
            ),
        });
    }
    Ok(view.as_i64_slice()?.to_vec())
}

/// Read one `I64` scalar (rank 0, or rank 1 with one element).
pub(crate) fn read_i64_scalar(
    op: OpId,
    view: &TensorView<'_>,
    what: &'static str,
) -> Result<i64, Fn3Error> {
    require_dtype(op, view.meta(), DType::I64)?;
    let values = view.as_i64_slice()?;
    match values {
        [value] => Ok(*value),
        _ => Err(Fn3Error::InvalidShape {
            context: what,
            detail: format!(
                "op {op}: expected a scalar i64 tensor, found shape {} with {} elements",
                view.shape(),
                values.len()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_is_row_major() {
        let mut coordinates = [0u64; 3];
        for linear in 0..24u64 {
            decompose(linear, &[2, 3, 4], &mut coordinates);
            let expected = [linear / 12, (linear / 4) % 3, linear % 4];
            assert_eq!(coordinates, expected, "linear {linear}");
        }
    }

    #[test]
    fn checked_product_reports_the_overflowing_axis() {
        let error = checked_product(&[2, u64::MAX, 2], "test").unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::ShapeOverflow { axis: Some(1), .. }
        ));
    }

    #[test]
    fn axis_normalisation_rejects_out_of_range() {
        assert_eq!(normalize_axis(OpId::Softmax, -1, 3, "test"), Ok(2));
        assert_eq!(normalize_axis(OpId::Softmax, 2, 3, "test"), Ok(2));
        let error = normalize_axis(OpId::Softmax, 3, 3, "test").unwrap_err();
        assert!(error.to_string().contains("axis 3"), "{error}");
        let error = normalize_axis(OpId::Softmax, -4, 3, "test").unwrap_err();
        assert!(error.to_string().contains("axis -4"), "{error}");
    }

    #[test]
    fn locator_promotes_size_one_axes_with_a_zero_stride() {
        let meta = TensorMeta::contiguous(
            Shape::from_dims([1, 3]),
            DType::F32,
            crate::fn3::device::DeviceId::Cpu,
        )
        .unwrap();
        let locator = Locator::new(OpId::Add, &meta, &Shape::from_dims([4, 3])).unwrap();
        assert_eq!(locator.index(OpId::Add, &[0, 0]).unwrap(), 0);
        assert_eq!(locator.index(OpId::Add, &[3, 2]).unwrap(), 2);
    }

    #[test]
    fn attribute_helpers_name_the_op() {
        let error = require_no_attributes(OpId::Add, &OpParams::default().with_axes([0]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("accepts no attributes"), "{error}");
        assert!(error.contains("op add"), "{error}");
        assert!(require_no_attributes(OpId::Add, &OpParams::default()).is_ok());
        assert!(require_only_axes(OpId::Softmax, &OpParams::default().with_axes([-1])).is_ok());
        assert!(
            require_only_axes(OpId::Softmax, &OpParams::default().with_keep_dims(true)).is_err()
        );
    }
}
