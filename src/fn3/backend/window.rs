//! Window geometry shared by every CPU provider.
//!
//! Convolution and pooling need the same three decisions before a single
//! multiply:
//!
//! 1. **which** kernel/strides/dilations the op declares ([`window_axes`]);
//! 2. **how much** `auto_pad` resolves to on each side ([`resolve_pads`]);
//! 3. **what** output extent that window produces ([`window_output`]).
//!
//! The scalar kernels in [`crate::inference::runtime::kernels`] and the
//! optional oneDNN engine ask these questions here, so a provider cannot
//! disagree with the typed shape pass (`ir::infer`) about the geometry the plan
//! reserved. This module sits under `backend` because both callers are
//! providers: the runtime re-exports these functions rather than owning a
//! second copy, and the dependency arrow stays downward.
//!
//! # Padding is explicit before a provider sees it
//!
//! oneDNN takes explicit padding only, and the scalar kernels want the same
//! form. [`resolve_pads`] therefore turns `AutoPad::SameUpper`, `SameLower`,
//! `Valid` and `NotSet` into one `(begin, end)` pair per spatial axis *before*
//! any provider is chosen. A negative explicit pad is rejected here: `Conv` and
//! the pool family only pad outward (`Pad` is the op that crops).

use crate::inference::backend::{AutoPad, OpParams};
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;

/// `(kernel_shape, strides, dilations)`, one extent per spatial axis.
pub(crate) type WindowAxes = (Vec<usize>, Vec<usize>, Vec<usize>);

/// Per-axis `kernel_shape`, `strides` and `dilations`, validated against the
/// spatial rank.
pub(crate) fn window_axes(
    op: OpId,
    params: &OpParams,
    spatial: usize,
) -> Result<WindowAxes, Fn3Error> {
    let read =
        |values: &[i64], default: usize, what: &'static str| -> Result<Vec<usize>, Fn3Error> {
            if values.is_empty() {
                return Ok(vec![default; spatial]);
            }
            if values.len() != spatial {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime kernel window axes",
                    detail: format!(
                        "op {op} has {} {what} extents for a {spatial}-axis spatial rank",
                        values.len()
                    ),
                });
            }
            values
                .iter()
                .map(|&value| {
                    usize::try_from(value)
                        .ok()
                        .filter(|&value| value > 0)
                        .ok_or_else(|| Fn3Error::InvalidShape {
                            context: "runtime kernel window axes",
                            detail: format!("op {op} has the non-positive {what} extent {value}"),
                        })
                })
                .collect()
        };
    if params.kernel_shape.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "runtime kernel window axes",
            detail: format!("op {op} needs kernel_shape for its {spatial} spatial axis/axes"),
        });
    }
    let kernel = read(&params.kernel_shape, 1, "kernel_shape")?;
    let strides = read(&params.strides, 1, "strides")?;
    let dilations = read(&params.dilations, 1, "dilations")?;
    Ok((kernel, strides, dilations))
}

/// Resolve `auto_pad` or explicit `pads` into signed begin/end pairs.
///
/// This is the same rule the typed shape pass applies (`ir::infer`), so a
/// kernel's padding cannot disagree with the output shape the plan reserved. A
/// negative explicit pad is rejected: convolution and pooling only pad outward
/// (`Pad` is the op that crops).
pub(crate) fn resolve_pads(
    op: OpId,
    params: &OpParams,
    input: &[usize],
    kernel: &[usize],
    strides: &[usize],
    dilations: &[usize],
) -> Result<Vec<(i64, i64)>, Fn3Error> {
    let spatial = input.len();
    match params.auto_pad {
        AutoPad::NotSet => {
            if params.pads.is_empty() {
                return Ok(vec![(0, 0); spatial]);
            }
            if params.pads.len() != 2 * spatial {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime kernel pads",
                    detail: format!(
                        "op {op} has {} pads for a {spatial}-axis spatial rank; ONNX supplies \
                         {spatial} begin extents followed by {spatial} end extents",
                        params.pads.len()
                    ),
                });
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let (begin, end) = (params.pads[axis], params.pads[spatial + axis]);
                if begin < 0 || end < 0 {
                    return Err(Fn3Error::InvalidShape {
                        context: "runtime kernel pads",
                        detail: format!(
                            "op {op} has the negative pads ({begin}, {end}) on spatial axis \
                             {axis}; convolution and pooling pads are non-negative"
                        ),
                    });
                }
                pairs.push((begin, end));
            }
            Ok(pairs)
        }
        AutoPad::Valid => {
            if !params.pads.is_empty() {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime kernel pads",
                    detail: format!(
                        "op {op} sets auto_pad = valid and explicit pads; ONNX forbids combining \
                         them"
                    ),
                });
            }
            Ok(vec![(0, 0); spatial])
        }
        same @ (AutoPad::SameUpper | AutoPad::SameLower) => {
            if !params.pads.is_empty() {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime kernel pads",
                    detail: format!(
                        "op {op} sets auto_pad = {} and explicit pads; ONNX forbids combining them",
                        same.name()
                    ),
                });
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                let out = input[axis].div_ceil(strides[axis]);
                let total = (out.saturating_sub(1))
                    .saturating_mul(strides[axis])
                    .saturating_add(effective)
                    .saturating_sub(input[axis]);
                let (begin, end) = if same == AutoPad::SameUpper {
                    (total / 2, total - total / 2)
                } else {
                    (total - total / 2, total / 2)
                };
                pairs.push((
                    i64::try_from(begin).map_err(|_| Fn3Error::ShapeOverflow {
                        context: "runtime kernel auto_pad",
                        axis: Some(axis),
                        detail: "begin pad does not fit in i64".to_string(),
                    })?,
                    i64::try_from(end).map_err(|_| Fn3Error::ShapeOverflow {
                        context: "runtime kernel auto_pad",
                        axis: Some(axis),
                        detail: "end pad does not fit in i64".to_string(),
                    })?,
                ));
            }
            Ok(pairs)
        }
    }
}

/// One output extent of a windowed op, under the ONNX rule.
///
/// Identical to the typed shape pass's `window_output`, so a kernel can prove
/// the plan reserved the geometry it computes.
pub(crate) fn window_output(
    op: OpId,
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    begin: i64,
    end: i64,
    ceil_mode: bool,
) -> Result<usize, Fn3Error> {
    let effective = dilation * (kernel - 1) + 1;
    let total = input as i128 + begin as i128 + end as i128;
    let stride = stride as i128;
    let input = input as i128;
    let delta = total - effective as i128;
    let mut out = if ceil_mode {
        -((-delta).div_euclid(stride))
    } else {
        delta.div_euclid(stride)
    } + 1;
    if ceil_mode && out > 0 && (out - 1) * stride >= input + begin as i128 {
        out -= 1;
    }
    if out < 0 {
        out = 0;
    }
    usize::try_from(out).map_err(|_| Fn3Error::ShapeOverflow {
        context: "runtime kernel window output",
        axis: None,
        detail: format!("op {op}: {out} output extents does not fit in usize"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_output_matches_the_ir_rule() {
        // 5x5 kernel, stride 1, pad 2, input 8 -> 8 (the MaxPool in YOLOv8).
        assert_eq!(
            window_output(OpId::MaxPool, 8, 5, 1, 1, 2, 2, false).unwrap(),
            8
        );
        // 2x2 kernel, stride 2, no pad, input 5: floor gives 2, ceil gives 3.
        assert_eq!(
            window_output(OpId::MaxPool, 5, 2, 2, 1, 0, 0, false).unwrap(),
            2
        );
        assert_eq!(
            window_output(OpId::MaxPool, 5, 2, 2, 1, 0, 0, true).unwrap(),
            3
        );
        // 3x3 kernel, stride 2, pad 1, input 4 -> 2.
        assert_eq!(
            window_output(OpId::Conv, 4, 3, 2, 1, 1, 1, false).unwrap(),
            2
        );
        // Dilation widens the window: 3x3 with dilation 2 is 5 wide.
        assert_eq!(
            window_output(OpId::Conv, 6, 3, 1, 2, 0, 0, false).unwrap(),
            2
        );
    }

    #[test]
    fn auto_pad_same_upper_and_lower_split_the_extra_extent() {
        let mut params = OpParams::new();
        params.auto_pad = AutoPad::SameUpper;
        let pads = resolve_pads(OpId::Conv, &params, &[8], &[3], &[1], &[1]).unwrap();
        assert_eq!(pads, vec![(1, 1)]);
        // An even total puts the extra element at the end for SAME_UPPER.
        let pads = resolve_pads(OpId::Conv, &params, &[4], &[2], &[1], &[1]).unwrap();
        assert_eq!(pads, vec![(0, 1)]);
        params.auto_pad = AutoPad::SameLower;
        let pads = resolve_pads(OpId::Conv, &params, &[4], &[2], &[1], &[1]).unwrap();
        assert_eq!(pads, vec![(1, 0)]);
    }

    #[test]
    fn negative_explicit_pads_are_rejected_for_windows() {
        let params = OpParams::new().with_pads([-1, 0]);
        let error = resolve_pads(OpId::MaxPool, &params, &[4], &[2], &[1], &[1]).unwrap_err();
        assert!(error.to_string().contains("negative pads"), "{error}");
    }
}
