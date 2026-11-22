//! Pure segment geometry for the oneDNN engine.
//!
//! The same functions answer two questions, and that is deliberate:
//!
//! - the **planner** asks whether a segment is expressible in oneDNN's
//!   vocabulary (2-D NCHW convolution and pooling, 2-D matmul, `f32`);
//! - **session preparation** builds the exact descriptors from the answer.
//!
//! A rule that lives in only one of them would let a segment be claimed and
//! then fail to prepare, so every check here is shared. Nothing in this module
//! touches oneDNN: it is shape and attribute arithmetic over
//! [`OpParams`](crate::inference::backend::OpParams), and it reuses the same
//! pad/dilation/output-extent helpers as the scalar kernels
//! ([`crate::inference::backend::window`]).
//!
//! # What is claimed, and what falls back
//!
//! | Op | Claimed | Falls back |
//! |---|---|---|
//! | `Conv` | 4-D `f32` NCHW, groups, strides, dilations, asymmetric explicit pads, optional bias | 1-D/3-D spatial, other dtypes, negative pads |
//! | `MaxPool`, `AveragePool` | 4-D `f32`, explicit pads, strides, `count_include_pad` | `ceil_mode = 1`, dilation, 1-D/3-D |
//! | `GlobalAveragePool` | 4-D `f32` (as average pooling with a full window) | other ranks/dtypes |
//! | `MatMul` | 2-D `f32` (the decoder's `Gemm` form) | batched/rank-3+, broadcasting |
//! | everything else | — | scalar kernels |
//!
//! Every reason string is the plan's fallback report, so it names the operand
//! and the property rather than saying "unsupported".

use crate::inference::backend::window::{resolve_pads, window_axes, window_output};
use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;

/// Smallest output-channel count worth handing to another worker.
///
/// Below eight channels a block's fixed cost (a Rayon task, one `execute`, one
/// `dnnl_stream_wait`) is a large fraction of its arithmetic, and on a
/// depthwise convolution an eight-channel block is still one output plane per
/// thread.
pub(crate) const MIN_BLOCK_CHANNELS: usize = 8;

/// Smallest output payload, in `f32` elements, worth another worker.
///
/// 16 Ki elements is 64 KiB: one cache line per worker on a 16-core host and
/// roughly the point where a task hand-off stops dominating a block.
pub(crate) const MIN_BLOCK_ELEMENTS: usize = 16 * 1024;

fn dims(shape: &Shape, what: &'static str) -> Result<Vec<usize>, Fn3Error> {
    shape
        .dims()
        .iter()
        .map(|&extent| {
            usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
                context: what,
                axis: None,
                detail: format!("extent {extent} does not fit in usize"),
            })
        })
        .collect()
}

/// A 2-D convolution, resolved to the exact window the kernel will use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Conv {
    /// Batch extent.
    pub batch: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Input height and width.
    pub in_hw: [usize; 2],
    /// Output channels.
    pub out_channels: usize,
    /// Input channels per group (`in_channels / group`).
    pub in_per_group: usize,
    /// Groups.
    pub group: usize,
    /// Kernel height and width.
    pub kernel: [usize; 2],
    /// Stride per spatial axis.
    pub strides: [usize; 2],
    /// Dilation per spatial axis.
    pub dilations: [usize; 2],
    /// Padding per spatial axis: `(begin, end)`, already resolved from
    /// `auto_pad`.
    pub pads: [(i64, i64); 2],
    /// Output height and width the plan reserved.
    pub out_hw: [usize; 2],
}

impl Conv {
    /// Resolve and validate `Conv` from the plan's static operand shapes.
    pub(crate) fn new(
        x: &Shape,
        w: &Shape,
        out: &Shape,
        params: &OpParams,
    ) -> Result<Conv, Fn3Error> {
        let op = OpId::Conv;
        if x.rank() != 4 || w.rank() != 4 || out.rank() != 4 {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN convolution",
                detail: format!(
                    "op {op} is claimed for 4-D NCHW/NCHW/NCHW operands only; found input \
                     {x}, weight {w}, output {out}"
                ),
            });
        }
        let x_dims = dims(x, "oneDNN convolution input")?;
        let w_dims = dims(w, "oneDNN convolution weight")?;
        let out_dims = dims(out, "oneDNN convolution output")?;
        let (kernel, strides, dilations) = window_axes(op, params, 2)?;

        let group = match params.group {
            0 | 1 => 1usize,
            positive if positive > 0 => {
                usize::try_from(positive).map_err(|_| Fn3Error::InvalidShape {
                    context: "oneDNN convolution",
                    detail: format!("op {op} group {positive} is too large"),
                })?
            }
            negative => {
                return Err(Fn3Error::InvalidShape {
                    context: "oneDNN convolution",
                    detail: format!("op {op} group {negative} is negative"),
                })
            }
        };
        let (batch, in_channels) = (x_dims[0], x_dims[1]);
        let out_channels = w_dims[0];
        let in_per_group = w_dims[1];
        if out_dims[0] != batch || out_dims[1] != out_channels {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("an output [{batch}, {out_channels}, ...]"),
                actual: format!("{out}"),
            });
        }
        if group == 0
            || in_channels % group != 0
            || out_channels % group != 0
            || in_per_group != in_channels / group
        {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN convolution",
                detail: format!(
                    "op {op} needs {in_channels} input channels = group {group} * the weight's \
                     {in_per_group} input channels, and {out_channels} output channels divisible \
                     by {group}"
                ),
            });
        }
        for axis in 0..2 {
            if w_dims[axis + 2] != kernel[axis] {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("the weight's kernel extent {} on axis {axis}", kernel[axis]),
                    actual: format!("{}", w_dims[axis + 2]),
                });
            }
        }
        let in_hw = [x_dims[2], x_dims[3]];
        let pads = resolve_pads(op, params, &in_hw, &kernel, &strides, &dilations)?;
        let mut out_hw = [0usize; 2];
        for axis in 0..2 {
            let (begin, end) = pads[axis];
            let computed = window_output(
                op,
                in_hw[axis],
                kernel[axis],
                strides[axis],
                dilations[axis],
                begin,
                end,
                false,
            )?;
            if computed != out_dims[axis + 2] {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!(
                        "the planned output extent {} on spatial axis {axis}",
                        out_dims[axis + 2]
                    ),
                    actual: format!("the kernel's rule gives {computed}"),
                });
            }
            out_hw[axis] = computed;
        }
        if batch == 0
            || in_channels == 0
            || out_channels == 0
            || in_hw.contains(&0)
            || out_hw.contains(&0)
        {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN convolution",
                detail: format!(
                    "op {op} has a zero extent (input {x}, weight {w}, output {out}); an empty \
                     convolution is left to the scalar kernels"
                ),
            });
        }
        Ok(Conv {
            batch,
            in_channels,
            in_hw,
            out_channels,
            in_per_group,
            group,
            kernel: [kernel[0], kernel[1]],
            strides: [strides[0], strides[1]],
            dilations: [dilations[0], dilations[1]],
            pads: [(pads[0].0, pads[0].1), (pads[1].0, pads[1].1)],
            out_hw,
        })
    }

    /// Output elements one block writes for `channels` output channels.
    pub(crate) fn block_elements(&self, channels: usize) -> usize {
        self.batch * channels * self.out_hw[0] * self.out_hw[1]
    }
}

/// A 2-D pooling window, resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Pool {
    /// Batch extent.
    pub batch: usize,
    /// Channels (equal in and out).
    pub channels: usize,
    /// Input height and width.
    pub in_hw: [usize; 2],
    /// Output height and width the plan reserved.
    pub out_hw: [usize; 2],
    /// Kernel height and width.
    pub kernel: [usize; 2],
    /// Stride per spatial axis.
    pub strides: [usize; 2],
    /// Padding per spatial axis `(begin, end)`.
    pub pads: [(i64, i64); 2],
    /// `true` for `GlobalAveragePool`, which is average pooling over the whole
    /// input plane.
    pub global: bool,
    /// `true` when padded cells count in an average denominator.
    pub count_include_pad: bool,
}

impl Pool {
    /// Resolve and validate one of the three pooling ids.
    pub(crate) fn new(
        op: OpId,
        x: &Shape,
        out: &Shape,
        params: &OpParams,
    ) -> Result<Pool, Fn3Error> {
        if x.rank() != 4 || out.rank() != 4 {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN pooling",
                detail: format!(
                    "op {op} is claimed for 4-D NCHW operands only; found input {x}, output {out}"
                ),
            });
        }
        let x_dims = dims(x, "oneDNN pooling input")?;
        let out_dims = dims(out, "oneDNN pooling output")?;
        if op == OpId::GlobalAveragePool {
            let kernel = [x_dims[2], x_dims[3]];
            for axis in 0..2 {
                if out_dims[axis + 2] != 1 {
                    return Err(Fn3Error::ShapeMismatch {
                        op: Some(op),
                        expected: "a 1x1 pooled plane".to_string(),
                        actual: format!("output {out}"),
                    });
                }
            }
            if x_dims.contains(&0) {
                return Err(Fn3Error::InvalidShape {
                    context: "oneDNN pooling",
                    detail: format!("op {op} has a zero extent in {x}"),
                });
            }
            return Ok(Pool {
                batch: x_dims[0],
                channels: x_dims[1],
                in_hw: [x_dims[2], x_dims[3]],
                out_hw: [1, 1],
                kernel,
                strides: [1, 1],
                pads: [(0, 0), (0, 0)],
                global: true,
                count_include_pad: false,
            });
        }
        if params.ceil_mode {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN pooling",
                detail: format!(
                    "op {op} sets ceil_mode = 1; oneDNN pooling has no ceil rule, so the \
                     segment stays scalar"
                ),
            });
        }
        let (kernel, strides, dilations) = window_axes(op, params, 2)?;
        if dilations != [1, 1] {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN pooling",
                detail: format!(
                    "op {op} has dilations {dilations:?}; the oneDNN engine claims the \
                     undilated pool forms only"
                ),
            });
        }
        let in_hw = [x_dims[2], x_dims[3]];
        let pads = resolve_pads(op, params, &in_hw, &kernel, &strides, &dilations)?;
        let mut out_hw = [0usize; 2];
        for axis in 0..2 {
            let (begin, end) = pads[axis];
            let computed = window_output(
                op,
                in_hw[axis],
                kernel[axis],
                strides[axis],
                dilations[axis],
                begin,
                end,
                params.ceil_mode,
            )?;
            if computed != out_dims[axis + 2] {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!(
                        "the planned output extent {} on spatial axis {axis}",
                        out_dims[axis + 2]
                    ),
                    actual: format!("the kernel's rule gives {computed}"),
                });
            }
            out_hw[axis] = computed;
        }
        if x_dims.contains(&0) || out_dims.contains(&0) {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN pooling",
                detail: format!("op {op} has a zero extent (input {x}, output {out})"),
            });
        }
        Ok(Pool {
            batch: x_dims[0],
            channels: x_dims[1],
            in_hw,
            out_hw,
            kernel: [kernel[0], kernel[1]],
            strides: [strides[0], strides[1]],
            pads: [(pads[0].0, pads[0].1), (pads[1].0, pads[1].1)],
            global: false,
            count_include_pad: params.count_include_pad,
        })
    }

    /// Output elements one block writes for `channels` channels.
    pub(crate) fn block_elements(&self, channels: usize) -> usize {
        self.batch * channels * self.out_hw[0] * self.out_hw[1]
    }
}

/// A 2-D matrix multiply: `[m, k] x [k, n] -> [m, n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMul {
    /// Rows of the left operand and of the result.
    pub m: usize,
    /// Shared reduction extent.
    pub k: usize,
    /// Columns of the right operand and of the result.
    pub n: usize,
}

impl MatMul {
    /// Resolve and validate a 2-D `MatMul`.
    pub(crate) fn new(a: &Shape, b: &Shape, out: &Shape) -> Result<MatMul, Fn3Error> {
        let op = OpId::MatMul;
        if a.rank() != 2 || b.rank() != 2 || out.rank() != 2 {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN matmul",
                detail: format!(
                    "op {op} is claimed for 2-D operands only; found {a} x {b} -> {out}"
                ),
            });
        }
        let a_dims = dims(a, "oneDNN matmul left")?;
        let b_dims = dims(b, "oneDNN matmul right")?;
        let out_dims = dims(out, "oneDNN matmul output")?;
        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        if k != k2 || out_dims != [m, n] {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("[{m}, {k2}] -> [{m}, {}]", b_dims[1]),
                actual: format!("{a} x {b} -> {out}"),
            });
        }
        if m == 0 || k == 0 || n == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN matmul",
                detail: format!("op {op} has a zero extent ({a} x {b} -> {out})"),
            });
        }
        Ok(MatMul { m, k, n })
    }

    /// Output elements one block writes for `columns` columns.
    pub(crate) fn block_elements(&self, columns: usize) -> usize {
        self.m * columns
    }
}

/// A contiguous range of convolution output channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvBlock {
    /// First group this block covers.
    pub group_start: usize,
    /// Whole groups in the block.
    pub groups: usize,
    /// First output channel (in the parent `[O, ...]` layout).
    pub out_start: usize,
    /// Output channels in the block.
    pub out_channels: usize,
    /// First input channel the block reads.
    pub in_start: usize,
    /// Input channels the block reads.
    pub in_channels: usize,
}

/// Split `units` work units into at most `workers` contiguous, evenly sized
/// groups and return `(start, count)` per block.
///
/// `units` counts indivisible pieces (channels for a plain convolution, groups
/// for a grouped one) and `per_unit` says how many output channels one piece
/// writes. The result is deterministic: the same `(units, per_unit,
/// block_elements, workers)` always gives the same split, irrespective of the
/// host, because the only host input is `workers`, which the plan records
/// through the session's thread count.
fn split_units(
    units: usize,
    per_unit: usize,
    block_elements: impl Fn(usize) -> usize,
    workers: usize,
) -> Vec<(usize, usize)> {
    debug_assert!(units > 0 && per_unit > 0);
    let by_channels = MIN_BLOCK_CHANNELS.div_ceil(per_unit).max(1);
    // The minimum unit count that keeps one block at or above the payload
    // floor. Computed by a small search because `block_elements` is not
    // necessarily affine in the unit count (batching multiplies it).
    let mut by_payload = 1usize;
    while by_payload < units && block_elements(by_payload * per_unit) < MIN_BLOCK_ELEMENTS {
        by_payload += 1;
    }
    let min_units = by_channels.max(by_payload).min(units);
    let blocks = workers.max(1).min(units / min_units).max(1);
    let base = units / blocks;
    let remainder = units % blocks;
    let mut result = Vec::with_capacity(blocks);
    let mut start = 0usize;
    for index in 0..blocks {
        let count = base + usize::from(index < remainder);
        result.push((start, count));
        start += count;
    }
    result
}

impl Conv {
    /// The block split for this convolution and this session's worker count.
    ///
    /// A **plain** convolution splits its output channels directly. A
    /// **grouped** one splits whole groups, because the weights of one group
    /// only multiply that group's input channels; a boundary inside a group
    /// would make each block a smaller one-group convolution with a partial
    /// weight row, which oneDNN cannot express as a submemory of the ONNX
    /// `[O, C/g, kh, kw]` payload. Depthwise convolutions (`C_out/g ==
    /// 1`) therefore still split one channel per block.
    pub(crate) fn blocks(&self, workers: usize) -> Vec<ConvBlock> {
        if self.group == 1 {
            let split = split_units(
                self.out_channels,
                1,
                |channels| self.block_elements(channels),
                workers,
            );
            return split
                .into_iter()
                .map(|(out_start, out_channels)| ConvBlock {
                    group_start: 0,
                    groups: 1,
                    out_start,
                    out_channels,
                    in_start: 0,
                    in_channels: self.in_channels,
                })
                .collect();
        }
        let per_group = self.out_channels / self.group;
        let split = split_units(
            self.group,
            per_group,
            |groups| self.block_elements(groups * per_group),
            workers,
        );
        split
            .into_iter()
            .map(|(group_start, groups)| ConvBlock {
                group_start,
                groups,
                out_start: group_start * per_group,
                out_channels: groups * per_group,
                in_start: group_start * self.in_per_group,
                in_channels: groups * self.in_per_group,
            })
            .collect()
    }
}

impl Pool {
    /// The block split for this pooling op.
    pub(crate) fn blocks(&self, workers: usize) -> Vec<(usize, usize)> {
        split_units(
            self.channels,
            1,
            |channels| self.block_elements(channels),
            workers,
        )
    }
}

impl MatMul {
    /// The block split, along the result columns.
    ///
    /// Splitting `n` keeps the left operand whole, so its memory descriptor is
    /// the plain one; the right operand and the result become column
    /// submemories. The alternative (splitting `m`) would make the right
    /// operand whole and the left a row submemory; both are expressible, and
    /// columns keep the prepacked weight's inner stride intact.
    pub(crate) fn blocks(&self, workers: usize) -> Vec<(usize, usize)> {
        split_units(self.n, 1, |columns| self.block_elements(columns), workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> OpParams {
        OpParams::new()
            .with_kernel_shape([3, 3])
            .with_strides([1, 1])
            .with_pads([1, 1, 1, 1])
            .with_dilations([1, 1])
    }

    #[test]
    fn a_plain_convolution_splits_channels_evenly_and_deterministically() {
        let conv = Conv::new(
            &Shape::from_dims([1, 64, 40, 40]),
            &Shape::from_dims([64, 64, 3, 3]),
            &Shape::from_dims([1, 64, 40, 40]),
            &params(),
        )
        .unwrap();
        // 64 channels of 40x40 is 102 Ki elements; the 16 Ki payload floor
        // keeps one block to at least 11 channels, so eight workers get five
        // blocks rather than eight.
        let blocks = conv.blocks(8);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks.iter().map(|b| b.out_channels).sum::<usize>(), 64);
        assert_eq!(
            blocks.iter().map(|b| b.out_channels).collect::<Vec<_>>(),
            vec![13, 13, 13, 13, 12]
        );
        assert_eq!(blocks[3].out_start, 39);
        assert_eq!(conv.blocks(8), blocks);
        // One worker is one block.
        assert_eq!(conv.blocks(1).len(), 1);
        assert_eq!(conv.blocks(1)[0].out_channels, 64);
    }

    #[test]
    fn a_grouped_convolution_keeps_block_boundaries_on_groups() {
        let mut p = params();
        p.group = 4;
        let conv = Conv::new(
            &Shape::from_dims([1, 64, 40, 40]),
            &Shape::from_dims([64, 16, 3, 3]),
            &Shape::from_dims([1, 64, 40, 40]),
            &p,
        )
        .unwrap();
        let blocks = conv.blocks(8);
        assert_eq!(blocks.len(), 4, "one block per group at most");
        for (index, block) in blocks.iter().enumerate() {
            assert_eq!(block.out_start, index * 16);
            assert_eq!(block.out_channels, 16);
            assert_eq!(block.groups, 1);
            assert_eq!(block.in_start, index * 16);
            assert_eq!(block.in_channels, 16);
        }
    }

    #[test]
    fn a_depthwise_convolution_splits_one_channel_per_block() {
        let mut p = params();
        p.group = 64;
        let conv = Conv::new(
            &Shape::from_dims([1, 64, 32, 32]),
            &Shape::from_dims([64, 1, 3, 3]),
            &Shape::from_dims([1, 64, 32, 32]),
            &p,
        )
        .unwrap();
        // One group per channel, 32x32 each: the payload floor wants 16
        // channels per block, so eight workers get four blocks.
        let blocks = conv.blocks(8);
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks.iter().map(|b| b.out_channels).sum::<usize>(), 64);
        assert!(blocks.iter().all(|b| b.out_channels == 16));
        assert_eq!(blocks[1].group_start, 16);
    }

    #[test]
    fn a_small_problem_stays_one_block() {
        let mut p = OpParams::new()
            .with_kernel_shape([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]);
        p.group = 1;
        let conv = Conv::new(
            &Shape::from_dims([1, 4, 2, 2]),
            &Shape::from_dims([4, 4, 1, 1]),
            &Shape::from_dims([1, 4, 2, 2]),
            &p,
        )
        .unwrap();
        assert_eq!(conv.blocks(8).len(), 1);
    }

    #[test]
    fn ceil_mode_and_batched_matmul_fall_back() {
        let mut p = OpParams::new()
            .with_kernel_shape([2, 2])
            .with_strides([2, 2])
            .with_ceil_mode(true);
        let error = Pool::new(
            OpId::MaxPool,
            &Shape::from_dims([1, 8, 5, 5]),
            &Shape::from_dims([1, 8, 3, 3]),
            &p,
        )
        .unwrap_err();
        assert!(error.to_string().contains("ceil_mode"), "{error}");
        let error = MatMul::new(
            &Shape::from_dims([2, 3, 4]),
            &Shape::from_dims([2, 4, 5]),
            &Shape::from_dims([2, 3, 5]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("2-D"), "{error}");
    }

    #[test]
    fn global_average_pooling_is_a_full_window_average() {
        let pool = Pool::new(
            OpId::GlobalAveragePool,
            &Shape::from_dims([1, 7, 5, 9]),
            &Shape::from_dims([1, 7, 1, 1]),
            &OpParams::new(),
        )
        .unwrap();
        assert!(pool.global);
        assert_eq!(pool.kernel, [5, 9]);
        assert_eq!(pool.out_hw, [1, 1]);
        assert_eq!(
            pool.blocks(8).len(),
            1,
            "7 channels and 7 elements is small"
        );
    }
}
