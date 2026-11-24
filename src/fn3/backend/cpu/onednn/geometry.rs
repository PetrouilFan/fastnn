//! Pure segment geometry for the oneDNN engine.
//!
//! The same functions answer two questions, and that is deliberate:
//!
//! - the **planner** asks whether a segment is expressible in oneDNN's
//!   vocabulary (2-D NCHW convolution, 2-D matmul, `f32`);
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
/// 4 Ki elements is 16 KiB of output: a block that writes less than that does
/// not pay back a task hand-off, an `execute` and a `dnnl_stream_wait`.
///
/// The phase-3a value was **16 Ki elements**, chosen on the reasoning that a
/// block should have one cache line per worker on a 16-core host. Measured, it
/// was leaving the machine idle: yolo11n and yolov8n at 320x320 are graphs of
/// *small* layers, and 16 Ki elements per block means a 64-channel 20x20
/// convolution (25 600 outputs) is **one block**, i.e. one worker while seven
/// wait. The median convolution in the yolov8n plan had one block at eight
/// workers, and the plan reported 146 blocks for 64 convolutions.
///
/// This floor was picked from a sweep on the Ryzen 7 3700X, eight workers,
/// yolov8n at `[1,3,320,320]`, five warm-ups and thirty forwards, best of three
/// interleaved rounds (the host is shared, so the minimum is the estimate):
///
/// | floor | blocks per conv (min/median/max) | wall | peak RSS |
/// |---|---|---|---|
/// | 16 Ki elements | 1 / 1 / 7 | 31.68 ms | 105 MB |
/// | **4 Ki elements** | 1 / 4 / 8 | **20.08 ms** | 125 MB |
/// | 2 Ki elements | 1 / 4 / 8 | 20.5 ms | 164 MB |
/// | 1 Ki element | 1 / 4 / 8 | 20.6 ms | 174 MB |
///
/// 4 Ki is where the wall time stops improving; below it the extra blocks only
/// cost staging buffers. The split stays inside the <=2x-of-ORT goal at one
/// worker (1.86x) and narrows the eight-worker gap from 4.1x to 2.6x; the
/// remaining gap is convolution efficiency, not block count — see
/// `docs/v3/backends.md`.
pub(crate) const MIN_BLOCK_ELEMENTS: usize = 4 * 1024;

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

/// A contiguous range of convolution output.
///
/// A block is split along exactly one dimension, recorded in
/// [`ConvSplit::axis`]:
///
/// - [`ConvSplitAxis::Channels`]: `out_start..out_start + out_channels` and,
///   for a grouped convolution, `in_start..in_start + in_channels`; the row
///   fields describe the whole output.
/// - [`ConvSplitAxis::Rows`]: the whole channel range, output rows
///   `out_row_start..out_row_start + out_rows`, and the input rows
///   `in_row_start..in_row_start + in_rows` that those output rows need,
///   including the halo but excluding rows outside the input (which
///   `pad_top`/`pad_bottom` let the convolution's own padding supply).
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
    /// First output row of the block.
    pub out_row_start: usize,
    /// Output rows the block writes.
    pub out_rows: usize,
    /// First input row of the block's source submemory.
    pub in_row_start: usize,
    /// Input rows in the block's source submemory.
    pub in_rows: usize,
    /// Padding rows oneDNN must add before `in_row_start` (a first block that
    /// begins inside the parent's top padding).
    pub pad_top: usize,
    /// Padding rows oneDNN must add after the submemory (a last block that
    /// ends inside the parent's bottom padding).
    pub pad_bottom: usize,
}

/// The dimension a convolution segment is split along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvSplitAxis {
    /// Contiguous output-channel ranges (the phase-3a split).
    Channels,
    /// Contiguous output-row ranges with an input halo (phase 3b).
    Rows,
}

/// A convolution's complete block plan for one worker count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConvSplit {
    /// Which dimension the blocks divide.
    pub axis: ConvSplitAxis,
    /// The blocks, in ascending output order.
    pub blocks: Vec<ConvBlock>,
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
    /// A plain convolution can divide its output two ways:
    ///
    /// - **channels**, the phase-3a split: each block reads the full input (a
    ///   plain convolution has no channel submemory), so `B` blocks move `B`
    ///   times the input bytes. Correct, but on a memory-bound graph eight
    ///   blocks scale like one.
    /// - **rows with a halo**, phase 3b: each block reads only the input rows
    ///   its output rows need. Consecutive blocks overlap by the kernel's
    ///   halo, so the input moves about `(H + halo) / H` times instead of `B`
    ///   times. A block near the top or bottom of the input is clamped to the
    ///   real memory and the missing rows are supplied by oneDNN's
    ///   convolution padding (`pad_top`/`pad_bottom`), so no block ever reads
    ///   outside the source slot.
    ///
    /// The choice is deterministic from `(shape, params, workers)`:
    ///
    /// 1. split channels when row splitting would produce fewer blocks than the
    ///    channel split (parallelism is never traded away);
    /// 2. otherwise split rows when their estimated input traffic is lower.
    ///
    /// A grouped convolution's channel blocks read only their own channels, so
    /// the traffic estimate keeps it on channels; a depthwise convolution
    /// therefore never pays for a halo it does not need. A one-worker session
    /// gets one block either way.
    pub(crate) fn split(&self, workers: usize) -> ConvSplit {
        let channels = self.channel_blocks(workers);
        let rows = self.row_blocks(workers);
        if self.prefer_rows(&channels, &rows) {
            ConvSplit {
                axis: ConvSplitAxis::Rows,
                blocks: rows,
            }
        } else {
            ConvSplit {
                axis: ConvSplitAxis::Channels,
                blocks: channels,
            }
        }
    }

    /// The row split alone, for a caller that needs dense row chunks (a
    /// channels-last destination is dense per row range, a channel range is
    /// not).
    pub(crate) fn row_split(&self, workers: usize) -> ConvSplit {
        ConvSplit {
            axis: ConvSplitAxis::Rows,
            blocks: self.row_blocks(workers),
        }
    }

    /// The channel split alone, for a caller that cannot use rows.
    ///
    /// A fused `sum` needs the destination pre-filled with the residual, and a
    /// row block's destination is a dense staging buffer rather than a range of
    /// the output slot; the channel split is the one whose destination
    /// submemory sees the pre-filled bytes. It is also the conservative split
    /// for a chain of blocked layouts, where a row submemory of a blocked
    /// parent takes oneDNN off its JIT path exactly as it did for NCHW.
    pub(crate) fn channel_split(&self, workers: usize) -> ConvSplit {
        ConvSplit {
            axis: ConvSplitAxis::Channels,
            blocks: self.channel_blocks(workers),
        }
    }

    fn channel_blocks(&self, workers: usize) -> Vec<ConvBlock> {
        let whole_output =
            |out_start, out_channels, group_start, groups, in_start, in_channels| ConvBlock {
                group_start,
                groups,
                out_start,
                out_channels,
                in_start,
                in_channels,
                out_row_start: 0,
                out_rows: self.out_hw[0],
                in_row_start: 0,
                in_rows: self.in_hw[0],
                pad_top: 0,
                pad_bottom: 0,
            };
        if self.group == 1 {
            let split = split_units(
                self.out_channels,
                1,
                |channels| self.block_elements(channels),
                workers,
            );
            return split
                .into_iter()
                .map(|(out_start, out_channels)| {
                    whole_output(out_start, out_channels, 0, 1, 0, self.in_channels)
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
            .map(|(group_start, groups)| {
                whole_output(
                    group_start * per_group,
                    groups * per_group,
                    group_start,
                    groups,
                    group_start * self.in_per_group,
                    groups * self.in_per_group,
                )
            })
            .collect()
    }

    /// The row split: at most `workers` contiguous output-row ranges, each with
    /// the input rows its dilated windows touch.
    fn row_blocks(&self, workers: usize) -> Vec<ConvBlock> {
        let out_rows_total = self.out_hw[0];
        if out_rows_total == 0 {
            return Vec::new();
        }
        let row_elements = self.batch * self.out_channels * self.out_hw[1];
        let payload_rows = MIN_BLOCK_ELEMENTS.div_ceil(row_elements.max(1)).max(1);
        // A block that covers less than one dilated kernel span would read more
        // halo than output. That is a floor on rows, not a size tuning knob.
        let kernel_rows = ((self.kernel[0] - 1) * self.dilations[0]).div_ceil(self.strides[0]) + 1;
        let min_rows = payload_rows.max(kernel_rows).min(out_rows_total);
        let blocks = workers.max(1).min(out_rows_total / min_rows).max(1);
        let base = out_rows_total / blocks;
        let remainder = out_rows_total % blocks;
        let stride = self.strides[0] as i64;
        let dilation = self.dilations[0] as i64;
        let kernel = self.kernel[0] as i64;
        let pad_top = self.pads[0].0;
        let in_rows_total = self.in_hw[0] as i64;
        let mut result = Vec::with_capacity(blocks);
        let mut out_row_start = 0usize;
        for index in 0..blocks {
            let out_rows = base + usize::from(index < remainder);
            // The parent convolution reads input row `r * stride - pad_top`
            // through `r * stride - pad_top + (kh - 1) * dilation` for output
            // row `r`; a block needs the union over its own rows.
            let ideal_start = out_row_start as i64 * stride - pad_top;
            let ideal_end =
                (out_row_start + out_rows - 1) as i64 * stride + (kernel - 1) * dilation + 1
                    - pad_top;
            let row_start = ideal_start.max(0);
            let row_end = ideal_end.min(in_rows_total);
            debug_assert!(row_end > row_start, "an existing output row has a window");
            result.push(ConvBlock {
                group_start: 0,
                groups: self.group,
                out_start: 0,
                out_channels: self.out_channels,
                in_start: 0,
                in_channels: self.in_channels,
                out_row_start,
                out_rows,
                in_row_start: row_start as usize,
                in_rows: (row_end - row_start) as usize,
                // Only the padding outside the real input is handed back to
                // oneDNN; the submemory starts at a real row.
                pad_top: (-ideal_start).max(0) as usize,
                pad_bottom: (ideal_end - in_rows_total).max(0) as usize,
            });
            out_row_start += out_rows;
        }
        result
    }

    /// `true` when the row split moves less input and keeps the parallelism.
    fn prefer_rows(&self, channels: &[ConvBlock], rows: &[ConvBlock]) -> bool {
        if rows.len() <= 1 || rows.len() < channels.len() {
            return false;
        }
        let channel_cells: usize = channels
            .iter()
            .map(|block| block.in_channels * self.in_hw[0] * self.in_hw[1] * self.batch)
            .sum();
        let row_cells: usize = rows
            .iter()
            .map(|block| block.in_channels * block.in_rows * self.in_hw[1] * self.batch)
            .sum();
        row_cells < channel_cells
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
    fn a_plain_convolution_prefers_rows_when_channels_do_not_fill_the_pool() {
        let conv = Conv::new(
            &Shape::from_dims([1, 64, 40, 40]),
            &Shape::from_dims([64, 64, 3, 3]),
            &Shape::from_dims([1, 64, 40, 40]),
            &params(),
        )
        .unwrap();
        // 64 channels of 40x40 is 102 Ki elements, so at the 4 Ki payload floor
        // both splits fill eight workers: eight channel blocks of eight
        // channels, or eight row blocks of five output rows. The row split wins
        // because it reads 54 input rows instead of 8 x 40.
        let split = conv.split(8);
        assert_eq!(split.axis, ConvSplitAxis::Rows);
        assert_eq!(split.blocks.len(), 8);
        assert_eq!(
            split.blocks.iter().map(|b| b.out_rows).collect::<Vec<_>>(),
            vec![5; 8]
        );
        assert_eq!(
            split.blocks.iter().map(|b| b.in_rows).sum::<usize>(),
            54,
            "five output rows need seven input rows, and each edge block one fewer"
        );
        assert_eq!(split.blocks[0].pad_top, 1);
        assert_eq!(split.blocks[0].in_row_start, 0);
        assert_eq!(split.blocks[7].pad_bottom, 1);
        for block in &split.blocks {
            assert_eq!(block.out_channels, 64);
            assert_eq!(block.in_channels, 64);
            assert!(block.in_row_start + block.in_rows <= 40);
        }
        // Deterministic for the same thread count.
        assert_eq!(conv.split(8), split);
        // One worker is one channel block, never a row split.
        let one = conv.split(1);
        assert_eq!(one.axis, ConvSplitAxis::Channels);
        assert_eq!(one.blocks.len(), 1);
        assert_eq!(one.blocks[0].out_channels, 64);
    }

    #[test]
    fn a_first_convolution_splits_rows_to_fill_the_pool() {
        // YOLOv8n's first convolution: 3 -> 16 channels, 320x320, stride 2. The
        // channel split cannot make more than two blocks (the 16 Ki payload
        // floor wants eight channels), so rows are the only way to use eight
        // workers.
        let conv = Conv::new(
            &Shape::from_dims([1, 3, 320, 320]),
            &Shape::from_dims([16, 3, 3, 3]),
            &Shape::from_dims([1, 16, 160, 160]),
            &OpParams::new()
                .with_kernel_shape([3, 3])
                .with_strides([2, 2])
                .with_pads([1, 1, 1, 1])
                .with_dilations([1, 1]),
        )
        .unwrap();
        let split = conv.split(8);
        assert_eq!(split.axis, ConvSplitAxis::Rows);
        assert_eq!(split.blocks.len(), 8);
        assert_eq!(split.blocks.iter().map(|b| b.out_rows).sum::<usize>(), 160);
        let first = split.blocks.first().unwrap();
        assert_eq!(first.in_row_start, 0);
        assert_eq!(first.pad_top, 1, "the parent's top pad is one row");
        let last = split.blocks.last().unwrap();
        assert_eq!(
            last.pad_bottom, 0,
            "160 is even; the last window ends at 320"
        );
        assert!(last.in_row_start + last.in_rows <= 320);
        // The halo is read twice, not the full input eight times.
        let halo_rows: usize = split.blocks.iter().map(|b| b.in_rows).sum();
        assert!(halo_rows < 8 * 320, "{halo_rows}");
        assert!(halo_rows <= 320 + 8 * 2, "{halo_rows}");
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
        // A grouped block reads only its own input channels, so the channel
        // split already moves each input row once; rows would add halo traffic
        // for no bandwidth win.
        let split = conv.split(8);
        assert_eq!(split.axis, ConvSplitAxis::Channels);
        assert_eq!(split.blocks.len(), 4, "one block per group at most");
        for (index, block) in split.blocks.iter().enumerate() {
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
        // One group per channel, 32x32 each: eight workers get eight blocks of
        // eight channels, and rows are rejected because a channel block already
        // reads only its own input channels (a halo would add traffic).
        let split = conv.split(8);
        assert_eq!(split.axis, ConvSplitAxis::Channels);
        assert_eq!(split.blocks.len(), 8);
        assert_eq!(
            split.blocks.iter().map(|b| b.out_channels).sum::<usize>(),
            64
        );
        assert!(split.blocks.iter().all(|b| b.out_channels == 8));
        assert_eq!(split.blocks[1].group_start, 8);
    }

    #[test]
    fn a_small_problem_stays_one_block() {
        let p = OpParams::new()
            .with_kernel_shape([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]);
        let conv = Conv::new(
            &Shape::from_dims([1, 4, 2, 2]),
            &Shape::from_dims([4, 4, 1, 1]),
            &Shape::from_dims([1, 4, 2, 2]),
            &p,
        )
        .unwrap();
        let split = conv.split(8);
        assert_eq!(split.axis, ConvSplitAxis::Channels);
        assert_eq!(split.blocks.len(), 1);
    }

    #[test]
    fn batched_matmul_falls_back() {
        let error = MatMul::new(
            &Shape::from_dims([2, 3, 4]),
            &Shape::from_dims([2, 4, 5]),
            &Shape::from_dims([2, 3, 5]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("2-D"), "{error}");
    }
}
