//! Packing FP32 projection weights into the locked W8A16 representation.
//!
//! The container in [`crate::inference::quant`] is produced by the independent Python
//! encoder in `tools/quant/fn3_quant.py`. A CPU backend that wants a packed
//! weight has to run the same policy; this module is the Rust implementation of
//! `docs/quantization/fn3-quant-format-v1.md` §4.5 for a weight already in
//! memory:
//!
//! ```text
//! span       = max - min
//! asym_scale = span / 255
//! asym_zero  = -128 - round_half_away(min / asym_scale)
//!
//! if the group is constant:
//!     scale = (value == 0) ? 1.0 : max(|value| / 127, 1e-6); zero = 0
//! elif -128 <= asym_zero <= 127:
//!     scale = asym_scale; zero = asym_zero
//! else:
//!     scale = max(|min|, |max|) / 127; zero = 0
//!
//! scale is rounded to binary16 first; codes are computed against the widened
//! binary16 value with round-half-away and clipped to [-128, 127].
//! ```
//!
//! # Orientation
//!
//! GPT-2 exports `Gemm(A, W)` with `W` shaped `[K, N]` (the ONNX `MatMul`
//! right-hand side), while the format packs rows along the *output* axis:
//! `W8A16[N, K]`. `pack_f32_rhs` therefore consumes the `[K, N]` source
//! exactly as the graph carries it and writes the transposed, grouped layout in
//! one pass. No intermediate `[N, K]` FP32 copy is created.
//!
//! # What is stored
//!
//! A [`PackedWeight`] owns three byte vectors: `n_rows * k` signed codes, then
//! `n_rows * groups_per_row` little-endian binary16 scales, then the same
//! number of little-endian int16 zero-points. That is byte-for-byte the payload
//! layout of the container, minus the container's per-row alignment padding
//! (which exists so a file's rows start on 16-byte boundaries; an in-memory
//! `Vec<u8>` needs no such padding). [`PackedWeight::format_bytes`] reports the
//! padded container size for storage comparisons.
//!
//! # Validation
//!
//! Every non-finite input value, an f16 scale that rounds to zero or infinity,
//! and a shape whose byte counts overflow are reported as `Fn3Error` before a
//! packed weight is returned. The encoder never panics.

use half::f16;

use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::quant::GROUP_SIZE;
use crate::inference::tensor::{Tensor, TensorView};

use super::kernel::{groups_per_row, PackedView};

/// Smallest K the quantize policy accepts (one full group).
pub const MIN_K: usize = GROUP_SIZE;
/// Smallest element count the quantize policy accepts.
pub const MIN_NUMEL: u64 = 4096;

/// Accuracy and storage evidence produced while packing one weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackingReport {
    /// Elements in the source weight.
    pub elements: u64,
    /// Source size in bytes at FP32.
    pub fp32_bytes: u64,
    /// Code bytes written.
    pub code_bytes: u64,
    /// Scale + zero-point bytes written.
    pub metadata_bytes: u64,
    /// Groups across every row.
    pub groups: u64,
    /// Groups encoded asymmetrically.
    pub asymmetric_groups: u64,
    /// Groups encoded symmetrically (`zero_point == 0`).
    pub symmetric_groups: u64,
    /// Constant groups (also symmetric).
    pub constant_groups: u64,
    /// `||W - dequant(W)||_F / ||W||_F`, computed in `f64`.
    pub relative_frobenius: f64,
    /// Largest `|w - dequant(w)|`.
    pub max_abs_error: f32,
}

impl PackingReport {
    /// Bytes a resident packed weight occupies.
    pub fn packed_bytes(&self) -> u64 {
        self.code_bytes + self.metadata_bytes
    }

    /// Packed bytes as a fraction of FP32.
    pub fn storage_ratio(&self) -> f64 {
        if self.fp32_bytes == 0 {
            0.0
        } else {
            self.packed_bytes() as f64 / self.fp32_bytes as f64
        }
    }
}

/// An owned packed W8A16 weight.
#[derive(Debug, Clone)]
pub struct PackedWeight {
    name: String,
    n_rows: usize,
    k: usize,
    groups_per_row: usize,
    row_stride: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    zeros: Vec<u8>,
    source_shape: [u64; 2],
    report: PackingReport,
}

impl PackedWeight {
    /// Pack a row-major `[K, N]` FP32 right-hand side.
    ///
    /// `values.len()` must be `k * n`; `n` is the number of output rows and `k`
    /// the reduction length. The source is read with stride `n`, so no
    /// transposed copy is materialized.
    pub fn from_f32_rhs(
        name: impl Into<String>,
        k: usize,
        n: usize,
        values: &[f32],
    ) -> Result<PackedWeight, Fn3Error> {
        pack_f32_rhs(name, k, n, values)
    }

    /// Pack the right-hand side of one 2-D FP32 model weight.
    ///
    /// The tensor is the ONNX `MatMul` operand, `[K, N]`; it must be contiguous
    /// and on the host. A rank other than 2, another dtype, or a strided tensor
    /// is rejected rather than silently repacked.
    pub fn from_tensor(name: impl Into<String>, tensor: &Tensor) -> Result<PackedWeight, Fn3Error> {
        let name = name.into();
        if tensor.dtype() != DType::F32 {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(crate::inference::ops::OpId::MatMul),
                dtype: tensor.dtype(),
                context: "W8A16 packing needs an f32 weight",
            });
        }
        if tensor.shape().rank() != 2 {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 packing",
                detail: format!(
                    "weight {name:?} has rank {}, the format packs a 2-D MatMul right-hand side",
                    tensor.shape().rank()
                ),
            });
        }
        let dims = tensor.shape().dims();
        let k = usize::try_from(dims[0]).map_err(|_| Fn3Error::ShapeOverflow {
            context: "W8A16 packing k",
            axis: Some(0),
            detail: format!("k {} does not fit in usize", dims[0]),
        })?;
        let n = usize::try_from(dims[1]).map_err(|_| Fn3Error::ShapeOverflow {
            context: "W8A16 packing n",
            axis: Some(1),
            detail: format!("n {} does not fit in usize", dims[1]),
        })?;
        let view: TensorView<'_> = tensor.as_view()?;
        let values = view.as_f32_slice()?;
        pack_f32_rhs(name, k, n, values)
    }

    /// Borrow the packed bytes as a kernel view.
    pub fn view(&self) -> PackedView<'_> {
        PackedView {
            n_rows: self.n_rows,
            k: self.k,
            groups_per_row: self.groups_per_row,
            row_stride: self.row_stride,
            codes: &self.codes,
            scales: &self.scales,
            zeros: &self.zeros,
        }
    }

    /// Tensor name the weight came from.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Output rows (N).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Reduction length (K).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Groups per row.
    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }

    /// Source shape `[K, N]`, the graph's right-hand side.
    pub fn source_shape(&self) -> [u64; 2] {
        self.source_shape
    }

    /// Packing evidence for this weight.
    pub fn report(&self) -> &PackingReport {
        &self.report
    }

    /// Container bytes including the per-row 16-byte alignment padding.
    pub fn format_bytes(&self) -> u64 {
        let row_stride = self.k.div_ceil(16) * 16;
        let codes = (self.n_rows * row_stride) as u64;
        codes + self.report.metadata_bytes
    }
}

fn pack_f32_rhs(
    name: impl Into<String>,
    k: usize,
    n: usize,
    values: &[f32],
) -> Result<PackedWeight, Fn3Error> {
    let name = name.into();
    if k == 0 || n == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "W8A16 packing",
            detail: format!("weight {name:?} is {n} x {k}; both extents must be positive"),
        });
    }
    let elements = k.checked_mul(n).ok_or_else(|| Fn3Error::ShapeOverflow {
        context: "W8A16 packing elements",
        axis: None,
        detail: format!("{n} rows x {k} columns"),
    })?;
    if values.len() != elements {
        return Err(Fn3Error::InvalidShape {
            context: "W8A16 packing",
            detail: format!(
                "weight {name:?} is declared [{k}, {n}] ({elements} values) but {} were supplied",
                values.len()
            ),
        });
    }

    let groups = groups_per_row(k);
    let mut codes = vec![0u8; elements];
    let mut scales = Vec::with_capacity(n * groups * 2);
    let mut zeros = Vec::with_capacity(n * groups * 2);
    let mut report = PackingReport {
        elements: elements as u64,
        fp32_bytes: (elements as u64) * 4,
        code_bytes: elements as u64,
        metadata_bytes: (n * groups * 2) as u64 * 2,
        groups: (n * groups) as u64,
        asymmetric_groups: 0,
        symmetric_groups: 0,
        constant_groups: 0,
        relative_frobenius: 0.0,
        max_abs_error: 0.0,
    };

    let mut squared_signal = 0.0f64;
    let mut squared_error = 0.0f64;

    for row in 0..n {
        let code_base = row * k;
        for group in 0..groups {
            let start = group * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(k);

            // The source is [K, N]; this row's group walks k with stride n.
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for column in start..end {
                let value = values[column * n + row];
                if !value.is_finite() {
                    return Err(Fn3Error::InvalidShape {
                        context: "W8A16 packing",
                        detail: format!(
                            "weight {name:?} row {row} column {column} is {value}; the format \
                             has no representation for a non-finite value"
                        ),
                    });
                }
                low = low.min(value);
                high = high.max(value);
            }

            let span = high - low;
            let degenerate = span == 0.0;
            let asym_scale = (span / 255.0).max(1e-30);
            let asym_zero = -128.0 - low / asym_scale;
            let use_asym = !degenerate && (-128.0..=127.0).contains(&asym_zero);
            let absolute = low.abs().max(high.abs());
            let sym_scale = if absolute == 0.0 {
                1.0
            } else {
                (absolute / 127.0).max(1e-6)
            };
            let scale = if use_asym { asym_scale } else { sym_scale };
            let scale_bits = f16::from_f32(scale);
            let scale = scale_bits.to_f32();
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Fn3Error::InvalidShape {
                    context: "W8A16 packing",
                    detail: format!(
                        "weight {name:?} row {row} group {group} produced scale {scale}, which \
                         is not a finite positive binary16 value"
                    ),
                });
            }
            let zero = if use_asym {
                round_half_away(-128.0 - low / scale).clamp(-128, 127) as i16
            } else {
                0
            };

            if use_asym {
                report.asymmetric_groups += 1;
            } else {
                report.symmetric_groups += 1;
                if degenerate {
                    report.constant_groups += 1;
                }
            }

            scales.extend_from_slice(&scale_bits.to_bits().to_le_bytes());
            zeros.extend_from_slice(&zero.to_le_bytes());

            for column in start..end {
                let value = values[column * n + row];
                let quantized = round_half_away(value / scale) + i64::from(zero);
                let code = quantized.clamp(-128, 127) as i8;
                codes[code_base + column] = code as u8;
                let dequantized = (i32::from(code) - i32::from(zero)) as f32 * scale;
                let error = f64::from(value) - f64::from(dequantized);
                squared_signal += f64::from(value) * f64::from(value);
                squared_error += error * error;
                let absolute_error = (value - dequantized).abs();
                if absolute_error > report.max_abs_error {
                    report.max_abs_error = absolute_error;
                }
            }
        }
    }

    report.relative_frobenius = if squared_signal == 0.0 {
        0.0
    } else {
        (squared_error / squared_signal).sqrt()
    };

    Ok(PackedWeight {
        name,
        n_rows: n,
        k,
        groups_per_row: groups,
        row_stride: k,
        codes,
        scales,
        zeros,
        source_shape: [k as u64, n as u64],
        report,
    })
}

/// Round to nearest, ties away from zero, matching the Python reference.
///
/// The ratio is an `f32` (exactly what `weights / scale_f16` produces); the
/// rounding itself is done in `f64`, as `tools/quant/fn3_quant.py` does, so a
/// tie cannot depend on single-precision addition.
fn round_half_away(value: f32) -> i64 {
    let value = f64::from(value);
    (value + 0.5f64.copysign(value)).trunc() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_away_matches_the_reference_rule() {
        assert_eq!(round_half_away(0.5), 1);
        assert_eq!(round_half_away(-0.5), -1);
        assert_eq!(round_half_away(1.4999), 1);
        assert_eq!(round_half_away(-1.5), -2);
        assert_eq!(round_half_away(2.5), 3);
    }

    #[test]
    fn a_tail_group_packs_the_short_final_range() {
        let k = 70;
        let n = 1;
        let values: Vec<f32> = (0..k).map(|column| column as f32 * 0.5 - 8.0).collect();
        let weight = pack_f32_rhs("tail", k, n, &values).unwrap();
        assert_eq!(weight.groups_per_row(), 2);
        assert_eq!(weight.view().group_range(0), 0..64);
        assert_eq!(weight.view().group_range(1), 64..70);
        assert_eq!(weight.report().elements, 70);
        assert!(weight.report().relative_frobenius < 0.01);
    }
}
