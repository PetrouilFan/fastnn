//! Canonical value, storage, and quantization representation types.
//!
//! These types deliberately separate mathematical scalar semantics from physical
//! packing and compiler policy. Backend kernel selection does not belong here.

use crate::{FastnnError, FastnnResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    F32,
    F64,
    F16,
    BF16,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
    I64,
    I32,
    I8,
    U8,
    I4,
    U4,
    Bool,
}

impl ScalarType {
    pub const fn bit_width(self) -> u8 {
        match self {
            Self::F64 | Self::I64 => 64,
            Self::F32 | Self::I32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::I8 | Self::U8 => 8,
            Self::Fp4E2M1 | Self::I4 | Self::U4 => 4,
            Self::Bool => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F64
                | Self::F16
                | Self::BF16
                | Self::Fp8E4M3
                | Self::Fp8E5M2
                | Self::Fp4E2M1
        )
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I64 | Self::I32 | Self::I8 | Self::U8 | Self::I4 | Self::U4
        )
    }

    pub const fn plain_byte_width(self) -> Option<usize> {
        match self {
            Self::Bool => Some(1),
            Self::Fp4E2M1 | Self::I4 | Self::U4 => None,
            _ => Some((self.bit_width() / 8) as usize),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageEncoding {
    Plain,
    Packed { word_bits: u8, lanes: u8 },
}

pub const PACKED_WORD_BYTES: usize = std::mem::size_of::<u32>();
pub const PACKED_SIMD_MARGIN_WORDS: usize = 16;
pub const PACKED_SIMD_MARGIN_BYTES: usize = PACKED_SIMD_MARGIN_WORDS * PACKED_WORD_BYTES;

/// Physical allocation rules layered on top of an encoded scalar payload.
///
/// Prefix/suffix bytes are allocation capacity, not logical tensor elements.
/// `row_packed` means each innermost row starts on a complete packed word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorStorageLayout {
    pub encoding: StorageEncoding,
    pub row_packed: bool,
    pub prefix_bytes: usize,
    pub suffix_bytes: usize,
}

impl TensorStorageLayout {
    pub const fn contiguous(encoding: StorageEncoding) -> Self {
        Self {
            encoding,
            row_packed: false,
            prefix_bytes: 0,
            suffix_bytes: 0,
        }
    }

    pub fn allocation_bytes(self, scalar: ScalarType, shape: &[usize]) -> FastnnResult<usize> {
        self.encoding.validate_for(scalar)?;
        if self.row_packed && matches!(self.encoding, StorageEncoding::Plain) {
            return Err(FastnnError::dtype(
                "row-packed allocation requires a packed storage encoding",
            ));
        }
        let numel = shape.iter().try_fold(1usize, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or_else(|| FastnnError::Overflow("tensor element count overflow".into()))
        })?;
        let payload_bytes = match (self.encoding, self.row_packed, shape.last().copied()) {
            (StorageEncoding::Packed { word_bits, lanes }, true, Some(inner)) => {
                let rows =
                    shape[..shape.len() - 1]
                        .iter()
                        .try_fold(1usize, |total, &dimension| {
                            total.checked_mul(dimension).ok_or_else(|| {
                                FastnnError::Overflow("tensor row count overflow".into())
                            })
                        })?;
                rows.checked_mul(inner.div_ceil(usize::from(lanes)))
                    .and_then(|words| words.checked_mul(usize::from(word_bits / 8)))
                    .ok_or_else(|| {
                        FastnnError::Overflow("row-packed tensor payload overflow".into())
                    })?
            }
            _ => self.encoding.storage_bytes(scalar, numel)?,
        };
        self.prefix_bytes
            .checked_add(payload_bytes)
            .and_then(|bytes| bytes.checked_add(self.suffix_bytes))
            .ok_or_else(|| FastnnError::Overflow("tensor allocation size overflow".into()))
    }
}

impl StorageEncoding {
    pub fn validate_for(self, scalar: ScalarType) -> FastnnResult<()> {
        match self {
            Self::Plain if scalar.plain_byte_width().is_none() => Err(FastnnError::dtype(format!(
                "{}-bit scalar requires an explicit packed storage encoding",
                scalar.bit_width()
            ))),
            Self::Plain => Ok(()),
            Self::Packed { word_bits, lanes } => {
                if word_bits == 0 || lanes == 0 {
                    return Err(FastnnError::dtype(
                        "packed storage word_bits and lanes must be non-zero",
                    ));
                }
                let packed_bits = u16::from(scalar.bit_width()) * u16::from(lanes);
                if packed_bits != u16::from(word_bits) {
                    return Err(FastnnError::dtype(format!(
                        "{} lanes of {}-bit values require a {}-bit word, got {}",
                        lanes,
                        scalar.bit_width(),
                        packed_bits,
                        word_bits
                    )));
                }
                if !word_bits.is_multiple_of(8) {
                    return Err(FastnnError::dtype(format!(
                        "packed word width {word_bits} is not byte-addressable"
                    )));
                }
                Ok(())
            }
        }
    }

    pub fn storage_bytes(self, scalar: ScalarType, numel: usize) -> FastnnResult<usize> {
        self.validate_for(scalar)?;
        match self {
            Self::Plain => numel
                .checked_mul(scalar.plain_byte_width().ok_or_else(|| {
                    FastnnError::dtype("plain storage requires a byte-addressable scalar")
                })?)
                .ok_or_else(|| FastnnError::Overflow("tensor storage size overflow".into())),
            Self::Packed { word_bits, lanes } => numel
                .div_ceil(usize::from(lanes))
                .checked_mul(usize::from(word_bits / 8))
                .ok_or_else(|| FastnnError::Overflow("packed tensor storage size overflow".into())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    PerTensor,
    PerAxis { axis: usize },
    PerGroup { axis: usize, group_size: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    Symmetric,
    Asymmetric,
    Codebook { entries: Vec<Vec<f32>> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub scheme: QuantizationScheme,
    pub granularity: QuantizationGranularity,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
}

impl QuantizationSpec {
    pub fn validate(&self) -> FastnnResult<()> {
        if !matches!(self.scheme, QuantizationScheme::Codebook { .. }) && self.scales.is_empty() {
            return Err(FastnnError::dtype(
                "quantization requires at least one scale",
            ));
        }
        if self
            .scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(FastnnError::dtype(
                "quantization scales must be finite and positive",
            ));
        }
        if let QuantizationGranularity::PerGroup { group_size, .. } = self.granularity {
            if group_size == 0 {
                return Err(FastnnError::dtype(
                    "per-group quantization requires a non-zero group size",
                ));
            }
        }
        match &self.scheme {
            QuantizationScheme::Symmetric if !self.zero_points.is_empty() => Err(
                FastnnError::dtype("symmetric quantization must not store zero points"),
            ),
            QuantizationScheme::Asymmetric if self.zero_points.len() != self.scales.len() => {
                Err(FastnnError::dtype(format!(
                    "asymmetric quantization has {} scales but {} zero points",
                    self.scales.len(),
                    self.zero_points.len()
                )))
            }
            QuantizationScheme::Codebook { entries }
                if entries.is_empty()
                    || entries.iter().any(|entry| {
                        entry.is_empty() || entry.iter().any(|value| !value.is_finite())
                    }) =>
            {
                Err(FastnnError::dtype(
                    "codebook quantization requires non-empty finite codebooks",
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RepresentationTransform {
    None,
    AffineQuantization(QuantizationSpec),
    /// Legacy packed formats use `real = q * scale + offset`, where offset
    /// is a floating dequantization parameter rather than an integer zero point.
    AffineDequantization {
        granularity: QuantizationGranularity,
        scales: Vec<f32>,
        offsets: Vec<f32>,
    },
    Codebook {
        granularity: QuantizationGranularity,
        entries: Vec<Vec<f32>>,
        scales: Vec<f32>,
        offsets: Vec<f32>,
    },
    /// Quantization parameters are carried in the runtime tensor payload.
    RuntimeAffineQuantization {
        granularity: QuantizationGranularity,
    },
    /// Floating scale/offset parameters are carried alongside the runtime payload.
    RuntimeScaledAffine {
        granularity: QuantizationGranularity,
    },
    Scaled {
        scales: Vec<f32>,
    },
    /// Floating storage with `real = stored_float * scale + offset`.
    ScaledAffine {
        granularity: QuantizationGranularity,
        scales: Vec<f32>,
        offsets: Vec<f32>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueRepresentation {
    /// Mathematical value domain exposed to graph semantics.
    pub logical: ScalarType,
    /// Scalar format physically stored in memory.
    pub storage: ScalarType,
    pub encoding: StorageEncoding,
    pub transform: RepresentationTransform,
}

impl ValueRepresentation {
    pub fn validate(&self) -> FastnnResult<()> {
        self.encoding.validate_for(self.storage)?;
        match &self.transform {
            RepresentationTransform::None => {
                if self.logical != self.storage {
                    return Err(FastnnError::dtype(
                        "an untransformed representation must use the same logical and storage scalar",
                    ));
                }
                Ok(())
            }
            RepresentationTransform::AffineQuantization(quantization) => {
                if !self.logical.is_float() {
                    return Err(FastnnError::dtype(
                        "quantized representation requires a floating logical type",
                    ));
                }
                if !self.storage.is_integer() {
                    return Err(FastnnError::dtype(
                        "quantized representation requires an integer stored type",
                    ));
                }
                quantization.validate()
            }
            RepresentationTransform::RuntimeAffineQuantization { .. } => {
                if !self.logical.is_float() || !self.storage.is_integer() {
                    return Err(FastnnError::dtype(
                        "runtime affine quantization requires floating logical and integer storage types",
                    ));
                }
                Ok(())
            }
            RepresentationTransform::RuntimeScaledAffine { .. } => {
                if !self.logical.is_float() || !self.storage.is_float() {
                    return Err(FastnnError::dtype(
                        "runtime scaled affine representation requires floating logical and storage types",
                    ));
                }
                Ok(())
            }
            RepresentationTransform::AffineDequantization {
                scales, offsets, ..
            } => validate_scales_and_offsets(scales, offsets),
            RepresentationTransform::Codebook {
                entries,
                scales,
                offsets,
                ..
            } => {
                if entries.is_empty()
                    || entries.iter().any(|entry| {
                        entry.is_empty() || entry.iter().any(|value| !value.is_finite())
                    })
                {
                    return Err(FastnnError::dtype(
                        "codebook representation requires non-empty finite entries",
                    ));
                }
                if scales
                    .iter()
                    .any(|scale| !scale.is_finite() || *scale <= 0.0)
                {
                    return Err(FastnnError::dtype(
                        "codebook representation scales must be finite and positive",
                    ));
                }
                if !offsets.is_empty()
                    && (offsets.len() != scales.len()
                        || offsets.iter().any(|offset| !offset.is_finite()))
                {
                    return Err(FastnnError::dtype(
                        "codebook representation offsets must be finite and match scale count",
                    ));
                }
                Ok(())
            }
            RepresentationTransform::Scaled { scales } => {
                if !self.logical.is_float() || !self.storage.is_float() {
                    return Err(FastnnError::dtype(
                        "scaled representation requires floating logical and storage types",
                    ));
                }
                if scales.is_empty()
                    || scales
                        .iter()
                        .any(|scale| !scale.is_finite() || *scale <= 0.0)
                {
                    return Err(FastnnError::dtype(
                        "scaled representation requires finite positive scales",
                    ));
                }
                Ok(())
            }
            RepresentationTransform::ScaledAffine {
                scales, offsets, ..
            } => {
                if !self.logical.is_float() || !self.storage.is_float() {
                    return Err(FastnnError::dtype(
                        "scaled affine representation requires floating logical and storage types",
                    ));
                }
                validate_scales_and_optional_offsets(scales, offsets)
            }
        }
    }

    pub fn storage_bytes(&self, numel: usize) -> FastnnResult<usize> {
        self.validate()?;
        self.encoding.storage_bytes(self.storage, numel)
    }
}

fn validate_scales_and_offsets(scales: &[f32], offsets: &[f32]) -> FastnnResult<()> {
    if scales.is_empty()
        || scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(FastnnError::dtype(
            "affine dequantization requires finite positive scales",
        ));
    }
    if offsets.len() != scales.len() || offsets.iter().any(|offset| !offset.is_finite()) {
        return Err(FastnnError::dtype(format!(
            "affine dequantization has {} scales but {} finite offsets are required",
            scales.len(),
            offsets.len()
        )));
    }
    Ok(())
}

fn validate_scales_and_optional_offsets(scales: &[f32], offsets: &[f32]) -> FastnnResult<()> {
    if offsets.is_empty() {
        if scales.is_empty()
            || scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(FastnnError::dtype(
                "scaled affine representation requires finite positive scales",
            ));
        }
        return Ok(());
    }
    validate_scales_and_offsets(scales, offsets)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantTarget {
    I8,
    U8,
    I4,
    U4,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
    I4Codebook,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompileTarget {
    Native,
    WeightOnly(QuantTarget),
    IntegerInference(QuantTarget),
    TrainingMixedPrecision {
        compute: ScalarType,
        accumulator: ScalarType,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_storage_rounds_to_whole_words() {
        let encoding = StorageEncoding::Packed {
            word_bits: 32,
            lanes: 8,
        };
        assert_eq!(encoding.storage_bytes(ScalarType::U4, 1).unwrap(), 4);
        assert_eq!(encoding.storage_bytes(ScalarType::U4, 8).unwrap(), 4);
        assert_eq!(encoding.storage_bytes(ScalarType::U4, 9).unwrap(), 8);
    }

    #[test]
    fn plain_bool_storage_uses_one_byte_per_value() {
        assert_eq!(
            StorageEncoding::Plain
                .storage_bytes(ScalarType::Bool, 3)
                .unwrap(),
            3
        );
    }

    #[test]
    fn allocation_layout_separates_row_padding_and_capacity_margin() {
        let layout = TensorStorageLayout {
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: 64,
        };
        assert_eq!(
            layout.allocation_bytes(ScalarType::U4, &[2, 9]).unwrap(),
            80
        );
        assert_eq!(layout.allocation_bytes(ScalarType::U4, &[18]).unwrap(), 76);
    }

    #[test]
    fn rejects_mismatched_packing_geometry() {
        let encoding = StorageEncoding::Packed {
            word_bits: 32,
            lanes: 4,
        };
        assert!(encoding.validate_for(ScalarType::U4).is_err());
    }

    #[test]
    fn validates_asymmetric_quantization_parameter_counts() {
        let spec = QuantizationSpec {
            scheme: QuantizationScheme::Asymmetric,
            granularity: QuantizationGranularity::PerAxis { axis: 0 },
            scales: vec![0.5, 0.25],
            zero_points: vec![1],
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn validates_integer_backed_quantized_representation() {
        let representation = ValueRepresentation {
            logical: ScalarType::F32,
            storage: ScalarType::U4,
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            transform: RepresentationTransform::AffineQuantization(QuantizationSpec {
                scheme: QuantizationScheme::Asymmetric,
                granularity: QuantizationGranularity::PerTensor,
                scales: vec![0.125],
                zero_points: vec![8],
            }),
        };
        assert_eq!(representation.storage_bytes(17).unwrap(), 12);
    }

    #[test]
    fn validates_scaled_packed_float_representation() {
        let representation = ValueRepresentation {
            logical: ScalarType::F32,
            storage: ScalarType::Fp8E4M3,
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 4,
            },
            transform: RepresentationTransform::Scaled { scales: vec![0.25] },
        };
        assert_eq!(representation.storage_bytes(5).unwrap(), 8);
    }
}
