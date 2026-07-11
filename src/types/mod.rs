//! Canonical value, storage, and quantization representation types.
//!
//! These types deliberately separate mathematical scalar semantics from physical
//! packing and compiler policy. Backend kernel selection does not belong here.

use crate::{FastnnError, FastnnResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    F32,
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
            Self::F32 | Self::I32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::I8 | Self::U8 => 8,
            Self::Fp4E2M1 | Self::I4 | Self::U4 => 4,
            Self::I64 => 64,
            Self::Bool => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp4E2M1
        )
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I64 | Self::I32 | Self::I8 | Self::U8 | Self::I4 | Self::U4
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageEncoding {
    Plain,
    Packed { word_bits: u8, lanes: u8 },
}

impl StorageEncoding {
    pub fn validate_for(self, scalar: ScalarType) -> FastnnResult<()> {
        match self {
            Self::Plain if scalar.bit_width() < 8 => Err(FastnnError::dtype(format!(
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
                .checked_mul(usize::from(scalar.bit_width() / 8))
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
    Scaled {
        scales: Vec<f32>,
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
