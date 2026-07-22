//! Symbolic shapes and tensor value types used by the IR.

use super::graph::{ComputeGraph, NodeId};
use crate::types::{
    QuantizationGranularity, RepresentationTransform, ScalarType, StorageEncoding,
    TensorStorageLayout, ValueRepresentation, PACKED_SIMD_MARGIN_BYTES,
};
use crate::FastnnResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default maximum extent assumed for a purely symbolic dimension (no Bounded bound).
/// Used by the memory planner to allocate sufficient arena space for graphs that
/// contain unbounded Symbol dims.  Callers that know tighter bounds should use
/// [`DimExpr::Bounded`] instead.
///
/// This is an atomic so it can be adjusted at runtime (e.g. from a config file
/// or by [`set_symbol_dim_max`]).
///
/// The default of 8192 balances compile-time arena sizing for typical models
/// (batch ≤ 256, feature dims ≤ 4096, spatial dims ≤ 2048).  Users with larger
/// expected dimensions should call `set_symbol_dim_max` before graph construction.
/// Use [`DimExpr::Bounded`] for per-dimension limits to avoid overallocation
/// when multiple symbolic dims appear in the same tensor shape.
pub static SYMBOL_DIM_MAX: AtomicU64 = AtomicU64::new(8192);

/// Override the global [`SYMBOL_DIM_MAX`] value.  Call before constructing
/// graphs whose unbounded Symbol dims may exceed 4096.
pub fn set_symbol_dim_max(val: u64) {
    SYMBOL_DIM_MAX.store(val, Ordering::Relaxed);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DimExpr {
    Known(u64),
    Symbol(String),
    Bounded { sym: String, max: u64 },
}

impl DimExpr {
    pub fn is_known(&self) -> bool {
        matches!(self, DimExpr::Known(_))
    }

    pub fn evaluate(&self) -> Option<u64> {
        match self {
            DimExpr::Known(v) => Some(*v),
            DimExpr::Bounded { max, .. } => Some(*max),
            DimExpr::Symbol(_) => None,
        }
    }

    /// Symbolic multiplication of two DimExpr values.
    pub fn mul(&self, other: &DimExpr) -> DimExpr {
        match (self, other) {
            (DimExpr::Known(0), _) | (_, DimExpr::Known(0)) => DimExpr::Known(0),
            (DimExpr::Known(1), other) | (other, DimExpr::Known(1)) => other.clone(),
            (DimExpr::Known(va), DimExpr::Known(vb)) => DimExpr::Known(va * vb),
            (DimExpr::Known(v), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Known(v)) => DimExpr::Bounded {
                sym: sym.clone(),
                max: max * v,
            },
            (DimExpr::Known(v), DimExpr::Symbol(s)) | (DimExpr::Symbol(s), DimExpr::Known(v)) => {
                DimExpr::Bounded {
                    sym: s.clone(),
                    max: *v,
                }
            }
            (
                DimExpr::Bounded { sym, max },
                DimExpr::Bounded {
                    sym: sym_b,
                    max: mb,
                },
            ) => {
                let sym = if sym == sym_b {
                    format!("{}^2", sym)
                } else {
                    canonical_commutative("*", sym, sym_b)
                };
                DimExpr::Bounded { sym, max: max * mb }
            }
            // Same symbol squared → canonical "N^2" instead of stringy "(N*N)"
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) if s == t => {
                let m = SYMBOL_DIM_MAX.load(Ordering::Relaxed);
                DimExpr::Bounded {
                    sym: format!("{}^2", s),
                    max: m * m,
                }
            }
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) => {
                DimExpr::Symbol(canonical_commutative("*", s, t))
            }
            (DimExpr::Symbol(s), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Symbol(s)) => {
                let sym = if s == sym {
                    format!("{}^2", s)
                } else {
                    canonical_commutative("*", s, sym)
                };
                DimExpr::Bounded { sym, max: *max }
            }
        }
    }

    /// Symbolic addition of two DimExpr values.
    pub fn add(&self, other: &DimExpr) -> DimExpr {
        match (self, other) {
            (DimExpr::Known(0), other) | (other, DimExpr::Known(0)) => other.clone(),
            (DimExpr::Known(va), DimExpr::Known(vb)) => DimExpr::Known(va + vb),
            (DimExpr::Known(v), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Known(v)) => DimExpr::Bounded {
                sym: sym.clone(),
                max: max + v,
            },
            (DimExpr::Known(v), DimExpr::Symbol(s)) | (DimExpr::Symbol(s), DimExpr::Known(v)) => {
                DimExpr::Bounded {
                    sym: s.clone(),
                    max: *v,
                }
            }
            (
                DimExpr::Bounded { sym, max },
                DimExpr::Bounded {
                    sym: sym_b,
                    max: mb,
                },
            ) => {
                let sym = if sym == sym_b {
                    format!("2*{}", sym)
                } else {
                    canonical_commutative("+", sym, sym_b)
                };
                DimExpr::Bounded { sym, max: max + mb }
            }
            // Same symbol + itself → canonical "2*N" instead of stringy "(N+N)"
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) if s == t => {
                let max = SYMBOL_DIM_MAX.load(Ordering::Relaxed);
                DimExpr::Bounded {
                    sym: format!("2*{}", s),
                    max: 2 * max,
                }
            }
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) => {
                DimExpr::Symbol(canonical_commutative("+", s, t))
            }
            (DimExpr::Symbol(s), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Symbol(s)) => DimExpr::Bounded {
                sym: if s == sym {
                    format!("2*{}", s)
                } else {
                    canonical_commutative("+", s, sym)
                },
                max: *max,
            },
        }
    }
}

// =============================================================================
// DimExpr arithmetic helpers
// =============================================================================

/// Build a canonical string for a commutative binary operation.
/// Operands are sorted alphabetically so `N+10` and `10+N` produce the same string.
fn canonical_commutative(op: &str, a: &str, b: &str) -> String {
    let mut v: Vec<&str> = vec![a, b];
    v.sort_unstable();
    format!("({}{}{})", v[0], op, v[1])
}

impl DimExpr {
    /// Symbolic floor division (affine expression support, e.g. `T/2`).
    ///
    /// Only division by a [`Known`](DimExpr::Known) divisor is supported;
    /// division by a [`Symbol`](DimExpr::Symbol) or
    /// [`Bounded`](DimExpr::Bounded) returns a [`Known`](DimExpr::Known) 1
    /// fallback since the result is rarely affine in a useful way.
    pub fn floordiv(&self, divisor: &DimExpr) -> DimExpr {
        match (self, divisor) {
            (DimExpr::Known(va), DimExpr::Known(vb)) if *vb > 0 => DimExpr::Known(va / vb),
            (DimExpr::Symbol(s), DimExpr::Known(v)) if *v > 0 => {
                let max = SYMBOL_DIM_MAX.load(Ordering::Relaxed);
                DimExpr::Bounded {
                    sym: format!("{}/{}", s, v),
                    max: max / v,
                }
            }
            (DimExpr::Bounded { sym, max }, DimExpr::Known(v)) if *v > 0 => DimExpr::Bounded {
                sym: format!("{}/{}", sym, v),
                max: max / v,
            },
            _ => DimExpr::Known(1),
        }
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Known(v) => write!(f, "{}", v),
            DimExpr::Symbol(s) => write!(f, "{}", s),
            DimExpr::Bounded { sym, max } => write!(f, "{}∈[1,{}]", sym, max),
        }
    }
}

// =============================================================================
// ShapeEnv — runtime shape environment that resolves Symbol → concrete value
// =============================================================================

/// Runtime mapping from symbolic dimension names to concrete integer values.
///
/// Created by the executor before dispatch by inspecting input buffer sizes
/// against the shapes declared in `GraphBuilder::input_with_dims()`.
#[derive(Debug, Clone, Default)]
pub struct ShapeEnv {
    symbols: HashMap<String, u64>,
}

impl ShapeEnv {
    pub fn new() -> Self {
        ShapeEnv {
            symbols: HashMap::new(),
        }
    }

    /// Bind a symbol name to a concrete value, returning an error on conflicts.
    pub fn try_bind(&mut self, name: &str, value: u64) -> Result<(), String> {
        if let Some(&existing) = self.symbols.get(name) {
            if existing != value {
                return Err(format!(
                    "ShapeEnv: symbol '{}' bound to {} then inconsistently to {}",
                    name, existing, value
                ));
            }
        }
        self.symbols.insert(name.to_string(), value);
        Ok(())
    }

    /// Resolve a symbol name to its concrete value, if bound.
    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    /// Build a ShapeEnv by matching input byte sizes against graph input node shapes.
    ///
    /// Returns an error if:
    /// - Input byte count is not evenly divisible by the known-stride element count.
    /// - Two inputs infer different values for the same symbol.
    ///
    /// The algorithm has two passes:
    ///
    /// 1. **Single-symbol inputs** — if an input shape has exactly one
    ///    [`DimExpr::Symbol`] or [`DimExpr::Bounded`] dimension, its value is
    ///    computed directly from `data_bytes / (known_numel * elem_size)`.
    ///
    /// 2. **Multi-symbol inputs** — after pass 1, any already-bound symbols
    ///    are treated as known.  If exactly one symbol remains unbound, it is
    ///    inferred from the remaining unknown element count.
    ///
    /// This handles the common case of `[B, T, D]` where `B` is also present
    /// in another input (e.g. a bias of shape `[B]`).
    pub fn from_graph_inputs(graph: &ComputeGraph, inputs: &[&[u8]]) -> Result<Self, String> {
        let mut env = ShapeEnv::new();
        let mut input_infos: Vec<(NodeId, usize, usize, Vec<String>)> = Vec::new();

        // Pass 1: collect shape info, bind single-symbol inputs
        for (i, &input_id) in graph.inputs.iter().enumerate() {
            let data_bytes = inputs.get(i).map(|b| b.len()).unwrap_or(0);
            if data_bytes == 0 {
                continue;
            }
            if let Some(node) = graph.get_node(input_id) {
                let elem_size = node
                    .output_type
                    .plain_storage_byte_width()
                    .map_err(|error| format!("ShapeEnv: input {i} dtype: {error}"))?
                    .ok_or_else(|| {
                        format!("ShapeEnv: input {i} uses packed storage without a scalar stride")
                    })?;
                let known_numel: usize = node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Known(v) => Some(*v as usize),
                        _ => None,
                    })
                    .product();
                if known_numel == 0 || node.output_type.shape.is_empty() {
                    continue;
                }
                let stride = elem_size * known_numel;
                if data_bytes % stride != 0 {
                    return Err(format!(
                        "ShapeEnv: input {} node {}: data size {} not divisible by known-stride {} (shape {:?}, elem_size {})",
                        i, input_id, data_bytes, stride,
                        node.output_type.shape, elem_size
                    ));
                }
                let total_numel = data_bytes / elem_size;
                let unknown_numel = total_numel / known_numel;
                if unknown_numel == 0 {
                    continue;
                }
                let symbolic: Vec<String> = node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Symbol(s) => Some(s.clone()),
                        DimExpr::Bounded { sym, .. } => Some(sym.clone()),
                        _ => None,
                    })
                    .collect();

                if symbolic.len() == 1 {
                    env.try_bind(&symbolic[0], unknown_numel as u64)?;
                } else if symbolic.len() > 1 {
                    input_infos.push((input_id, known_numel, total_numel, symbolic));
                }
            }
        }

        // Pass 2: multi-symbol inputs — resolve remaining symbols using
        // already-bound values.
        for (_id, known_numel, total_numel, symbolic) in &input_infos {
            let mut known_product = *known_numel;
            let mut unbound: Vec<&str> = Vec::new();
            for sym in symbolic {
                if let Some(val) = env.resolve(sym) {
                    known_product *= val as usize;
                } else {
                    unbound.push(sym);
                }
            }
            if unbound.len() == 1 && known_product > 0 {
                if total_numel % known_product != 0 {
                    return Err(format!(
                        "ShapeEnv: multi-symbol input node {}: remaining numel {} not divisible by known_product {}",
                        _id, total_numel, known_product
                    ));
                }
                let val = total_numel / known_product;
                if val > 0 {
                    env.try_bind(unbound[0], val as u64)?;
                }
            }
        }

        Ok(env)
    }
}

impl DimExpr {
    /// Evaluate with runtime shape environment.  Bounded dims try resolving
    /// their symbol name first, falling back to the compile-time max bound.
    ///
    /// # Panics
    /// If a [`DimExpr::Bounded`] symbol resolves but the value exceeds `max`,
    /// indicating the runtime shape overflows the compile-time bound contract.
    /// Evaluate the dimension expression using a runtime [`ShapeEnv`].
    ///
    /// Returns:
    /// - `Ok(value)` if the expression resolves successfully
    /// - `Err(message)` if a [`DimExpr::Symbol`] is unresolved, or a
    ///   [`DimExpr::Bounded`] resolves to a value exceeding its compile-time
    ///   `max` bound.
    pub fn evaluate_with_env(&self, env: &ShapeEnv) -> Result<u64, String> {
        match self {
            DimExpr::Known(v) => Ok(*v),
            DimExpr::Bounded { sym, max } => match env.resolve(sym) {
                Some(v) => {
                    if v > *max {
                        Err(format!(
                            "DimExpr::Bounded: symbol '{}' resolved to {}, exceeds bound {}",
                            sym, v, max
                        ))
                    } else {
                        Ok(v)
                    }
                }
                None => Ok(*max),
            },
            DimExpr::Symbol(s) => env
                .resolve(s)
                .ok_or_else(|| format!("DimExpr::Symbol '{}' is not bound in the ShapeEnv", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrDType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    Bool,
    /// Signed 8-bit integer (used for INT8 activation quantization).
    I8,
    /// Packed 4-bit (I4x8): 8 values per u32 word.
    /// `scales` and `dequant_offsets` are per-output-channel vectors.
    /// When `codebooks` is non-empty, bits 0-3 are an unsigned 4-bit index
    /// (0-15) into `codebook[block_idx]`.  Dequant: `codebook[blk][nibble]`.
    /// Replaces scales/dequant_offsets when present (they are ignored).
    I4 {
        scales: Vec<f32>,
        dequant_offsets: Vec<f32>,
        codebooks: Vec<[f32; 16]>,
    },
    /// Packed 8-bit (I8x4): 4 values per u32 word.
    /// `scales` and `dequant_offsets` are per-output-channel vectors.
    /// Formerly named `U8` (misleading — this was always signed I8x4).
    I8Scaled {
        scales: Vec<f32>,
        dequant_offsets: Vec<f32>,
    },
    /// Unsigned packed 4-bit (U4x8): 8 values per u32 word.
    /// `scales` and `dequant_offsets` are per-output-channel vectors.
    U4Scaled {
        scales: Vec<f32>,
        dequant_offsets: Vec<f32>,
    },
    /// Unsigned packed 8-bit (U8x4): 4 values per u32 word.
    /// `scales` and `dequant_offsets` are per-output-channel vectors.
    U8Scaled {
        scales: Vec<f32>,
        dequant_offsets: Vec<f32>,
    },

    /// FP8 E4M3 (no zero-point, 2-term dequant).
    F8 {
        scales: Vec<f32>,
    },
    /// FP8 E5M2 range variant (wider range, for gradients).
    F8R {
        scales: Vec<f32>,
    },
    /// FP4 E2M1 (NVFP4-style), 8 values per u32 word.
    /// Uses 256-entry LUT for dot product. Block scales stored in PackedTensor.
    /// When `codebooks` is non-empty, bits 0-2 are a magnitude index into
    /// `codebook[block_idx]` and bit 3 is the sign.
    /// Dequant: `sign * codebook[block][magnitude]`. Overrides scales/dequant_offsets.
    F4 {
        scales: Vec<f32>,
        dequant_offsets: Vec<f32>,
        codebooks: Vec<[f32; 16]>,
    },
}

impl IrDType {
    /// Return floating affine-dequantization parameters for legacy packed
    /// integer representations.
    ///
    /// These values implement `real = q * scale + offset`; the second slice is
    /// not an integer quantizer zero point. Codebook-backed I4 values return
    /// `None` because their lookup table owns the value transform.
    pub fn affine_dequantization(&self) -> Option<(&[f32], &[f32])> {
        match self {
            Self::I4 {
                scales,
                dequant_offsets,
                codebooks,
            } if codebooks.is_empty() => Some((scales, dequant_offsets)),
            Self::I8Scaled {
                scales,
                dequant_offsets,
            }
            | Self::U4Scaled {
                scales,
                dequant_offsets,
            }
            | Self::U8Scaled {
                scales,
                dequant_offsets,
            } => Some((scales, dequant_offsets)),
            _ => None,
        }
    }
}

/// Macro that generates simple per-variant constant lookup methods for IrDType.
macro_rules! impl_ir_dtype_props {
    ($(($variant:pat, $as_str:expr)),* $(,)?) => {
        impl IrDType {
            pub fn as_str(&self) -> &'static str {
                match self { $( $variant => $as_str, )* }
            }
        }
    };
}

impl_ir_dtype_props!(
    (Self::F32, "f32"),
    (Self::F16, "f16"),
    (Self::BF16, "bf16"),
    (Self::I32, "i32"),
    (Self::I64, "i64"),
    (Self::Bool, "bool"),
    (Self::I8, "i8"),
    (Self::I4 { .. }, "i4"),
    (Self::I8Scaled { .. }, "i8"),
    (Self::F8 { .. }, "f8"),
    (Self::F8R { .. }, "f8r"),
    (Self::F4 { .. }, "f4"),
    (Self::U4Scaled { .. }, "u4"),
    (Self::U8Scaled { .. }, "u8"),
);

fn quantization_granularity(parameter_count: usize) -> QuantizationGranularity {
    if parameter_count <= 1 {
        QuantizationGranularity::PerTensor
    } else {
        QuantizationGranularity::PerAxis { axis: 0 }
    }
}

fn scaled_float_representation(
    storage: ScalarType,
    lanes: u8,
    scales: &[f32],
) -> FastnnResult<ValueRepresentation> {
    Ok(ValueRepresentation {
        logical: ScalarType::F32,
        storage,
        encoding: StorageEncoding::Packed {
            word_bits: 32,
            lanes,
        },
        transform: RepresentationTransform::Scaled {
            scales: scales.to_vec(),
        },
    })
}

fn codebook_representation(
    storage: ScalarType,
    lanes: u8,
    scales: &[f32],
    offsets: &[f32],
    codebooks: &[[f32; 16]],
) -> FastnnResult<ValueRepresentation> {
    Ok(ValueRepresentation {
        logical: ScalarType::F32,
        storage,
        encoding: StorageEncoding::Packed {
            word_bits: 32,
            lanes,
        },
        transform: RepresentationTransform::Codebook {
            granularity: QuantizationGranularity::PerGroup {
                axis: 0,
                group_size: 1,
            },
            entries: codebooks.iter().map(|entry| entry.to_vec()).collect(),
            scales: scales.to_vec(),
            offsets: offsets.to_vec(),
        },
    })
}

impl IrDType {
    /// Translate the legacy IR dtype into the canonical orthogonal representation.
    pub fn value_representation(&self) -> FastnnResult<ValueRepresentation> {
        let packed = |lanes| StorageEncoding::Packed {
            word_bits: 32,
            lanes,
        };
        let affine = |storage, lanes, scales: &Vec<f32>, offsets: &Vec<f32>| {
            ValueRepresentation::packed_affine_dequantization(
                storage,
                lanes,
                quantization_granularity(scales.len()),
                scales.clone(),
                offsets.clone(),
            )
        };

        match self {
            Self::F32 => Ok(ValueRepresentation::native(ScalarType::F32)),
            Self::F16 => Ok(ValueRepresentation::native(ScalarType::F16)),
            Self::BF16 => Ok(ValueRepresentation::native(ScalarType::BF16)),
            Self::I32 => Ok(ValueRepresentation::native(ScalarType::I32)),
            Self::I64 => Ok(ValueRepresentation::native(ScalarType::I64)),
            Self::Bool => Ok(ValueRepresentation::native(ScalarType::Bool)),
            Self::I8 => ValueRepresentation::runtime_affine(ScalarType::I8, StorageEncoding::Plain),
            Self::I4 {
                scales,
                dequant_offsets,
                codebooks,
            } if !codebooks.is_empty() => Ok(codebook_representation(
                ScalarType::I4,
                8,
                scales,
                dequant_offsets,
                codebooks,
            )?),
            Self::I4 {
                scales,
                dequant_offsets,
                ..
            } => affine(ScalarType::I4, 8, scales, dequant_offsets),
            Self::I8Scaled {
                scales,
                dequant_offsets,
            } => affine(ScalarType::I8, 4, scales, dequant_offsets),
            Self::U4Scaled {
                scales,
                dequant_offsets,
            } => affine(ScalarType::U4, 8, scales, dequant_offsets),
            Self::U8Scaled {
                scales,
                dequant_offsets,
            } => affine(ScalarType::U8, 4, scales, dequant_offsets),
            Self::F8 { scales } => scaled_float_representation(ScalarType::Fp8E4M3, 4, scales),
            Self::F8R { scales } => scaled_float_representation(ScalarType::Fp8E5M2, 4, scales),
            Self::F4 {
                scales,
                dequant_offsets,
                codebooks,
            } if !codebooks.is_empty() => {
                codebook_representation(ScalarType::I4, 8, scales, dequant_offsets, codebooks)
            }
            Self::F4 {
                scales,
                dequant_offsets,
                ..
            } => Ok(ValueRepresentation {
                logical: ScalarType::F32,
                storage: ScalarType::Fp4E2M1,
                encoding: packed(8),
                transform: RepresentationTransform::ScaledAffine {
                    granularity: quantization_granularity(scales.len()),
                    scales: scales.clone(),
                    offsets: dequant_offsets.clone(),
                },
            }),
        }
    }

    /// Physical byte width for byte-addressable scalar storage.
    /// Packed encodings return `None`; callers must use encoded payload sizing.
    pub fn plain_storage_byte_width(&self) -> FastnnResult<Option<usize>> {
        let representation = self.value_representation()?;
        Ok(match representation.encoding {
            StorageEncoding::Plain => representation.storage.plain_byte_width(),
            StorageEncoding::Packed { .. } => None,
        })
    }

    /// Canonical physical allocation layout for this legacy IR dtype.
    pub fn storage_layout(&self) -> FastnnResult<TensorStorageLayout> {
        let representation = self.value_representation()?;
        Ok(match (&representation.encoding, self) {
            (StorageEncoding::Plain, Self::I8) => TensorStorageLayout {
                encoding: representation.encoding,
                row_packed: false,
                prefix_bytes: 8,
                suffix_bytes: 0,
            },
            (StorageEncoding::Packed { .. }, _) => TensorStorageLayout {
                encoding: representation.encoding,
                row_packed: true,
                prefix_bytes: 0,
                suffix_bytes: PACKED_SIMD_MARGIN_BYTES,
            },
            (StorageEncoding::Plain, _) => TensorStorageLayout::contiguous(representation.encoding),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<DimExpr>,
    pub dtype: IrDType,
}

impl TensorType {
    pub fn new(shape: Vec<DimExpr>, dtype: IrDType) -> Self {
        TensorType { shape, dtype }
    }

    /// Return the same tensor contract with a different logical shape.
    ///
    /// Shape-only rewrites must use this instead of rebuilding a tensor type
    /// from its dtype identity. Once representation and layout are owned
    /// directly by `TensorType`, this preserves those fields automatically.
    pub fn with_shape(&self, shape: Vec<DimExpr>) -> Self {
        Self {
            shape,
            ..self.clone()
        }
    }

    /// Resolve the canonical logical/storage/transform contract for this tensor.
    ///
    /// Callers should use this tensor-level API rather than reaching through the
    /// legacy dtype field. That keeps representation ownership movable without
    /// coupling compiler and backend code to the transitional IR schema.
    pub fn value_representation(&self) -> FastnnResult<ValueRepresentation> {
        self.dtype.value_representation()
    }

    pub fn storage_layout(&self) -> FastnnResult<TensorStorageLayout> {
        self.dtype.storage_layout()
    }

    pub fn plain_storage_byte_width(&self) -> FastnnResult<Option<usize>> {
        let representation = self.value_representation()?;
        Ok(match representation.encoding {
            StorageEncoding::Plain => representation.storage.plain_byte_width(),
            StorageEncoding::Packed { .. } => None,
        })
    }

    pub fn affine_dequantization(&self) -> Option<(&[f32], &[f32])> {
        self.dtype.affine_dequantization()
    }

    pub fn is_native_scalar(&self, scalar: ScalarType) -> bool {
        self.value_representation().is_ok_and(|representation| {
            representation.logical == scalar
                && representation.storage == scalar
                && matches!(representation.encoding, StorageEncoding::Plain)
                && matches!(representation.transform, RepresentationTransform::None)
        })
    }

    pub fn is_native_float(&self) -> bool {
        [ScalarType::F32, ScalarType::F16, ScalarType::BF16]
            .into_iter()
            .any(|scalar| self.is_native_scalar(scalar))
    }

    pub fn numel(&self) -> Option<u64> {
        let mut total = 1u64;
        for dim in &self.shape {
            let value = dim.evaluate()?;
            total = total.checked_mul(value)?;
        }
        Some(total)
    }

    /// Compute the byte size of this tensor's storage, using `SYMBOL_DIM_MAX`
    /// as the fallback extent for pure [`DimExpr::Symbol`] dimensions.
    pub fn byte_size(&self) -> usize {
        self.byte_size_with_env(None)
    }

    /// Compute the byte size of this tensor's storage, resolving symbolic
    /// dimensions against the provided [`ShapeEnv`] when available.
    /// Pure [`DimExpr::Symbol`] dimensions (not [`DimExpr::Bounded`]) still
    /// fall back to `SYMBOL_DIM_MAX`.
    ///
    /// For packed types (U4/U8), this computes the actual packed storage size
    /// including the SIMD margin, which is needed for correct arena allocation.
    pub fn try_byte_size_with_env(&self, env: Option<&ShapeEnv>) -> Option<usize> {
        let symbol_max = SYMBOL_DIM_MAX.load(Ordering::Relaxed) as usize;
        let shape = self
            .shape
            .iter()
            .map(|dimension| {
                Some(match dimension {
                    DimExpr::Known(value) => usize::try_from(*value).ok()?,
                    DimExpr::Bounded { max, .. } => usize::try_from(*max).ok()?,
                    DimExpr::Symbol(_) => match env {
                        Some(env) => {
                            usize::try_from(dimension.evaluate_with_env(env).ok()?).ok()?
                        }
                        None => symbol_max,
                    },
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let representation = self.value_representation().ok()?;
        self.storage_layout()
            .ok()?
            .allocation_bytes(representation.storage, &shape)
            .ok()
    }

    pub fn byte_size_with_env(&self, env: Option<&ShapeEnv>) -> usize {
        self.try_byte_size_with_env(env)
            .expect("tensor allocation size overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_plain_width_is_distinct_from_packed_payload_sizing() {
        assert_eq!(IrDType::Bool.plain_storage_byte_width().unwrap(), Some(1));
        assert_eq!(IrDType::F32.plain_storage_byte_width().unwrap(), Some(4));
        assert_eq!(
            IrDType::U4Scaled {
                scales: vec![1.0],
                dequant_offsets: vec![0.0],
            }
            .plain_storage_byte_width()
            .unwrap(),
            None
        );
    }

    #[test]
    fn ir_dtype_exposes_runtime_headers_and_packed_margins_as_layout() {
        let activation = IrDType::I8.storage_layout().unwrap();
        assert_eq!(activation.prefix_bytes, 8);
        assert_eq!(activation.suffix_bytes, 0);

        let packed = IrDType::U4Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        }
        .storage_layout()
        .unwrap();
        assert!(packed.row_packed);
        assert_eq!(packed.suffix_bytes, 64);
    }

    #[test]
    fn tensor_type_allocation_uses_canonical_layout() {
        let shape = vec![DimExpr::Known(2), DimExpr::Known(9)];
        let packed = TensorType::new(
            shape.clone(),
            IrDType::U4Scaled {
                scales: vec![1.0],
                dequant_offsets: vec![0.0],
            },
        );
        assert_eq!(packed.try_byte_size_with_env(None), Some(80));
        assert_eq!(TensorType::new(shape.clone(), IrDType::I8).byte_size(), 26);
        assert_eq!(TensorType::new(shape, IrDType::F32).byte_size(), 72);

        let symbolic = TensorType::new(
            vec![DimExpr::Known(2), DimExpr::Symbol("K".into())],
            IrDType::U4Scaled {
                scales: vec![1.0],
                dequant_offsets: vec![0.0],
            },
        );
        let mut env = ShapeEnv::new();
        env.try_bind("K", 9).unwrap();
        assert_eq!(symbolic.try_byte_size_with_env(Some(&env)), Some(80));
    }

    #[test]
    fn maps_unsigned_affine_ir_dtype_to_canonical_representation() {
        let representation = IrDType::U4Scaled {
            scales: vec![0.25, 0.5],
            dequant_offsets: vec![8.0, 7.0],
        }
        .value_representation()
        .unwrap();

        assert_eq!(representation.logical, ScalarType::F32);
        assert_eq!(representation.storage, ScalarType::U4);
        assert_eq!(
            representation.encoding,
            StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            }
        );
        assert!(matches!(
            representation.transform,
            RepresentationTransform::AffineDequantization {
                granularity: QuantizationGranularity::PerAxis { axis: 0 },
                ..
            }
        ));
        representation.validate().unwrap();
    }

    #[test]
    fn maps_runtime_activation_quantization_without_inventing_static_scales() {
        let representation = IrDType::I8.value_representation().unwrap();
        assert!(matches!(
            representation.transform,
            RepresentationTransform::RuntimeAffineQuantization
        ));
        representation.validate().unwrap();
    }

    #[test]
    fn preserves_fractional_legacy_dequantization_offsets() {
        let dtype = IrDType::I8Scaled {
            scales: vec![0.25],
            dequant_offsets: vec![1.5],
        };
        let (scales, offsets) = dtype.affine_dequantization().unwrap();
        assert_eq!(scales, &[0.25]);
        assert_eq!(offsets, &[1.5]);

        let representation = dtype.value_representation().unwrap();
        assert!(matches!(
            representation.transform,
            RepresentationTransform::AffineDequantization { ref offsets, .. }
                if offsets == &[1.5]
        ));
        representation.validate().unwrap();
    }

    #[test]
    fn tensor_type_exposes_canonical_representation_queries() {
        let tensor_type = TensorType::new(
            vec![DimExpr::Known(2), DimExpr::Known(8)],
            IrDType::I8Scaled {
                scales: vec![0.25],
                dequant_offsets: vec![1.5],
            },
        );

        let representation = tensor_type.value_representation().unwrap();
        assert_eq!(representation.storage, ScalarType::I8);
        assert!(matches!(
            representation.transform,
            RepresentationTransform::AffineDequantization { ref offsets, .. }
                if offsets == &[1.5]
        ));
        assert_eq!(tensor_type.plain_storage_byte_width().unwrap(), None);
        assert!(tensor_type.storage_layout().unwrap().row_packed);
        assert_eq!(
            tensor_type.affine_dequantization(),
            Some((&[0.25][..], &[1.5][..]))
        );
    }

    #[test]
    fn shape_rewrite_preserves_complete_tensor_contract() {
        let tensor_type = TensorType::new(
            vec![DimExpr::Known(2), DimExpr::Known(4)],
            IrDType::U4Scaled {
                scales: vec![0.25, 0.5],
                dequant_offsets: vec![-1.0, -2.0],
            },
        );
        let reshaped = tensor_type.with_shape(vec![DimExpr::Known(8)]);

        assert_eq!(reshaped.shape, vec![DimExpr::Known(8)]);
        assert_eq!(
            reshaped.value_representation().unwrap(),
            tensor_type.value_representation().unwrap()
        );
        assert_eq!(
            reshaped.storage_layout().unwrap(),
            tensor_type.storage_layout().unwrap()
        );
    }

    #[test]
    fn f4_preserves_affine_offsets_in_canonical_representation() {
        let representation = IrDType::F4 {
            scales: vec![0.25],
            dequant_offsets: vec![1.5],
            codebooks: vec![],
        }
        .value_representation()
        .unwrap();

        assert_eq!(
            representation.transform,
            RepresentationTransform::ScaledAffine {
                granularity: QuantizationGranularity::PerTensor,
                scales: vec![0.25],
                offsets: vec![1.5],
            }
        );
        representation.validate().unwrap();
    }

    #[test]
    fn codebook_dtype_does_not_expose_affine_dequantization() {
        let dtype = IrDType::I4 {
            scales: vec![1.0],
            dequant_offsets: vec![3.5],
            codebooks: vec![[0.0; 16]],
        };
        assert!(dtype.affine_dequantization().is_none());
    }
}
