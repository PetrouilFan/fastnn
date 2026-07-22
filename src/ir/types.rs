//! Symbolic shapes and tensor value types used by the IR.

use super::graph::{ComputeGraph, NodeId};
use crate::types::{
    RepresentationTransform, ScalarType, StorageEncoding, TensorStorageLayout, ValueRepresentation,
    PACKED_SIMD_MARGIN_BYTES,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Quantization metadata now lives in `ValueRepresentation`.
    I4,
    /// Packed 8-bit (I8x4): 4 values per u32 word.
    /// Formerly named `U8` (misleading — this was always signed I8x4).
    I8Scaled,
    /// Unsigned packed 4-bit (U4x8): 8 values per u32 word.
    U4Scaled,
    /// Unsigned packed 8-bit (U8x4): 4 values per u32 word.
    U8Scaled,
    /// FP8 E4M3 (no zero-point, 2-term dequant).
    F8,
    /// FP8 E5M2 range variant (wider range, for gradients).
    F8R,
    /// FP4 E2M1 (NVFP4-style), 8 values per u32 word.
    F4,
}

impl IrDType {
    /// Return floating affine-dequantization parameters for legacy packed
    /// integer representations.
    ///
    /// These values implement `real = q * scale + offset`; the second slice is
    /// not an integer quantizer zero point. Codebook-backed I4 values return
    /// `None` because their lookup table owns the value transform.
    ///
    /// Since IrDType is now metadata-free, this always returns None.
    /// Use `TensorType::affine_dequantization()` which reads from the
    /// owned `ValueRepresentation`.
    pub fn affine_dequantization(&self) -> Option<(&[f32], &[f32])> {
        None
    }
}

impl IrDType {
    /// Derive an identity IrDType from the canonical representation.
    /// Uses storage scalar type and encoding lanes to distinguish packed
    /// weight variants (I8Scaled, U4Scaled, U8Scaled) from runtime
    /// activation I8 (which uses Plain encoding).
    pub fn from_representation(representation: &ValueRepresentation) -> Self {
        match representation.storage {
            ScalarType::F32 => Self::F32,
            ScalarType::F16 => Self::F16,
            ScalarType::BF16 => Self::BF16,
            ScalarType::I32 => Self::I32,
            ScalarType::I64 => Self::I64,
            ScalarType::Bool => Self::Bool,
            ScalarType::I8 => match representation.encoding {
                StorageEncoding::Plain => Self::I8,
                StorageEncoding::Packed { lanes: 4, .. } => Self::I8Scaled,
                _ => Self::I8,
            },
            ScalarType::I4 => Self::I4,
            ScalarType::U4 => Self::U4Scaled,
            ScalarType::U8 => Self::U8Scaled,
            ScalarType::Fp8E4M3 => Self::F8,
            ScalarType::Fp8E5M2 => Self::F8R,
            ScalarType::Fp4E2M1 => Self::F4,
            ScalarType::F64 => panic!("F64 has no executable IR dtype identity"),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I4 => "i4",
            Self::I8Scaled => "i8",
            Self::F8 => "f8",
            Self::F8R => "f8r",
            Self::F4 => "f4",
            Self::U4Scaled => "u4",
            Self::U8Scaled => "u8",
        }
    }
}

impl IrDType {
    /// Translate the identity IR dtype into the canonical orthogonal representation.
    /// Since IrDType is metadata-free, quantized types use runtime transform variants
    /// that expected metadata to be filled in later (e.g. during quantization passes).
    pub fn value_representation(&self) -> FastnnResult<ValueRepresentation> {
        match self {
            Self::F32 => Ok(ValueRepresentation::native(ScalarType::F32)),
            Self::F16 => Ok(ValueRepresentation::native(ScalarType::F16)),
            Self::BF16 => Ok(ValueRepresentation::native(ScalarType::BF16)),
            Self::I32 => Ok(ValueRepresentation::native(ScalarType::I32)),
            Self::I64 => Ok(ValueRepresentation::native(ScalarType::I64)),
            Self::Bool => Ok(ValueRepresentation::native(ScalarType::Bool)),
            Self::I8 => ValueRepresentation::runtime_affine(ScalarType::I8, StorageEncoding::Plain),
            Self::I4 => ValueRepresentation::packed_runtime_affine(ScalarType::I4, 8),
            Self::I8Scaled => ValueRepresentation::packed_runtime_affine(ScalarType::I8, 4),
            Self::U4Scaled => ValueRepresentation::packed_runtime_affine(ScalarType::U4, 8),
            Self::U8Scaled => ValueRepresentation::packed_runtime_affine(ScalarType::U8, 4),
            Self::F8 => ValueRepresentation::packed_runtime_scaled_affine(ScalarType::Fp8E4M3, 4),
            Self::F8R => ValueRepresentation::packed_runtime_scaled_affine(ScalarType::Fp8E5M2, 4),
            Self::F4 => ValueRepresentation::packed_runtime_scaled_affine(ScalarType::Fp4E2M1, 8),
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

    /// Canonical physical allocation layout for this IR dtype identity.
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
    pub representation: ValueRepresentation,
    pub layout: TensorStorageLayout,
}

impl TensorType {
    /// Construct a TensorType from an identity IrDType.
    /// The canonical `ValueRepresentation` and `TensorStorageLayout` are
    /// derived from the dtype identity.
    pub fn new(shape: Vec<DimExpr>, dtype: IrDType) -> Self {
        let representation = dtype
            .value_representation()
            .expect("IrDType identity always maps to a valid representation");
        let layout = dtype
            .storage_layout()
            .expect("IrDType identity always maps to a valid storage layout");
        TensorType {
            shape,
            representation,
            layout,
        }
    }

    /// Construct a TensorType with an explicit representation and layout.
    pub fn try_from_parts(
        shape: Vec<DimExpr>,
        representation: ValueRepresentation,
        layout: TensorStorageLayout,
    ) -> FastnnResult<Self> {
        representation.validate()?;
        if layout.encoding != representation.encoding {
            return Err(crate::FastnnError::dtype(
                "tensor layout encoding does not match value representation encoding",
            ));
        }
        if layout.row_packed && matches!(layout.encoding, StorageEncoding::Plain) {
            return Err(crate::FastnnError::dtype(
                "row-packed tensor layout requires packed storage encoding",
            ));
        }
        Ok(TensorType {
            shape,
            representation,
            layout,
        })
    }

    pub fn from_parts(
        shape: Vec<DimExpr>,
        representation: ValueRepresentation,
        layout: TensorStorageLayout,
    ) -> Self {
        Self::try_from_parts(shape, representation, layout).expect("invalid canonical tensor type")
    }

    /// The IrDType identity derived from the representation.
    /// This is lossy for quantized types (runtime vs static metadata).
    pub fn dtype(&self) -> IrDType {
        IrDType::from_representation(&self.representation)
    }

    /// Return the same tensor contract with a different logical shape.
    pub fn with_shape(&self, shape: Vec<DimExpr>) -> Self {
        Self {
            shape,
            ..self.clone()
        }
    }

    pub fn value_representation(&self) -> FastnnResult<ValueRepresentation> {
        Ok(self.representation.clone())
    }

    pub fn storage_layout(&self) -> FastnnResult<TensorStorageLayout> {
        Ok(self.layout)
    }

    pub fn plain_storage_byte_width(&self) -> FastnnResult<Option<usize>> {
        Ok(match self.representation.encoding {
            StorageEncoding::Plain => self.representation.storage.plain_byte_width(),
            StorageEncoding::Packed { .. } => None,
        })
    }

    pub fn affine_dequantization(&self) -> Option<(&[f32], &[f32])> {
        match &self.representation.transform {
            RepresentationTransform::AffineDequantization {
                scales, offsets, ..
            } => Some((scales.as_slice(), offsets.as_slice())),
            _ => None,
        }
    }

    pub fn is_native_scalar(&self, scalar: ScalarType) -> bool {
        self.representation.logical == scalar
            && self.representation.storage == scalar
            && matches!(self.representation.encoding, StorageEncoding::Plain)
            && matches!(self.representation.transform, RepresentationTransform::None)
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
        self.layout
            .allocation_bytes(self.representation.storage, &shape)
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
        assert_eq!(IrDType::U4Scaled.plain_storage_byte_width().unwrap(), None);
    }

    #[test]
    fn ir_dtype_exposes_runtime_headers_and_packed_margins_as_layout() {
        let activation = IrDType::I8.storage_layout().unwrap();
        assert_eq!(activation.prefix_bytes, 8);
        assert_eq!(activation.suffix_bytes, 0);

        let packed = IrDType::U4Scaled.storage_layout().unwrap();
        assert!(packed.row_packed);
        assert_eq!(packed.suffix_bytes, 64);
    }

    #[test]
    fn tensor_type_allocation_uses_canonical_layout() {
        let shape = vec![DimExpr::Known(2), DimExpr::Known(9)];
        let packed = TensorType::new(shape.clone(), IrDType::U4Scaled);
        assert_eq!(packed.try_byte_size_with_env(None), Some(80));
        assert_eq!(TensorType::new(shape.clone(), IrDType::I8).byte_size(), 26);
        assert_eq!(TensorType::new(shape, IrDType::F32).byte_size(), 72);

        let symbolic = TensorType::new(
            vec![DimExpr::Known(2), DimExpr::Symbol("K".into())],
            IrDType::U4Scaled,
        );
        let mut env = ShapeEnv::new();
        env.try_bind("K", 9).unwrap();
        assert_eq!(symbolic.try_byte_size_with_env(Some(&env)), Some(80));
    }

    #[test]
    fn maps_unsigned_affine_ir_dtype_to_canonical_representation() {
        let representation = IrDType::U4Scaled.value_representation().unwrap();

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
            RepresentationTransform::RuntimeAffineQuantization
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
        let representation = ValueRepresentation::packed_affine_dequantization(
            ScalarType::I8,
            4,
            crate::types::QuantizationGranularity::PerTensor,
            vec![0.25],
            vec![1.5],
        )
        .unwrap();
        assert!(matches!(
            representation.transform,
            RepresentationTransform::AffineDequantization { ref offsets, .. }
                if offsets == &[1.5]
        ));
        representation.validate().unwrap();
    }

    #[test]
    fn tensor_type_exposes_canonical_representation_queries() {
        let representation = ValueRepresentation::packed_affine_dequantization(
            ScalarType::I8,
            4,
            crate::types::QuantizationGranularity::PerTensor,
            vec![0.25],
            vec![1.5],
        )
        .unwrap();
        let layout = TensorStorageLayout {
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 4,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: PACKED_SIMD_MARGIN_BYTES,
        };
        let tensor_type = TensorType::from_parts(
            vec![DimExpr::Known(2), DimExpr::Known(8)],
            representation.clone(),
            layout,
        );

        assert_eq!(tensor_type.dtype(), IrDType::I8Scaled);
        assert_eq!(tensor_type.value_representation().unwrap(), representation);
        assert_eq!(tensor_type.plain_storage_byte_width().unwrap(), None);
        assert!(tensor_type.storage_layout().unwrap().row_packed);
        assert_eq!(
            tensor_type.affine_dequantization(),
            Some((&[0.25][..], &[1.5][..]))
        );
    }

    #[test]
    fn shape_rewrite_preserves_complete_tensor_contract() {
        let representation = ValueRepresentation::packed_affine_dequantization(
            ScalarType::U4,
            8,
            crate::types::QuantizationGranularity::PerAxis { axis: 0 },
            vec![0.25, 0.5],
            vec![-1.0, -2.0],
        )
        .unwrap();
        let layout = TensorStorageLayout {
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: PACKED_SIMD_MARGIN_BYTES,
        };
        let tensor_type = TensorType::from_parts(
            vec![DimExpr::Known(2), DimExpr::Known(4)],
            representation,
            layout,
        );
        let reshaped = tensor_type.with_shape(vec![DimExpr::Known(8)]);

        assert_eq!(reshaped.shape, vec![DimExpr::Known(8)]);
    }

    #[test]
    fn f4_preserves_affine_offsets_in_canonical_representation() {
        // F4 now maps to RuntimeScaledAffine in metadata-free mode
        let representation = IrDType::F4.value_representation().unwrap();

        assert!(matches!(
            representation.transform,
            RepresentationTransform::RuntimeScaledAffine
        ));
        representation.validate().unwrap();
    }

    #[test]
    fn codebook_dtype_does_not_expose_affine_dequantization() {
        // IrDType is metadata-free - affine_dequantization always returns None
        assert!(IrDType::I4.affine_dequantization().is_none());
    }
}
