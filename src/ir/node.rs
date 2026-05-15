#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Conv2d,
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Neg,
    Abs,
    LeakyRelu,
    Elu,
    Softplus,
    Hardswish,
    Clamp,
    Sign,
    Maximum,
    Minimum,
    LogicalNot,
    Reshape,
    Transpose,
    Concat,
    Slice,
    Flatten,
    Squeeze,
    Unsqueeze,
    ReduceSum,
    ReduceMean,
    ReduceMax,
    LogSoftmax,
    MaxPool,
    AvgPool,
    BatchNorm,
    LayerNorm,
    Softmax,
    BiasAdd,
    Pad,
    Gather,
    ScatterNd,
    Constant(TensorValue),
    Input,
    // ── v2.1 additions (new ops added to complete the AOT pipeline) ────────
    Conv1d,
    Conv3d,
    ConvTranspose2d,
    Prelu,
    RMSNorm,
    Embedding,
    Pow,
    GtScalar,
    LtScalar,
    EqScalar,
    AddScalar,
    ArgMax,
    MulScalar,
    DivScalar,
    /// Mish activation: x * tanh(softplus(x))
    Mish,
    UpsampleNearest2d,
    UpsampleBilinear2d,
    AdaptiveAvgPool2d,
    Repeat,
    CumSum,
    Erf,
    Flip,
    /// Element-wise where(condition, x, y) — selects from x or y based on condition
    Where,
    /// Fused Top-K: produces both values and indices in one kernel.
    TopK,
    // ── v2.1 shape + metadata ops ────────────────────────────────────
    /// Returns the shape of the input tensor as a 1D I64 tensor.
    Shape,
    /// Casts the input tensor to a target dtype (specified via `"to"` attr).
    Cast,
    // ── v2.1 type conversion ops (quantization + precision) ──────────
    /// Quantize F32 → U4/U8 with per-channel scales/zero_points (attribute `"bit_width"`).
    Quantize,
    /// Dequantize U4/U8 → F32 using scales/zero_points from input dtype.
    Dequantize,
    /// Convert F32 → F16 (half-precision).
    ToF16,
    /// Convert F16 → F32 (half-precision).
    ToF32,
    /// Quantize activations F32 → INT8 (per-tensor symmetric, attribute `"scale"`).
    QuantizeActivations,
    /// Dequantize activations INT8 → F32 (per-tensor symmetric).
    DequantizeActivations,
    /// Broadcasts the input tensor to a target shape (second input).
    Expand,
    /// Repeats the input tensor along each axis (repeats from second input).
    Tile,
    /// Range(start, limit, step) — produces a 1D F32 tensor [start, start+step, ..., limit).
    Range,
    // ── v2.1 optimizer opcodes (CPU training via IR) ──────────────────
    /// SGD weight update: weight -= lr * (grad + weight_decay * weight)
    SgdUpdate,
    /// Adam weight update: full Adam optimizer step
    AdamUpdate,
    /// AdamW weight update: Adam with decoupled weight decay
    AdamWUpdate,
    /// Muon weight update: orthogonalized gradient update
    MuonUpdate,
    /// Lion weight update: sign( momentum ) based optimizer
    LionUpdate,
    /// RMSprop weight update: RMSprop optimizer step
    RmspropUpdate,
    /// Scale gradient by a constant factor (for loss scaling / gradient unscaling).
    /// Backward: dx = dy * scale (correctly scales the gradient).
    /// Attribute `"scale"`: f32 scale factor.
    GradientScale,
    // ── v2.2 fusion opcodes ────────────────────────────────────────
    /// Fused residual add + layer_norm/rms_norm: output = norm(input + residual)
    FusedResidualAddNorm,
}

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

    /// Bind a symbol name to a concrete value.
    ///
    /// # Panics
    /// Panics if the symbol is already bound to a *different* value, to prevent
    /// silently inconsistent runtime shapes.
    pub fn bind(&mut self, name: &str, value: u64) {
        if let Some(&existing) = self.symbols.get(name) {
            assert_eq!(
                existing, value,
                "ShapeEnv: symbol '{}' bound to {} then inconsistently to {}",
                name, existing, value
            );
        }
        self.symbols.insert(name.to_string(), value);
    }

    /// Resolve a symbol name to its concrete value, if bound.
    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    /// Build a ShapeEnv by matching input byte sizes against graph input node shapes.
    ///
    /// Returns an error if:
    /// - Input byte count is not evenly divisible by the known-stride element count.
    /// - Two inputs infer different values for the same symbol (via [`bind`](Self::bind)).
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
                let elem_size = node.output_type.dtype.byte_size();
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
                    env.bind(&symbolic[0], unknown_numel as u64);
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
                    env.bind(unbound[0], val as u64);
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
    /// Packed 4-bit (U4x8): 8 values per u32 word.
    /// `scales` and `zero_points` are per-output-channel vectors.
    U4 {
        scales: Vec<f32>,
        zero_points: Vec<f32>,
    },
    /// Packed 8-bit (U8x4): 4 values per u32 word.
    /// `scales` and `zero_points` are per-output-channel vectors.
    U8 {
        scales: Vec<f32>,
        zero_points: Vec<f32>,
    },
}

impl IrDType {
    /// Logical byte size per element for arena planning.
    /// For packed types this is a conservative over-estimate (safe for
    /// worst-case memory planning).  Use [`packed_byte_size`] for the
    /// actual storage size needed for a given logical element count.
    pub fn byte_size(&self) -> usize {
        match self {
            IrDType::F32 => 4,
            IrDType::F16 => 2,
            IrDType::BF16 => 2,
            IrDType::I32 => 4,
            IrDType::I64 => 8,
            IrDType::Bool => 1,
            IrDType::I8 => 1,
            // Conservative logical overestimate (packed data fits in 1 byte/elem).
            IrDType::U4 { .. } => 1,
            IrDType::U8 { .. } => 1,
        }
    }

    /// Actual packed storage size in bytes for a given logical element
    /// count.  For F32/F16/etc. this equals `numel * byte_size()`.
    /// For packed types it computes word-level packing plus the SIMD margin
    /// that [`PackedTensor`] allocates (16 extra u32 words = 64 bytes).
    pub fn packed_byte_size(&self, numel: usize) -> usize {
        match self {
            IrDType::F32 => numel * 4,
            IrDType::F16 | IrDType::BF16 => numel * 2,
            IrDType::I32 => numel * 4,
            IrDType::I64 => numel * 8,
            IrDType::Bool => numel,
            IrDType::I8 => numel,
            // U4x8: 8 nibbles per u32 word (4 bytes)
            // packed_len = ceil(numel / 8) words + SIMD_MARGIN(16)
            IrDType::U4 { .. } => {
                let words = numel.div_ceil(8) + 16; // +16 SIMD_MARGIN
                words * 4
            }
            // U8x4: 4 values per u32 word (4 bytes)
            // packed_len = ceil(numel / 4) words + SIMD_MARGIN(16)
            IrDType::U8 { .. } => {
                let words = numel.div_ceil(4) + 16; // +16 SIMD_MARGIN
                words * 4
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            IrDType::F32 => "f32",
            IrDType::F16 => "f16",
            IrDType::BF16 => "bf16",
            IrDType::I32 => "i32",
            IrDType::I64 => "i64",
            IrDType::Bool => "bool",
            IrDType::I8 => "i8",
            IrDType::U4 { .. } => "u4",
            IrDType::U8 { .. } => "u8",
        }
    }

    pub fn bit_width(&self) -> usize {
        match self {
            IrDType::F32 => 32,
            IrDType::F16 => 16,
            IrDType::BF16 => 16,
            IrDType::I32 => 32,
            IrDType::I64 => 64,
            IrDType::I8 => 8,
            IrDType::Bool => 1,
            IrDType::U4 { .. } => 4,
            IrDType::U8 { .. } => 8,
        }
    }

    pub fn items_per_word(&self) -> usize {
        match self {
            IrDType::F32 => 1,
            IrDType::F16 => 2,
            IrDType::BF16 => 2,
            IrDType::I32 => 1,
            IrDType::I64 => 1,
            IrDType::I8 => 4,
            IrDType::Bool => 32,
            IrDType::U4 { .. } => 8,
            IrDType::U8 { .. } => 4,
        }
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

    pub fn numel(&self) -> Option<u64> {
        let mut total = 1u64;
        for dim in &self.shape {
            match dim.evaluate() {
                Some(v) => total = total.checked_mul(v)?,
                None => return None,
            }
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
    pub fn byte_size_with_env(&self, env: Option<&ShapeEnv>) -> usize {
        let symbol_max = SYMBOL_DIM_MAX.load(Ordering::Relaxed) as usize;
        let numel: usize = self
            .shape
            .iter()
            .map(|d| match d {
                DimExpr::Known(v) => *v as usize,
                DimExpr::Bounded { max, .. } => *max as usize,
                DimExpr::Symbol(_) => match env {
                    Some(e) => d.evaluate_with_env(e).ok().unwrap_or(symbol_max as u64) as usize,
                    None => symbol_max,
                },
            })
            .product();
        // For packed types, use the actual packed byte size (accounts for
        // word-level packing and SIMD margin) rather than the logical per-element
        // overestimate.  The packed data stored in TensorValue::Data already
        // includes the SIMD margin, so the slot must be large enough to hold it.
        match &self.dtype {
            IrDType::U4 { .. } | IrDType::U8 { .. } => self.dtype.packed_byte_size(numel),
            _ => numel * self.dtype.byte_size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorValue {
    Float(f32),
    Int(i64),
    Data {
        bytes: Vec<u8>,
        tensor_type: TensorType,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IRNode {
    pub id: NodeId,
    pub opcode: Opcode,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
    pub secondary_output_type: Option<TensorType>,
    pub attrs: HashMap<String, String>,
    pub name: String,
}

impl IRNode {
    pub fn num_outputs(&self) -> usize {
        if self.secondary_output_type.is_some() {
            2
        } else {
            1
        }
    }
    pub fn output_type_for_index(&self, idx: usize) -> &TensorType {
        match idx {
            0 => &self.output_type,
            1 => self
                .secondary_output_type
                .as_ref()
                .expect("secondary output requested but not set"),
            _ => panic!("output index {} out of range (max 1)", idx),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeGraph {
    pub nodes: Vec<IRNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub required_nodes: Vec<NodeId>,
    pub next_id: NodeId,
}

impl ComputeGraph {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ComputeGraph {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            required_nodes: Vec::new(),
            next_id: 1,
        }
    }

    pub fn add_node(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs: HashMap::new(),
            name: String::new(),
        });
        id
    }

    pub fn add_node_with_name(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        name: &str,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs: HashMap::new(),
            name: name.to_string(),
        });
        id
    }

    /// Add a node with attributes set.
    pub fn add_node_with_attrs(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        attrs: HashMap<String, String>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs,
            name: String::new(),
        });
        id
    }

    pub fn add_constant(&mut self, value: TensorValue) -> NodeId {
        let tensor_type = match &value {
            TensorValue::Float(_) => TensorType::new(Vec::new(), IrDType::F32),
            TensorValue::Int(_) => TensorType::new(Vec::new(), IrDType::I64),
            TensorValue::Data { tensor_type, .. } => tensor_type.clone(),
        };
        self.add_node(Opcode::Constant(value), Vec::new(), tensor_type)
    }

    pub fn add_node_with_secondary_output(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        primary_type: TensorType,
        secondary_type: TensorType,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type: primary_type,
            secondary_output_type: Some(secondary_type),
            attrs: HashMap::new(),
            name: String::new(),
        });
        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<&IRNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut IRNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_inputs(&mut self, inputs: Vec<NodeId>) {
        self.inputs = inputs;
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    pub fn add_required_node(&mut self, node_id: NodeId) {
        if !self.required_nodes.contains(&node_id) {
            self.required_nodes.push(node_id);
        }
    }

    pub fn topological_sort(&self) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }

        for node in &self.nodes {
            for &input_id in &node.inputs {
                if adjacency.contains_key(&input_id) {
                    adjacency
                        .get_mut(&input_id)
                        .expect("adjacency entry for input_id")
                        .push(node.id);
                    *in_degree
                        .get_mut(&node.id)
                        .expect("in_degree entry for node.id") += 1;
                }
            }
        }

        let mut queue: Vec<NodeId> = {
            let mut zero_deg: Vec<NodeId> = in_degree
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&id, _)| id)
                .collect();
            zero_deg.sort(); // deterministic order for zero-degree nodes
            zero_deg
        };

        let mut sorted = Vec::with_capacity(self.nodes.len());
        while let Some(node_id) = queue.pop() {
            sorted.push(node_id);
            if let Some(children) = adjacency.get(&node_id) {
                for &child_id in children {
                    if let Some(deg) = in_degree.get_mut(&child_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(child_id);
                        }
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            panic!("ComputeGraph::topological_sort: cycle detected in the computation graph");
        }

        sorted
    }

    pub fn consumers(&self, node_id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.contains(&node_id))
            .map(|n| n.id)
            .collect()
    }

    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        for node in &mut self.nodes {
            node.inputs.retain(|&i| i != id);
        }
        self.inputs.retain(|&i| i != id);
        self.outputs.retain(|&i| i != id);
        self.required_nodes.retain(|&i| i != id);
    }

    /// Save the ComputeGraph to a .fnn binary file.
    pub fn save_fnn(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a ComputeGraph from a .fnn binary file created by [`save_fnn`](Self::save_fnn).
    pub fn load_fnn(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let graph: ComputeGraph = bincode::deserialize(&bytes)?;
        Ok(graph)
    }
}
