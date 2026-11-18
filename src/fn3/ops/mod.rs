//! Stable operation identifiers for the 3.0 inference core.
//!
//! An [`OpId`] is the only thing a plan, an importer or a backend may use to
//! name an operation. Discriminants are part of the serialized plan contract
//! and of the capability-query contract, so they are explicit and grouped into
//! reserved ranges rather than derived from declaration order.
//!
//! # Ranges
//!
//! | Range | Owner |
//! |---|---|
//! | `0x0001 ..= 0x00FF` | alpha/beta scalar and tensor ops (this file) |
//! | `0x0100 ..= 0x01FF` | reserved for quantized ops (post-alpha) |
//! | `0x0200 ..= 0x02FF` | reserved for fused/attention variants (FN3-15) |
//! | `0x0300 ..= 0x03FF` | reserved for structural/planner ops (FN3-10) |
//! | `0x0400 ..= 0xFFFF` | unassigned; must not be used without an ADR update |
//!
//! Adding an op is a two-part change: append a variant with the next free
//! discriminant in its range, and extend [`OpId::ALL`]. Never reuse or
//! renumber a discriminant, and never let two variants share one.
//!
//! # Embedding
//!
//! There is deliberately no `Embedding` op id. A GPT-2 embedding lookup is
//! `Gather` with `axis = 0` over an `F32` weight table and `I64` indices, which
//! keeps the importer mapping one-to-one with ONNX.

use std::fmt;

/// Stable identifier of a graph operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum OpId {
    /// Embedding lookup / ONNX `Gather`: pick slices of `data` along `axis`
    /// using integer indices. Table lookup, not a compute op.
    Gather = 1,
    /// Matrix product. Alpha contract: rank-2 `[M, K] x [K, N] -> [M, N]`.
    MatMul = 2,
    /// Elementwise addition with right-aligned NumPy broadcasting.
    Add = 3,
    /// Elementwise subtraction with broadcasting.
    Sub = 4,
    /// Elementwise multiplication with broadcasting.
    Mul = 5,
    /// Elementwise division with broadcasting.
    Div = 6,
    /// Elementwise negation.
    Neg = 7,
    /// Elementwise power.
    Pow = 8,
    /// Elementwise square root.
    Sqrt = 9,
    /// Elementwise Gauss error function.
    Erf = 10,
    /// Elementwise hyperbolic tangent.
    Tanh = 11,
    /// Gaussian error linear unit, exact (`erf`) form.
    Gelu = 12,
    /// Rectified linear unit.
    Relu = 13,
    /// Numerically stable softmax over one axis.
    Softmax = 14,
    /// Layer normalization over the trailing axes, as a composite contract
    /// (`mean`, `variance`, `scale`, `bias`, `epsilon`).
    LayerNorm = 15,
    /// Mean reduction over a set of axes.
    ReduceMean = 16,
    /// Sum reduction over a set of axes.
    ReduceSum = 17,
    /// Index of the maximum along an axis; produces `I64`.
    ArgMax = 18,
    /// Axis permutation; the only strided view operation in alpha.
    Transpose = 19,
    /// Shape change with identical element count.
    Reshape = 20,
    /// Collapse to a rank-2 (or rank-1) view; no data movement.
    Flatten = 21,
    /// Remove size-1 axes.
    Squeeze = 22,
    /// Insert size-1 axes.
    Unsqueeze = 23,
    /// Join tensors along an axis.
    Concat = 24,
    /// Strided slice with starts/ends/steps/axes.
    Slice = 25,
    /// Split one tensor into several along an axis.
    Split = 26,
    /// Materialize the runtime shape as an `I64` tensor.
    Shape = 27,
    /// Allocate a tensor filled with one value, shaped by an `I64` tensor.
    ConstantOfShape = 28,
    /// Broadcast to a target shape given by an `I64` tensor.
    Expand = 29,
    /// Generate a 1-D sequence from start/limit/delta scalars.
    Range = 30,
    /// Dtype conversion; the only sanctioned dtype-changing op.
    Cast = 31,
    /// Elementwise equality producing a `Bool` mask.
    Equal = 32,
    /// Elementwise less-than producing a `Bool` mask.
    Less = 33,
    /// Elementwise greater-than producing a `Bool` mask.
    Greater = 34,
    /// Select between two tensors using a `Bool` mask, with broadcasting.
    Where = 35,
    /// Triangular mask / selection with a diagonal offset.
    Trilu = 36,
    /// Elementwise sine (RoPE).
    Sin = 37,
    /// Elementwise cosine (RoPE).
    Cos = 38,
    /// Scatter updates into a copy of a tensor at integer index tuples.
    ScatterND = 39,
    /// Fused causal multi-head attention with an explicit KV cache extent.
    ///
    /// Reserved as a composite for GPU backends (FN3-15 / #61). The alpha CPU
    /// backend may compose it from `MatMul`/`Softmax`/`Where`, but must report
    /// `Support::Unsupported` for the fused id rather than approximating it.
    MaskedAttention = 40,
}

impl OpId {
    /// Every op id defined in this revision, in discriminant order.
    pub const ALL: [OpId; 40] = [
        OpId::Gather,
        OpId::MatMul,
        OpId::Add,
        OpId::Sub,
        OpId::Mul,
        OpId::Div,
        OpId::Neg,
        OpId::Pow,
        OpId::Sqrt,
        OpId::Erf,
        OpId::Tanh,
        OpId::Gelu,
        OpId::Relu,
        OpId::Softmax,
        OpId::LayerNorm,
        OpId::ReduceMean,
        OpId::ReduceSum,
        OpId::ArgMax,
        OpId::Transpose,
        OpId::Reshape,
        OpId::Flatten,
        OpId::Squeeze,
        OpId::Unsqueeze,
        OpId::Concat,
        OpId::Slice,
        OpId::Split,
        OpId::Shape,
        OpId::ConstantOfShape,
        OpId::Expand,
        OpId::Range,
        OpId::Cast,
        OpId::Equal,
        OpId::Less,
        OpId::Greater,
        OpId::Where,
        OpId::Trilu,
        OpId::Sin,
        OpId::Cos,
        OpId::ScatterND,
        OpId::MaskedAttention,
    ];

    /// Stable numeric discriminant. This is the value written into plans.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Stable lowercase name used in diagnostics and capability reports.
    pub const fn name(self) -> &'static str {
        match self {
            OpId::Gather => "gather",
            OpId::MatMul => "matmul",
            OpId::Add => "add",
            OpId::Sub => "sub",
            OpId::Mul => "mul",
            OpId::Div => "div",
            OpId::Neg => "neg",
            OpId::Pow => "pow",
            OpId::Sqrt => "sqrt",
            OpId::Erf => "erf",
            OpId::Tanh => "tanh",
            OpId::Gelu => "gelu",
            OpId::Relu => "relu",
            OpId::Softmax => "softmax",
            OpId::LayerNorm => "layernorm",
            OpId::ReduceMean => "reduce_mean",
            OpId::ReduceSum => "reduce_sum",
            OpId::ArgMax => "argmax",
            OpId::Transpose => "transpose",
            OpId::Reshape => "reshape",
            OpId::Flatten => "flatten",
            OpId::Squeeze => "squeeze",
            OpId::Unsqueeze => "unsqueeze",
            OpId::Concat => "concat",
            OpId::Slice => "slice",
            OpId::Split => "split",
            OpId::Shape => "shape",
            OpId::ConstantOfShape => "constant_of_shape",
            OpId::Expand => "expand",
            OpId::Range => "range",
            OpId::Cast => "cast",
            OpId::Equal => "equal",
            OpId::Less => "less",
            OpId::Greater => "greater",
            OpId::Where => "where",
            OpId::Trilu => "trilu",
            OpId::Sin => "sin",
            OpId::Cos => "cos",
            OpId::ScatterND => "scatter_nd",
            OpId::MaskedAttention => "masked_attention",
        }
    }

    /// One-line description of the alpha contract for this op. Used verbatim in
    /// unsupported-op diagnostics so an error explains what was requested.
    pub const fn description(self) -> &'static str {
        match self {
            OpId::Gather => "table lookup with integer indices along one axis",
            OpId::MatMul => "rank-2 f32 matrix product",
            OpId::Add => "elementwise add with broadcasting",
            OpId::Sub => "elementwise subtract with broadcasting",
            OpId::Mul => "elementwise multiply with broadcasting",
            OpId::Div => "elementwise divide with broadcasting",
            OpId::Neg => "elementwise negation",
            OpId::Pow => "elementwise power",
            OpId::Sqrt => "elementwise square root",
            OpId::Erf => "elementwise Gauss error function",
            OpId::Tanh => "elementwise hyperbolic tangent",
            OpId::Gelu => "exact (erf) Gaussian error linear unit",
            OpId::Relu => "rectified linear unit",
            OpId::Softmax => "numerically stable softmax over one axis",
            OpId::LayerNorm => "layer normalization composite over trailing axes",
            OpId::ReduceMean => "mean reduction over axes",
            OpId::ReduceSum => "sum reduction over axes",
            OpId::ArgMax => "index of the maximum along an axis (i64)",
            OpId::Transpose => "axis permutation view",
            OpId::Reshape => "shape change with equal element count",
            OpId::Flatten => "collapse to a rank-2 view",
            OpId::Squeeze => "remove size-1 axes",
            OpId::Unsqueeze => "insert size-1 axes",
            OpId::Concat => "join along an axis",
            OpId::Slice => "strided slice",
            OpId::Split => "split along an axis",
            OpId::Shape => "materialize the runtime shape as an i64 tensor",
            OpId::ConstantOfShape => "filled tensor shaped by an i64 tensor",
            OpId::Expand => "broadcast to an i64 tensor shape",
            OpId::Range => "arithmetic sequence from scalars",
            OpId::Cast => "explicit dtype conversion",
            OpId::Equal => "elementwise equality mask",
            OpId::Less => "elementwise less-than mask",
            OpId::Greater => "elementwise greater-than mask",
            OpId::Where => "masked select with broadcasting",
            OpId::Trilu => "triangular mask with diagonal offset",
            OpId::Sin => "elementwise sine",
            OpId::Cos => "elementwise cosine",
            OpId::ScatterND => "indexed scatter into a copy",
            OpId::MaskedAttention => "fused causal attention over a KV cache extent",
        }
    }

    /// Coarse family, used by capability tables and by the planner to group
    /// instructions. Purely descriptive: it never changes execution semantics.
    pub const fn family(self) -> OpFamily {
        match self {
            OpId::Gather
            | OpId::Transpose
            | OpId::Reshape
            | OpId::Flatten
            | OpId::Squeeze
            | OpId::Unsqueeze
            | OpId::Concat
            | OpId::Slice
            | OpId::Split
            | OpId::Expand => OpFamily::DataMovement,
            OpId::Shape | OpId::ConstantOfShape | OpId::Range => OpFamily::Shape,
            OpId::MatMul => OpFamily::LinearAlgebra,
            OpId::Add
            | OpId::Sub
            | OpId::Mul
            | OpId::Div
            | OpId::Neg
            | OpId::Pow
            | OpId::Sqrt
            | OpId::Erf
            | OpId::Tanh
            | OpId::Gelu
            | OpId::Relu
            | OpId::Sin
            | OpId::Cos
            | OpId::Cast => OpFamily::Elementwise,
            OpId::Softmax | OpId::LayerNorm => OpFamily::Normalization,
            OpId::ReduceMean | OpId::ReduceSum | OpId::ArgMax => OpFamily::Reduction,
            OpId::Equal
            | OpId::Less
            | OpId::Greater
            | OpId::Where
            | OpId::Trilu
            | OpId::ScatterND => OpFamily::Comparison,
            OpId::MaskedAttention => OpFamily::Attention,
        }
    }

    /// `true` for ops whose semantics are a fixed composition of other ops.
    /// A backend may implement a composite directly (fused) or decompose it,
    /// but it must never approximate it.
    pub const fn is_composite(self) -> bool {
        matches!(
            self,
            OpId::Gelu | OpId::LayerNorm | OpId::Softmax | OpId::MaskedAttention
        )
    }

    /// Decode a stable discriminant. `None` means the code is not assigned in
    /// this revision (including reserved ranges).
    pub const fn from_u16(code: u16) -> Option<OpId> {
        match code {
            1 => Some(OpId::Gather),
            2 => Some(OpId::MatMul),
            3 => Some(OpId::Add),
            4 => Some(OpId::Sub),
            5 => Some(OpId::Mul),
            6 => Some(OpId::Div),
            7 => Some(OpId::Neg),
            8 => Some(OpId::Pow),
            9 => Some(OpId::Sqrt),
            10 => Some(OpId::Erf),
            11 => Some(OpId::Tanh),
            12 => Some(OpId::Gelu),
            13 => Some(OpId::Relu),
            14 => Some(OpId::Softmax),
            15 => Some(OpId::LayerNorm),
            16 => Some(OpId::ReduceMean),
            17 => Some(OpId::ReduceSum),
            18 => Some(OpId::ArgMax),
            19 => Some(OpId::Transpose),
            20 => Some(OpId::Reshape),
            21 => Some(OpId::Flatten),
            22 => Some(OpId::Squeeze),
            23 => Some(OpId::Unsqueeze),
            24 => Some(OpId::Concat),
            25 => Some(OpId::Slice),
            26 => Some(OpId::Split),
            27 => Some(OpId::Shape),
            28 => Some(OpId::ConstantOfShape),
            29 => Some(OpId::Expand),
            30 => Some(OpId::Range),
            31 => Some(OpId::Cast),
            32 => Some(OpId::Equal),
            33 => Some(OpId::Less),
            34 => Some(OpId::Greater),
            35 => Some(OpId::Where),
            36 => Some(OpId::Trilu),
            37 => Some(OpId::Sin),
            38 => Some(OpId::Cos),
            39 => Some(OpId::ScatterND),
            40 => Some(OpId::MaskedAttention),
            _ => None,
        }
    }

    /// `true` if `code` falls inside a range reserved for a later milestone.
    /// Reserved codes decode to `None` today but must not be reassigned.
    pub const fn is_reserved(code: u16) -> bool {
        matches!(code, 0x0100..=0x03FF)
    }

    /// Range of discriminants reserved for quantized ops.
    pub const QUANTIZED_RANGE: std::ops::RangeInclusive<u16> = 0x0100..=0x01FF;
    /// Range of discriminants reserved for fused/attention variants.
    pub const FUSED_RANGE: std::ops::RangeInclusive<u16> = 0x0200..=0x02FF;
    /// Range of discriminants reserved for structural/planner ops.
    pub const STRUCTURAL_RANGE: std::ops::RangeInclusive<u16> = 0x0300..=0x03FF;
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Coarse grouping of ops. Descriptive only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpFamily {
    /// Layout / indexing operations that move or reinterpret elements.
    DataMovement,
    /// Operations that produce or consume shape tensors.
    Shape,
    /// Matrix products and contractions.
    LinearAlgebra,
    /// Pointwise arithmetic and transcendental functions.
    Elementwise,
    /// Softmax and normalization.
    Normalization,
    /// Reductions and arg-reductions.
    Reduction,
    /// Comparison, masking and scatter.
    Comparison,
    /// Attention.
    Attention,
}

impl OpFamily {
    /// Stable name for capability reports.
    pub const fn name(self) -> &'static str {
        match self {
            OpFamily::DataMovement => "data_movement",
            OpFamily::Shape => "shape",
            OpFamily::LinearAlgebra => "linear_algebra",
            OpFamily::Elementwise => "elementwise",
            OpFamily::Normalization => "normalization",
            OpFamily::Reduction => "reduction",
            OpFamily::Comparison => "comparison",
            OpFamily::Attention => "attention",
        }
    }
}

/// Static metadata for an op. Looked up through [`OpId::metadata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpMetadata {
    /// The identifier this metadata describes.
    pub id: OpId,
    /// Stable name; always equal to `id.name()`.
    pub name: &'static str,
    /// Coarse family.
    pub family: OpFamily,
    /// Whether the op is a documented composition of other ops.
    pub composite: bool,
    /// One-line alpha contract.
    pub description: &'static str,
}

impl OpId {
    /// Static metadata for this op.
    pub const fn metadata(self) -> OpMetadata {
        OpMetadata {
            id: self,
            name: self.name(),
            family: self.family(),
            composite: self.is_composite(),
            description: self.description(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_is_complete_and_unique() {
        let mut seen = HashSet::new();
        for op in OpId::ALL {
            assert!(seen.insert(op.code()), "duplicate discriminant {op}");
            assert_eq!(OpId::from_u16(op.code()), Some(op));
            assert_eq!(op.metadata().id, op);
            assert_eq!(op.metadata().name, op.name());
        }
        assert_eq!(seen.len(), OpId::ALL.len());
    }

    #[test]
    fn reserved_codes_are_not_assigned() {
        for code in 0x0100..=0x03FFu16 {
            assert!(OpId::from_u16(code).is_none(), "code {code} is assigned");
            assert!(OpId::is_reserved(code));
        }
        assert_eq!(OpId::from_u16(0), None);
        assert_eq!(OpId::from_u16(u16::MAX), None);
    }
}
