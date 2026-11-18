//! Shapes, element strides and bounded symbolic dimensions.
//!
//! # Shape rules
//!
//! - A shape is an ordered list of extents, outermost first (row-major).
//! - Rank 0 is a scalar and has exactly one element.
//! - A zero extent is legal: the tensor has zero elements. Zero extents do not
//!   propagate into neighbour arithmetic (see [`Shape::numel`]).
//! - Every element count and byte count is checked against both `u64` and
//!   `usize`. Overflow is [`Fn3Error::ShapeOverflow`], never a wrap and never a
//!   truncation on a 32-bit target.
//!
//! # Stride rules (alpha views)
//!
//! Strides are measured in **elements**, not bytes, and are signed so that an
//! unsupported layout can be *reported* instead of reinterpreted.
//!
//! Alpha supports exactly two families:
//!
//! 1. contiguous row-major ([`Shape::default_strides`]), and
//! 2. explicitly-strided views with positive strides that do not overlap,
//!    which covers transposes and slices.
//!
//! Negative strides, zero strides in a writable view, and overlapping strides
//! are rejected with [`Fn3Error::InvalidStride`]. A zero stride is accepted only
//! through [`Strides::validate_broadcast_for`], i.e. when the view is explicitly
//! declared as a broadcast view; such a view is read-only (see
//! [`crate::fn3::tensor::TensorMeta`]).
//!
//! # Dynamic bounds
//!
//! [`ShapeEnvelope`] is the per-session bound on the seven symbols that a
//! decoder-shaped model needs. Bounds are inclusive and capped by
//! [`DimBound::MAX_EXTENT`]. A request outside the envelope is
//! [`Fn3Error::DimOutOfEnvelope`], which names the symbol and the bound.

use std::fmt;

use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;

/// Convert a `u64` count to `usize` with an explicit overflow report.
///
/// On a 32-bit target a count can fit in `u64` and still not fit in `usize`;
/// this is the one place that conversion is allowed to happen, and it never
/// truncates.
pub(crate) fn checked_usize(
    value: u64,
    context: &'static str,
    what: &'static str,
) -> Result<usize, Fn3Error> {
    usize::try_from(value).map_err(|_| Fn3Error::ShapeOverflow {
        context,
        axis: None,
        detail: format!(
            "{what} ({value}) does not fit in a usize on this {}-bit target",
            usize::BITS
        ),
    })
}

/// An ordered, row-major tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    dims: Vec<u64>,
}

impl Shape {
    /// The rank-0 scalar shape. Has exactly one element.
    pub fn scalar() -> Self {
        Shape { dims: Vec::new() }
    }

    /// Construct from extents, outermost first.
    pub fn from_dims<I>(dims: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        Shape {
            dims: dims.into_iter().collect(),
        }
    }

    /// The extents, outermost first.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// `true` for rank 0.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// `true` if any extent is zero, i.e. the tensor has no elements.
    pub fn has_zero_extent(&self) -> bool {
        self.dims.contains(&0)
    }

    /// Extent of one axis.
    pub fn dim(&self, axis: usize) -> Option<u64> {
        self.dims.get(axis).copied()
    }

    /// Total element count.
    ///
    /// A scalar is 1. A shape with any zero extent is 0. Otherwise the product
    /// of the extents is checked against `u64` overflow and reported as
    /// [`Fn3Error::ShapeOverflow`].
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        // Checked before multiplying so that `[u64::MAX, 2, 0]` reports 0
        // elements instead of a spurious overflow.
        if self.dims.contains(&0) {
            return Ok(0);
        }
        let mut product: u64 = 1;
        for (axis, &dim) in self.dims.iter().enumerate() {
            product = product.checked_mul(dim).ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "Shape::numel",
                axis: Some(axis),
                detail: format!(
                    "element count of {self} exceeds u64::MAX while multiplying axis {axis} by {dim}"
                ),
            })?;
        }
        Ok(product)
    }

    /// Total element count as a `usize`.
    ///
    /// Checked against both `u64` and `usize` overflow. On a 32-bit target a
    /// count such as `2^32` elements is [`Fn3Error::ShapeOverflow`], never a
    /// truncation to zero. Use this before allocating or iterating; use
    /// [`Shape::numel`] when the count itself is the value of interest.
    pub fn numel_usize(&self) -> Result<usize, Fn3Error> {
        checked_usize(self.numel()?, "Shape::numel_usize", "element count")
    }

    /// Total byte length for `dtype`.
    ///
    /// Checked against both `u64` and `usize` overflow. Returns 0 for a
    /// zero-element shape.
    pub fn byte_len(&self, dtype: DType) -> Result<usize, Fn3Error> {
        let numel = self.numel()?;
        if numel == 0 {
            return Ok(0);
        }
        let elements_per_element = dtype.size_bytes() as u64;
        let bytes = numel
            .checked_mul(elements_per_element)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "Shape::byte_len",
                axis: None,
                detail: format!(
                    "{numel} elements of {dtype} ({elements_per_element} bytes each) exceed u64::MAX"
                ),
            })?;
        checked_usize(bytes, "Shape::byte_len", "byte length")
    }

    /// Canonical contiguous row-major strides, in elements.
    ///
    /// Fails if a stride does not fit in `i64`, which can only happen for a
    /// shape that could never be allocated.
    pub fn default_strides(&self) -> Result<Strides, Fn3Error> {
        let mut strides = vec![0i64; self.rank()];
        let mut accumulator: u64 = 1;
        for axis in (0..self.rank()).rev() {
            strides[axis] = i64::try_from(accumulator).map_err(|_| Fn3Error::ShapeOverflow {
                context: "Shape::default_strides",
                axis: Some(axis),
                detail: format!("stride {accumulator} does not fit in i64 for {self}"),
            })?;
            accumulator = accumulator.checked_mul(self.dims[axis]).ok_or_else(|| {
                Fn3Error::ShapeOverflow {
                    context: "Shape::default_strides",
                    axis: Some(axis),
                    detail: format!("extent product of {self} exceeds u64::MAX"),
                }
            })?;
        }
        Ok(Strides { strides })
    }

    /// `true` if `strides` describe this shape in contiguous row-major order.
    ///
    /// Strides on axes of extent 1 are ignored, matching the usual definition
    /// of contiguity. A zero-element shape is contiguous when the rank matches:
    /// it has no element order to violate, so this never divides or subtracts
    /// on a zero extent. Returns `false` (never panics) on rank mismatch,
    /// negative strides or internal overflow.
    pub fn is_contiguous(&self, strides: &Strides) -> bool {
        if strides.rank() != self.rank() {
            return false;
        }
        if self.has_zero_extent() {
            return true;
        }
        let mut expected: u64 = 1;
        for axis in (0..self.rank()).rev() {
            let stride = strides.strides[axis];
            if self.dims[axis] != 1 && u64::try_from(stride).ok() != Some(expected) {
                return false;
            }
            match expected.checked_mul(self.dims[axis]) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    /// Rank, extents and axis order of `perm` applied to this shape.
    pub fn permuted(&self, perm: &[usize]) -> Result<Shape, Fn3Error> {
        Shape::validate_permutation(perm, self.rank())?;
        Ok(Shape {
            dims: perm.iter().map(|&axis| self.dims[axis]).collect(),
        })
    }

    /// Validate that `perm` is a permutation of `0..rank`.
    pub fn validate_permutation(perm: &[usize], rank: usize) -> Result<(), Fn3Error> {
        if perm.len() != rank {
            return Err(Fn3Error::InvalidShape {
                context: "Shape::validate_permutation",
                detail: format!(
                    "permutation has {} axes but the tensor has rank {rank}",
                    perm.len()
                ),
            });
        }
        let mut seen = vec![false; rank];
        for (position, &axis) in perm.iter().enumerate() {
            if axis >= rank {
                return Err(Fn3Error::InvalidShape {
                    context: "Shape::validate_permutation",
                    detail: format!(
                        "permutation position {position} names axis {axis} of rank {rank}"
                    ),
                });
            }
            if seen[axis] {
                return Err(Fn3Error::InvalidShape {
                    context: "Shape::validate_permutation",
                    detail: format!("permutation repeats axis {axis}"),
                });
            }
            seen[axis] = true;
        }
        Ok(())
    }

    /// `true` if the two shapes have the same extents in some axis order.
    ///
    /// This is the transpose-compatibility check: `[2, 3]` and `[3, 2]` are
    /// permutations of each other, while `[2, 3]` and `[4, 3]` are not.
    pub fn is_permutation_of(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        let mut mine = self.dims.clone();
        let mut theirs = other.dims.clone();
        mine.sort_unstable();
        theirs.sort_unstable();
        mine == theirs
    }

    /// Right-aligned NumPy broadcasting of two shapes for `op`.
    ///
    /// Extents are equal or one of them is 1. Zero extents follow NumPy:
    /// `[0, 3]` broadcasts with `[1, 3]` to `[0, 3]`, but `[0, 3]` and `[2, 3]`
    /// are incompatible.
    pub fn broadcast_with(&self, other: &Shape, op: OpId) -> Result<Shape, Fn3Error> {
        let rank = self.rank().max(other.rank());
        let mut dims = vec![0u64; rank];
        for position in 0..rank {
            let left = self.extent_from_right(position);
            let right = other.extent_from_right(position);
            let extent = match (left, right) {
                (Some(left), Some(right)) => {
                    if left == right {
                        left
                    } else if left == 1 {
                        right
                    } else if right == 1 {
                        left
                    } else {
                        return Err(Fn3Error::ShapeMismatch {
                            op: Some(op),
                            expected: format!("a shape broadcast-compatible with {self}"),
                            actual: format!(
                                "{other} (axis {position} from the right: {left} vs {right})"
                            ),
                        });
                    }
                }
                (Some(extent), None) | (None, Some(extent)) => extent,
                (None, None) => {
                    unreachable!("position < max rank always names at least one operand")
                }
            };
            dims[rank - 1 - position] = extent;
        }
        Ok(Shape { dims })
    }

    fn extent_from_right(&self, position: usize) -> Option<u64> {
        let rank = self.rank();
        position
            .checked_add(1)
            .filter(|&depth| depth <= rank)
            .map(|depth| self.dims[rank - depth])
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (index, dim) in self.dims.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}

impl From<Vec<u64>> for Shape {
    fn from(dims: Vec<u64>) -> Self {
        Shape { dims }
    }
}

impl From<&[u64]> for Shape {
    fn from(dims: &[u64]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }
}

/// Element strides of a view, one per axis.
///
/// Signed on purpose: the type must be able to *carry* a negative stride so
/// validation can name it, even though the alpha view contract rejects it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    strides: Vec<i64>,
}

impl Strides {
    /// Canonical contiguous strides for `shape`.
    pub fn contiguous(shape: &Shape) -> Result<Strides, Fn3Error> {
        shape.default_strides()
    }

    /// Construct from raw element strides without validating them.
    ///
    /// Callers must pass the result through [`Strides::validate_for`] or
    /// [`Strides::validate_broadcast_for`] before building a tensor view.
    pub fn from_elements(strides: Vec<i64>) -> Self {
        Strides { strides }
    }

    /// The raw element strides.
    pub fn as_slice(&self) -> &[i64] {
        &self.strides
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.strides.len()
    }

    /// Stride of one axis.
    pub fn get(&self, axis: usize) -> Option<i64> {
        self.strides.get(axis).copied()
    }

    /// `true` if these strides are contiguous for `shape`.
    pub fn is_contiguous_for(&self, shape: &Shape) -> bool {
        shape.is_contiguous(self)
    }

    /// Number of elements the view spans: `1 + sum((extent - 1) * stride)`.
    ///
    /// Zero for a zero-element shape. Checked. Negative strides are reported as
    /// [`Fn3Error::InvalidStride`].
    pub fn required_len(&self, shape: &Shape) -> Result<u64, Fn3Error> {
        if self.rank() != shape.rank() {
            return Err(Fn3Error::RankMismatch {
                op: None,
                expected: shape.rank(),
                actual: self.rank(),
            });
        }
        if shape.has_zero_extent() {
            return Ok(0);
        }
        let mut len: u64 = 1;
        for (axis, (&dim, &stride)) in shape.dims().iter().zip(&self.strides).enumerate() {
            let stride = u64::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
                axis,
                stride,
                detail: "negative strides are not part of the 3.0 alpha view contract".to_string(),
            })?;
            let last_index = dim.saturating_sub(1);
            let span = last_index
                .checked_mul(stride)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "Strides::required_len",
                    axis: Some(axis),
                    detail: format!("({dim} - 1) * {stride} exceeds u64::MAX for shape {shape}"),
                })?;
            len = len
                .checked_add(span)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "Strides::required_len",
                    axis: Some(axis),
                    detail: format!("span of {shape} exceeds u64::MAX"),
                })?;
        }
        Ok(len)
    }

    /// [`Strides::required_len`] as a `usize`, checked on both widths.
    pub fn required_len_usize(&self, shape: &Shape) -> Result<usize, Fn3Error> {
        checked_usize(
            self.required_len(shape)?,
            "Strides::required_len_usize",
            "spanning element count",
        )
    }

    /// Bytes spanned by this view from element 0, for `dtype`.
    ///
    /// Checked in `u64` and `usize`. This is the span a storage allocation must
    /// provide at offset 0; a view at a non-zero offset additionally needs its
    /// offset (see [`crate::fn3::tensor::TensorMeta::required_bytes`]).
    pub fn required_bytes(&self, shape: &Shape, dtype: DType) -> Result<usize, Fn3Error> {
        let elements = self.required_len(shape)?;
        let bytes = elements
            .checked_mul(dtype.size_bytes() as u64)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "Strides::required_bytes",
                axis: None,
                detail: format!(
                    "spanning {elements} elements of {dtype} ({} bytes each) exceeds u64::MAX",
                    dtype.size_bytes()
                ),
            })?;
        checked_usize(bytes, "Strides::required_bytes", "spanning byte count")
    }

    /// Validate these strides as a writable, non-overlapping view of `shape`.
    ///
    /// Requires positive strides and rejects any axis whose stride is smaller
    /// than the number of elements already covered by lower-stride axes.
    pub fn validate_for(&self, shape: &Shape) -> Result<(), Fn3Error> {
        if self.rank() != shape.rank() {
            return Err(Fn3Error::RankMismatch {
                op: None,
                expected: shape.rank(),
                actual: self.rank(),
            });
        }
        if shape.has_zero_extent() {
            return Ok(());
        }
        for (axis, &stride) in self.strides.iter().enumerate() {
            if stride == 0 {
                return Err(Fn3Error::InvalidStride {
                    axis,
                    stride,
                    detail: "zero strides are only accepted for explicitly broadcast views; \
                             see Strides::validate_broadcast_for"
                        .to_string(),
                });
            }
            if stride < 0 {
                return Err(Fn3Error::InvalidStride {
                    axis,
                    stride,
                    detail: "negative strides are not part of the 3.0 alpha view contract"
                        .to_string(),
                });
            }
        }
        self.check_no_overlap(shape, false)
    }

    /// Validate these strides as an explicitly broadcast view of `shape`.
    ///
    /// Zero strides are accepted (they re-read one element), negative strides
    /// are still rejected, and overlap is checked only between axes that claim
    /// distinct elements. Views validated this way are read-only; see
    /// [`crate::fn3::tensor::TensorMeta`].
    pub fn validate_broadcast_for(&self, shape: &Shape) -> Result<(), Fn3Error> {
        if self.rank() != shape.rank() {
            return Err(Fn3Error::RankMismatch {
                op: None,
                expected: shape.rank(),
                actual: self.rank(),
            });
        }
        if shape.has_zero_extent() {
            return Ok(());
        }
        for (axis, &stride) in self.strides.iter().enumerate() {
            if stride < 0 {
                return Err(Fn3Error::InvalidStride {
                    axis,
                    stride,
                    detail: "negative strides are not part of the 3.0 alpha view contract"
                        .to_string(),
                });
            }
        }
        self.check_no_overlap(shape, true)
    }

    /// Strides of the axes in `perm` order.
    pub fn permuted(&self, perm: &[usize]) -> Result<Strides, Fn3Error> {
        Shape::validate_permutation(perm, self.rank())?;
        Ok(Strides {
            strides: perm.iter().map(|&axis| self.strides[axis]).collect(),
        })
    }

    fn check_no_overlap(&self, shape: &Shape, allow_zero_strides: bool) -> Result<(), Fn3Error> {
        // An axis of extent 1 addresses only coordinate 0, so it can never
        // overlap another axis and must not grow the covered element count.
        // Counting it made a contiguous axis look overlapping with a preceding
        // equal-stride axis: `[2, 1, 4, 4]` has default strides `[16, 16, 4, 1]`,
        // and axis 1 would be rejected against a coverage of 32 contributed by
        // axis 0. Extent-1 axes are ubiquitous in transformer shapes (`[N, 1]`,
        // `[2, 1, 4, 4]`, `[1, 3, 1]`), so this is not an edge case.
        let mut axes: Vec<usize> = (0..self.rank())
            .filter(|&axis| shape.dims()[axis] != 1)
            .filter(|&axis| !(allow_zero_strides && self.strides[axis] == 0))
            .collect();
        axes.sort_by_key(|&axis| self.strides[axis]);
        let mut covered: u64 = 1;
        for axis in axes {
            let stride = self.strides[axis] as u64;
            if stride < covered {
                return Err(Fn3Error::InvalidStride {
                    axis,
                    stride: self.strides[axis],
                    detail: format!(
                        "strides overlap: axis {axis} advances {stride} elements but lower-stride \
                         axes already cover {covered} elements of shape {shape}"
                    ),
                });
            }
            covered =
                covered
                    .checked_mul(shape.dims()[axis])
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "Strides::check_no_overlap",
                        axis: Some(axis),
                        detail: format!("covered element count of {shape} exceeds u64::MAX"),
                    })?;
        }
        Ok(())
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("strides[")?;
        for (index, stride) in self.strides.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{stride}")?;
        }
        f.write_str("]")
    }
}

/// Named dynamic dimensions of a decoder-shaped model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Symbol {
    /// Batch size.
    Batch = 0,
    /// Number of new tokens in this call.
    Seq = 1,
    /// Number of tokens already in the cache.
    Past = 2,
    /// `past + seq`.
    Total = 3,
    /// Query head count.
    NHeads = 4,
    /// Key/value head count (grouped-query attention).
    NKvHeads = 5,
    /// Per-head width.
    HeadDim = 6,
}

impl Symbol {
    /// Number of symbols.
    pub const COUNT: usize = 7;

    /// Every symbol, in declaration order.
    pub const ALL: [Symbol; Symbol::COUNT] = [
        Symbol::Batch,
        Symbol::Seq,
        Symbol::Past,
        Symbol::Total,
        Symbol::NHeads,
        Symbol::NKvHeads,
        Symbol::HeadDim,
    ];

    /// Stable name used in plans and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Symbol::Batch => "batch",
            Symbol::Seq => "seq",
            Symbol::Past => "past",
            Symbol::Total => "total",
            Symbol::NHeads => "n_heads",
            Symbol::NKvHeads => "n_kv_heads",
            Symbol::HeadDim => "head_dim",
        }
    }

    /// Parse a stable name.
    pub fn from_name(name: &str) -> Option<Symbol> {
        Symbol::ALL.into_iter().find(|symbol| symbol.name() == name)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Inclusive bounds on one dynamic dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimBound {
    min: u64,
    max: u64,
}

impl DimBound {
    /// Hard upper limit for any single dynamic extent.
    ///
    /// This is a policy cap, not a hardware limit: it keeps a bad envelope from
    /// authorising a multi-terabyte allocation. It is high enough for a 16M
    /// token cache.
    pub const MAX_EXTENT: u64 = 1 << 24;

    /// Construct inclusive bounds. Requires `min <= max <= MAX_EXTENT`.
    pub fn new(min: u64, max: u64) -> Result<DimBound, Fn3Error> {
        if min > max {
            return Err(Fn3Error::InvalidShape {
                context: "DimBound::new",
                detail: format!("minimum {min} exceeds maximum {max}"),
            });
        }
        if max > DimBound::MAX_EXTENT {
            return Err(Fn3Error::InvalidShape {
                context: "DimBound::new",
                detail: format!(
                    "maximum {max} exceeds the hard cap of {}",
                    DimBound::MAX_EXTENT
                ),
            });
        }
        Ok(DimBound { min, max })
    }

    /// Bounds that allow exactly `extent`.
    pub fn exact(extent: u64) -> Result<DimBound, Fn3Error> {
        DimBound::new(extent, extent)
    }

    /// Inclusive lower bound.
    pub const fn min(self) -> u64 {
        self.min
    }

    /// Inclusive upper bound.
    pub const fn max(self) -> u64 {
        self.max
    }

    /// `true` if `value` is inside the bounds.
    pub const fn contains(self, value: u64) -> bool {
        value >= self.min && value <= self.max
    }

    /// Check one concrete extent against these bounds and `symbol`.
    pub fn check(self, symbol: Symbol, value: u64) -> Result<u64, Fn3Error> {
        if self.contains(value) {
            Ok(value)
        } else {
            Err(Fn3Error::DimOutOfEnvelope {
                symbol: symbol.name(),
                requested: value,
                min: self.min,
                max: self.max,
            })
        }
    }
}

impl fmt::Display for DimBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.min, self.max)
    }
}

/// Inclusive bounds for every dynamic symbol of one prepared model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeEnvelope {
    bounds: [DimBound; Symbol::COUNT],
}

impl ShapeEnvelope {
    /// Construct from bounds in [`Symbol::ALL`] order:
    /// `batch, seq, past, total, n_heads, n_kv_heads, head_dim`.
    pub const fn new(bounds: [DimBound; Symbol::COUNT]) -> ShapeEnvelope {
        ShapeEnvelope { bounds }
    }

    /// Every symbol bounded by `[1, max]`, except `past` and `total`, which
    /// start at 0. Useful for tests and for a first concrete plan.
    pub fn uniform(max: u64) -> Result<ShapeEnvelope, Fn3Error> {
        let full = DimBound::new(1, max)?;
        let from_zero = DimBound::new(0, max)?;
        Ok(ShapeEnvelope::new([
            full, full, from_zero, from_zero, full, full, full,
        ]))
    }

    /// An envelope where every symbol is pinned to the value in `values`.
    pub fn exact(values: &LiveDims) -> Result<ShapeEnvelope, Fn3Error> {
        let mut bounds = [DimBound::exact(0)?; Symbol::COUNT];
        for symbol in Symbol::ALL {
            bounds[symbol as usize] = DimBound::exact(values.get(symbol))?;
        }
        Ok(ShapeEnvelope::new(bounds))
    }

    /// Bound of one symbol.
    pub const fn get(&self, symbol: Symbol) -> DimBound {
        self.bounds[symbol as usize]
    }

    /// Bound of one symbol, by name.
    pub fn bound(&self, symbol: Symbol) -> DimBound {
        self.bounds[symbol as usize]
    }

    /// Copy with one bound replaced.
    pub fn with(mut self, symbol: Symbol, bound: DimBound) -> Self {
        self.bounds[symbol as usize] = bound;
        self
    }

    /// Validate the envelope itself: every bound is well formed.
    pub fn validate(&self) -> Result<(), Fn3Error> {
        for symbol in Symbol::ALL {
            DimBound::new(self.get(symbol).min, self.get(symbol).max)?;
        }
        Ok(())
    }

    /// Validate a concrete shape environment against this envelope.
    pub fn check(&self, values: &LiveDims) -> Result<(), Fn3Error> {
        for symbol in Symbol::ALL {
            self.get(symbol).check(symbol, values.get(symbol))?;
        }
        Ok(())
    }

    /// Validate relationships that must hold for a decoder-shaped model:
    /// `total == past + seq` and `n_heads % n_kv_heads == 0` with
    /// `n_kv_heads >= 1`.
    pub fn check_consistency(&self, values: &LiveDims) -> Result<(), Fn3Error> {
        values.check_total()?;
        if values.n_kv_heads == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "ShapeEnvelope::check_consistency",
                detail: "n_kv_heads must be at least 1".to_string(),
            });
        }
        if !values.n_heads.is_multiple_of(values.n_kv_heads) {
            return Err(Fn3Error::InvalidShape {
                context: "ShapeEnvelope::check_consistency",
                detail: format!(
                    "n_heads ({}) must be a multiple of n_kv_heads ({}) for grouped-query attention",
                    values.n_heads, values.n_kv_heads
                ),
            });
        }
        Ok(())
    }
}

impl Default for ShapeEnvelope {
    /// A small, explicitly bounded decoder envelope. Not model-specific; use
    /// [`ShapeEnvelope::new`] or [`ShapeEnvelope::exact`] for real plans.
    fn default() -> Self {
        ShapeEnvelope::uniform(1024).expect("1024 is within DimBound::MAX_EXTENT")
    }
}

/// Concrete values of the dynamic symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveDims {
    /// Batch size.
    pub batch: u64,
    /// New tokens in this call.
    pub seq: u64,
    /// Tokens already cached.
    pub past: u64,
    /// `past + seq`.
    pub total: u64,
    /// Query heads.
    pub n_heads: u64,
    /// Key/value heads.
    pub n_kv_heads: u64,
    /// Per-head width.
    pub head_dim: u64,
}

impl LiveDims {
    /// Construct with `total = past + seq` (checked).
    pub fn new(
        batch: u64,
        seq: u64,
        past: u64,
        n_heads: u64,
        n_kv_heads: u64,
        head_dim: u64,
    ) -> Result<LiveDims, Fn3Error> {
        let total = past
            .checked_add(seq)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "LiveDims::new",
                axis: None,
                detail: format!("past ({past}) + seq ({seq}) exceeds u64::MAX"),
            })?;
        Ok(LiveDims {
            batch,
            seq,
            past,
            total,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Value of one symbol.
    pub const fn get(&self, symbol: Symbol) -> u64 {
        match symbol {
            Symbol::Batch => self.batch,
            Symbol::Seq => self.seq,
            Symbol::Past => self.past,
            Symbol::Total => self.total,
            Symbol::NHeads => self.n_heads,
            Symbol::NKvHeads => self.n_kv_heads,
            Symbol::HeadDim => self.head_dim,
        }
    }

    /// Set one symbol.
    pub fn set(&mut self, symbol: Symbol, value: u64) {
        match symbol {
            Symbol::Batch => self.batch = value,
            Symbol::Seq => self.seq = value,
            Symbol::Past => self.past = value,
            Symbol::Total => self.total = value,
            Symbol::NHeads => self.n_heads = value,
            Symbol::NKvHeads => self.n_kv_heads = value,
            Symbol::HeadDim => self.head_dim = value,
        }
    }

    /// Check that `total == past + seq`.
    pub fn check_total(&self) -> Result<(), Fn3Error> {
        let expected = self
            .past
            .checked_add(self.seq)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "LiveDims::check_total",
                axis: None,
                detail: format!("past ({}) + seq ({}) exceeds u64::MAX", self.past, self.seq),
            })?;
        if self.total != expected {
            return Err(Fn3Error::InvalidShape {
                context: "LiveDims::check_total",
                detail: format!(
                    "total ({}) must equal past ({}) + seq ({}) = {expected}",
                    self.total, self.past, self.seq
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_is_one_element_and_has_no_strides() {
        let scalar = Shape::scalar();
        assert_eq!(scalar.numel(), Ok(1));
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.default_strides().unwrap().rank(), 0);
        assert_eq!(
            scalar.default_strides().unwrap().required_len(&scalar),
            Ok(1)
        );
        assert!(scalar.is_contiguous(&Strides::from_elements(vec![])));
    }

    #[test]
    fn zero_extent_is_checked_before_multiplication() {
        let empty = Shape::from_dims([u64::MAX, 2, 0]);
        assert_eq!(empty.numel(), Ok(0));
        assert_eq!(empty.byte_len(DType::I64), Ok(0));
    }

    #[test]
    fn numel_overflow_is_reported() {
        let huge = Shape::from_dims([u64::MAX, 2]);
        let error = huge.numel().unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::ShapeOverflow {
                context: "Shape::numel",
                axis: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn transpose_strides_are_contiguous_permutation() {
        let shape = Shape::from_dims([2, 3]);
        let strides = shape.default_strides().unwrap();
        let permuted_shape = shape.permuted(&[1, 0]).unwrap();
        let permuted_strides = strides.permuted(&[1, 0]).unwrap();
        assert_eq!(permuted_shape, Shape::from_dims([3, 2]));
        assert_eq!(permuted_strides.as_slice(), &[1, 3]);
        assert!(!permuted_shape.is_contiguous(&permuted_strides));
        assert_eq!(permuted_strides.required_len(&permuted_shape), Ok(6));
        assert!(shape.is_permutation_of(&permuted_shape));
    }

    #[test]
    fn overlapping_and_zero_strides_are_rejected() {
        let shape = Shape::from_dims([2, 2]);
        let overlap = Strides::from_elements(vec![1, 1]);
        assert!(matches!(
            overlap.validate_for(&shape),
            Err(Fn3Error::InvalidStride { axis: 1, .. })
        ));
        let zero = Strides::from_elements(vec![2, 0]);
        assert!(matches!(
            zero.validate_for(&shape),
            Err(Fn3Error::InvalidStride { .. })
        ));
        zero.validate_broadcast_for(&shape).unwrap();
        let negative = Strides::from_elements(vec![-2, 1]);
        assert!(matches!(
            negative.validate_for(&shape),
            Err(Fn3Error::InvalidStride { stride: -2, .. })
        ));
    }

    #[test]
    fn broadcasting_is_right_aligned() {
        let a = Shape::from_dims([2, 3]);
        let b = Shape::from_dims([3]);
        assert_eq!(
            a.broadcast_with(&b, OpId::Add).unwrap(),
            Shape::from_dims([2, 3])
        );
        let incompatible =
            Shape::from_dims([4, 3]).broadcast_with(&Shape::from_dims([2]), OpId::Add);
        let error = incompatible.unwrap_err();
        assert_eq!(error.op(), Some(OpId::Add));
        assert!(error.to_string().contains("op add"), "{error}");
    }

    #[test]
    fn envelope_rejects_out_of_bounds_with_symbol_and_bound() {
        let envelope = ShapeEnvelope::uniform(8).unwrap();
        let values = LiveDims::new(2, 9, 0, 4, 2, 8).unwrap();
        let error = envelope.check(&values).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("seq = 9"), "{text}");
        assert!(text.contains("[1, 8]"), "{text}");
        assert!(matches!(
            error,
            Fn3Error::DimOutOfEnvelope {
                symbol: "seq",
                requested: 9,
                min: 1,
                max: 8
            }
        ));
    }

    #[test]
    fn dim_bound_rejects_inverted_and_oversized_ranges() {
        assert!(DimBound::new(5, 4).is_err());
        assert!(DimBound::new(0, DimBound::MAX_EXTENT + 1).is_err());
        assert_eq!(DimBound::new(0, 4).unwrap().max(), 4);
    }
}
