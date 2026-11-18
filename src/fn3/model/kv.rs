//! Session-owned key/value cache: bounded buffers, a live extent and an
//! append that commits or changes nothing.
//!
//! # Why the cache is its own module
//!
//! A decoder's KV cache is the only mutable model state a session owns, and it
//! is the only place where a step can be *partially* applied. Everything about
//! it is therefore explicit here:
//!
//! ```text
//! KvCacheState                      one layer
//!   ├── config        layers/batch/heads/head_dim/capacity/dtype/layout
//!   ├── keys          [batch, n_kv_heads, capacity_tokens, head_dim]
//!   ├── values        [batch, n_kv_heads, capacity_tokens, head_dim]
//!   └── live_len      the semantic extent; capacity_tokens is the bound
//! ```
//!
//! `capacity_tokens` is an allocation bound, never a semantic length. Reads the
//! model sees ([`KvCacheState::live_keys`], [`KvCacheState::live_values`])
//! always stop at `live_len`, so stale bytes past the extent are unobservable.
//!
//! # The append protocol
//!
//! One append writes a newly computed `present` slice at the live extent and
//! advances the extent by exactly that many tokens. It is two-phase:
//!
//! 1. [`KvCacheState::check_append`] validates the extent, the capacity and the
//!    writability of the buffers, and [`KvCacheState::check_present`] validates
//!    the incoming slice's dtype, device and shape. Neither mutates anything.
//! 2. [`KvCacheState::append`] repeats both checks and then copies. All index
//!    arithmetic is done and bounds-checked *before* the first byte is written,
//!    so a passing append cannot fail halfway through.
//!
//! A session calls `check_append` + `check_present` for **every** layer before
//! it appends to any of them, which makes the multi-layer commit atomic at the
//! step level: an instruction or output failure leaves every layer's extent
//! where it was.
//!
//! # Overflow is a typed error, never a truncation
//!
//! Appending past `capacity_tokens` returns
//! [`Fn3Error::CapacityExceeded`] naming the requested extent and the capacity.
//! Nothing is truncated, nothing is reallocated and no token is dropped.
//!
//! # Storage
//!
//! Alpha tensor storage is host memory on every device (see
//! [`crate::fn3::tensor::Tensor::from_storage`]), so a cache built for a Vulkan
//! device is host-staged: the bytes live in an [`crate::fn3::storage::AlignedBuffer`]
//! and the Vulkan backend stages each op that reads them. That is recorded by
//! the backend as `host_staged = true`; it is not a CPU fallback, because no
//! kernel runs on the host. Device-resident cache storage is the follow-up
//! described in `docs/reports/fn3-12-session-kv.md`.

use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::shape::{DimBound, Shape, ShapeEnvelope};
use crate::fn3::storage::Storage;
use crate::fn3::tensor::{Tensor, TensorMeta, TensorView};

/// Layout of the key and value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// `[batch, n_kv_heads, capacity_tokens, head_dim]`, keys and values
    /// identical. The innermost axis is `head_dim`, which keeps an appended
    /// token's vector contiguous.
    BatchHeadsCapacityHeadDim,
}

impl KvCacheLayout {
    /// Stable name used in diagnostics and reports.
    pub const fn name(self) -> &'static str {
        match self {
            KvCacheLayout::BatchHeadsCapacityHeadDim => "batch,heads,capacity,head_dim",
        }
    }
}

/// Largest layer count a KV cache configuration may declare.
///
/// This is a sanity bound: it keeps a corrupt or hostile model header from
/// asking a session to build an unbounded number of cache buffers before the
/// allocator reports a failure.
pub const MAX_KV_LAYERS: u64 = 1024;

/// Geometry and dtype of one layer's key/value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Number of transformer layers that each own a key and value cache.
    pub layers: u64,
    /// Cache capacity in tokens (the safety bound, not the live extent).
    pub capacity_tokens: u64,
    /// Batch size.
    pub batch: u64,
    /// Key/value head count.
    pub n_kv_heads: u64,
    /// Per-head width.
    pub head_dim: u64,
    /// Element type of the cache.
    pub dtype: DType,
    /// Layout of keys and values.
    pub layout: KvCacheLayout,
}

impl KvCacheConfig {
    /// Validate against the model envelope and compute nothing.
    pub fn validate(&self, envelope: &ShapeEnvelope) -> Result<(), Fn3Error> {
        if self.layers == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheConfig::validate",
                detail: "a model with a KV cache needs at least one layer".to_string(),
            });
        }
        if self.layers > MAX_KV_LAYERS {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache layer count",
                requested: self.layers,
                capacity: MAX_KV_LAYERS,
            });
        }
        if self.batch == 0 || self.n_kv_heads == 0 || self.head_dim == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheConfig::validate",
                detail: format!(
                    "batch, n_kv_heads and head_dim must be positive, found {}, {}, {}",
                    self.batch, self.n_kv_heads, self.head_dim
                ),
            });
        }
        if self.capacity_tokens == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheConfig::validate",
                detail: "a KV cache needs capacity for at least one token".to_string(),
            });
        }
        let total = envelope.get(crate::fn3::shape::Symbol::Total);
        if self.capacity_tokens > total.max() {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache capacity vs envelope total",
                requested: self.capacity_tokens,
                capacity: total.max(),
            });
        }
        DimBound::new(self.capacity_tokens, self.capacity_tokens)?;
        Ok(())
    }

    /// Shape of one full-capacity cache buffer.
    pub fn shape(&self) -> Shape {
        Shape::from_dims([
            self.batch,
            self.n_kv_heads,
            self.capacity_tokens,
            self.head_dim,
        ])
    }

    /// Shape of `tokens` tokens of one cache: the shape of an appended slice
    /// and of a live prefix.
    pub fn shape_for(&self, tokens: u64) -> Shape {
        Shape::from_dims([self.batch, self.n_kv_heads, tokens, self.head_dim])
    }

    /// Bytes of one cache buffer (keys or values).
    pub fn buffer_bytes(&self) -> Result<u64, Fn3Error> {
        let elements = self.shape().numel()?;
        elements
            .checked_mul(self.dtype.size_bytes() as u64)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "KvCacheConfig::buffer_bytes",
                axis: None,
                detail: format!("{} elements of {}", self.shape(), self.dtype),
            })
    }

    /// Bytes of one layer (keys plus values).
    pub fn layer_bytes(&self) -> Result<u64, Fn3Error> {
        self.buffer_bytes()?
            .checked_mul(2)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "KvCacheConfig::layer_bytes",
                axis: None,
                detail: format!(
                    "two {}-byte buffers",
                    self.buffer_bytes().unwrap_or(u64::MAX)
                ),
            })
    }

    /// Bytes of every layer (the whole session cache).
    pub fn total_bytes(&self) -> Result<u64, Fn3Error> {
        self.layer_bytes()?
            .checked_mul(self.layers)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "KvCacheConfig::total_bytes",
                axis: None,
                detail: format!("{} layers", self.layers),
            })
    }
}

/// Allocate a zero-filled, contiguous host tensor tagged for `device`.
///
/// This is the session's allocation primitive. [`Tensor::zeros`] deliberately
/// refuses a non-CPU device because device-resident storage has no owner yet;
/// a session-owned buffer is explicit host staging instead: the bytes live in
/// host memory, the metadata names the device, and the backend moves them per
/// op and records `host_staged = true`. There is no path from here to a CPU
/// kernel.
pub(super) fn allocate(shape: Shape, dtype: DType, device: DeviceId) -> Result<Tensor, Fn3Error> {
    let meta = TensorMeta::contiguous(shape, dtype, device)?;
    let storage = Storage::zeroed(meta.required_bytes()?)?;
    Tensor::from_storage(storage, meta)
}

/// Mutable per-layer cache state owned by a [`super::Session`].
///
/// `capacity_tokens` is the allocation bound; `live_len` is the semantic
/// extent. Reads must never address beyond `live_len`.
#[derive(Debug)]
pub struct KvCacheState {
    config: KvCacheConfig,
    keys: Tensor,
    values: Tensor,
    live_len: u64,
}

impl KvCacheState {
    /// Allocate zero-filled key and value buffers at full capacity.
    ///
    /// The buffers are host memory tagged for `device` (see
    /// [`allocate`]); on a Vulkan device that is host staging, not a CPU
    /// compute path.
    pub fn new(config: KvCacheConfig, device: DeviceId) -> Result<KvCacheState, Fn3Error> {
        let shape = config.shape();
        let keys = allocate(shape.clone(), config.dtype, device)?;
        let values = allocate(shape, config.dtype, device)?;
        Ok(KvCacheState {
            config,
            keys,
            values,
            live_len: 0,
        })
    }

    /// Configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Key buffer, shape `[batch, n_kv_heads, capacity, head_dim]`.
    pub fn keys(&self) -> &Tensor {
        &self.keys
    }

    /// Value buffer, shape `[batch, n_kv_heads, capacity, head_dim]`.
    pub fn values(&self) -> &Tensor {
        &self.values
    }

    /// Device the buffers are tagged for.
    pub fn device(&self) -> DeviceId {
        self.keys.device()
    }

    /// Allocated capacity in tokens.
    pub fn capacity(&self) -> u64 {
        self.config.capacity_tokens
    }

    /// Live semantic extent in tokens.
    pub fn live_len(&self) -> u64 {
        self.live_len
    }

    /// Bytes the session holds for this layer's keys and values.
    pub fn allocated_bytes(&self) -> u64 {
        (self.keys.storage().len() + self.values.storage().len()) as u64
    }

    /// Clear the live extent without reallocating or zeroing bytes.
    ///
    /// Stale bytes beyond `live_len` are never readable through the session
    /// contract; a reader uses the live extent, not the buffer length.
    pub fn reset(&mut self) {
        self.live_len = 0;
    }

    /// Check that `additional` more tokens fit, without mutating anything.
    pub fn check_capacity(&self, additional: u64) -> Result<(), Fn3Error> {
        let requested =
            self.live_len
                .checked_add(additional)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "KvCacheState::check_capacity",
                    axis: None,
                    detail: format!("{} + {additional} exceeds u64::MAX", self.live_len),
                })?;
        if requested > self.config.capacity_tokens {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                requested,
                capacity: self.config.capacity_tokens,
            });
        }
        Ok(())
    }

    /// Check everything an append of `additional` tokens needs except the
    /// incoming data: capacity, an addressable layout and writable buffers.
    pub fn check_append(&self, additional: u64) -> Result<(), Fn3Error> {
        self.check_capacity(additional)?;
        if self.config.dtype != DType::F32 {
            return Err(Fn3Error::UnsupportedDType {
                op: None,
                dtype: self.config.dtype,
                context: "KvCacheState::append (the alpha cache is f32)",
            });
        }
        if self.config.layout != KvCacheLayout::BatchHeadsCapacityHeadDim {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheState::append",
                detail: format!(
                    "unsupported KV cache layout {:?}; the alpha session writes \
                     [batch, n_kv_heads, capacity_tokens, head_dim]",
                    self.config.layout
                ),
            });
        }
        // A step writes the buffers in place, so they must be exclusively owned.
        // A caller-held clone is reported before any bytes are touched rather
        // than halfway through the commit.
        for tensor in [&self.keys, &self.values] {
            let storage = tensor.storage();
            if !storage.is_exclusive() {
                return Err(Fn3Error::StorageShared {
                    bytes: storage.len(),
                    handles: storage.handle_count(),
                });
            }
        }
        Ok(())
    }

    /// Validate the slice an append would write: `[batch, n_kv_heads, tokens,
    /// head_dim]`, same dtype and device as the cache.
    pub fn check_present(
        &self,
        keys: &TensorView<'_>,
        values: &TensorView<'_>,
    ) -> Result<u64, Fn3Error> {
        let tokens = keys.shape().dim(2).ok_or_else(|| Fn3Error::RankMismatch {
            op: None,
            expected: 4,
            actual: keys.shape().rank(),
        })?;
        let expected = self.config.shape_for(tokens);
        for (half, view) in [("keys", keys), ("values", values)] {
            if view.shape() != &expected {
                return Err(Fn3Error::ShapeMismatch {
                    op: None,
                    expected: format!(
                        "{expected} (the appended `present` {half} slice for {tokens} token(s))"
                    ),
                    actual: format!("{} ({half})", view.shape()),
                });
            }
            if view.dtype() != self.config.dtype {
                return Err(Fn3Error::DTypeMismatch {
                    op: None,
                    expected: self.config.dtype,
                    actual: view.dtype(),
                });
            }
            if view.device() != self.keys.device() {
                return Err(Fn3Error::DeviceMismatch {
                    op: None,
                    expected: self.keys.device(),
                    actual: view.device(),
                });
            }
            if view.meta().offset() % self.config.head_dim as usize != 0 {
                return Err(Fn3Error::InvalidShape {
                    context: "KvCacheState::check_present",
                    detail: format!(
                        "the {half} slice starts at element offset {}, which is not a whole \
                         number of {}-wide rows; a `present` slice must start on a token boundary",
                        view.meta().offset(),
                        self.config.head_dim
                    ),
                });
            }
        }
        if keys.shape() != values.shape() {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{} (keys)", keys.shape()),
                actual: format!("{} (values)", values.shape()),
            });
        }
        Ok(tokens)
    }

    /// Append a newly computed `present` slice at the live extent.
    ///
    /// `keys` and `values` are the *new* tokens only, shaped
    /// `[batch, n_kv_heads, tokens, head_dim]`; a caller that has a full
    /// `[batch, n_kv_heads, total, head_dim]` `present` output slices it from
    /// `live_len` first (the session does exactly that). On success the live
    /// extent advances by `tokens` and the new value is returned. On failure
    /// nothing changes: the checks run before the first byte is written.
    pub fn append(
        &mut self,
        keys: &TensorView<'_>,
        values: &TensorView<'_>,
    ) -> Result<u64, Fn3Error> {
        let tokens = self.check_present(keys, values)?;
        self.check_append(tokens)?;
        let start = self.live_len;
        // Every destination and source row is bounds-checked here, before the
        // buffers are borrowed mutably and written.
        let key_rows = self.append_rows(&self.keys, keys, tokens)?;
        let value_rows = self.append_rows(&self.values, values, tokens)?;
        write_rows(&mut self.keys, keys, &key_rows)?;
        write_rows(&mut self.values, values, &value_rows)?;
        self.live_len = start + tokens;
        Ok(self.live_len)
    }

    /// Live keys as a contiguous `[batch, n_kv_heads, live_len, head_dim]`
    /// copy on the cache's device.
    ///
    /// A copy, not a view: the cache rows of one token are contiguous but a
    /// prefix of `[batch, heads, capacity, head_dim]` is not, and the plan's
    /// `past_key_values` input must be a contiguous tensor on every backend
    /// (the Vulkan kernels reject strided operands). The copy is
    /// `live_len * head_dim * 4` bytes per head per layer.
    pub fn live_keys(&self) -> Result<Tensor, Fn3Error> {
        self.live_prefix(&self.keys)
    }

    /// Live values, shaped like [`KvCacheState::live_keys`].
    pub fn live_values(&self) -> Result<Tensor, Fn3Error> {
        self.live_prefix(&self.values)
    }

    fn live_prefix(&self, buffer: &Tensor) -> Result<Tensor, Fn3Error> {
        let shape = self.config.shape_for(self.live_len);
        let mut out = allocate(shape, self.config.dtype, buffer.device())?;
        if self.live_len == 0 {
            return Ok(out);
        }
        let head_dim = self.config.head_dim as usize;
        let rows = self.live_rows(buffer)?;
        let source = buffer.as_view()?;
        let source = source.storage_slice::<f32>()?;
        let mut view = out.as_mut_view()?;
        let destination = view.storage_slice_mut::<f32>()?;
        for (source_row, destination_row) in rows {
            destination[destination_row..destination_row + head_dim]
                .copy_from_slice(&source[source_row..source_row + head_dim]);
        }
        Ok(out)
    }

    /// `(source_row, destination_row)` pairs for the live prefix, validated.
    fn live_rows(&self, buffer: &Tensor) -> Result<Vec<(usize, usize)>, Fn3Error> {
        let head_dim = self.config.head_dim as usize;
        let rows = self.config.batch * self.config.n_kv_heads * self.live_len;
        let source_len = buffer.meta().required_elements()? as usize;
        let destination_len = rows as usize * head_dim;
        let mut pairs = Vec::with_capacity(rows as usize);
        for batch in 0..self.config.batch {
            for head in 0..self.config.n_kv_heads {
                for token in 0..self.live_len {
                    let source_row = buffer.meta().checked_index(&[batch, head, token, 0])?;
                    let destination_row = ((batch * self.config.n_kv_heads + head) * self.live_len
                        + token) as usize
                        * head_dim;
                    if source_row + head_dim > source_len
                        || destination_row + head_dim > destination_len
                    {
                        return Err(Fn3Error::OutOfBounds {
                            what: "KvCacheState::live_prefix",
                            index: (source_row + head_dim) as u64,
                            len: source_len as u64,
                        });
                    }
                    pairs.push((source_row, destination_row));
                }
            }
        }
        Ok(pairs)
    }

    /// `(source_row, destination_row)` pairs for an append, validated.
    fn append_rows(
        &self,
        buffer: &Tensor,
        present: &TensorView<'_>,
        tokens: u64,
    ) -> Result<Vec<(usize, usize)>, Fn3Error> {
        let head_dim = self.config.head_dim as usize;
        let destination_len = buffer.meta().required_elements()? as usize;
        // A `present` slice is a strided view into the whole `present` output,
        // so its first addressed element is at `offset`, not at zero. The bound
        // an index must satisfy is therefore `offset + span`, not the view's
        // element count.
        let source_len = present.meta().required_elements()? as usize;
        let rows = self.config.batch * self.config.n_kv_heads * tokens;
        let mut pairs = Vec::with_capacity(rows as usize);
        for batch in 0..self.config.batch {
            for head in 0..self.config.n_kv_heads {
                for token in 0..tokens {
                    let source_row = present.meta().checked_index(&[batch, head, token, 0])?;
                    let destination_row =
                        buffer
                            .meta()
                            .checked_index(&[batch, head, self.live_len + token, 0])?;
                    if source_row + head_dim > source_len
                        || destination_row + head_dim > destination_len
                    {
                        return Err(Fn3Error::OutOfBounds {
                            what: "KvCacheState::append",
                            index: (destination_row + head_dim) as u64,
                            len: destination_len as u64,
                        });
                    }
                    pairs.push((source_row, destination_row));
                }
            }
        }
        Ok(pairs)
    }
}

/// Copy validated rows into `buffer`.
///
/// The rows were produced and bounds-checked by [`KvCacheState::append_rows`]
/// or [`KvCacheState::live_rows`] against the same buffer and source, so the
/// slice indexing below cannot go out of bounds; an error here would be a bug
/// in that pairing, not caller input.
fn write_rows(
    buffer: &mut Tensor,
    present: &TensorView<'_>,
    rows: &[(usize, usize)],
) -> Result<(), Fn3Error> {
    let head_dim = present
        .meta()
        .shape()
        .dim(3)
        .ok_or_else(|| Fn3Error::RankMismatch {
            op: None,
            expected: 4,
            actual: present.meta().shape().rank(),
        })? as usize;
    let source = present.storage_slice::<f32>()?;
    let mut view = buffer.as_mut_view()?;
    let destination = view.storage_slice_mut::<f32>()?;
    for &(source_row, destination_row) in rows {
        destination[destination_row..destination_row + head_dim]
            .copy_from_slice(&source[source_row..source_row + head_dim]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::shape::ShapeEnvelope;

    fn config(capacity: u64) -> KvCacheConfig {
        KvCacheConfig {
            layers: 1,
            capacity_tokens: capacity,
            batch: 1,
            n_kv_heads: 2,
            head_dim: 4,
            dtype: DType::F32,
            layout: KvCacheLayout::BatchHeadsCapacityHeadDim,
        }
    }

    fn envelope() -> ShapeEnvelope {
        ShapeEnvelope::uniform(8).unwrap()
    }

    fn slice(rows: &[f32], tokens: u64) -> Tensor {
        Tensor::from_f32(config(8).shape_for(tokens), rows).unwrap()
    }

    #[test]
    fn append_writes_the_new_rows_at_the_live_extent() {
        let mut cache = KvCacheState::new(config(8), DeviceId::Cpu).unwrap();
        assert_eq!(cache.live_len(), 0);
        assert_eq!(cache.capacity(), 8);
        assert_eq!(cache.keys().shape(), &Shape::from_dims([1, 2, 8, 4]));

        // Token 0: head 0 = 1..=4, head 1 = 5..=8.
        let first = slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1);
        let second = slice(&[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], 1);
        assert_eq!(
            cache.append(&first.as_view().unwrap(), &second.as_view().unwrap()),
            Ok(1)
        );
        assert_eq!(cache.live_len(), 1);
        assert_eq!(
            cache.live_keys().unwrap().as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            cache.live_values().unwrap().as_f32_slice().unwrap(),
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        );

        // Token 1 lands directly after it, per head.
        let third = slice(&[21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0], 1);
        let fourth = slice(&[31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0], 1);
        cache
            .append(&third.as_view().unwrap(), &fourth.as_view().unwrap())
            .unwrap();
        assert_eq!(cache.live_len(), 2);
        assert_eq!(
            cache.live_keys().unwrap().as_f32_slice().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 21.0, 22.0, 23.0, 24.0, 5.0, 6.0, 7.0, 8.0, 25.0, 26.0, 27.0,
                28.0
            ]
        );
    }

    #[test]
    fn appending_past_capacity_is_a_typed_error_and_changes_nothing() {
        let mut cache = KvCacheState::new(config(2), DeviceId::Cpu).unwrap();
        let two = slice(&[1.0; 16], 2);
        cache
            .append(&two.as_view().unwrap(), &two.as_view().unwrap())
            .unwrap();
        assert_eq!(cache.live_len(), 2);

        let one = slice(&[9.0; 8], 1);
        let error = cache
            .append(&one.as_view().unwrap(), &one.as_view().unwrap())
            .unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                requested: 3,
                capacity: 2
            }
        ));
        assert_eq!(cache.live_len(), 2, "a rejected append changes nothing");
        assert_eq!(
            cache.live_keys().unwrap().as_f32_slice().unwrap(),
            &[1.0; 16],
            "the rejected bytes were never written"
        );

        cache.reset();
        assert_eq!(cache.live_len(), 0);
        assert_eq!(cache.live_keys().unwrap().numel(), Ok(0));
    }

    #[test]
    fn a_shared_buffer_is_refused_before_any_write() {
        let mut cache = KvCacheState::new(config(4), DeviceId::Cpu).unwrap();
        let held = cache.keys().clone();
        let one = slice(&[1.0; 8], 1);
        let error = cache
            .append(&one.as_view().unwrap(), &one.as_view().unwrap())
            .unwrap_err();
        assert!(matches!(error, Fn3Error::StorageShared { .. }), "{error}");
        assert_eq!(cache.live_len(), 0);
        drop(held);
        cache
            .append(&one.as_view().unwrap(), &one.as_view().unwrap())
            .unwrap();
    }

    #[test]
    fn present_shape_and_dtype_are_validated() {
        let mut cache = KvCacheState::new(config(8), DeviceId::Cpu).unwrap();
        let wrong_tokens = slice(&[1.0; 16], 2);
        let one = slice(&[1.0; 8], 1);
        let error = cache
            .append(&wrong_tokens.as_view().unwrap(), &one.as_view().unwrap())
            .unwrap_err();
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert_eq!(cache.live_len(), 0);
    }

    #[test]
    fn config_bounds_are_checked() {
        let mut bad = config(9);
        let error = bad.validate(&envelope()).unwrap_err();
        assert!(
            matches!(error, Fn3Error::CapacityExceeded { .. }),
            "{error}"
        );
        bad.capacity_tokens = 8;
        bad.validate(&envelope()).unwrap();
        bad.layers = 0;
        assert!(bad.validate(&envelope()).is_err());
        bad.layers = 1;
        bad.batch = 0;
        assert!(bad.validate(&envelope()).is_err());
        bad.batch = 1;
        bad.capacity_tokens = 0;
        assert!(bad.validate(&envelope()).is_err());
        assert_eq!(config(8).buffer_bytes().unwrap(), 8 * 4 * 2 * 4);
    }
}
