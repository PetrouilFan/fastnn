//! Aligned host storage for the 3.0 inference core.
//!
//! # Why not `Vec<u8>`
//!
//! `Vec<u8>` only guarantees 1-byte alignment, so reinterpreting its bytes as
//! `&[f32]` is undefined behaviour even when the length happens to be a
//! multiple of four. [`AlignedBuffer`] owns a 64-byte-aligned allocation (the
//! cache-line size on the targets the alpha CPU backend measures) and exposes
//! element views only through checked casts.
//!
//! # Aliasing rule
//!
//! - [`AlignedBuffer`] is an owning, uniquely-owned allocation. Mutation is
//!   only possible through `&mut self`, so the borrow checker statically
//!   excludes any live `&[T]` alias while a kernel writes.
//! - [`Storage`] is a cheap `Arc` handle for sharing a buffer between a
//!   `PreparedModel`, its plans and every [`crate::fn3::tensor::Tensor`] view.
//!   Shared access is **read-only**: `Storage::as_slice` takes `&self`.
//! - Mutation through a shared handle requires exclusive ownership:
//!   [`Storage::try_make_mut`] fails with [`Fn3Error::StorageShared`] while any
//!   other handle is alive. There is no interior mutability anywhere in this
//!   module, so a `Storage` that is shared cannot be written behind another
//!   handle's back.
//!
//! Buffers are never resized in place. Growing a tensor allocates a new
//! `AlignedBuffer`, which keeps every existing view valid.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use bytemuck::Pod;

use crate::fn3::error::Fn3Error;

/// Alignment of every [`AlignedBuffer`] allocation, in bytes.
pub const STORAGE_ALIGNMENT: usize = 64;

/// An owned, 64-byte-aligned, zero-initialised byte buffer.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` uniquely owns its allocation (no other handle can
// read or write it while it is owned by value), it contains no interior
// mutability, and all element access is bounds- and alignment-checked before a
// reference is created. Sending it to another thread therefore transfers the
// only access path, which is exactly the `Send` contract.
unsafe impl Send for AlignedBuffer {}

// SAFETY: sharing `&AlignedBuffer` only exposes `&[u8]` and checked `&[T]`
// views; there is no method that mutates through `&self`. Concurrent readers
// therefore only ever read immutable bytes, which is `Sync`.
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `len` zeroed bytes with [`STORAGE_ALIGNMENT`] alignment.
    ///
    /// A zero length allocates nothing and returns a dangling-but-aligned
    /// buffer.
    pub fn zeroed(len: usize) -> Result<AlignedBuffer, Fn3Error> {
        if len == 0 {
            return Ok(AlignedBuffer::empty());
        }
        let layout = Layout::from_size_align(len, STORAGE_ALIGNMENT).map_err(|error| {
            Fn3Error::Allocation {
                bytes: len as u64,
                detail: format!(
                    "length is not a valid allocation size for {STORAGE_ALIGNMENT}-byte alignment: {error}"
                ),
            }
        })?;
        // SAFETY: `layout` has a non-zero size and a valid power-of-two
        // alignment. `alloc_zeroed` returns either a pointer to a fresh
        // allocation of exactly this layout, owned by this `AlignedBuffer`
        // until `Drop`, or null, which is reported as an error below.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or_else(|| Fn3Error::Allocation {
            bytes: len as u64,
            detail: format!("allocator returned null for a {len}-byte buffer"),
        })?;
        Ok(AlignedBuffer { ptr, layout })
    }

    /// Copy `bytes` into a new aligned buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<AlignedBuffer, Fn3Error> {
        let mut buffer = AlignedBuffer::zeroed(bytes.len())?;
        buffer.as_bytes_mut().copy_from_slice(bytes);
        Ok(buffer)
    }

    /// Copy a typed slice into a new aligned buffer.
    pub fn from_slice<T: Pod>(values: &[T]) -> Result<AlignedBuffer, Fn3Error> {
        AlignedBuffer::from_bytes(bytemuck::cast_slice(values))
    }

    fn empty() -> AlignedBuffer {
        // SAFETY: constructing a `Layout` with size 0 and a valid alignment
        // cannot fail. The pointer is aligned and non-null but is never
        // dereferenced because every access path checks `len == 0` first.
        let layout = unsafe { Layout::from_size_align_unchecked(0, STORAGE_ALIGNMENT) };
        AlignedBuffer {
            ptr: NonNull::new(std::ptr::without_provenance_mut(STORAGE_ALIGNMENT))
                .expect("64 is non-zero"),
            layout,
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    /// `true` for a zero-length buffer.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start address of the allocation. Always [`STORAGE_ALIGNMENT`]-aligned.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// The buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` addresses `len` initialised bytes owned by `self`, and
        // the returned slice borrows `self`, so the allocation outlives it.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// The buffer as mutable bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        // SAFETY: as `as_bytes`, plus `&mut self` proves exclusive access to
        // the allocation for the duration of the returned slice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// View the whole buffer as elements of `T`.
    ///
    /// Fails when `T` has zero size, when the start address is not aligned for
    /// `T` (it always is for the alpha dtypes), or when the length is not a
    /// whole multiple of `size_of::<T>()`.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        let element = std::any::type_name::<T>();
        if std::mem::size_of::<T>() == 0 {
            return Err(Fn3Error::StorageCast {
                element,
                byte_len: self.len(),
                detail: "zero-sized element types are not supported".to_string(),
            });
        }
        bytemuck::try_cast_slice(self.as_bytes()).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len: self.len(),
            detail: format!(
                "{error:?}; the buffer length must be a multiple of {} and the start \
                 must be {}-byte aligned",
                std::mem::size_of::<T>(),
                std::mem::align_of::<T>()
            ),
        })
    }

    /// Mutable view of the whole buffer as elements of `T`.
    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        let element = std::any::type_name::<T>();
        let byte_len = self.len();
        if std::mem::size_of::<T>() == 0 {
            return Err(Fn3Error::StorageCast {
                element,
                byte_len,
                detail: "zero-sized element types are not supported".to_string(),
            });
        }
        bytemuck::try_cast_slice_mut(self.as_bytes_mut()).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len,
            detail: format!(
                "{error:?}; the buffer length must be a multiple of {} and the start \
                 must be {}-byte aligned",
                std::mem::size_of::<T>(),
                std::mem::align_of::<T>()
            ),
        })
    }

    /// Fill the buffer with zero bytes.
    pub fn zero_fill(&mut self) {
        self.as_bytes_mut().fill(0);
    }

    /// Copy `source` into the buffer. Lengths must match exactly.
    pub fn copy_from_slice(&mut self, source: &[u8]) -> Result<(), Fn3Error> {
        if source.len() != self.len() {
            return Err(Fn3Error::CapacityExceeded {
                what: "AlignedBuffer::copy_from_slice",
                requested: source.len() as u64,
                capacity: self.len() as u64,
            });
        }
        self.as_bytes_mut().copy_from_slice(source);
        Ok(())
    }
}

impl fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len())
            .field("align", &self.layout.align())
            .finish_non_exhaustive()
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut copy =
            AlignedBuffer::zeroed(self.len()).expect("cloning a live buffer cannot fail");
        copy.as_bytes_mut().copy_from_slice(self.as_bytes());
        copy
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: `self.layout` is exactly the layout that was passed to
            // `alloc_zeroed` in `zeroed`, and this is the only deallocation of
            // that pointer.
            unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

/// A shared, read-only handle to an [`AlignedBuffer`].
///
/// Cloning is cheap (`Arc`). Writes require [`Storage::try_make_mut`], which
/// only succeeds when this is the last handle. See the module-level aliasing
/// rule.
#[derive(Clone)]
pub struct Storage {
    buffer: Arc<AlignedBuffer>,
}

impl Storage {
    /// Take ownership of a buffer.
    pub fn new(buffer: AlignedBuffer) -> Storage {
        Storage {
            buffer: Arc::new(buffer),
        }
    }

    /// Allocate `len` zeroed bytes.
    pub fn zeroed(len: usize) -> Result<Storage, Fn3Error> {
        Ok(Storage::new(AlignedBuffer::zeroed(len)?))
    }

    /// Copy `bytes` into new storage.
    pub fn from_bytes(bytes: &[u8]) -> Result<Storage, Fn3Error> {
        Ok(Storage::new(AlignedBuffer::from_bytes(bytes)?))
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// `true` for zero-length storage.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Read the whole buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Read the whole buffer as elements of `T`.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        self.buffer.as_slice()
    }

    /// Number of live handles to the same buffer.
    pub fn handle_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
    }

    /// `true` when no other handle exists, i.e. mutation is possible.
    pub fn is_exclusive(&self) -> bool {
        self.handle_count() == 1
    }

    /// Exclusive access to the buffer, or [`Fn3Error::StorageShared`].
    ///
    /// This is the only write path through a shared handle.
    pub fn try_make_mut(&mut self) -> Result<&mut AlignedBuffer, Fn3Error> {
        let bytes = self.buffer.len();
        let handles = Arc::strong_count(&self.buffer);
        Arc::get_mut(&mut self.buffer).ok_or(Fn3Error::StorageShared { bytes, handles })
    }

    /// Consume the handle and return the buffer when it is the last one.
    pub fn try_into_buffer(self) -> Result<AlignedBuffer, Fn3Error> {
        let bytes = self.buffer.len();
        let handles = Arc::strong_count(&self.buffer);
        Arc::try_unwrap(self.buffer).map_err(|_| Fn3Error::StorageShared { bytes, handles })
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.len())
            .field("handles", &self.handle_count())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_aligned_and_zeroed() {
        let buffer = AlignedBuffer::zeroed(96).unwrap();
        assert_eq!(buffer.as_ptr() as usize % STORAGE_ALIGNMENT, 0);
        assert!(buffer.as_bytes().iter().all(|&byte| byte == 0));
        assert_eq!(buffer.as_slice::<f32>().unwrap().len(), 24);
    }

    #[test]
    fn misaligned_lengths_are_reported() {
        let buffer = AlignedBuffer::from_bytes(&[0u8; 6]).unwrap();
        let error = buffer.as_slice::<f32>().unwrap_err();
        let text = error.to_string();
        assert!(text.contains("6 bytes"), "{text}");
        assert!(text.contains("f32"), "{text}");
    }

    #[test]
    fn shared_storage_cannot_be_written() {
        let mut storage = Storage::zeroed(16).unwrap();
        let clone = storage.clone();
        assert!(matches!(
            storage.try_make_mut(),
            Err(Fn3Error::StorageShared {
                bytes: 16,
                handles: 2
            })
        ));
        drop(clone);
        assert!(storage.try_make_mut().is_ok());
    }

    #[test]
    fn empty_buffers_are_aligned_and_never_dereferenced() {
        let buffer = AlignedBuffer::zeroed(0).unwrap();
        assert!(buffer.is_empty());
        assert_eq!(buffer.as_ptr() as usize % STORAGE_ALIGNMENT, 0);
        assert_eq!(buffer.as_slice::<f32>().unwrap().len(), 0);
    }
}
