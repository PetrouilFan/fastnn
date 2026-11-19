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
//!
//! # Offsets
//!
//! [`StorageWindow`] is a checked byte window `[offset, offset + len)` over
//! shared storage. It is the only way to build a tensor view at a non-zero
//! storage offset, and it enforces both invariants at construction:
//!
//! 1. `offset % dtype.alignment_hint() == 0`, so the window can be reinterpreted
//!    as `dtype` elements without an unaligned load, and
//! 2. `offset + len <= storage.len()`, with the addition checked in `usize`
//!    rather than wrapped.
//!
//! # Zero-length buffers
//!
//! A zero-length [`AlignedBuffer`] owns no allocation at all: it holds a
//! dangling, correctly aligned, non-null pointer and `Drop` deallocates
//! nothing. The allocator is never called with a zero-size layout.
//!
//! # Device-resident storage (FN3-26 / #72)
//!
//! [`Storage`] can describe an allocation whose authoritative bytes live in
//! *device* memory instead of host memory. That is the residency attachment:
//! `Storage::from_resident` takes a [`DeviceResidency`] (an opaque handle owned
//! by the backend that allocated the buffer) and the logical byte length.
//!
//! A resident storage has **no host bytes**. Its host `AlignedBuffer` is empty,
//! so there is nothing stale to return and no hidden copy of the data:
//!
//! - every host byte accessor ([`Storage::as_bytes`], [`Storage::as_slice`],
//!   [`Storage::byte_range`], [`StorageWindow::as_bytes`],
//!   [`Storage::try_make_mut`], [`Storage::try_into_buffer`]) is either empty
//!   or returns [`Fn3Error::NotImplemented`], which names the remedy
//!   ([`Storage::download`] or, at tensor level, `Tensor::to_host`);
//! - [`Storage::download`] is the one explicit materialisation path, and it
//!   asks the owning backend for the bytes.
//!
//! This makes "never return stale bytes" structural: a device-resident tensor
//! cannot be read through the host API at all, and a caller that wants the
//! bytes must pay for an explicit transfer. `Storage::len` stays the *logical*
//! length on both kinds so window and bounds checks are identical.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use bytemuck::Pod;

use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;

/// What a resident storage tells its owner.
///
/// Implemented by a backend (the Vulkan platform layer is the first) so that
/// `storage.rs` keeps no dependency on `ash`. The object must be immutable: a
/// resident storage is never resized or moved, and the backend destroys the
/// device buffer when the last `Arc` to this handle drops.
///
/// # Contract
///
/// - [`DeviceResidency::bytes`] never changes and always equals the
///   [`Storage`]'s logical byte length.
/// - [`DeviceResidency::download`] copies current device bytes into `dst`; it
///   is the only way host code may observe a resident allocation.
/// - `as_any` returns the concrete handle so the owning backend can recover the
///   buffer it created; no other caller may assume a concrete type.
pub trait DeviceResidency: fmt::Debug + Send + Sync {
    /// Backend that owns the buffer (`"vulkan"`).
    fn backend(&self) -> &'static str;

    /// Logical byte length of the device allocation.
    fn bytes(&self) -> usize;

    /// One-line description used in diagnostics and error messages.
    fn describe(&self) -> String;

    /// The concrete handle, for the backend that created it.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Copy `dst.len()` device bytes starting at `offset` into `dst`.
    ///
    /// `offset + dst.len()` must lie inside [`DeviceResidency::bytes`]; a
    /// caller that violates that gets an error, never a truncated copy.
    fn download(&self, offset: usize, dst: &mut [u8]) -> Result<(), Fn3Error>;
}

/// The typed error for a host byte access to device-resident storage.
///
/// It is a hard error, never a panic and never an empty slice read as data: the
/// remedy is an explicit [`Storage::download`] (or `Tensor::to_host`, which
/// wraps it). `what` names that remedy because `Fn3Error::NotImplemented`
/// carries no detail field; the concrete buffer is visible through
/// [`Storage::residency_label`].
fn host_access_denied(what: &'static str) -> Fn3Error {
    Fn3Error::NotImplemented {
        what: match what {
            "read" => {
                "reading device-resident storage through the host byte API; download it \
                 explicitly with Storage::download or Tensor::to_host"
            }
            "write" => {
                "writing device-resident storage through the host byte API; device kernels \
                 write it on the device"
            }
            _ => "host byte access to device-resident storage",
        },
        owner: "FN3-26 (#72)",
    }
}

/// Alignment of every [`AlignedBuffer`] allocation, in bytes.
pub const STORAGE_ALIGNMENT: usize = 64;

/// Alignment hint of a dtype: the byte offset into storage that a view of
/// `dtype` elements must start on.
///
/// For the alpha dtype set the hint equals [`DType::size_bytes`], so an aligned
/// byte offset is always an exact element offset. It is a separate name
/// because it is a *storage* property: a future dtype may be readable from a
/// smaller alignment than its size.
pub trait DTypeAlignment {
    /// Alignment, in bytes, required of a storage offset for this dtype.
    fn alignment_hint(self) -> usize;
}

impl DTypeAlignment for DType {
    fn alignment_hint(self) -> usize {
        self.size_bytes()
    }
}

/// Validate that `byte_len` bytes at `address` can be viewed as `T`.
///
/// Returns a human-readable reason on failure; callers wrap it in
/// [`Fn3Error::StorageCast`] with their own context.
fn check_element_layout<T>(address: *const u8, byte_len: usize) -> Result<(), String> {
    let size = std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>();
    if size == 0 {
        return Err("zero-sized element types are not supported".to_string());
    }
    if !byte_len.is_multiple_of(size) {
        return Err(format!(
            "{byte_len} bytes is not a whole number of {size}-byte elements"
        ));
    }
    if !(address as usize).is_multiple_of(align) {
        return Err(format!(
            "start address {address:p} is not {align}-byte aligned for this element type"
        ));
    }
    Ok(())
}

/// An owned, 64-byte-aligned, zero-initialised byte buffer.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
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
    /// A zero length performs no allocation and returns a
    /// dangling-but-aligned buffer; it never calls the allocator with a
    /// zero-size layout.
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
        // SAFETY: `layout` has a non-zero size (`len != 0` was checked above)
        // and a valid power-of-two alignment (`STORAGE_ALIGNMENT`). `alloc_zeroed`
        // returns either a pointer to a fresh, zeroed allocation of exactly this
        // layout, owned exclusively by this `AlignedBuffer` until `Drop`, or
        // null, which is reported as an error below.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or_else(|| Fn3Error::Allocation {
            bytes: len as u64,
            detail: format!("allocator returned null for a {len}-byte buffer"),
        })?;
        Ok(AlignedBuffer { ptr, len })
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
        AlignedBuffer {
            // A dangling but aligned, non-null pointer. It is never dereferenced
            // (`len == 0` short-circuits every access path) and never passed to
            // `dealloc`.
            ptr: NonNull::new(std::ptr::without_provenance_mut(STORAGE_ALIGNMENT))
                .expect("STORAGE_ALIGNMENT is non-zero"),
            len: 0,
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` for a zero-length buffer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start address of the allocation. Always [`STORAGE_ALIGNMENT`]-aligned.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// The buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is non-null and aligned (dangling and never
        // dereferenced when `len == 0`); otherwise it addresses `len`
        // initialised bytes owned by `self`. The returned slice borrows `self`,
        // so the allocation outlives it and no other reference can mutate it
        // while it lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The buffer as mutable bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len;
        // SAFETY: as `as_bytes`, plus `&mut self` proves exclusive access to
        // the allocation for the lifetime of the returned slice: no other
        // reference (shared or mutable) can exist at the same time.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// View the whole buffer as elements of `T`.
    ///
    /// Fails when `T` has zero size, when the start address is not aligned for
    /// `T`, or when the length is not a whole multiple of `size_of::<T>()`.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        let element = std::any::type_name::<T>();
        check_element_layout::<T>(self.as_ptr(), self.len).map_err(|detail| {
            Fn3Error::StorageCast {
                element,
                byte_len: self.len,
                detail,
            }
        })?;
        bytemuck::try_cast_slice(self.as_bytes()).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len: self.len,
            detail: format!("{error:?}"),
        })
    }

    /// Mutable view of the whole buffer as elements of `T`.
    ///
    /// Applies the same checks as [`AlignedBuffer::as_slice`].
    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        let element = std::any::type_name::<T>();
        let byte_len = self.len;
        check_element_layout::<T>(self.as_ptr(), byte_len).map_err(|detail| {
            Fn3Error::StorageCast {
                element,
                byte_len,
                detail,
            }
        })?;
        bytemuck::try_cast_slice_mut(self.as_bytes_mut()).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len,
            detail: format!("{error:?}"),
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
            .field("len", &self.len)
            .field("align", &STORAGE_ALIGNMENT)
            .finish_non_exhaustive()
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut copy = AlignedBuffer::zeroed(self.len).expect("cloning a live buffer cannot fail");
        copy.as_bytes_mut().copy_from_slice(self.as_bytes());
        copy
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: `len` is non-zero, so this layout has a non-zero size and
            // the same 64-byte alignment that was passed to `alloc_zeroed` in
            // `zeroed`. `self.ptr` was returned by that call, and this is the
            // only deallocation of it: the buffer is uniquely owned and
            // `Drop` runs once.
            unsafe {
                dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.len, STORAGE_ALIGNMENT),
                )
            };
        }
    }
}

/// A shared, read-only handle to an [`AlignedBuffer`], or to a device buffer.
///
/// Cloning is cheap (`Arc`). Writes require [`Storage::try_make_mut`], which
/// only succeeds when this is the last handle. See the module-level aliasing
/// rule.
///
/// A handle is either host-backed ([`Storage::new`]) or device-resident
/// ([`Storage::from_resident`]). Both report the same logical length, so
/// window and bounds checks do not branch; only the byte accessors do. A
/// device-resident handle has no host bytes and returns
/// [`Fn3Error::NotImplemented`] from every host byte accessor — the data is
/// never stale, it is simply not visible without [`Storage::download`].
#[derive(Clone)]
pub struct Storage {
    buffer: Arc<AlignedBuffer>,
    residency: Option<Arc<dyn DeviceResidency>>,
    len: usize,
}

impl Storage {
    /// Take ownership of a host buffer.
    pub fn new(buffer: AlignedBuffer) -> Storage {
        Storage {
            len: buffer.len(),
            buffer: Arc::new(buffer),
            residency: None,
        }
    }

    /// Take ownership of a device allocation of `len` bytes.
    ///
    /// `residency` is the backend's handle; it must describe exactly `len`
    /// bytes ([`DeviceResidency::bytes`] is checked). The handle owns the
    /// device buffer, so it is released when the last [`Storage`] clone drops.
    pub fn from_resident(
        len: usize,
        residency: Arc<dyn DeviceResidency>,
    ) -> Result<Storage, Fn3Error> {
        if residency.bytes() != len {
            return Err(Fn3Error::OutOfBounds {
                what: "Storage::from_resident (device bytes)",
                index: residency.bytes() as u64,
                len: len as u64,
            });
        }
        Ok(Storage {
            buffer: Arc::new(AlignedBuffer::zeroed(0)?),
            residency: Some(residency),
            len,
        })
    }

    /// Allocate `len` zeroed host bytes.
    pub fn zeroed(len: usize) -> Result<Storage, Fn3Error> {
        Ok(Storage::new(AlignedBuffer::zeroed(len)?))
    }

    /// Copy `bytes` into new host storage.
    pub fn from_bytes(bytes: &[u8]) -> Result<Storage, Fn3Error> {
        Ok(Storage::new(AlignedBuffer::from_bytes(bytes)?))
    }

    /// Length in bytes. Identical for host and device storage.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` for zero-length storage.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` when the authoritative bytes live in device memory.
    pub fn is_resident(&self) -> bool {
        self.residency.is_some()
    }

    /// The device handle, when this storage is device-resident.
    pub fn residency(&self) -> Option<&Arc<dyn DeviceResidency>> {
        self.residency.as_ref()
    }

    /// Backend name of a resident storage (`"host"` otherwise).
    pub fn residency_label(&self) -> &'static str {
        self.residency
            .as_ref()
            .map(|residency| residency.backend())
            .unwrap_or("host")
    }

    /// A process-unique identity for the allocation.
    ///
    /// Two handles with the same identity address the same bytes; a backend
    /// uses it to key persistent uploads. It is stable for the life of the
    /// allocation and is never reused for a different buffer.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.buffer) as *const u8 as usize
    }

    /// Read the whole buffer as bytes.
    ///
    /// Empty for device-resident storage; use [`Storage::is_resident`] or
    /// [`Storage::download`]. Host byte accessors on a resident handle are
    /// rejected by [`Storage::host_bytes`] and the tensor layer, never silently
    /// satisfied with a stale copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Read the whole buffer as bytes, rejecting device-resident storage.
    pub fn host_bytes(&self) -> Result<&[u8], Fn3Error> {
        if self.residency.is_some() {
            return Err(host_access_denied("read"));
        }
        Ok(self.buffer.as_bytes())
    }

    /// Copy the whole allocation into `dst`, which must be exactly [`Storage::len`].
    ///
    /// This is the one explicit materialisation path for a device-resident
    /// storage. A host-backed storage copies from its buffer; both forms cost
    /// exactly one copy and never share bytes with the storage.
    pub fn download(&self, dst: &mut [u8]) -> Result<(), Fn3Error> {
        if dst.len() != self.len {
            return Err(Fn3Error::CapacityExceeded {
                what: "Storage::download",
                requested: dst.len() as u64,
                capacity: self.len as u64,
            });
        }
        self.download_range(0, dst)
    }

    /// Copy `dst.len()` bytes starting at `offset` into `dst`.
    ///
    /// The ranged form of [`Storage::download`]: it is what a caller uses to
    /// materialise one row of a large resident tensor (the final logits, for
    /// example) without transferring the rest of it. The range is checked
    /// against [`Storage::len`] on both widths.
    pub fn download_range(&self, offset: usize, dst: &mut [u8]) -> Result<(), Fn3Error> {
        let end = checked_window_end("Storage::download_range", offset, dst.len(), self.len)?;
        let _ = end;
        match &self.residency {
            None => {
                dst.copy_from_slice(&self.buffer.as_bytes()[offset..offset + dst.len()]);
                Ok(())
            }
            Some(residency) => residency.download(offset, dst),
        }
    }

    /// Read the whole buffer as elements of `T`.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        if self.residency.is_some() {
            return Err(host_access_denied("read"));
        }
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

    /// `true` when both handles refer to the same allocation.
    ///
    /// Two handles with equal bytes but distinct allocations are not the same
    /// storage. This is the identity test a backend uses before deciding
    /// whether a write could alias a read.
    pub fn ptr_eq(&self, other: &Storage) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }

    /// A checked sub-range of this storage as bytes.
    ///
    /// `offset + len` is checked in `usize`, never wrapped, and the range must
    /// lie inside the buffer. No alignment is required: raw payload slicing
    /// (for example a serialized weight blob) is a byte operation.
    pub fn byte_range(&self, offset: usize, len: usize) -> Result<&[u8], Fn3Error> {
        let end = checked_window_end("Storage::byte_range", offset, len, self.len())?;
        if self.residency.is_some() {
            return Err(host_access_denied("read"));
        }
        self.as_bytes()
            .get(offset..end)
            .ok_or_else(|| Fn3Error::StorageWindow {
                what: "Storage::byte_range",
                offset: offset as u64,
                len: len as u64,
                storage_len: self.len() as u64,
                detail: format!("byte range [{offset}, {end}) is outside the buffer"),
            })
    }

    /// A checked byte window into this storage for `dtype` elements.
    ///
    /// Enforces both offset invariants at construction and returns
    /// [`Fn3Error::StorageWindow`] instead of panicking:
    ///
    /// 1. `offset % dtype.alignment_hint() == 0`, and
    /// 2. `offset + len <= self.len()`, with the addition checked.
    ///
    /// The window shares the allocation through `Arc`; it never copies. Use it
    /// with [`crate::fn3::tensor::TensorView::from_window`] or
    /// [`crate::fn3::tensor::Tensor::from_window`] to build a tensor whose
    /// element offset refers to the same allocation.
    pub fn window(
        &self,
        offset: usize,
        len: usize,
        dtype: DType,
    ) -> Result<StorageWindow, Fn3Error> {
        let hint = dtype.alignment_hint();
        if !offset.is_multiple_of(hint) {
            return Err(Fn3Error::StorageWindow {
                what: "Storage::window",
                offset: offset as u64,
                len: len as u64,
                storage_len: self.len() as u64,
                detail: format!(
                    "offset {offset} is not a multiple of the {hint}-byte {dtype} alignment hint"
                ),
            });
        }
        let end = checked_window_end("Storage::window", offset, len, self.len())?;
        Ok(StorageWindow {
            storage: self.clone(),
            offset,
            len: end - offset,
        })
    }

    /// Exclusive access to the buffer, or [`Fn3Error::StorageShared`].
    ///
    /// This is the only write path through a shared handle. Device-resident
    /// storage is not host-writable and returns
    /// [`Fn3Error::NotImplemented`]; a resident output is written by a device
    /// kernel, not by a host byte copy.
    pub fn try_make_mut(&mut self) -> Result<&mut AlignedBuffer, Fn3Error> {
        if self.residency.is_some() {
            return Err(host_access_denied("write"));
        }
        let bytes = self.buffer.len();
        let handles = Arc::strong_count(&self.buffer);
        Arc::get_mut(&mut self.buffer).ok_or(Fn3Error::StorageShared { bytes, handles })
    }

    /// Consume the handle and return the buffer when it is the last one.
    ///
    /// Device-resident storage has no host buffer and returns
    /// [`Fn3Error::NotImplemented`].
    pub fn try_into_buffer(self) -> Result<AlignedBuffer, Fn3Error> {
        if self.residency.is_some() {
            return Err(host_access_denied("read"));
        }
        let bytes = self.buffer.len();
        let handles = Arc::strong_count(&self.buffer);
        Arc::try_unwrap(self.buffer).map_err(|_| Fn3Error::StorageShared { bytes, handles })
    }
}

/// Compute `offset + len` and check it against `storage_len`.
///
/// The addition is checked, so a window at `usize::MAX` with a non-zero length
/// is an error rather than a wrap to a small in-bounds offset.
fn checked_window_end(
    what: &'static str,
    offset: usize,
    len: usize,
    storage_len: usize,
) -> Result<usize, Fn3Error> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Fn3Error::StorageWindow {
            what,
            offset: offset as u64,
            len: len as u64,
            storage_len: storage_len as u64,
            detail: "offset + len overflows usize".to_string(),
        })?;
    if end > storage_len {
        return Err(Fn3Error::StorageWindow {
            what,
            offset: offset as u64,
            len: len as u64,
            storage_len: storage_len as u64,
            detail: format!("offset + len = {end} exceeds the storage length {storage_len}"),
        });
    }
    Ok(end)
}

/// A checked byte window `[offset, offset + len)` into shared [`Storage`].
///
/// Created by [`Storage::window`]; it never copies bytes. The offset is
/// guaranteed to satisfy the alignment hint of the dtype it was created for and
/// the window is guaranteed to lie inside the parent buffer.
#[derive(Clone)]
pub struct StorageWindow {
    storage: Storage,
    offset: usize,
    len: usize,
}

impl StorageWindow {
    /// The parent allocation.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Byte offset of the window from the start of the parent allocation.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Length of the window in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` for a zero-length window.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte offset one past the end of the window, from the start of the parent
    /// allocation.
    ///
    /// This cannot overflow: `Storage::window` checked the addition.
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// The window as bytes.
    ///
    /// A window over device-resident storage has no host bytes and returns
    /// [`Fn3Error::NotImplemented`]; see [`StorageWindow::try_as_bytes`] for the
    /// fallible form used by callers that must handle both kinds.
    pub fn as_bytes(&self) -> &[u8] {
        // The range was validated in `Storage::window`, and storage length and
        // pointer never change after construction. A resident storage's host
        // buffer is empty, so the slice is empty rather than stale.
        if self.storage.is_resident() {
            return &[];
        }
        self.storage
            .as_bytes()
            .get(self.offset..self.end())
            .expect("StorageWindow bounds are validated at construction")
    }

    /// The window as bytes, rejecting device-resident storage.
    pub fn try_as_bytes(&self) -> Result<&[u8], Fn3Error> {
        if self.storage.is_resident() {
            return Err(host_access_denied("read"));
        }
        Ok(self.as_bytes())
    }

    /// The window as elements of `T`, with the same checks as
    /// [`AlignedBuffer::as_slice`].
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        if self.storage.is_resident() {
            return Err(host_access_denied("read"));
        }
        let element = std::any::type_name::<T>();
        let byte_len = self.len;
        check_element_layout::<T>(self.as_bytes().as_ptr(), byte_len).map_err(|detail| {
            Fn3Error::StorageCast {
                element,
                byte_len,
                detail,
            }
        })?;
        bytemuck::try_cast_slice(self.as_bytes()).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len,
            detail: format!("{error:?}"),
        })
    }

    /// Element offset of the window relative to the parent storage.
    ///
    /// Exact by construction when the window was created for `dtype`: the
    /// alignment hint equals the element size for every alpha dtype. A caller
    /// that asks for a different `dtype` gets [`Fn3Error::StorageWindow`] if the
    /// byte offset is not a multiple of that dtype's element size.
    pub fn element_offset(&self, dtype: DType) -> Result<usize, Fn3Error> {
        let size = dtype.size_bytes();
        if !self.offset.is_multiple_of(size) {
            return Err(Fn3Error::StorageWindow {
                what: "StorageWindow::element_offset",
                offset: self.offset as u64,
                len: self.len as u64,
                storage_len: self.storage.len() as u64,
                detail: format!(
                    "byte offset {} is not a multiple of the {size}-byte {dtype} element size",
                    self.offset
                ),
            });
        }
        Ok(self.offset / size)
    }

    /// Consume the window and return a handle to the parent allocation.
    ///
    /// The returned handle shares the same allocation; no bytes are copied.
    pub fn into_storage(self) -> Storage {
        self.storage
    }
}

impl fmt::Debug for StorageWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageWindow")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("storage_len", &self.storage.len())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.len())
            .field("handles", &self.handle_count())
            .field("residency", &self.residency_label())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_aligned_and_zeroed() {
        for len in [1usize, 3, 63, 64, 65, 96, 127, 4096] {
            let buffer = AlignedBuffer::zeroed(len).unwrap();
            assert_eq!(buffer.as_ptr() as usize % STORAGE_ALIGNMENT, 0, "len {len}");
            assert_eq!(buffer.len(), len);
            assert!(buffer.as_bytes().iter().all(|&byte| byte == 0));
        }
        let buffer = AlignedBuffer::zeroed(96).unwrap();
        assert_eq!(buffer.as_slice::<f32>().unwrap().len(), 24);
    }

    #[test]
    fn misaligned_lengths_are_reported() {
        let buffer = AlignedBuffer::from_bytes(&[0u8; 6]).unwrap();
        let error = buffer.as_slice::<f32>().unwrap_err();
        let text = error.to_string();
        assert!(text.contains("6 bytes"), "{text}");
        assert!(text.contains("f32"), "{text}");
        assert!(matches!(error, Fn3Error::StorageCast { .. }));

        let mut buffer = AlignedBuffer::from_bytes(&[0u8; 6]).unwrap();
        assert!(buffer.as_mut_slice::<u32>().is_err());
    }

    #[test]
    fn shared_storage_cannot_be_written() {
        let mut storage = Storage::zeroed(16).unwrap();
        let clone = storage.clone();
        assert!(!storage.is_exclusive());
        assert!(storage.ptr_eq(&clone));
        assert!(matches!(
            storage.try_make_mut(),
            Err(Fn3Error::StorageShared {
                bytes: 16,
                handles: 2
            })
        ));
        drop(clone);
        assert!(storage.is_exclusive());
        assert!(storage.try_make_mut().is_ok());
    }

    #[test]
    fn empty_buffers_are_aligned_and_never_dereferenced() {
        let buffer = AlignedBuffer::zeroed(0).unwrap();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.as_ptr() as usize % STORAGE_ALIGNMENT, 0);
        assert_eq!(buffer.as_slice::<f32>().unwrap().len(), 0);
        assert!(buffer.as_bytes().is_empty());
        assert_eq!(AlignedBuffer::from_bytes(&[]).unwrap().len(), 0);
        assert_eq!(Storage::zeroed(0).unwrap().len(), 0);
    }

    #[test]
    fn clone_copies_into_a_distinct_allocation() {
        let buffer = AlignedBuffer::from_bytes(&[1, 2, 3, 4]).unwrap();
        let copy = buffer.clone();
        assert_ne!(buffer.as_ptr(), copy.as_ptr());
        assert_eq!(buffer.as_bytes(), copy.as_bytes());
    }

    #[test]
    fn windows_check_alignment_and_bounds() {
        let storage = Storage::zeroed(32).unwrap();
        let window = storage.window(8, 8, DType::F32).unwrap();
        assert_eq!(window.offset(), 8);
        assert_eq!(window.len(), 8);
        assert_eq!(window.end(), 16);
        assert_eq!(window.element_offset(DType::F32).unwrap(), 2);
        assert_eq!(window.as_slice::<f32>().unwrap().len(), 2);
        assert_ne!(window.as_slice::<f32>().unwrap().as_ptr() as usize, 0);
        assert_eq!(window.as_bytes().len(), 8);
        assert!(storage.window(32, 0, DType::F32).is_ok());
        assert!(storage.window(0, 32, DType::Bool).is_ok());

        let misaligned = storage.window(6, 8, DType::F32).unwrap_err();
        assert!(matches!(misaligned, Fn3Error::StorageWindow { .. }));
        let text = misaligned.to_string();
        assert!(text.contains('6'), "{text}");
        assert!(text.contains("f32"), "{text}");

        assert!(storage.window(8, 32, DType::Bool).is_err());
        assert!(storage.window(33, 0, DType::Bool).is_err());
    }

    #[test]
    fn windows_never_wrap_the_offset_addition() {
        let storage = Storage::zeroed(32).unwrap();
        // Bool has a 1-byte alignment hint, so `usize::MAX` reaches the
        // checked addition instead of being rejected for misalignment.
        let error = storage.window(usize::MAX, 1, DType::Bool).unwrap_err();
        assert!(matches!(error, Fn3Error::StorageWindow { .. }));
        assert!(error.to_string().contains("overflows"), "{error}");
    }

    /// A stand-in for a backend's device allocation: bytes plus the download
    /// contract, no Vulkan needed.
    #[derive(Debug)]
    struct FakeResident {
        bytes: Vec<u8>,
    }

    impl DeviceResidency for FakeResident {
        fn backend(&self) -> &'static str {
            "fake"
        }

        fn bytes(&self) -> usize {
            self.bytes.len()
        }

        fn describe(&self) -> String {
            format!("{} fake bytes", self.bytes.len())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn download(&self, offset: usize, dst: &mut [u8]) -> Result<(), Fn3Error> {
            let end = offset.checked_add(dst.len()).ok_or(Fn3Error::OutOfBounds {
                what: "FakeResident::download",
                index: offset as u64,
                len: dst.len() as u64,
            })?;
            let source = self.bytes.get(offset..end).ok_or(Fn3Error::OutOfBounds {
                what: "FakeResident::download",
                index: end as u64,
                len: self.bytes.len() as u64,
            })?;
            dst.copy_from_slice(source);
            Ok(())
        }
    }

    fn resident_storage(values: &[f32]) -> Storage {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = bytes.len();
        Storage::from_resident(len, Arc::new(FakeResident { bytes })).unwrap()
    }

    #[test]
    fn resident_storage_has_no_host_bytes_and_downloads_explicitly() {
        let mut storage = resident_storage(&[1.5, -2.0, 3.25, 4.0]);
        assert!(storage.is_resident());
        assert_eq!(storage.len(), 16);
        assert_eq!(storage.residency_label(), "fake");
        assert!(
            storage.as_bytes().is_empty(),
            "no host copy, so no stale read"
        );
        let error = storage.host_bytes().unwrap_err();
        assert!(matches!(error, Fn3Error::NotImplemented { .. }), "{error}");
        assert!(error.to_string().contains("to_host"), "{error}");
        assert!(storage.as_slice::<f32>().is_err());
        assert!(storage.byte_range(0, 4).is_err());
        assert!(storage.try_make_mut().is_err());

        let mut bytes = [0u8; 16];
        storage.download(&mut bytes).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert_eq!(values, vec![1.5, -2.0, 3.25, 4.0]);

        // A download of the wrong length is refused, never truncated.
        assert!(matches!(
            storage.download(&mut [0u8; 8]),
            Err(Fn3Error::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn a_resident_window_keeps_bounds_and_refuses_host_bytes() {
        let storage = resident_storage(&[1.0, 2.0, 3.0, 4.0]);
        let window = storage.window(4, 8, DType::F32).unwrap();
        assert_eq!(window.len(), 8);
        assert_eq!(window.element_offset(DType::F32).unwrap(), 1);
        assert!(window.try_as_bytes().is_err());
        assert!(window.as_slice::<f32>().is_err());
        // Bounds and alignment behave exactly as on host storage.
        assert!(storage.window(6, 8, DType::F32).is_err());
        assert!(storage.window(4, 16, DType::F32).is_err());
        assert_eq!(window.into_storage().len(), 16);
    }

    #[test]
    fn from_resident_rejects_a_length_that_disagrees_with_the_handle() {
        let residency = Arc::new(FakeResident { bytes: vec![0; 8] });
        let error = Storage::from_resident(4, residency).unwrap_err();
        assert!(matches!(error, Fn3Error::OutOfBounds { .. }), "{error}");
    }

    #[test]
    fn alignment_hint_is_the_element_size_for_every_dtype() {
        for dtype in DType::ALL {
            assert_eq!(dtype.alignment_hint(), dtype.size_bytes());
            assert!(dtype.alignment_hint() > 0);
        }
    }
}
