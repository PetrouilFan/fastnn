//! Tensor metadata, views and owned tensors.
//!
//! # Device affinity
//!
//! Every tensor carries a [`DeviceId`]. Metadata construction does not require
//! the device to be executable in this build (a plan may be validated for
//! Vulkan before a Vulkan backend exists), but allocation
//! ([`Tensor::zeros`]) and execution do: both report
//! [`Fn3Error::DeviceUnavailable`] instead of substituting the CPU.
//!
//! # Views
//!
//! [`TensorMeta`] is the single source of truth for shape, strides, offset,
//! dtype and device. Three containers use it:
//!
//! | Type | Ownership | Writable |
//! |---|---|---|
//! | [`Tensor`] | owns its [`Storage`] | yes, when storage is exclusive |
//! | [`TensorView`] | borrows a `Storage` | no |
//! | [`TensorMut`] | borrows an `AlignedBuffer` exclusively | yes |
//!
//! A view validated as a broadcast view ([`TensorMeta::is_broadcast`]) is
//! read-only: several coordinates address the same element.
//!
//! # Offsets
//!
//! `offset` is measured in **elements** from the start of the storage. Kernels
//! index storage with `offset + sum(coord[axis] * stride[axis])`, so a view of
//! a slice never needs to copy.
//!
//! An offset that is measured in **bytes** first goes through
//! [`crate::fn3::storage::StorageWindow`], which checks
//! `offset % dtype.alignment_hint() == 0` and `offset + len <= storage.len()`
//! before any metadata exists. [`TensorView::from_window`] and
//! [`Tensor::from_window`] then derive the element offset from that window, so
//! a byte offset can never point between two elements or past the allocation.
//!
//! # Aliasing and mutation
//!
//! Shared storage is read-only: [`TensorView`] and [`Tensor::as_view`] only
//! ever read, and [`Storage::try_make_mut`] succeeds only when the handle is
//! the last one. The rule a backend must obey is therefore:
//!
//! > **A write must not alias a read.** An output element that an input view
//! > can also address must not be written, in either order.
//!
//! The safe API makes that structural rather than a convention:
//!
//! - [`Tensor::as_mut_view`] fails with [`Fn3Error::StorageShared`] while any
//!   other `Storage` handle exists, which includes every live [`Tensor`] clone
//!   and every live [`TensorView`] of the same allocation.
//! - [`TensorMut`] borrows an [`AlignedBuffer`] mutably, so no `&Storage`
//!   pointing at that buffer can exist at the same time.
//! - [`Tensor::shares_storage_with`] and [`Storage::ptr_eq`] let a backend
//!   assert allocation identity before deciding that two operands are
//!   distinct.
//!
//! A backend that computes raw indices must preserve this rule when it writes:
//! it may not write an output slot through a pointer obtained from an input,
//! and it may not assume that two `TensorMut` outputs point at distinct
//! allocations unless it checked. Plan construction rejects a slot that is both
//! input and output ([`crate::fn3::model::ExecutionPlan`]).
//!
//! # Materialisation
//!
//! A strided view is fully supported by metadata indexing, but some operations
//! need one contiguous window of bytes. [`Tensor::to_contiguous`] and
//! [`TensorView::to_contiguous`] return an owned contiguous tensor whose bytes
//! are the logical row-major order of the view; they never reinterpret a
//! strided window as if it were contiguous.
//!
//! # Device-resident tensors (FN3-26 / #72)
//!
//! A tensor whose [`Storage`] is device-resident
//! ([`crate::fn3::storage::Storage::from_resident`]) carries its authoritative
//! bytes in device memory. It is the same tensor to the planner and the
//! kernels — same [`TensorMeta`], same strides, same device — but it has no
//! host bytes:
//!
//! - host byte accessors ([`Tensor::as_f32_slice`], [`TensorView::as_slice`],
//!   [`TensorMut::as_mut_slice`], …) return [`Fn3Error::NotImplemented`] and
//!   name the remedy instead of returning empty or stale bytes;
//! - [`Tensor::to_host`] downloads the allocation explicitly and returns a
//!   host tensor with identical metadata;
//! - a device kernel writes a resident output through
//!   [`Tensor::as_mut_view`], which is still an exclusive-borrow operation:
//!   the storage must be uniquely owned.
//!
//! CPU tensors are never resident, so the CPU backend's semantics are
//! unaffected: every code path it takes is the host path above.

use std::fmt;

use bytemuck::Pod;

use crate::fn3::device::{DeviceId, DeviceResolveError};
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::{checked_usize, Shape, Strides};
use crate::fn3::storage::{AlignedBuffer, Storage, StorageWindow};

/// Validated description of a tensor: dtype, device, shape, strides, offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    dtype: DType,
    device: DeviceId,
    shape: Shape,
    strides: Strides,
    offset: usize,
    broadcast: bool,
}

impl TensorMeta {
    /// Contiguous row-major metadata for `shape`.
    pub fn contiguous(
        shape: Shape,
        dtype: DType,
        device: DeviceId,
    ) -> Result<TensorMeta, Fn3Error> {
        let strides = shape.default_strides()?;
        TensorMeta::build(shape, strides, 0, dtype, device, false)
    }

    /// Explicitly-strided metadata. Strides must be positive and non-overlapping.
    pub fn strided(
        shape: Shape,
        strides: Strides,
        offset: usize,
        dtype: DType,
        device: DeviceId,
    ) -> Result<TensorMeta, Fn3Error> {
        TensorMeta::build(shape, strides, offset, dtype, device, false)
    }

    /// Explicitly broadcast metadata. Zero strides are allowed and the
    /// resulting view is read-only.
    pub fn broadcast(
        shape: Shape,
        strides: Strides,
        offset: usize,
        dtype: DType,
        device: DeviceId,
    ) -> Result<TensorMeta, Fn3Error> {
        TensorMeta::build(shape, strides, offset, dtype, device, true)
    }

    fn build(
        shape: Shape,
        strides: Strides,
        offset: usize,
        dtype: DType,
        device: DeviceId,
        broadcast: bool,
    ) -> Result<TensorMeta, Fn3Error> {
        if broadcast {
            strides.validate_broadcast_for(&shape)?;
        } else {
            strides.validate_for(&shape)?;
        }
        let meta = TensorMeta {
            dtype,
            device,
            shape,
            strides,
            offset,
            broadcast,
        };
        // Reject metadata that can never address a buffer on this target:
        // offset + span must fit in u64 (and in usize once scaled to bytes).
        // The element offset is always aligned to the dtype's alignment hint
        // because the hint equals the element size for every alpha dtype.
        meta.required_bytes()?;
        Ok(meta)
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device the tensor lives on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Logical shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Element strides.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Element offset into the storage.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// `true` if this is an explicitly broadcast (read-only) view.
    pub fn is_broadcast(&self) -> bool {
        self.broadcast
    }

    /// `true` if the view may be written element by element.
    pub fn is_writable(&self) -> bool {
        !self.broadcast
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of elements in the view.
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        self.shape.numel()
    }

    /// Number of elements in the view, checked against `usize`.
    pub fn numel_usize(&self) -> Result<usize, Fn3Error> {
        self.shape.numel_usize()
    }

    /// Byte length of the view's elements (not of its span).
    pub fn byte_len(&self) -> Result<usize, Fn3Error> {
        self.shape.byte_len(self.dtype)
    }

    /// Checked byte offset of the first addressed element from the start of
    /// storage: `offset * dtype.size_bytes()`.
    ///
    /// Fails with [`Fn3Error::ShapeOverflow`] instead of wrapping when the
    /// element offset cannot be scaled to bytes in a `usize`.
    pub fn byte_offset(&self) -> Result<usize, Fn3Error> {
        self.offset
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "TensorMeta::byte_offset",
                axis: None,
                detail: format!(
                    "element offset {} times {} bytes per {} exceeds usize::MAX",
                    self.offset,
                    self.dtype.size_bytes(),
                    self.dtype
                ),
            })
    }

    /// Elements addressed from the start of storage: `offset + span`.
    ///
    /// A zero-element view reports its offset, not zero: the view still starts
    /// somewhere inside its storage, and a later window or slice operation must
    /// be able to see that start.
    pub fn required_elements(&self) -> Result<u64, Fn3Error> {
        let span = self.strides.required_len(&self.shape)?;
        let offset = self.offset as u64;
        offset
            .checked_add(span)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "TensorMeta::required_elements",
                axis: None,
                detail: format!("offset {offset} plus span {span} exceeds u64::MAX"),
            })
    }

    /// Bytes that storage must provide for this view, checked on both widths.
    pub fn required_bytes(&self) -> Result<usize, Fn3Error> {
        let elements = self.required_elements()?;
        if elements == 0 {
            return Ok(0);
        }
        let bytes = elements
            .checked_mul(self.dtype.size_bytes() as u64)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "TensorMeta::required_bytes",
                axis: None,
                detail: format!(
                    "{elements} elements of {} ({} bytes each) exceed u64::MAX",
                    self.dtype,
                    self.dtype.size_bytes()
                ),
            })?;
        checked_usize(bytes, "TensorMeta::required_bytes", "required byte count")
    }

    /// Checked element index of `coordinates` from the start of storage.
    ///
    /// Every coordinate must be inside its extent. The returned index is what a
    /// kernel passes to `storage_slice::<T>()`; the addition and the
    /// coordinate-times-stride product are both checked, and the result is
    /// converted to `usize` without truncation.
    pub fn checked_index(&self, coordinates: &[u64]) -> Result<usize, Fn3Error> {
        if coordinates.len() != self.rank() {
            return Err(Fn3Error::RankMismatch {
                op: None,
                expected: self.rank(),
                actual: coordinates.len(),
            });
        }
        let mut index = self.offset as u64;
        for (axis, (&coordinate, &extent)) in coordinates.iter().zip(self.shape.dims()).enumerate()
        {
            if coordinate >= extent {
                return Err(Fn3Error::InvalidShape {
                    context: "TensorMeta::checked_index",
                    detail: format!(
                        "coordinate {coordinate} on axis {axis} is outside extent {extent} \
                         of shape {}",
                        self.shape
                    ),
                });
            }
            let stride = self
                .strides
                .get(axis)
                .ok_or_else(|| Fn3Error::RankMismatch {
                    op: None,
                    expected: self.rank(),
                    actual: axis + 1,
                })?;
            let stride = u64::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
                axis,
                stride,
                detail: "negative strides are not addressable in the alpha view contract"
                    .to_string(),
            })?;
            let delta = coordinate
                .checked_mul(stride)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "TensorMeta::checked_index",
                    axis: Some(axis),
                    detail: format!(
                        "coordinate {coordinate} times stride {stride} exceeds u64::MAX"
                    ),
                })?;
            index = index
                .checked_add(delta)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "TensorMeta::checked_index",
                    axis: Some(axis),
                    detail: format!("element index of shape {} exceeds u64::MAX", self.shape),
                })?;
        }
        checked_usize(index, "TensorMeta::checked_index", "element index")
    }

    /// `true` if the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
    }

    /// Require a contiguous row-major view, naming the view when it is not.
    ///
    /// A strided view is not invalid; it simply cannot be exposed as one
    /// contiguous slice. Materialise it with
    /// [`crate::fn3::tensor::TensorView::to_contiguous`] first.
    pub fn require_contiguous(&self, context: &'static str) -> Result<(), Fn3Error> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err(Fn3Error::NonContiguous {
                context,
                detail: format!("{self} uses {}", self.strides),
            })
        }
    }

    /// Require this tensor to be on `expected`.
    pub fn require_device(&self, op: Option<OpId>, expected: DeviceId) -> Result<(), Fn3Error> {
        if self.device == expected {
            Ok(())
        } else {
            Err(Fn3Error::DeviceMismatch {
                op,
                expected,
                actual: self.device,
            })
        }
    }

    /// Require two operands to share a device.
    pub fn check_same_device(&self, op: OpId, other: &TensorMeta) -> Result<(), Fn3Error> {
        if self.device == other.device {
            Ok(())
        } else {
            Err(Fn3Error::DeviceMismatch {
                op: Some(op),
                expected: self.device,
                actual: other.device,
            })
        }
    }

    /// Require a dtype, reporting the actual dtype when it differs.
    pub fn require_dtype(&self, op: Option<OpId>, expected: DType) -> Result<(), Fn3Error> {
        if self.dtype == expected {
            Ok(())
        } else {
            Err(Fn3Error::DTypeMismatch {
                op,
                expected,
                actual: self.dtype,
            })
        }
    }
}

impl fmt::Display for TensorMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dtype)?;
        write!(f, "{}", self.shape)?;
        write!(f, " on {}", self.device)?;
        if !self.is_contiguous() {
            write!(f, " ({})", self.strides)?;
        }
        if self.broadcast {
            f.write_str(" (broadcast)")?;
        }
        Ok(())
    }
}

/// Build contiguous metadata for `shape` placed at `window`'s offset.
///
/// Checks that the whole view fits inside the window, not merely inside the
/// parent storage: a window is a caller's promise about which part of a shared
/// allocation a tensor may address.
fn meta_from_window(
    window: &StorageWindow,
    shape: Shape,
    dtype: DType,
    device: DeviceId,
    what: &'static str,
) -> Result<TensorMeta, Fn3Error> {
    let element_offset = window.element_offset(dtype)?;
    let strides = shape.default_strides()?;
    let meta = TensorMeta::build(shape, strides, element_offset, dtype, device, false)?;
    let required = meta.required_bytes()?;
    if required > window.end() {
        return Err(Fn3Error::StorageWindow {
            what,
            offset: window.offset() as u64,
            len: window.len() as u64,
            storage_len: window.storage().len() as u64,
            detail: format!(
                "the {dtype}{} tensor needs {required} bytes from the start of storage, \
                 which is beyond the window end {}",
                meta.shape(),
                window.end()
            ),
        });
    }
    Ok(meta)
}

/// A read-only, borrowed view of a tensor.
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    storage: &'a Storage,
    meta: TensorMeta,
}

impl<'a> TensorView<'a> {
    /// Bind `meta` to `storage`, checking that the storage is large enough.
    pub fn new(storage: &'a Storage, meta: TensorMeta) -> Result<TensorView<'a>, Fn3Error> {
        let required = meta.required_bytes()?;
        if required > storage.len() {
            return Err(Fn3Error::OutOfBounds {
                what: "TensorView::new (storage bytes)",
                index: required as u64,
                len: storage.len() as u64,
            });
        }
        Ok(TensorView { storage, meta })
    }

    /// A contiguous view of `shape` whose first element starts at `window`'s
    /// byte offset.
    ///
    /// The window's alignment was checked against `dtype` when it was created,
    /// and this constructor additionally checks that the whole tensor fits
    /// inside the window. The view shares the parent allocation; no bytes are
    /// copied.
    pub fn from_window(
        window: &'a StorageWindow,
        shape: Shape,
        dtype: DType,
        device: DeviceId,
    ) -> Result<TensorView<'a>, Fn3Error> {
        let meta = meta_from_window(window, shape, dtype, device, "TensorView::from_window")?;
        TensorView::new(window.storage(), meta)
    }

    /// Metadata of the view.
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }

    /// Device.
    pub fn device(&self) -> DeviceId {
        self.meta.device
    }

    /// Shape.
    pub fn shape(&self) -> &Shape {
        self.meta.shape()
    }

    /// Element strides.
    pub fn strides(&self) -> &Strides {
        self.meta.strides()
    }

    /// Number of elements.
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        self.meta.numel()
    }

    /// `true` for a contiguous row-major view.
    pub fn is_contiguous(&self) -> bool {
        self.meta.is_contiguous()
    }

    /// The shared storage the view addresses.
    ///
    /// The whole allocation, not just the view's window; pair it with
    /// [`TensorMeta::offset`] and the strides.
    pub fn storage(&self) -> &'a Storage {
        self.storage
    }

    /// The whole underlying storage as bytes.
    ///
    /// Kernels use this together with [`TensorMeta::offset`] and the strides;
    /// it is deliberately *not* restricted to the view's window. A
    /// device-resident view has no host bytes: use [`TensorView::host_bytes`],
    /// which rejects it, or [`TensorView::to_host`].
    pub fn storage_bytes(&self) -> &'a [u8] {
        self.storage.as_bytes()
    }

    /// The whole underlying storage as bytes, rejecting device-resident
    /// storage.
    pub fn host_bytes(&self) -> Result<&'a [u8], Fn3Error> {
        self.storage.host_bytes()
    }

    /// `true` when the view's bytes live in device memory.
    pub fn is_device_resident(&self) -> bool {
        self.storage.is_resident()
    }

    /// The whole underlying storage as elements of `T`.
    pub fn storage_slice<T: Pod>(&self) -> Result<&'a [T], Fn3Error> {
        self.storage.as_slice()
    }

    /// Download this view's bytes and return an owned host tensor.
    ///
    /// A host view returns a cheap storage-sharing clone; a device-resident
    /// view is copied device → host through
    /// [`Storage::download`](crate::fn3::storage::Storage::download), so the
    /// result is a snapshot, never a stale alias.
    pub fn to_host(&self) -> Result<Tensor, Fn3Error> {
        if !self.storage.is_resident() {
            return Tensor::from_storage(self.storage.clone(), self.meta.clone());
        }
        let bytes = self.materialised_bytes()?;
        Tensor::from_storage(Storage::from_bytes(&bytes)?, self.meta.clone())
    }

    /// The whole resident allocation as host bytes, or an error for a host
    /// allocation (which is already host-readable).
    fn materialised_bytes(&self) -> Result<Vec<u8>, Fn3Error> {
        let mut bytes = vec![0u8; self.storage.len()];
        self.storage.download(&mut bytes)?;
        Ok(bytes)
    }

    /// The view's window as elements of `T`.
    ///
    /// Requires a contiguous view whose element size matches `T`. A valid
    /// transposed or sliced view returns [`Fn3Error::NonContiguous`]; call
    /// [`TensorView::to_contiguous`] to materialise it.
    pub fn as_slice<T: Pod>(&self) -> Result<&'a [T], Fn3Error> {
        self.meta.require_contiguous("TensorView::as_slice")?;
        self.contiguous_window::<T>("TensorView::as_slice")
    }

    /// The view's window as `f32`.
    pub fn as_f32_slice(&self) -> Result<&'a [f32], Fn3Error> {
        self.meta.require_dtype(None, DType::F32)?;
        self.as_slice::<f32>()
    }

    /// The view's window as `i64`.
    pub fn as_i64_slice(&self) -> Result<&'a [i64], Fn3Error> {
        self.meta.require_dtype(None, DType::I64)?;
        self.as_slice::<i64>()
    }

    /// The view's window as `i32`.
    pub fn as_i32_slice(&self) -> Result<&'a [i32], Fn3Error> {
        self.meta.require_dtype(None, DType::I32)?;
        self.as_slice::<i32>()
    }

    fn contiguous_window<T: Pod>(&self, context: &'static str) -> Result<&'a [T], Fn3Error> {
        if self.storage.is_resident() {
            return Err(device_resident_read(context));
        }
        if std::mem::size_of::<T>() != self.meta.dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element: std::any::type_name::<T>(),
                byte_len: self.meta.byte_len().unwrap_or(0),
                detail: format!(
                    "element size does not match dtype {} ({} bytes)",
                    self.meta.dtype,
                    self.meta.dtype.size_bytes()
                ),
            });
        }
        let start = self.meta.byte_offset()?;
        let len = self.meta.byte_len()?;
        if len == 0 {
            return Ok(&[]);
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context,
                axis: None,
                detail: format!("window end {start} + {len} exceeds usize::MAX"),
            })?;
        let bytes =
            self.storage
                .as_bytes()
                .get(start..end)
                .ok_or_else(|| Fn3Error::OutOfBounds {
                    what: "TensorView contiguous window (storage bytes)",
                    index: end as u64,
                    len: self.storage.len() as u64,
                })?;
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: std::any::type_name::<T>(),
            byte_len: bytes.len(),
            detail: format!("{error:?}"),
        })
    }

    /// Materialise this view into an owned contiguous tensor.
    ///
    /// A contiguous view shares its storage (no copy). A strided view is copied
    /// element by element in row-major logical order into a fresh allocation.
    /// An empty view produces an empty tensor with no allocation. A
    /// device-resident view is downloaded first.
    pub fn to_contiguous(&self) -> Result<Tensor, Fn3Error> {
        let host = self.to_host()?;
        materialise_contiguous(host.storage(), host.meta())
    }
}

/// Which allocation a writable view addresses.
#[derive(Debug)]
enum TensorMutBacking<'a> {
    /// An exclusively borrowed host buffer.
    Host(&'a mut AlignedBuffer),
    /// A device-resident storage, written by a device kernel.
    Device(&'a Storage),
}

/// A writable, exclusively borrowed view of a tensor.
#[derive(Debug)]
pub struct TensorMut<'a> {
    backing: TensorMutBacking<'a>,
    meta: TensorMeta,
}

impl<'a> TensorMut<'a> {
    /// Bind `meta` to an exclusively borrowed buffer.
    ///
    /// Rejects broadcast metadata: a broadcast view addresses the same element
    /// from several coordinates and therefore cannot be written safely.
    pub fn new(buffer: &'a mut AlignedBuffer, meta: TensorMeta) -> Result<TensorMut<'a>, Fn3Error> {
        if !meta.is_writable() {
            return Err(Fn3Error::InvalidShape {
                context: "TensorMut::new",
                detail: format!("broadcast view {meta} is read-only"),
            });
        }
        let required = meta.required_bytes()?;
        if required > buffer.len() {
            return Err(Fn3Error::OutOfBounds {
                what: "TensorMut::new (storage bytes)",
                index: required as u64,
                len: buffer.len() as u64,
            });
        }
        Ok(TensorMut {
            backing: TensorMutBacking::Host(buffer),
            meta,
        })
    }

    /// Bind `meta` to an exclusively borrowed device-resident storage.
    ///
    /// The exclusivity rule is the same as for host storage: the caller must
    /// hold the only handle ([`Storage::is_exclusive`]), which is what
    /// [`Tensor::as_mut_view`] checks. A device kernel writes the buffer; no
    /// host bytes are involved.
    pub fn resident(storage: &'a Storage, meta: TensorMeta) -> Result<TensorMut<'a>, Fn3Error> {
        if !meta.is_writable() {
            return Err(Fn3Error::InvalidShape {
                context: "TensorMut::resident",
                detail: format!("broadcast view {meta} is read-only"),
            });
        }
        if !storage.is_resident() {
            return Err(Fn3Error::InvalidShape {
                context: "TensorMut::resident",
                detail: format!(
                    "the storage for {meta} is host memory; bind it with TensorMut::new and an \
                     exclusively borrowed buffer"
                ),
            });
        }
        if !storage.is_exclusive() {
            return Err(Fn3Error::StorageShared {
                bytes: storage.len(),
                handles: storage.handle_count(),
            });
        }
        let required = meta.required_bytes()?;
        if required > storage.len() {
            return Err(Fn3Error::OutOfBounds {
                what: "TensorMut::resident (storage bytes)",
                index: required as u64,
                len: storage.len() as u64,
            });
        }
        Ok(TensorMut {
            backing: TensorMutBacking::Device(storage),
            meta,
        })
    }

    /// Metadata of the view.
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }

    /// Device.
    pub fn device(&self) -> DeviceId {
        self.meta.device
    }

    /// Shape.
    pub fn shape(&self) -> &Shape {
        self.meta.shape()
    }

    /// Element strides.
    pub fn strides(&self) -> &Strides {
        self.meta.strides()
    }

    /// Number of elements.
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        self.meta.numel()
    }

    /// `true` for a contiguous row-major view.
    pub fn is_contiguous(&self) -> bool {
        self.meta.is_contiguous()
    }

    /// `true` when this output lives in device memory.
    pub fn is_device_resident(&self) -> bool {
        matches!(self.backing, TensorMutBacking::Device(_))
    }

    /// The device-resident storage this output writes, when it is resident.
    pub fn resident_storage(&self) -> Option<&'a Storage> {
        match &self.backing {
            TensorMutBacking::Device(storage) => Some(storage),
            TensorMutBacking::Host(_) => None,
        }
    }

    /// The whole underlying buffer as bytes.
    ///
    /// Empty for a device-resident output; [`TensorMut::host_bytes`] rejects
    /// that case explicitly.
    pub fn storage_bytes(&self) -> &[u8] {
        match &self.backing {
            TensorMutBacking::Host(buffer) => buffer.as_bytes(),
            TensorMutBacking::Device(_) => &[],
        }
    }

    /// The whole underlying buffer as mutable bytes, rejecting a resident output.
    pub fn host_bytes_mut(&mut self) -> Result<&mut [u8], Fn3Error> {
        match &mut self.backing {
            TensorMutBacking::Host(buffer) => Ok(buffer.as_bytes_mut()),
            TensorMutBacking::Device(_) => Err(device_resident_write("TensorMut::host_bytes_mut")),
        }
    }

    /// The whole underlying buffer as bytes, rejecting a resident output.
    pub fn host_bytes(&self) -> Result<&[u8], Fn3Error> {
        match &self.backing {
            TensorMutBacking::Host(buffer) => Ok(buffer.as_bytes()),
            TensorMutBacking::Device(_) => Err(device_resident_write("TensorMut::host_bytes")),
        }
    }

    /// The whole underlying buffer as mutable elements of `T`.
    pub fn storage_slice_mut<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        match &mut self.backing {
            TensorMutBacking::Host(buffer) => buffer.as_mut_slice::<T>(),
            TensorMutBacking::Device(_) => {
                Err(device_resident_write("TensorMut::storage_slice_mut"))
            }
        }
    }

    /// The view's window as elements of `T` (contiguous views only).
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        self.contiguous_window::<T>("TensorMut::as_slice")
    }

    /// The view's window as mutable elements of `T` (contiguous views only).
    ///
    /// A strided writable view can still be written element by element through
    /// [`TensorMut::storage_slice_mut`] and its metadata, but it is not one
    /// contiguous slice.
    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        let element = std::any::type_name::<T>();
        self.meta.require_contiguous("TensorMut::as_mut_slice")?;
        if std::mem::size_of::<T>() != self.meta.dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element,
                byte_len: self.meta.byte_len().unwrap_or(0),
                detail: format!(
                    "element size does not match dtype {} ({} bytes)",
                    self.meta.dtype,
                    self.meta.dtype.size_bytes()
                ),
            });
        }
        let start = self.meta.byte_offset()?;
        let len = self.meta.byte_len()?;
        if len == 0 {
            return Ok(&mut []);
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "TensorMut::as_mut_slice",
                axis: None,
                detail: format!("window end {start} + {len} exceeds usize::MAX"),
            })?;
        let buffer = match &mut self.backing {
            TensorMutBacking::Host(buffer) => buffer,
            TensorMutBacking::Device(_) => {
                return Err(device_resident_write("TensorMut::as_mut_slice"))
            }
        };
        let byte_len = buffer.len();
        let bytes = buffer
            .as_bytes_mut()
            .get_mut(start..end)
            .ok_or(Fn3Error::OutOfBounds {
                what: "TensorMut::as_mut_slice (storage bytes)",
                index: end as u64,
                len: byte_len as u64,
            })?;
        bytemuck::try_cast_slice_mut(bytes).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len: len,
            detail: format!("{error:?}"),
        })
    }

    fn contiguous_window<T: Pod>(&self, context: &'static str) -> Result<&[T], Fn3Error> {
        self.meta.require_contiguous(context)?;
        if std::mem::size_of::<T>() != self.meta.dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element: std::any::type_name::<T>(),
                byte_len: self.meta.byte_len().unwrap_or(0),
                detail: format!(
                    "element size does not match dtype {} ({} bytes)",
                    self.meta.dtype,
                    self.meta.dtype.size_bytes()
                ),
            });
        }
        let start = self.meta.byte_offset()?;
        let len = self.meta.byte_len()?;
        if len == 0 {
            return Ok(&[]);
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context,
                axis: None,
                detail: format!("window end {start} + {len} exceeds usize::MAX"),
            })?;
        let bytes = match &self.backing {
            TensorMutBacking::Host(buffer) => {
                buffer
                    .as_bytes()
                    .get(start..end)
                    .ok_or_else(|| Fn3Error::OutOfBounds {
                        what: "TensorMut contiguous window (storage bytes)",
                        index: end as u64,
                        len: buffer.len() as u64,
                    })?
            }
            TensorMutBacking::Device(_) => return Err(device_resident_write(context)),
        };
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: std::any::type_name::<T>(),
            byte_len: bytes.len(),
            detail: format!("{error:?}"),
        })
    }
}

/// An owned tensor: metadata plus a shared handle to its storage.
///
/// Cloning is cheap: it copies metadata and clones the `Arc`, so the clone
/// reads the same bytes and cannot be mutated while the original (or any other
/// handle or view) is alive. Use [`Tensor::as_mut_view`] for an exclusive
/// write.
#[derive(Debug, Clone)]
pub struct Tensor {
    meta: TensorMeta,
    storage: Storage,
}

impl Tensor {
    /// Allocate a zero-filled contiguous tensor.
    ///
    /// Fails with [`Fn3Error::DeviceUnavailable`] for any device this build has
    /// no allocator for. It never silently allocates host memory for a Vulkan
    /// request.
    pub fn zeros(shape: Shape, dtype: DType, device: DeviceId) -> Result<Tensor, Fn3Error> {
        if !device.is_cpu() {
            return Err(DeviceResolveError::unavailable_device(device).into());
        }
        let meta = TensorMeta::contiguous(shape, dtype, device)?;
        let storage = Storage::zeroed(meta.required_bytes()?)?;
        Ok(Tensor { meta, storage })
    }

    /// Wrap existing storage with validated metadata.
    ///
    /// The caller is responsible for the storage being valid on
    /// `meta.device()`. Host `AlignedBuffer` storage tagged for a Vulkan device
    /// is the staged path: the backend uploads it per dispatch, which is why
    /// `FN3_VK_STAGED=1` still produces it. Device-resident storage
    /// ([`crate::fn3::storage::Storage::from_resident`]) is the resident path
    /// and is written by kernels in place (FN3-26 / #72).
    ///
    /// The storage must provide every byte this metadata addresses:
    /// `meta.required_bytes() <= storage.len()`. The offset itself is checked
    /// at metadata construction.
    pub fn from_storage(storage: Storage, meta: TensorMeta) -> Result<Tensor, Fn3Error> {
        let required = meta.required_bytes()?;
        if required > storage.len() {
            return Err(Fn3Error::OutOfBounds {
                what: "Tensor::from_storage (storage bytes)",
                index: required as u64,
                len: storage.len() as u64,
            });
        }
        Ok(Tensor { meta, storage })
    }

    /// `true` when this tensor's bytes live in device memory.
    ///
    /// A resident tensor has no host bytes; read it with [`Tensor::to_host`].
    pub fn is_device_resident(&self) -> bool {
        self.storage.is_resident()
    }

    /// The device handle when this tensor is resident.
    pub fn residency(&self) -> Option<&std::sync::Arc<dyn crate::fn3::storage::DeviceResidency>> {
        self.storage.residency()
    }

    /// The tensor as host bytes, downloading first when it is resident.
    ///
    /// A host tensor returns a cheap storage-sharing clone (no copy). A
    /// device-resident tensor is copied device → host once and returned as a
    /// fresh host tensor with identical metadata, so the snapshot is never a
    /// stale alias of device state.
    pub fn to_host(&self) -> Result<Tensor, Fn3Error> {
        if !self.storage.is_resident() {
            return Tensor::from_storage(self.storage.clone(), self.meta.clone());
        }
        let mut bytes = vec![0u8; self.storage.len()];
        self.storage.download(&mut bytes)?;
        Tensor::from_storage(Storage::from_bytes(&bytes)?, self.meta.clone())
    }

    /// An owned contiguous tensor at `window`'s byte offset.
    ///
    /// The tensor shares the parent allocation through `Arc`; the window was
    /// already checked for alignment and bounds, and this constructor
    /// additionally checks that the tensor fits inside the window.
    pub fn from_window(
        window: StorageWindow,
        shape: Shape,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Tensor, Fn3Error> {
        let meta = meta_from_window(&window, shape, dtype, device, "Tensor::from_window")?;
        Tensor::from_storage(window.into_storage(), meta)
    }

    /// A 1-D `f32` tensor on the CPU.
    pub fn from_f32(shape: Shape, values: &[f32]) -> Result<Tensor, Fn3Error> {
        Tensor::from_slice(shape, DType::F32, values)
    }

    /// An `i64` tensor on the CPU.
    pub fn from_i64(shape: Shape, values: &[i64]) -> Result<Tensor, Fn3Error> {
        Tensor::from_slice(shape, DType::I64, values)
    }

    /// An `i32` tensor on the CPU.
    pub fn from_i32(shape: Shape, values: &[i32]) -> Result<Tensor, Fn3Error> {
        Tensor::from_slice(shape, DType::I32, values)
    }

    /// A `Bool` mask tensor on the CPU, stored as one 0/1 byte per element.
    pub fn from_mask(shape: Shape, values: &[bool]) -> Result<Tensor, Fn3Error> {
        let expected = shape.numel()?;
        if expected != values.len() as u64 {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{expected} mask elements for {shape}"),
                actual: format!("{} values", values.len()),
            });
        }
        let bytes: Vec<u8> = values.iter().map(|&value| u8::from(value)).collect();
        let meta = TensorMeta::contiguous(shape, DType::Bool, DeviceId::Cpu)?;
        Ok(Tensor {
            meta,
            storage: Storage::from_bytes(&bytes)?,
        })
    }

    /// A tensor on the CPU from a typed slice whose element size matches `dtype`.
    pub fn from_slice<T: Pod>(
        shape: Shape,
        dtype: DType,
        values: &[T],
    ) -> Result<Tensor, Fn3Error> {
        if std::mem::size_of::<T>() != dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element: std::any::type_name::<T>(),
                byte_len: std::mem::size_of::<T>(),
                detail: format!(
                    "element size does not match dtype {dtype} ({} bytes)",
                    dtype.size_bytes()
                ),
            });
        }
        let expected = shape.numel()?;
        if expected != values.len() as u64 {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{expected} elements of {dtype} for {shape}"),
                actual: format!("{} values", values.len()),
            });
        }
        let meta = TensorMeta::contiguous(shape, dtype, DeviceId::Cpu)?;
        Ok(Tensor {
            meta,
            storage: Storage::from_bytes(bytemuck::cast_slice(values))?,
        })
    }

    /// Metadata.
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    /// Shape.
    pub fn shape(&self) -> &Shape {
        self.meta.shape()
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.meta.dtype()
    }

    /// Device.
    pub fn device(&self) -> DeviceId {
        self.meta.device()
    }

    /// Number of elements.
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        self.meta.numel()
    }

    /// `true` for a contiguous row-major tensor.
    pub fn is_contiguous(&self) -> bool {
        self.meta.is_contiguous()
    }

    /// Shared handle to the storage. Cloning it does not copy bytes.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// `true` when both tensors address the same allocation.
    ///
    /// Two tensors with equal bytes in distinct allocations are not aliases.
    /// A backend can use this to assert that an output cannot observe an input
    /// before writing.
    pub fn shares_storage_with(&self, other: &Tensor) -> bool {
        self.storage.ptr_eq(&other.storage)
    }

    /// A read-only view of the whole tensor.
    pub fn as_view(&self) -> Result<TensorView<'_>, Fn3Error> {
        TensorView::new(&self.storage, self.meta.clone())
    }

    /// An exclusive writable view of the whole tensor.
    ///
    /// Fails with [`Fn3Error::StorageShared`] while any other handle to the
    /// storage is alive (for example a live [`TensorView`] or a weight shared
    /// with a `PreparedModel`).
    ///
    /// A device-resident tensor yields a resident [`TensorMut`]: the same
    /// exclusivity rule, and the write happens on the device rather than
    /// through host bytes.
    pub fn as_mut_view(&mut self) -> Result<TensorMut<'_>, Fn3Error> {
        let meta = self.meta.clone();
        if self.storage.is_resident() {
            return TensorMut::resident(&self.storage, meta);
        }
        let buffer = self.storage.try_make_mut()?;
        TensorMut::new(buffer, meta)
    }

    /// The tensor's window as elements of `T` (contiguous tensors only).
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        let view = self.as_view()?;
        view.as_slice::<T>()
    }

    /// The tensor's window as `f32`.
    pub fn as_f32_slice(&self) -> Result<&[f32], Fn3Error> {
        let view = self.as_view()?;
        view.as_f32_slice()
    }

    /// The tensor's window as `i64`.
    pub fn as_i64_slice(&self) -> Result<&[i64], Fn3Error> {
        let view = self.as_view()?;
        view.as_i64_slice()
    }

    /// The tensor's window as `i32`.
    pub fn as_i32_slice(&self) -> Result<&[i32], Fn3Error> {
        let view = self.as_view()?;
        view.as_i32_slice()
    }

    /// Materialise this tensor into an owned contiguous tensor.
    ///
    /// A contiguous tensor returns a cheap storage-sharing clone. A strided
    /// tensor is copied element by element in row-major order. An empty tensor
    /// produces an empty tensor with no allocation. A device-resident tensor is
    /// downloaded first.
    pub fn to_contiguous(&self) -> Result<Tensor, Fn3Error> {
        let host = self.to_host()?;
        materialise_contiguous(host.storage(), host.meta())
    }
}

/// The typed error for reading a device-resident allocation on the host.
fn device_resident_read(context: &'static str) -> Fn3Error {
    Fn3Error::NotImplemented {
        what: match context {
            "TensorView::as_slice" => {
                "reading a device-resident tensor through the host slice API; download it with \
                 Tensor::to_host"
            }
            _ => {
                "reading a device-resident tensor through the host byte API; download it with \
                 Tensor::to_host"
            }
        },
        owner: "FN3-26 (#72)",
    }
}

/// The typed error for writing a device-resident allocation on the host.
fn device_resident_write(context: &'static str) -> Fn3Error {
    Fn3Error::NotImplemented {
        what: match context {
            "TensorMut::as_mut_slice" | "TensorMut::storage_slice_mut" => {
                "writing a device-resident tensor through the host slice API; its kernels write \
                 it on the device"
            }
            _ => {
                "writing a device-resident tensor through the host byte API; its kernels write \
                 it on the device"
            }
        },
        owner: "FN3-26 (#72)",
    }
}

/// Copy a strided view into a fresh contiguous allocation.
///
/// The source is addressed through [`TensorMeta::checked_index`], so every
/// source byte is bounds-checked and every destination byte is written in
/// row-major logical order. The destination allocation is never indexed
/// unchecked.
fn materialise_contiguous(storage: &Storage, meta: &TensorMeta) -> Result<Tensor, Fn3Error> {
    let numel = meta.numel_usize()?;
    if numel == 0 {
        return Tensor::zeros(meta.shape().clone(), meta.dtype(), meta.device());
    }
    if meta.is_contiguous() {
        return Tensor::from_storage(storage.clone(), meta.clone());
    }
    if storage.is_resident() {
        return Err(device_resident_read("materialise_contiguous"));
    }

    let element_bytes = meta.dtype().size_bytes();
    let source = storage.as_bytes();
    let mut out = Tensor::zeros(meta.shape().clone(), meta.dtype(), meta.device())?;
    {
        let mut out_view = out.as_mut_view()?;
        let destination = out_view.storage_slice_mut::<u8>()?;
        let rank = meta.rank();
        let dims = meta.shape().dims();
        let mut coordinates = vec![0u64; rank];
        for linear in 0..numel {
            let mut remainder = linear;
            for axis in (0..rank).rev() {
                // `numel` fits in usize, so every non-zero extent does too;
                // a zero extent cannot reach this loop because numel == 0.
                let extent = checked_usize(
                    dims[axis],
                    "materialise_contiguous",
                    "axis extent for a non-empty tensor",
                )?;
                coordinates[axis] = (remainder % extent) as u64;
                remainder /= extent;
            }
            let index = meta.checked_index(&coordinates)?;
            let source_start =
                index
                    .checked_mul(element_bytes)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "materialise_contiguous (source)",
                        axis: None,
                        detail: format!(
                            "element index {index} times {element_bytes} bytes overflows"
                        ),
                    })?;
            let source_end =
                source_start
                    .checked_add(element_bytes)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "materialise_contiguous (source)",
                        axis: None,
                        detail: "source window end exceeds usize::MAX".to_string(),
                    })?;
            let destination_start =
                linear
                    .checked_mul(element_bytes)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "materialise_contiguous (destination)",
                        axis: None,
                        detail: format!(
                            "element index {linear} times {element_bytes} bytes overflows"
                        ),
                    })?;
            let destination_end =
                destination_start
                    .checked_add(element_bytes)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "materialise_contiguous (destination)",
                        axis: None,
                        detail: "destination window end exceeds usize::MAX".to_string(),
                    })?;
            let from = source
                .get(source_start..source_end)
                .ok_or(Fn3Error::OutOfBounds {
                    what: "materialise_contiguous (source bytes)",
                    index: source_end as u64,
                    len: source.len() as u64,
                })?;
            let destination_len = destination.len();
            let to = destination
                .get_mut(destination_start..destination_end)
                .ok_or(Fn3Error::OutOfBounds {
                    what: "materialise_contiguous (destination bytes)",
                    index: destination_end as u64,
                    len: destination_len as u64,
                })?;
            to.copy_from_slice(from);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_tensor_has_one_element() {
        let tensor = Tensor::from_f32(Shape::scalar(), &[2.5]).unwrap();
        assert_eq!(tensor.numel(), Ok(1));
        assert_eq!(tensor.as_f32_slice().unwrap(), &[2.5]);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn zero_element_tensor_is_legal() {
        let tensor = Tensor::zeros(Shape::from_dims([0, 3]), DType::F32, DeviceId::Cpu).unwrap();
        assert_eq!(tensor.numel(), Ok(0));
        assert_eq!(tensor.as_f32_slice().unwrap().len(), 0);
        assert_eq!(tensor.storage().len(), 0);
        assert_eq!(
            tensor
                .to_contiguous()
                .unwrap()
                .as_f32_slice()
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn non_cpu_allocation_is_rejected_not_substituted() {
        let error = Tensor::zeros(
            Shape::from_dims([2]),
            DType::F32,
            DeviceId::Vulkan(crate::fn3::device::PhysicalDeviceId::new(0)),
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("vulkan(0)"), "{text}");
        if cfg!(feature = "vulkan") {
            // The backend exists, so the reason a tensor cannot live on the
            // device is that alpha tensor storage is host memory, not that the
            // device is missing. Both remedies are honest; asserting the wrong
            // one would hide a regression in either direction.
            assert!(text.contains("host memory"), "{text}");
            assert!(text.contains("FN3-12"), "{text}");
        } else {
            assert!(text.contains("available devices: [cpu"), "{text}");
            assert!(text.contains("--features vulkan"), "{text}");
        }
    }

    #[test]
    fn shared_storage_rejects_mutation() {
        let mut tensor = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
        // A second `Storage` handle is what a weight shared with a
        // `PreparedModel` looks like.
        let shared = tensor.storage().clone();
        assert!(matches!(
            tensor.as_mut_view(),
            Err(Fn3Error::StorageShared { .. })
        ));
        drop(shared);
        tensor.as_mut_view().unwrap();
    }

    #[test]
    fn to_contiguous_materialises_a_transposed_view() {
        let tensor =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let shape = tensor.shape().permuted(&[1, 0]).unwrap();
        let strides = tensor.meta().strides().permuted(&[1, 0]).unwrap();
        let meta = TensorMeta::strided(shape, strides, 0, DType::F32, DeviceId::Cpu).unwrap();
        let view = TensorView::new(tensor.storage(), meta).unwrap();
        let contiguous = view.to_contiguous().unwrap();
        assert_eq!(contiguous.shape(), &Shape::from_dims([3, 2]));
        assert_eq!(
            contiguous.as_f32_slice().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn as_slice_rejects_a_strided_view() {
        let tensor = Tensor::from_f32(Shape::from_dims([2, 3]), &[0.0; 6]).unwrap();
        let meta = TensorMeta::strided(
            Shape::from_dims([3, 2]),
            Strides::from_elements(vec![1, 3]),
            0,
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        let view = TensorView::new(tensor.storage(), meta).unwrap();
        assert!(matches!(
            view.as_f32_slice(),
            Err(Fn3Error::NonContiguous { .. })
        ));
    }

    #[derive(Debug)]
    struct FakeResident {
        bytes: Vec<u8>,
    }

    impl crate::fn3::storage::DeviceResidency for FakeResident {
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
            dst.copy_from_slice(&self.bytes[offset..offset + dst.len()]);
            Ok(())
        }
    }

    fn resident_tensor(values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = bytes.len();
        let meta = TensorMeta::contiguous(
            Shape::from_dims([values.len() as u64]),
            DType::F32,
            DeviceId::Vulkan(crate::fn3::device::PhysicalDeviceId::new(0)),
        )
        .unwrap();
        let storage =
            Storage::from_resident(len, std::sync::Arc::new(FakeResident { bytes })).unwrap();
        Tensor::from_storage(storage, meta).unwrap()
    }

    #[test]
    fn a_resident_tensor_is_unreadable_on_the_host_and_downloads_explicitly() {
        let tensor = resident_tensor(&[1.0, 2.0, 3.0]);
        assert!(tensor.is_device_resident());
        assert_eq!(tensor.numel(), Ok(3));
        let error = tensor.as_f32_slice().unwrap_err();
        assert!(matches!(error, Fn3Error::NotImplemented { .. }), "{error}");
        assert!(tensor.storage().as_bytes().is_empty());

        let host = tensor.to_host().unwrap();
        assert!(!host.is_device_resident());
        assert_eq!(host.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(host.device(), tensor.device());
        assert_eq!(host.shape(), tensor.shape());

        // A host tensor's `to_host` is a cheap sharing clone.
        let again = host.to_host().unwrap();
        assert!(again.storage().ptr_eq(host.storage()));
    }

    #[test]
    fn a_resident_output_view_is_writable_only_in_exclusivity_and_on_the_device() {
        let mut tensor = resident_tensor(&[1.0, 2.0]);
        {
            let mut view = tensor.as_mut_view().unwrap();
            assert!(view.is_device_resident());
            assert!(matches!(
                view.as_mut_slice::<f32>(),
                Err(Fn3Error::NotImplemented { .. })
            ));
            assert!(view.storage_slice_mut::<u8>().is_err());
            assert!(view.host_bytes().is_err());
        }
        let shared = tensor.clone();
        let error = tensor.as_mut_view().unwrap_err();
        assert!(matches!(error, Fn3Error::StorageShared { .. }), "{error}");
        drop(shared);
        tensor.as_mut_view().unwrap();
    }

    #[test]
    fn window_views_share_the_allocation() {
        let storage = Storage::from_bytes(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0])).unwrap();
        let window = storage.window(4, 8, DType::F32).unwrap();
        let view =
            TensorView::from_window(&window, Shape::from_dims([2]), DType::F32, DeviceId::Cpu)
                .unwrap();
        assert_eq!(view.as_f32_slice().unwrap(), &[2.0, 3.0]);
        assert_eq!(view.meta().offset(), 1);

        let tensor =
            Tensor::from_window(window, Shape::from_dims([2]), DType::F32, DeviceId::Cpu).unwrap();
        assert_eq!(tensor.as_f32_slice().unwrap(), &[2.0, 3.0]);
        assert!(storage.ptr_eq(tensor.storage()));
    }
}
