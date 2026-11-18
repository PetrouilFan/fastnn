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

use std::fmt;

use bytemuck::Pod;

use crate::fn3::device::{DeviceId, DeviceResolveError};
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::{Shape, Strides};
use crate::fn3::storage::{AlignedBuffer, Storage};

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
        // Reject metadata that can never address a buffer on this target.
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

    /// Byte length of the view's elements (not of its span).
    pub fn byte_len(&self) -> Result<usize, Fn3Error> {
        self.shape.byte_len(self.dtype)
    }

    /// Elements addressed from the start of storage: `offset + span`.
    pub fn required_elements(&self) -> Result<u64, Fn3Error> {
        if self.shape.has_zero_extent() {
            return Ok(0);
        }
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

    /// Bytes that storage must provide for this view.
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
                detail: format!("{elements} elements of {} exceed u64::MAX", self.dtype),
            })?;
        usize::try_from(bytes).map_err(|_| Fn3Error::ShapeOverflow {
            context: "TensorMeta::required_bytes",
            axis: None,
            detail: format!("{bytes} bytes do not fit in a usize on this target"),
        })
    }

    /// `true` if the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
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

    /// The whole underlying storage as bytes.
    ///
    /// Kernels use this together with [`TensorMeta::offset`] and the strides;
    /// it is deliberately *not* restricted to the view's window.
    pub fn storage_bytes(&self) -> &'a [u8] {
        self.storage.as_bytes()
    }

    /// The whole underlying storage as elements of `T`.
    pub fn storage_slice<T: Pod>(&self) -> Result<&'a [T], Fn3Error> {
        self.storage.as_slice()
    }

    /// The view's window as elements of `T`.
    ///
    /// Requires a contiguous view whose element size matches `T`.
    pub fn as_slice<T: Pod>(&self) -> Result<&'a [T], Fn3Error> {
        if !self.is_contiguous() {
            return Err(Fn3Error::InvalidStride {
                axis: 0,
                stride: self.meta.strides.get(0).unwrap_or(0),
                detail: format!(
                    "as_slice requires a contiguous view, but {} is {}",
                    self.meta, self.meta.strides
                ),
            });
        }
        self.contiguous_window::<T>()
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

    fn contiguous_window<T: Pod>(&self) -> Result<&'a [T], Fn3Error> {
        if std::mem::size_of::<T>() != self.meta.dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element: std::any::type_name::<T>(),
                byte_len: std::mem::size_of::<T>(),
                detail: format!(
                    "element size does not match dtype {} ({})",
                    self.meta.dtype,
                    self.meta.dtype.size_bytes()
                ),
            });
        }
        let element_bytes = self.meta.dtype.size_bytes();
        let numel = usize::try_from(self.meta.numel()?).map_err(|_| Fn3Error::ShapeOverflow {
            context: "TensorView::contiguous_window",
            axis: None,
            detail: "element count does not fit in a usize".to_string(),
        })?;
        if numel == 0 {
            return Ok(&[]);
        }
        let start = self.meta.offset * element_bytes;
        let end = numel
            .checked_mul(element_bytes)
            .and_then(|len| start.checked_add(len))
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "TensorView::contiguous_window",
                axis: None,
                detail: "window end offset exceeds usize::MAX".to_string(),
            })?;
        let bytes = &self.storage.as_bytes()[start..end];
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: std::any::type_name::<T>(),
            byte_len: bytes.len(),
            detail: format!("{error:?}"),
        })
    }
}

/// A writable, exclusively borrowed view of a tensor.
#[derive(Debug)]
pub struct TensorMut<'a> {
    buffer: &'a mut AlignedBuffer,
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
        Ok(TensorMut { buffer, meta })
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

    /// The whole underlying buffer as bytes.
    pub fn storage_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// The whole underlying buffer as mutable elements of `T`.
    pub fn storage_slice_mut<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        self.buffer.as_mut_slice::<T>()
    }

    /// The view's window as elements of `T` (contiguous views only).
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], Fn3Error> {
        if std::mem::size_of::<T>() != self.meta.dtype.size_bytes() {
            return Err(Fn3Error::StorageCast {
                element: std::any::type_name::<T>(),
                byte_len: std::mem::size_of::<T>(),
                detail: format!(
                    "element size does not match dtype {} ({})",
                    self.meta.dtype,
                    self.meta.dtype.size_bytes()
                ),
            });
        }
        let element_bytes = self.meta.dtype.size_bytes();
        let start = self.meta.offset * element_bytes;
        let end = start + self.meta.byte_len()?;
        if start == end {
            return Ok(&[]);
        }
        let bytes = &self.buffer.as_bytes()[start..end];
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: std::any::type_name::<T>(),
            byte_len: bytes.len(),
            detail: format!("{error:?}"),
        })
    }

    /// The view's window as mutable elements of `T` (contiguous views only).
    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T], Fn3Error> {
        let element_bytes = self.meta.dtype.size_bytes();
        let start = self.meta.offset * element_bytes;
        let len = self.meta.byte_len()?;
        let element = std::any::type_name::<T>();
        let bytes = &mut self.buffer.as_bytes_mut()[start..start + len];
        bytemuck::try_cast_slice_mut(bytes).map_err(|error| Fn3Error::StorageCast {
            element,
            byte_len: len,
            detail: format!("{error:?}"),
        })
    }
}

/// An owned tensor: metadata plus a shared handle to its storage.
///
/// Cloning a `Tensor` is not implemented on purpose. Sharing is explicit
/// through [`Tensor::storage`] and [`Tensor::as_view`]; mutable access requires
/// exclusive storage ([`Tensor::as_mut_view`]).
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
    /// `meta.device()`. In the alpha build only `DeviceId::Cpu` storage is
    /// real; this constructor exists so device-affinity rules can be validated
    /// before the Vulkan allocator exists (FN3-11 / #57).
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
                    "element size does not match dtype {dtype} ({})",
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

    /// A read-only view of the whole tensor.
    pub fn as_view(&self) -> Result<TensorView<'_>, Fn3Error> {
        TensorView::new(&self.storage, self.meta.clone())
    }

    /// An exclusive writable view of the whole tensor.
    ///
    /// Fails with [`Fn3Error::StorageShared`] while any other handle to the
    /// storage is alive (for example a live [`TensorView`] or a weight shared
    /// with a `PreparedModel`).
    pub fn as_mut_view(&mut self) -> Result<TensorMut<'_>, Fn3Error> {
        let meta = self.meta.clone();
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

    /// Materialise this tensor into a fresh contiguous tensor.
    ///
    /// Contiguous tensors return a cheap storage-sharing copy. Strided views
    /// are copied element by element in row-major order.
    pub fn to_contiguous(&self) -> Result<Tensor, Fn3Error> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let mut out = Tensor::zeros(self.shape().clone(), self.dtype(), self.device())?;
        {
            let view = self.as_view()?;
            let mut out_view = out.as_mut_view()?;
            let element_bytes = self.dtype().size_bytes();
            let in_bytes = view.storage_bytes();
            let out_bytes = out_view.storage_slice_mut::<u8>()?;
            let numel = usize::try_from(self.numel()?).map_err(|_| Fn3Error::ShapeOverflow {
                context: "Tensor::to_contiguous",
                axis: None,
                detail: "element count does not fit in a usize".to_string(),
            })?;
            let rank = self.shape().rank();
            let in_shape = self.shape().dims();
            let in_strides = self.meta.strides.as_slice();
            let mut coordinates = vec![0u64; rank];
            for linear in 0..numel {
                let mut remainder = linear;
                for axis in (0..rank).rev() {
                    let extent = in_shape[axis] as usize;
                    coordinates[axis] = (remainder % extent) as u64;
                    remainder /= extent;
                }
                let mut source = self.meta.offset;
                for axis in 0..rank {
                    source += coordinates[axis] as usize * in_strides[axis] as usize;
                }
                let source = source * element_bytes;
                let destination = linear * element_bytes;
                out_bytes[destination..destination + element_bytes]
                    .copy_from_slice(&in_bytes[source..source + element_bytes]);
            }
        }
        Ok(out)
    }
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
        assert!(text.contains("available devices: [cpu]"), "{text}");
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
        let transposed =
            Tensor::from_storage(tensor.storage().clone(), view.meta().clone()).unwrap();
        let contiguous = transposed.to_contiguous().unwrap();
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
            Err(Fn3Error::InvalidStride { .. })
        ));
    }
}
