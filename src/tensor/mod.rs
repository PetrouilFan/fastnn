use crate::backend::cpu::CpuBackend;
use crate::backend::BackendError;
use crate::error::{FastnnError, FastnnResult};
use crate::ir::builder::GraphBuilder;
use crate::ir::{DimExpr, IrDType, TensorType};
use crate::storage::{DType, Device, Storage};
use crate::storage_pool::get_storage_pool;
use crate::types::{RepresentationTransform, ScalarType, ValueRepresentation};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::Arc;

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
use self::factories::memcpy_f32;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use self::factories::simd_copy_f32;
use self::shape::compute_strides;

mod device;
mod factories;
mod indexing;
mod ops;
mod reductions;
mod shape;

pub(crate) fn validate_tensor_shape(sizes: &[i64], dtype: DType) -> FastnnResult<(usize, usize)> {
    let mut numel = 1_usize;
    for (dimension, &size) in sizes.iter().enumerate() {
        if size < 0 {
            return Err(FastnnError::shape(format!(
                "tensor dimension {dimension} has negative size {size}"
            )));
        }
        numel = numel
            .checked_mul(size as usize)
            .ok_or_else(|| FastnnError::Overflow("tensor element count overflow".into()))?;
    }
    if numel > i64::MAX as usize {
        return Err(FastnnError::Overflow(
            "tensor element count exceeds i64::MAX".into(),
        ));
    }
    let nbytes = dtype.try_storage_bytes(numel)?;
    Ok((numel, nbytes))
}

pub struct TensorImpl {
    pub storage: Arc<Storage>,
    pub sizes: SmallVec<[i64; 8]>,
    pub strides: SmallVec<[i64; 8]>,
    pub storage_offset: i64,
    pub dtype: DType,
    pub device: Device,
    /// Cached contiguity: -1=unknown, 0=false, 1=true.
    /// Set to 1 when strides are known to be contiguous (new tensors from factories),
    /// set to -1 when strides may be non-contiguous (views, clones, device transfers).
    pub contiguous_cache: AtomicI8,
}

impl TensorImpl {
    pub fn new(storage: Arc<Storage>, sizes: SmallVec<[i64; 8]>, dtype: DType) -> Self {
        Self::try_new(storage, sizes, dtype).expect("TensorImpl::new failed")
    }

    pub fn try_new(
        storage: Arc<Storage>,
        sizes: SmallVec<[i64; 8]>,
        dtype: DType,
    ) -> FastnnResult<Self> {
        let device = storage.device();
        let (_, nbytes) = validate_tensor_shape(&sizes, dtype)?;
        let storage_nbytes = storage.nbytes();
        if nbytes > storage_nbytes {
            return Err(FastnnError::shape(format!(
                "shape {sizes:?} requires {nbytes} bytes of {dtype:?} storage, but only {storage_nbytes} bytes are available"
            )));
        }

        let strides = compute_strides(&sizes);

        Ok(TensorImpl {
            storage,
            sizes,
            strides,
            storage_offset: 0,
            dtype,
            device,
            contiguous_cache: AtomicI8::new(1),
        })
    }

    /// Create TensorImpl with specific device (ignoring storage device)
    pub fn new_with_device(
        storage: Arc<Storage>,
        sizes: SmallVec<[i64; 8]>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self::try_new_with_device(storage, sizes, device, dtype)
            .expect("TensorImpl::new_with_device failed")
    }

    pub fn try_new_with_device(
        storage: Arc<Storage>,
        sizes: SmallVec<[i64; 8]>,
        device: Device,
        dtype: DType,
    ) -> FastnnResult<Self> {
        let (_, nbytes) = validate_tensor_shape(&sizes, dtype)?;
        let storage_nbytes = storage.nbytes();
        if nbytes > storage_nbytes {
            return Err(FastnnError::shape(format!(
                "shape {sizes:?} requires {nbytes} bytes of {dtype:?} storage, but only {storage_nbytes} bytes are available"
            )));
        }

        let strides = compute_strides(&sizes);

        Ok(TensorImpl {
            storage,
            sizes,
            strides,
            storage_offset: 0,
            dtype,
            device,
            contiguous_cache: AtomicI8::new(1),
        })
    }

    pub fn id(&self) -> usize {
        let ptr = self as *const TensorImpl;
        ptr as usize
    }

    /// Create a view sharing storage from self.
    pub(crate) fn new_view_from(
        &self,
        sizes: SmallVec<[i64; 8]>,
        strides: SmallVec<[i64; 8]>,
        storage_offset: i64,
    ) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset,
            dtype: self.dtype,
            device: self.device,
            contiguous_cache: AtomicI8::new(-1), // strides may be non-contiguous
        }
    }

    /// Create a tensor on a different device.
    fn new_on_device(&self, storage: Arc<Storage>, device: Device) -> Self {
        Self {
            storage,
            sizes: self.sizes.clone(),
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device,
            contiguous_cache: AtomicI8::new(-1), // strides copied from self, may be non-contiguous
        }
    }

    #[track_caller]
    pub fn data_ptr(&self) -> *const u8 {
        self.try_data_ptr().expect("TensorImpl::data_ptr failed")
    }

    pub fn try_data_ptr(&self) -> FastnnResult<*const u8> {
        if self.storage_offset < 0 {
            return Err(FastnnError::shape(format!(
                "negative storage offset {} cannot produce a data pointer",
                self.storage_offset
            )));
        }
        let element_size = self.dtype.scalar_byte_width().ok_or_else(|| {
            FastnnError::dtype("raw pointer access requires plain scalar storage")
        })?;
        let byte_offset = (self.storage_offset as usize)
            .checked_mul(element_size)
            .ok_or_else(|| FastnnError::Overflow("data pointer byte offset overflow".into()))?;
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                if byte_offset > cpu.data.len() {
                    return Err(FastnnError::OutOfBounds(format!(
                        "data pointer byte offset {byte_offset} exceeds storage length {}",
                        cpu.data.len()
                    )));
                }
                // SAFETY: the checked offset is within or one past the CPU allocation,
                // and the pointer lifetime remains tied to the borrowed tensor storage.
                Ok(unsafe { cpu.data.as_ref().as_ptr().add(byte_offset) })
            }
        }
    }

    #[track_caller]
    pub fn data_ptr_f32(&self) -> *const f32 {
        self.try_data_ptr_f32()
            .expect("TensorImpl::data_ptr_f32 failed")
    }

    pub fn try_data_ptr_f32(&self) -> FastnnResult<*const f32> {
        if self.dtype != DType::F32 {
            return Err(FastnnError::dtype(format!(
                "F32 pointer access requires F32 storage, got {:?}",
                self.dtype
            )));
        }
        let ptr = self.try_data_ptr()? as *const f32;
        if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
            return Err(FastnnError::Internal(
                "CPU storage is not aligned for F32 pointer access".into(),
            ));
        }
        Ok(ptr)
    }

    pub fn data_ptr_f32_mut(&mut self) -> *mut f32 {
        self.try_data_ptr_f32_mut()
            .expect("TensorImpl::data_ptr_f32_mut failed")
    }

    pub fn try_data_ptr_f32_mut(&mut self) -> FastnnResult<*mut f32> {
        if self.dtype != DType::F32 {
            return Err(FastnnError::dtype(format!(
                "mutable F32 pointer access requires F32 storage, got {:?}",
                self.dtype
            )));
        }
        let ptr = self.try_data_ptr_mut()? as *mut f32;
        if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
            return Err(FastnnError::Internal(
                "CPU storage is not aligned for mutable F32 pointer access".into(),
            ));
        }
        Ok(ptr)
    }

    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.try_data_ptr_mut()
            .expect("TensorImpl::data_ptr_mut failed")
    }

    pub fn try_data_ptr_mut(&mut self) -> FastnnResult<*mut u8> {
        if self.storage_offset < 0 {
            return Err(FastnnError::shape(format!(
                "negative storage offset {} cannot produce a mutable data pointer",
                self.storage_offset
            )));
        }
        let element_size = self.dtype.scalar_byte_width().ok_or_else(|| {
            FastnnError::dtype("mutable raw pointer access requires plain scalar storage")
        })?;
        let byte_offset = (self.storage_offset as usize)
            .checked_mul(element_size)
            .ok_or_else(|| FastnnError::Overflow("mutable pointer byte offset overflow".into()))?;
        let storage = Arc::make_mut(&mut self.storage);
        match storage {
            Storage::Cpu(cpu) => {
                let data = Arc::make_mut(&mut cpu.data);
                if byte_offset > data.len() {
                    return Err(FastnnError::OutOfBounds(format!(
                        "mutable pointer byte offset {byte_offset} exceeds storage length {}",
                        data.len()
                    )));
                }
                // SAFETY: the checked offset is within or one past the allocation;
                // `&mut self` and `Arc::make_mut` establish exclusive backing access.
                Ok(unsafe { data.as_mut_ptr().add(byte_offset) })
            }
        }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        self.try_as_f32_slice()
            .expect("TensorImpl::as_f32_slice failed")
    }

    pub fn try_as_f32_slice(&self) -> FastnnResult<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(FastnnError::dtype(format!(
                "F32 slice access requires F32 storage, got {:?}",
                self.dtype
            )));
        }
        if !self.is_contiguous() {
            return Err(FastnnError::shape(
                "F32 slice access requires a contiguous tensor",
            ));
        }
        if self.storage_offset < 0 {
            return Err(FastnnError::shape(format!(
                "negative storage offset {} cannot be sliced",
                self.storage_offset
            )));
        }
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                let offset = self.storage_offset as usize;
                let numel = self.numel() as usize;
                let scalar_bytes = self.dtype.scalar_byte_width().ok_or_else(|| {
                    FastnnError::dtype("F32 slice access requires plain scalar storage")
                })?;
                let byte_offset = offset.checked_mul(scalar_bytes).ok_or_else(|| {
                    FastnnError::Overflow("F32 slice byte offset overflow".into())
                })?;
                let byte_len = numel.checked_mul(scalar_bytes).ok_or_else(|| {
                    FastnnError::Overflow("F32 slice byte length overflow".into())
                })?;
                let byte_end = byte_offset
                    .checked_add(byte_len)
                    .ok_or_else(|| FastnnError::Overflow("F32 slice byte range overflow".into()))?;
                if byte_end > cpu.data.len() {
                    return Err(FastnnError::OutOfBounds(format!(
                        "F32 slice byte range {byte_offset}..{byte_end} exceeds storage length {}",
                        cpu.data.len()
                    )));
                }
                bytemuck::try_cast_slice(&cpu.data[byte_offset..byte_end]).map_err(|error| {
                    FastnnError::Internal(format!("CPU storage cannot be viewed as F32: {error}"))
                })
            }
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        self.try_as_f32_slice_mut()
            .expect("TensorImpl::as_f32_slice_mut failed")
    }

    pub fn try_as_f32_slice_mut(&mut self) -> FastnnResult<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(FastnnError::dtype(format!(
                "mutable F32 slice access requires F32 storage, got {:?}",
                self.dtype
            )));
        }
        if !self.is_contiguous() {
            return Err(FastnnError::shape(
                "mutable F32 slice access requires a contiguous tensor",
            ));
        }
        if self.storage_offset < 0 {
            return Err(FastnnError::shape(format!(
                "negative storage offset {} cannot be sliced",
                self.storage_offset
            )));
        }
        let offset = self.storage_offset as usize;
        let numel = self.numel() as usize;
        let scalar_bytes = self.dtype.scalar_byte_width().ok_or_else(|| {
            FastnnError::dtype("mutable F32 slice access requires plain scalar storage")
        })?;
        let byte_offset = offset
            .checked_mul(scalar_bytes)
            .ok_or_else(|| FastnnError::Overflow("mutable F32 byte offset overflow".into()))?;
        let byte_len = numel
            .checked_mul(scalar_bytes)
            .ok_or_else(|| FastnnError::Overflow("mutable F32 byte length overflow".into()))?;
        let byte_end = byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| FastnnError::Overflow("mutable F32 byte range overflow".into()))?;

        let storage = Arc::make_mut(&mut self.storage);
        match storage {
            Storage::Cpu(cpu) => {
                let data = Arc::make_mut(&mut cpu.data);
                if byte_end > data.len() {
                    return Err(FastnnError::OutOfBounds(format!(
                        "mutable F32 byte range {byte_offset}..{byte_end} exceeds storage length {}",
                        data.len()
                    )));
                }
                bytemuck::try_cast_slice_mut(&mut data[byte_offset..byte_end]).map_err(|error| {
                    FastnnError::Internal(format!(
                        "CPU storage cannot be mutably viewed as F32: {error}"
                    ))
                })
            }
        }
    }
}

impl Clone for TensorImpl {
    fn clone(&self) -> Self {
        // Clone the storage by cloning the Arc<Mutex<Storage>>
        // This shares the same Mutex<Storage> between clones
        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes: self.sizes.clone(),
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            contiguous_cache: AtomicI8::new(self.contiguous_cache.load(Ordering::Relaxed)),
        }
    }
}

impl Drop for TensorImpl {
    fn drop(&mut self) {
        // If we are the last owner of the storage, return it to the pool
        if Arc::strong_count(&self.storage) == 1 {
            let storage = self.storage.clone();
            get_storage_pool().release(storage);
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub inner: Arc<TensorImpl>,
}

impl Tensor {
    pub fn new(inner: TensorImpl) -> Self {
        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn id(&self) -> usize {
        self.inner.id()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    pub fn item(&self) -> FastnnResult<f32> {
        if self.inner.numel() != 1 {
            return Err(FastnnError::shape(format!(
                "item() requires one element, got {}",
                self.inner.numel()
            )));
        }

        let ptr = match self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr(),
        };
        Ok(match self.inner.dtype {
            DType::F32 => {
                let f32_ptr = ptr as *const f32;
                // SAFETY: `self.inner.numel() == 1` was validated above. The pointer
                // is derived from the CPU storage allocation and `storage_offset` is
                // within bounds for a single element read.
                unsafe { *f32_ptr.add(self.inner.storage_offset as usize) }
            }
            DType::F64 => {
                let f64_ptr = ptr as *const f64;
                // SAFETY: Same as F32 case -- single element read from valid storage.
                unsafe { *f64_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::I32 => {
                let i32_ptr = ptr as *const i32;
                // SAFETY: Same as F32 case -- single element read from valid storage.
                unsafe { *i32_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::I64 => {
                let i64_ptr = ptr as *const i64;
                // SAFETY: Same as F32 case -- single element read from valid storage.
                unsafe { *i64_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::BF16 => {
                let bf16_ptr = ptr as *const half::bf16;
                // SAFETY: Same as F32 case -- single element read from valid storage.
                unsafe { f32::from(*bf16_ptr.add(self.inner.storage_offset as usize)) }
            }
            DType::F16 => {
                let f16_ptr = ptr as *const half::f16;
                // SAFETY: Same as F32 case -- single element read from valid storage.
                unsafe { f32::from(*f16_ptr.add(self.inner.storage_offset as usize)) }
            }
            dtype => {
                return Err(FastnnError::dtype(format!(
                    "item() does not support {dtype:?}"
                )))
            }
        })
    }

    pub fn to_numpy(&self) -> FastnnResult<Vec<f32>> {
        if self.inner.numel() == 0 {
            return Ok(Vec::new());
        }
        Ok(match &self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Fast path: contiguous F32 tensor - SIMD-accelerated copy
                if self.inner.dtype == DType::F32 && self.inner.is_contiguous() {
                    let slice = self.inner.as_f32_slice();
                    let mut result = vec![0.0f32; slice.len()];
                    let src = slice.as_ptr();
                    let dst = result.as_mut_ptr();
                    let len = slice.len();
                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                    {
                        simd_copy_f32(src, dst, len);
                    }
                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                    {
                        memcpy_f32(src, dst, len);
                    }
                    return Ok(result);
                }

                match self.inner.dtype {
                    DType::F32 => {
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ref().as_ptr() as *const f32;
                        let ndim = self.inner.ndim();
                        let strides = &self.inner.strides;
                        let sizes = &self.inner.sizes;
                        let offset = self.inner.storage_offset;

                        // Maintain running linear index and increment it
                        // instead of recomputing from scratch each iteration
                        let mut linear_idx = offset;
                        let mut indices = vec![0i64; ndim];

                        for _ in 0..self.inner.numel() {
                            unsafe {
                                result.push(*data.add(linear_idx as usize));
                            }

                            // Increment indices and update linear_idx incrementally
                            for dim in (0..ndim).rev() {
                                indices[dim] += 1;
                                linear_idx = linear_idx.saturating_add(strides[dim]);
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                // Wrap: subtract size * stride, reset to 0
                                linear_idx = linear_idx.saturating_sub(sizes[dim] * strides[dim]);
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::F64 => {
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ref().as_ptr() as *const f64;
                        let ndim = self.inner.ndim();
                        let strides = &self.inner.strides;
                        let sizes = &self.inner.sizes;
                        let offset = self.inner.storage_offset;
                        let mut linear_idx = offset;
                        let mut indices = vec![0i64; ndim];

                        for _ in 0..self.inner.numel() {
                            unsafe {
                                result.push(*data.add(linear_idx as usize) as f32);
                            }
                            for dim in (0..ndim).rev() {
                                indices[dim] += 1;
                                linear_idx = linear_idx.saturating_add(strides[dim]);
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                linear_idx = linear_idx.saturating_sub(sizes[dim] * strides[dim]);
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::I32 => {
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ref().as_ptr() as *const i32;
                        let ndim = self.inner.ndim();
                        let strides = &self.inner.strides;
                        let sizes = &self.inner.sizes;
                        let offset = self.inner.storage_offset;
                        let mut linear_idx = offset;
                        let mut indices = vec![0i64; ndim];

                        for _ in 0..self.inner.numel() {
                            unsafe {
                                result.push(*data.add(linear_idx as usize) as f32);
                            }
                            for dim in (0..ndim).rev() {
                                indices[dim] += 1;
                                linear_idx = linear_idx.saturating_add(strides[dim]);
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                linear_idx = linear_idx.saturating_sub(sizes[dim] * strides[dim]);
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::BF16 => {
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ref().as_ptr() as *const half::bf16;
                        let ndim = self.inner.ndim();
                        let strides = &self.inner.strides;
                        let sizes = &self.inner.sizes;
                        let offset = self.inner.storage_offset;
                        let mut linear_idx = offset;
                        let mut indices = vec![0i64; ndim];

                        for _ in 0..self.inner.numel() {
                            unsafe {
                                result.push(f32::from(*data.add(linear_idx as usize)));
                            }
                            for dim in (0..ndim).rev() {
                                indices[dim] += 1;
                                linear_idx += strides[dim];
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                linear_idx -= sizes[dim] * strides[dim];
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::F16 => {
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ref().as_ptr() as *const half::f16;
                        let ndim = self.inner.ndim();
                        let strides = &self.inner.strides;
                        let sizes = &self.inner.sizes;
                        let offset = self.inner.storage_offset;
                        let mut linear_idx = offset;
                        let mut indices = vec![0i64; ndim];

                        for _ in 0..self.inner.numel() {
                            unsafe {
                                result.push(f32::from(*data.add(linear_idx as usize)));
                            }
                            for dim in (0..ndim).rev() {
                                indices[dim] += 1;
                                linear_idx += strides[dim];
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                linear_idx -= sizes[dim] * strides[dim];
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    dtype => {
                        return Err(FastnnError::dtype(format!(
                            "to_numpy() does not support {dtype:?}"
                        )))
                    }
                }
            }
        })
    }

    #[track_caller]
    pub fn data_ptr(&self) -> *const u8 {
        self.try_data_ptr().expect("Tensor::data_ptr failed")
    }

    pub fn try_data_ptr(&self) -> FastnnResult<*const u8> {
        self.inner.try_data_ptr()
    }

    #[track_caller]
    pub fn data_ptr_f32(&self) -> *const f32 {
        self.try_data_ptr_f32()
            .expect("Tensor::data_ptr_f32 failed")
    }

    pub fn try_data_ptr_f32(&self) -> FastnnResult<*const f32> {
        self.inner.try_data_ptr_f32()
    }

    pub fn data_ptr_f32_mut(&mut self) -> *mut f32 {
        self.try_data_ptr_f32_mut()
            .expect("Tensor::data_ptr_f32_mut failed")
    }

    pub fn try_data_ptr_f32_mut(&mut self) -> FastnnResult<*mut f32> {
        Arc::make_mut(&mut self.inner).try_data_ptr_f32_mut()
    }

    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.try_data_ptr_mut()
            .expect("Tensor::data_ptr_mut failed")
    }

    pub fn try_data_ptr_mut(&mut self) -> FastnnResult<*mut u8> {
        Arc::make_mut(&mut self.inner).try_data_ptr_mut()
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        self.try_as_f32_slice()
            .expect("Tensor::as_f32_slice failed")
    }

    pub fn try_as_f32_slice(&self) -> FastnnResult<&[f32]> {
        self.inner.try_as_f32_slice()
    }

    /// Get a direct byte slice view of contiguous CPU tensor data.
    pub fn as_byte_slice(&self) -> Option<&[u8]> {
        self.try_as_bytes().ok()
    }

    pub fn try_as_bytes(&self) -> FastnnResult<&[u8]> {
        if !self.is_contiguous() {
            return Err(FastnnError::shape(
                "byte slice access requires a contiguous tensor",
            ));
        }
        if self.inner.storage_offset < 0 {
            return Err(FastnnError::shape(format!(
                "negative storage offset {} cannot be sliced",
                self.inner.storage_offset
            )));
        }
        match self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                let numel = self.inner.numel() as usize;
                let byte_len = self.inner.dtype.try_storage_bytes(numel)?;
                let start = if let Some(element_size) = self.inner.dtype.scalar_byte_width() {
                    (self.inner.storage_offset as usize)
                        .checked_mul(element_size)
                        .ok_or_else(|| FastnnError::Overflow("byte slice offset overflow".into()))?
                } else if self.inner.storage_offset == 0 {
                    0
                } else {
                    return Err(FastnnError::dtype(
                        "nonzero storage offsets are unsupported for packed byte slices",
                    ));
                };
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| FastnnError::Overflow("byte slice range overflow".into()))?;
                if end > cpu.data.len() {
                    return Err(FastnnError::OutOfBounds(format!(
                        "byte slice range {start}..{end} exceeds storage length {}",
                        cpu.data.len()
                    )));
                }
                Ok(&cpu.data[start..end])
            }
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        self.try_as_f32_slice_mut()
            .expect("Tensor::as_f32_slice_mut failed")
    }

    pub fn try_as_f32_slice_mut(&mut self) -> FastnnResult<&mut [f32]> {
        Arc::make_mut(&mut self.inner).try_as_f32_slice_mut()
    }
}

impl From<TensorImpl> for Tensor {
    fn from(inner: TensorImpl) -> Self {
        Tensor::new(inner)
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={})",
            self.shape(),
            self.dtype().as_str(),
            self.device().as_str()
        )
    }
}

pub fn einsum(equation: &str, tensors: &[Tensor]) -> Tensor {
    try_einsum(equation, tensors).expect("einsum received invalid inputs")
}

pub fn try_einsum(equation: &str, tensors: &[Tensor]) -> FastnnResult<Tensor> {
    let (inputs, output) = equation
        .split_once("->")
        .ok_or_else(|| FastnnError::shape("einsum equation must contain exactly one '->'"))?;
    if output.contains("->") {
        return Err(FastnnError::shape(
            "einsum equation must contain exactly one '->'",
        ));
    }
    let labels: Vec<&str> = inputs.split(',').map(str::trim).collect();
    if tensors.len() != 2 || labels.len() != 2 {
        return Err(FastnnError::shape(
            "einsum currently requires exactly two tensors and two input terms",
        ));
    }
    for (index, (term, tensor)) in labels.iter().zip(tensors).enumerate() {
        if term.len() != tensor.ndim() {
            return Err(FastnnError::shape(format!(
                "einsum input {index} has {} labels but tensor rank {}",
                term.len(),
                tensor.ndim()
            )));
        }
        let mut seen = [false; 26];
        for byte in term.bytes() {
            if !byte.is_ascii_lowercase() {
                return Err(FastnnError::shape(
                    "einsum labels must be lowercase ASCII letters",
                ));
            }
            let position = usize::from(byte - b'a');
            if std::mem::replace(&mut seen[position], true) {
                return Err(FastnnError::shape(
                    "repeated labels within one einsum input are not supported",
                ));
            }
        }
    }
    let mut output_seen = [false; 26];
    for byte in output.bytes() {
        if !byte.is_ascii_lowercase() {
            return Err(FastnnError::shape(
                "einsum labels must be lowercase ASCII letters",
            ));
        }
        let position = usize::from(byte - b'a');
        if std::mem::replace(&mut output_seen[position], true)
            || !labels.iter().any(|term| term.as_bytes().contains(&byte))
        {
            return Err(FastnnError::shape(
                "einsum output labels must be unique and present in an input",
            ));
        }
    }
    if tensors[0].dtype() != tensors[1].dtype() || tensors[0].device() != tensors[1].device() {
        return Err(FastnnError::dtype(
            "einsum input tensors must have matching dtypes and devices",
        ));
    }
    for (left, byte) in labels[0].bytes().enumerate() {
        if let Some(right) = labels[1].bytes().position(|candidate| candidate == byte) {
            if tensors[0].shape()[left] != tensors[1].shape()[right] {
                return Err(FastnnError::shape(format!(
                    "einsum label '{}' has mismatched dimensions {} and {}",
                    char::from(byte),
                    tensors[0].shape()[left],
                    tensors[1].shape()[right]
                )));
            }
        }
    }
    Ok(einsum_validated(equation, tensors))
}

fn einsum_validated(equation: &str, tensors: &[Tensor]) -> Tensor {
    let parts: Vec<&str> = equation.split("->").collect();
    let (input_str, output_str) = if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        (parts[0], "")
    };
    let input_tensors: Vec<&str> = input_str.split(',').map(|s| s.trim()).collect();
    if input_tensors.len() != tensors.len() {
        panic!(
            "einsum: number of tensors ({}) doesn't match equation ({})",
            tensors.len(),
            input_tensors.len()
        );
    }
    if input_tensors.len() != 2 {
        panic!("einsum: only 2-tensor einsum is supported");
    }

    let a = &tensors[0];
    let b = &tensors[1];
    let a_chars: Vec<char> = input_tensors[0].chars().collect();
    let b_chars: Vec<char> = input_tensors[1].chars().collect();
    let out_chars: Vec<char> = output_str.chars().collect();

    // Pre-compute character-to-position maps (ASCII lowercase)
    let mut a_pos = [-1i64; 256];
    for (i, &c) in a_chars.iter().enumerate() {
        a_pos[c as usize] = i as i64;
    }
    let mut b_pos = [-1i64; 256];
    for (i, &c) in b_chars.iter().enumerate() {
        b_pos[c as usize] = i as i64;
    }
    let mut out_pos = [-1i64; 256];
    for (i, &c) in out_chars.iter().enumerate() {
        out_pos[c as usize] = i as i64;
    }

    // Bitmask for b_chars membership
    let mut b_contains = [false; 256];
    for &c in &b_chars {
        b_contains[c as usize] = true;
    }

    // Find contracted dims (appear in both inputs but not in output)
    let mut contracted = [false; 256];
    let mut num_contracted = 0usize;
    for &c in &a_chars {
        if b_contains[c as usize] && out_pos[c as usize] == -1 {
            contracted[c as usize] = true;
            num_contracted += 1;
        }
    }

    // Build permutation for a: move contracted dims to the end
    let mut a_perm: SmallVec<[i64; 6]> = SmallVec::with_capacity(a_chars.len());
    for &c in &a_chars {
        if !contracted[c as usize] {
            a_perm.push(a_pos[c as usize]);
        }
    }
    for &c in &a_chars {
        if contracted[c as usize] {
            a_perm.push(a_pos[c as usize]);
        }
    }
    let a_permuted = if a_perm.iter().enumerate().all(|(i, &p)| p as usize == i) {
        a.clone()
    } else {
        a.permute(a_perm.to_vec())
    };

    // Build permutation for b: move contracted dims to the front
    let mut b_perm: SmallVec<[i64; 6]> = SmallVec::with_capacity(b_chars.len());
    for &c in &b_chars {
        if contracted[c as usize] {
            b_perm.push(b_pos[c as usize]);
        }
    }
    for &c in &b_chars {
        if !contracted[c as usize] {
            b_perm.push(b_pos[c as usize]);
        }
    }
    let b_permuted = if b_perm.iter().enumerate().all(|(i, &p)| p as usize == i) {
        b.clone()
    } else {
        b.permute(b_perm.to_vec())
    };

    // Reshape to 2D for matmul
    let a_batch: i64 = a_permuted.shape()[..a_permuted.ndim() - num_contracted]
        .iter()
        .product();
    let a_contract: i64 = a_permuted.shape()[a_permuted.ndim() - num_contracted..]
        .iter()
        .product();
    let b_contract: i64 = b_permuted.shape()[..num_contracted].iter().product();
    let b_rest: i64 = b_permuted.shape()[num_contracted..].iter().product();

    if a_contract != b_contract {
        panic!(
            "einsum: contracted dimensions don't match: {} vs {}",
            a_contract, b_contract
        );
    }

    let a_2d = a_permuted.reshape(vec![a_batch, a_contract]);
    let b_2d = b_permuted.reshape(vec![b_contract, b_rest]);
    let result_2d = a_2d.matmul(&b_2d);

    // Reshape to output shape using pre-computed maps (no linear scans)
    let mut out_shape: SmallVec<[i64; 6]> = SmallVec::with_capacity(out_chars.len());
    for &c in &out_chars {
        let pos = a_pos[c as usize];
        if pos != -1 {
            out_shape.push(a.inner.sizes[pos as usize]);
        } else {
            out_shape.push(b.inner.sizes[b_pos[c as usize] as usize]);
        }
    }
    if out_shape.is_empty() {
        result_2d.reshape(vec![])
    } else {
        result_2d.reshape(out_shape.to_vec())
    }
}

pub fn dtype_to_ir(dt: DType) -> FastnnResult<IrDType> {
    Ok(match dt {
        DType::F32 => IrDType::F32,
        DType::F64 => return Err(FastnnError::dtype("F64 is not supported in executable IR")),
        DType::I32 => IrDType::I32,
        DType::I64 => IrDType::I64,
        DType::Bool => IrDType::Bool,
        DType::F16 => IrDType::F16,
        DType::BF16 => IrDType::BF16,
        DType::I4
        | DType::I8Scaled
        | DType::F8
        | DType::F8R
        | DType::F4
        | DType::U4Scaled
        | DType::U8Scaled => {
            return Err(FastnnError::dtype(
                "packed eager dtype cannot enter executable IR without representation metadata",
            ))
        }
    })
}

pub fn representation_to_dtype(representation: &ValueRepresentation) -> FastnnResult<DType> {
    representation.validate()?;
    if matches!(
        representation.transform,
        RepresentationTransform::RuntimeAffineQuantization
            | RepresentationTransform::RuntimeScaledAffine
    ) {
        return Err(FastnnError::dtype(
            "runtime-described quantization is not a Tensor-level dtype",
        ));
    }
    Ok(match representation.storage {
        ScalarType::F64 => DType::F64,
        ScalarType::F32 => DType::F32,
        ScalarType::F16 => DType::F16,
        ScalarType::BF16 => DType::BF16,
        ScalarType::I64 => DType::I64,
        ScalarType::I32 => DType::I32,
        ScalarType::Bool => DType::Bool,
        ScalarType::I4 => DType::I4,
        ScalarType::I8 => DType::I8Scaled,
        ScalarType::U4 => DType::U4Scaled,
        ScalarType::U8 => DType::U8Scaled,
        ScalarType::Fp4E2M1 => DType::F4,
        ScalarType::Fp8E4M3 => DType::F8,
        ScalarType::Fp8E5M2 => DType::F8R,
    })
}

pub fn tensor_type_to_dtype(tensor_type: &TensorType) -> FastnnResult<DType> {
    representation_to_dtype(&tensor_type.value_representation()?)
}

impl Tensor {
    /// Expose the tensor's contiguous CPU data as raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.try_as_bytes().expect("Tensor::as_bytes failed")
    }

    /// Build a single-op graph, compile it with the AOT pipeline, execute,
    /// and return the output as a `Tensor`.
    ///
    /// `build_graph` receives the [`GraphBuilder`] and its graph-tensor inputs,
    /// and must return the output [`GraphTensor`]s for the operation.
    pub fn exec_aot<F>(inputs: &[&Tensor], build_graph: F) -> Result<Vec<Tensor>, BackendError>
    where
        F: FnOnce(
            &GraphBuilder,
            &[crate::ir::builder::GraphTensor],
        ) -> Vec<crate::ir::builder::GraphTensor>,
    {
        let g = GraphBuilder::new();
        let graph_inputs: Vec<_> = inputs
            .iter()
            .map(|t| {
                let dims: Vec<DimExpr> = t
                    .shape()
                    .iter()
                    .map(|&s| DimExpr::Known(s as u64))
                    .collect();
                let dtype = dtype_to_ir(t.dtype())
                    .map_err(|error| BackendError::Compilation(error.to_string()))?;
                Ok(g.input_with_dims(&dims, dtype))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        let graph_outputs = build_graph(&g, &graph_inputs);

        // A contiguous input is already in the layout the plan expects, and
        // `inputs` outlives the whole call, so borrow it instead of cloning.
        // The clone used to cost one full-size allocation and copy per input on
        // every eager module call, which at CNN feature-map sizes dominated the
        // kernel it was feeding. Only a strided input needs a materialized
        // contiguous copy.
        let materialized_inputs: Vec<Option<Tensor>> = inputs
            .iter()
            .map(|tensor| {
                if tensor.is_contiguous() {
                    Ok(None)
                } else {
                    tensor
                        .try_contiguous()
                        .map(Some)
                        .map_err(|error| BackendError::Dispatch(error.to_string()))
                }
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let input_sources: Vec<&Tensor> = inputs
            .iter()
            .zip(materialized_inputs.iter())
            .map(|(original, contiguous)| contiguous.as_ref().unwrap_or(original))
            .collect();
        let input_bytes: Vec<&[u8]> = input_sources
            .iter()
            .map(|tensor| {
                tensor
                    .try_as_bytes()
                    .map_err(|error| BackendError::Dispatch(error.to_string()))
            })
            .collect::<Result<_, _>>()?;
        let result_bytes = g.compile_and_execute::<CpuBackend>(
            &graph_outputs.iter().collect::<Vec<_>>(),
            CpuBackend,
            &input_bytes,
        )?;

        graph_outputs
            .into_iter()
            .zip(result_bytes)
            .map(|(gt, bytes)| -> Result<Tensor, BackendError> {
                let shape: SmallVec<[i64; 8]> = gt
                    .shape()
                    .iter()
                    .map(|d| match d {
                        DimExpr::Known(v) => *v as i64,
                        _ => unreachable!("AOT bridge: graph op output has concrete shape"),
                    })
                    .collect();
                let dt = tensor_type_to_dtype(gt.tensor_type())
                    .map_err(|error| BackendError::Compilation(error.to_string()))?;
                let numel: usize = shape.iter().map(|&s| s as usize).product();
                let expected_bytes = dt.storage_bytes(numel);
                let num_bytes = bytes.len();
                // Validate that the AOT pipeline produced the right number of
                // bytes for the shape inferred during graph construction.
                // A mismatch here indicates a shape-inference or memory-planning
                // bug in the compiler pipeline.
                assert_eq!(
                    num_bytes, expected_bytes,
                    "AOT bridge: output byte size mismatch for shape {:?}, dtype {:?}: \
                     got {} bytes but expected {} for {} logical elements. \
                     This is likely a shape-inference bug in the compiler pass.",
                    shape, dt, num_bytes, expected_bytes, numel
                );
                let storage = Storage::Cpu(crate::storage::CpuStorage::from_vec(bytes, num_bytes));
                Ok(Tensor::new(TensorImpl::new(Arc::new(storage), shape, dt)))
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantizationGranularity;

    #[test]
    fn canonical_representation_maps_to_eager_storage_dtype() {
        let u4 = ValueRepresentation::packed_affine_dequantization(
            ScalarType::U4,
            8,
            QuantizationGranularity::PerTensor,
            vec![0.25],
            vec![-1.0],
        )
        .unwrap();
        assert_eq!(representation_to_dtype(&u4).unwrap(), DType::U4Scaled);

        let f8 = ValueRepresentation {
            logical: ScalarType::F32,
            storage: ScalarType::Fp8E4M3,
            encoding: crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: 4,
            },
            transform: RepresentationTransform::Scaled { scales: vec![1.0] },
        };
        assert_eq!(representation_to_dtype(&f8).unwrap(), DType::F8);
    }

    #[test]
    fn runtime_quantization_is_not_an_eager_storage_dtype() {
        let runtime = ValueRepresentation::runtime_affine(
            ScalarType::I8,
            crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: 4,
            },
        )
        .unwrap();
        let error = representation_to_dtype(&runtime).unwrap_err();
        assert!(error.to_string().contains("runtime-described quantization"));
    }

    #[test]
    fn test_cat_dim0() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);
        let c = Tensor::cat(&[a, b], 0);
        assert_eq!(c.shape(), vec![3, 2]);
        let data = c.as_f32_slice();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[4] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_repeat() {
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let y = x.repeat(&[3, 2]);
        assert_eq!(y.shape(), vec![3, 4]);
    }

    #[test]
    fn test_where_tensor() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let cond = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
        let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
        let result = a.where_tensor(&cond, &b);
        let data = result.as_f32_slice();
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 30.0).abs() < 1e-5);
        assert!((data[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_einsum_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![3, 2]);
        let c = einsum("ij,jk->ik", &[a, b]);
        assert_eq!(c.shape(), vec![2, 2]);
    }

    #[test]
    fn test_einsum_dot() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        let result = einsum("i,i->", &[x, y]);
        assert!((result.item().unwrap() - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let y = x.leaky_relu(0.1);
        let data = y.as_f32_slice();
        assert!((data[0] - (-0.2)).abs() < 1e-5);
        assert!((data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus() {
        let x = Tensor::from_vec(vec![0.0], vec![1]);
        let y = x.softplus(1.0, 20.0);
        let data = y.as_f32_slice();
        // softplus(0) = ln(1 + exp(0)) / 1 = ln(2) ≈ 0.693
        assert!((data[0] - std::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn test_hardswish() {
        let x = Tensor::from_vec(vec![0.0], vec![1]);
        let y = x.hardswish();
        let data = y.as_f32_slice();
        // hardswish(0) = 0 * relu6(3) / 6 = 0
        assert!((data[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_flip() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = x.flip(&[0]);
        let data = y.as_f32_slice();
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maximum() {
        let a = Tensor::from_vec(vec![1.0, 5.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![2.0, 4.0, 4.0], vec![3]);
        let c = a.maximum(&b);
        let data = c.as_f32_slice();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = x.log_softmax(0);
        let data = y.as_f32_slice();
        // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
        // log_softmax = [-2.4076, -1.4076, -0.4076]
        assert!((data[0] - (-2.4076)).abs() < 1e-3);
        assert!((data[1] - (-1.4076)).abs() < 1e-3);
        assert!((data[2] - (-0.4076)).abs() < 1e-3);
    }
}
