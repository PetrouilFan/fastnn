use crate::autograd::{self, AutogradMeta};
use crate::backend::cpu::CpuBackend;
use crate::backend::BackendError;
use crate::error::{FastnnError, FastnnResult};
use crate::ir::builder::GraphBuilder;
use crate::ir::{DimExpr, IrDType};
use crate::storage::{DType, Device, Storage};
use crate::storage_pool::get_storage_pool;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicI8, AtomicU64, Ordering};
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
    pub version_counter: Arc<AtomicU64>,
    pub autograd_meta: Option<Arc<parking_lot::Mutex<AutogradMeta>>>,
    pub requires_grad: bool,
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
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: None,
            requires_grad: false,
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
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: None,
            requires_grad: false,
            contiguous_cache: AtomicI8::new(1),
        })
    }

    pub fn id(&self) -> usize {
        let ptr = self as *const TensorImpl;
        ptr as usize
    }

    /// Create a view sharing storage and version_counter from self.
    /// Creates a fresh autograd_meta for the view to prevent the view from modifying the source's grad_fn.
    pub(crate) fn new_view_from(
        &self,
        sizes: SmallVec<[i64; 8]>,
        strides: SmallVec<[i64; 8]>,
        storage_offset: i64,
    ) -> Self {
        let autograd_meta = self.autograd_meta.as_ref().map(|meta| {
            let lock = meta.lock();
            Arc::new(parking_lot::Mutex::new(AutogradMeta::new(
                lock.requires_grad,
            )))
        });
        Self {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta,
            requires_grad: self.requires_grad,
            contiguous_cache: AtomicI8::new(-1), // strides may be non-contiguous
        }
    }

    /// Create a tensor on a different device (fresh version_counter, shares autograd_meta).
    fn new_on_device(&self, storage: Arc<Storage>, device: Device) -> Self {
        Self {
            storage,
            sizes: self.sizes.clone(),
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device,
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: self.autograd_meta.clone(),
            requires_grad: self.requires_grad,
            contiguous_cache: AtomicI8::new(-1), // strides copied from self, may be non-contiguous
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set autograd_meta and sync the cached requires_grad bool in one call.
    /// Prefer this over writing to `autograd_meta` directly to keep the
    /// cached `requires_grad` field consistent.
    pub fn set_autograd_meta(&mut self, meta: AutogradMeta) {
        let rg = meta.requires_grad;
        self.autograd_meta = Some(Arc::new(parking_lot::Mutex::new(meta)));
        self.requires_grad = rg;
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if self.autograd_meta.is_none() {
            self.autograd_meta = Some(Arc::new(parking_lot::Mutex::new(AutogradMeta::new(
                requires_grad,
            ))));
        } else if let Some(meta) = &mut self.autograd_meta {
            let mut lock = meta.lock();
            lock.requires_grad = requires_grad;
        }
    }

    pub fn requires_grad_(mut self, requires_grad: bool) -> Tensor {
        self.requires_grad = requires_grad;
        if let Some(meta) = &mut self.autograd_meta {
            let mut lock = meta.lock();
            lock.requires_grad = requires_grad;
        }
        self.into()
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.autograd_meta.as_ref()?.lock().grad.clone()
    }

    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        if let Some(meta) = &mut self.autograd_meta {
            let mut lock = meta.lock();
            lock.grad = grad;
        }
    }

    pub fn set_grad_for_tensor(tensor: &Tensor, grad: Option<Tensor>) {
        if let Some(meta) = &tensor.inner.autograd_meta {
            let mut lock = meta.lock();
            lock.grad = grad;
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.autograd_meta
            .as_ref()
            .map(|m| m.lock())
            .map(|m| m.is_leaf)
            .unwrap_or(true)
    }

    pub fn grad_fn(&self) -> Option<Arc<autograd::NodeInfo>> {
        self.autograd_meta.as_ref()?.lock().grad_fn.clone()
    }

    pub fn detach(&self) -> Tensor {
        let mut new = self.clone();
        new.autograd_meta = None;
        new.requires_grad = false;
        new.into()
    }

    pub fn increment_version(&self) {
        self.version_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn version(&self) -> u64 {
        self.version_counter.load(Ordering::Relaxed)
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => Err(FastnnError::device(
                "cannot get a CPU pointer from GPU storage; move the tensor to CPU first",
            )),
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => Err(FastnnError::device(
                "cannot get a mutable CPU pointer from GPU storage; move the tensor to CPU first",
            )),
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => Err(FastnnError::device(
                "cannot slice GPU storage; move the tensor to CPU first",
            )),
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => Err(FastnnError::device(
                "cannot mutably slice GPU storage; move the tensor to CPU first",
            )),
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
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
            requires_grad: self.requires_grad,
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

    pub(crate) fn attach_grad_fn(mut output: Tensor, backward: Arc<autograd::NodeInfo>) -> Tensor {
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(backward);
        let inner = Arc::make_mut(&mut output.inner);
        inner.autograd_meta = Some(Arc::new(parking_lot::Mutex::new(meta)));
        inner.requires_grad = true;
        output
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn requires_grad_(&self, requires_grad: bool) -> Tensor {
        let mut inner = self.inner.clone();
        Arc::make_mut(&mut inner).set_requires_grad(requires_grad);
        Tensor { inner }
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad()
    }

    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(meta) = &self.inner.autograd_meta {
            let mut lock = meta.lock();
            lock.grad = grad;
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    pub fn grad_fn(&self) -> Option<Arc<autograd::NodeInfo>> {
        self.inner.grad_fn()
    }

    pub fn detach(&self) -> Tensor {
        self.inner.detach()
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => {
                return Err(FastnnError::device(
                    "item() requires CPU storage; transfer the tensor to CPU first",
                ));
            }
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => {
                return Err(FastnnError::device(
                    "to_numpy() requires CPU storage; transfer the tensor to CPU first",
                ));
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
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => Err(FastnnError::device(
                "cannot slice GPU storage; move the tensor to CPU first",
            )),
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        self.try_as_f32_slice_mut()
            .expect("Tensor::as_f32_slice_mut failed")
    }

    pub fn try_as_f32_slice_mut(&mut self) -> FastnnResult<&mut [f32]> {
        Arc::make_mut(&mut self.inner).try_as_f32_slice_mut()
    }

    pub fn increment_version(&self) {
        self.inner.increment_version();
    }

    pub fn version(&self) -> u64 {
        self.inner.version()
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

pub fn clip_grad_norm_(tensors: &[Tensor], max_norm: f32, norm_type: f32) -> f32 {
    try_clip_grad_norm_(tensors, max_norm, norm_type).expect("clip_grad_norm_ failed")
}

pub fn try_clip_grad_norm_(tensors: &[Tensor], max_norm: f32, norm_type: f32) -> FastnnResult<f32> {
    if !max_norm.is_finite() || max_norm < 0.0 {
        return Err(FastnnError::shape(
            "gradient maximum norm must be finite and non-negative",
        ));
    }
    if !norm_type.is_finite() || norm_type <= 0.0 {
        return Err(FastnnError::shape(
            "gradient norm type must be finite and positive",
        ));
    }
    let mut total_norm = 0.0f32;
    for t in tensors {
        if let Some(g) = t.grad() {
            let g_data = g.try_as_f32_slice()?;
            if norm_type == 2.0 {
                let param_norm = g_data.iter().map(|x| x * x).sum::<f32>().sqrt();
                total_norm += param_norm * param_norm;
            } else {
                total_norm += g_data.iter().map(|x| x.abs().powf(norm_type)).sum::<f32>();
            }
        }
    }
    if norm_type == 2.0 {
        total_norm = total_norm.sqrt();
    } else {
        total_norm = total_norm.powf(1.0 / norm_type);
    }
    if !total_norm.is_finite() {
        return Err(FastnnError::Computation(
            "gradient norm is not finite".into(),
        ));
    }
    let clip_coef = max_norm / total_norm.max(1e-6);
    if clip_coef < 1.0 {
        for t in tensors {
            if let Some(mut g) = t.grad() {
                for x in g.try_as_f32_slice_mut()?.iter_mut() {
                    *x *= clip_coef;
                }
                t.set_grad(Some(g));
            }
        }
    }
    Ok(total_norm)
}

pub fn clip_grad_value_(tensors: &[Tensor], clip_value: f32) {
    try_clip_grad_value_(tensors, clip_value).expect("clip_grad_value_ failed")
}

pub fn try_clip_grad_value_(tensors: &[Tensor], clip_value: f32) -> FastnnResult<()> {
    if !clip_value.is_finite() || clip_value < 0.0 {
        return Err(FastnnError::shape(
            "gradient clip value must be finite and non-negative",
        ));
    }
    for t in tensors {
        if let Some(mut g) = t.grad() {
            for x in g.try_as_f32_slice_mut()?.iter_mut() {
                *x = x.max(-clip_value).min(clip_value);
            }
            t.set_grad(Some(g));
        }
    }
    Ok(())
}

// =============================================================================
// AOT pipeline bridge — replace eager dispatcher calls with graph lowering
// =============================================================================

pub fn dtype_to_ir(dt: DType) -> FastnnResult<IrDType> {
    Ok(match dt {
        DType::F32 => IrDType::F32,
        DType::F64 => return Err(FastnnError::dtype("F64 is not supported in executable IR")),
        DType::I32 => IrDType::I32,
        DType::I64 => IrDType::I64,
        DType::Bool => IrDType::Bool,
        DType::F16 => IrDType::F16,
        DType::BF16 => IrDType::BF16,
        // I4/I8/F4/F8/F8R need per-channel scale/zp metadata that lives in the IR node,
        // not in the Tensor-level DType.  Use default values here; the actual
        // scales are filled in by the quantization compiler pass.
        DType::I4 => IrDType::I4 {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
            codebooks: vec![],
        },
        DType::I8Scaled => IrDType::I8Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        },
        DType::F8 => IrDType::F8 { scales: vec![1.0] },
        DType::F8R => IrDType::F8R { scales: vec![1.0] },
        DType::F4 => IrDType::F4 {
            scales: vec![1.0],
            dequant_offsets: vec![],
            codebooks: vec![],
        },
        DType::U4Scaled => IrDType::U4Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        },
        DType::U8Scaled => IrDType::U8Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        },
    })
}

pub fn ir_to_dtype(idt: IrDType) -> FastnnResult<DType> {
    Ok(match idt {
        IrDType::F32 => DType::F32,
        IrDType::F16 => DType::F16,
        IrDType::BF16 => DType::BF16,
        IrDType::I32 => DType::I32,
        IrDType::I64 => DType::I64,
        IrDType::Bool => DType::Bool,
        IrDType::I8 => {
            return Err(FastnnError::dtype(
                "runtime activation I8 is not a Tensor-level dtype",
            ))
        }
        // Packed types round-trip back to simple DType variants. Per-channel metadata
        // stays in the IR node; Tensor-level storage uses the packed dtype.
        IrDType::I4 { .. } => DType::I4,
        IrDType::I8Scaled { .. } => DType::I8Scaled,
        IrDType::F8 { .. } => DType::F8,
        IrDType::F8R { .. } => DType::F8R,
        IrDType::F4 { .. } => DType::F4,
        IrDType::U4Scaled { .. } => DType::U4Scaled,
        IrDType::U8Scaled { .. } => DType::U8Scaled,
    })
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

        let materialized_inputs: Vec<Tensor> = inputs
            .iter()
            .map(|tensor| {
                if tensor.is_contiguous() {
                    Ok((*tensor).clone())
                } else {
                    tensor
                        .try_contiguous()
                        .map_err(|error| BackendError::Dispatch(error.to_string()))
                }
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let input_bytes: Vec<&[u8]> = materialized_inputs
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
                let dt = ir_to_dtype(gt.dtype())
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
                let data = bytes.to_vec();
                let storage = Storage::Cpu(crate::storage::CpuStorage::from_vec(data, num_bytes));
                Ok(Tensor::new(TensorImpl::new(Arc::new(storage), shape, dt)))
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
