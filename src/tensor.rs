use crate::autograd::{self, AutogradMeta};
use crate::dispatcher::{device_to_dispatch_key, dispatch};
use crate::storage::{CpuStorage, DType, Device, GpuStorage, Storage};
use crate::storage_pool::get_storage_pool;
use parking_lot::RwLock;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

/// Cached scalar tensors for common dimension values (0-7)
/// Avoids heap allocation on every softmax/sum/mean/max call
fn dim_scalar(dim: i32) -> Tensor {
    use std::sync::OnceLock;
    static DIM_SCALARS: OnceLock<[Tensor; 8]> = OnceLock::new();
    let scalars =
        DIM_SCALARS.get_or_init(|| std::array::from_fn(|d| Tensor::from_scalar(d as f32)));
    let idx = dim as usize;
    if idx < scalars.len() {
        scalars[idx].clone()
    } else {
        Tensor::from_scalar(dim as f32)
    }
}

pub struct TensorImpl {
    pub storage: Arc<Storage>,
    pub sizes: SmallVec<[i64; 8]>,
    pub strides: SmallVec<[i64; 8]>,
    pub storage_offset: i64,
    pub dtype: DType,
    pub device: Device,
    pub version_counter: Arc<AtomicU64>,
    pub autograd_meta: Option<Arc<std::sync::Mutex<AutogradMeta>>>,
}

impl TensorImpl {
    pub fn new(storage: Arc<Storage>, sizes: SmallVec<[i64; 8]>, dtype: DType) -> Self {
        let device = storage.device(); // Get device from storage
        let numel: i64 = sizes.iter().product();
        let _nbytes = (numel * dtype.size() as i64) as usize;

        let strides = compute_strides(&sizes);

        TensorImpl {
            storage,
            sizes,
            strides,
            storage_offset: 0,
            dtype,
            device,
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: None,
        }
    }

    /// Create TensorImpl with specific device (ignoring storage device)
    pub fn new_with_device(
        storage: Arc<Storage>,
        sizes: SmallVec<[i64; 8]>,
        device: Device,
        dtype: DType,
    ) -> Self {
        let numel: i64 = sizes.iter().product();
        let _nbytes = (numel * dtype.size() as i64) as usize;

        let strides = compute_strides(&sizes);

        TensorImpl {
            storage,
            sizes,
            strides,
            storage_offset: 0,
            dtype,
            device,
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: None,
        }
    }

    pub fn from_data(data: &[f32], dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = smallvec![data.len() as i64];
        let storage = match dtype {
            DType::F32 => Arc::new(Storage::from_vec(data.to_vec(), DType::F32, device)),
            _ => panic!("Unsupported dtype for from_data"),
        };
        TensorImpl::new(storage, sizes, dtype)
    }

    pub fn id(&self) -> usize {
        let ptr = self as *const TensorImpl;
        ptr as usize
    }

    pub fn ndim(&self) -> usize {
        self.sizes.len()
    }

    pub fn numel(&self) -> i64 {
        self.sizes.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1i64;
        for (size, &stride) in self.sizes.iter().rev().zip(self.strides.iter().rev()) {
            if *size != 1 {
                if stride != expected_stride {
                    return false;
                }
                expected_stride *= *size;
            }
        }
        true
    }

    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return Tensor::new(self.clone());
        }
        // If not contiguous, we need to copy the data to a new contiguous layout
        let data = self.as_f32_slice().to_vec();
        let sizes = self.sizes.clone();
        let mut new_tensor = Tensor::from_vec(data, sizes.to_vec());

        // Preserve autograd metadata but without grad_fn (contiguous creates a copy)
        if let Some(meta) = &self.autograd_meta {
            let Ok(meta_lock) = meta.lock() else {
                return new_tensor;
            };
            if meta_lock.requires_grad {
                let new_meta = AutogradMeta::new_non_leaf(true);
                // Don't clone grad_fn - contiguous creates a copy, so it's a leaf
                Arc::make_mut(&mut new_tensor.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(new_meta)));
            }
        }

        new_tensor
    }

    pub fn view(&self, sizes: SmallVec<[i64; 8]>) -> TensorImpl {
        let mut new_sizes = sizes.clone();
        let mut product: i64 = 1;
        let mut minus_one_idx = None;

        for (i, s) in new_sizes.iter().enumerate() {
            if *s == -1 {
                if minus_one_idx.is_some() {
                    panic!("can only specify one unknown dimension in view");
                }
                minus_one_idx = Some(i);
            } else {
                product *= s;
            }
        }

        if let Some(idx) = minus_one_idx {
            if product == 0 {
                panic!("view: cannot infer dimension with zero-sized dimensions");
            }
            if self.numel() % product != 0 {
                panic!(
                    "size mismatch: view cannot infer dimension because {} is not divisible by {}",
                    self.numel(),
                    product
                );
            }
            new_sizes[idx] = self.numel() / product;
        }

        let numel: i64 = new_sizes.iter().product();
        if numel != self.numel() {
            panic!(
                "size mismatch: view size {} != numel {}",
                numel,
                self.numel()
            );
        }

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes: new_sizes.clone(),
            strides: compute_strides(&new_sizes),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
        }
    }

    pub fn reshape(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        let mut new_sizes = sizes.clone();
        let mut product: i64 = 1;
        let mut minus_one_idx = None;

        for (i, s) in new_sizes.iter().enumerate() {
            if *s == -1 {
                if minus_one_idx.is_some() {
                    panic!("can only specify one unknown dimension");
                }
                minus_one_idx = Some(i);
            } else if *s < 0 {
                panic!("reshape: negative size not allowed");
            } else {
                product *= s;
            }
        }

        if let Some(idx) = minus_one_idx {
            if product == 0 {
                panic!("reshape: cannot infer dimension with zero-sized dimensions");
            }
            if self.numel() % product != 0 {
                panic!(
                    "size mismatch: reshape cannot infer dimension because {} is not divisible by {}",
                    self.numel(),
                    product
                );
            }
            let known: i64 = self.numel() / product;
            new_sizes[idx] = known;
        } else {
            let numel: i64 = sizes.iter().product();
            if numel != self.numel() {
                panic!(
                    "size mismatch: reshape size {} != numel {}",
                    numel,
                    self.numel()
                );
            }
        }

        self.view(new_sizes).into()
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            panic!("transpose: dimension out of range");
        }

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();

        sizes.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
        }
        .into()
    }

    pub fn permute(&self, dims: SmallVec<[i64; 8]>) -> Tensor {
        let ndim = self.ndim();
        if dims.len() != ndim {
            panic!("permute: number of dimensions mismatch");
        }

        let mut seen = vec![false; ndim];
        for &d in &dims {
            if d < 0 || (d as usize) >= ndim || seen[d as usize] {
                panic!("permute: invalid permutation");
            }
            seen[d as usize] = true;
        }

        let mut sizes = SmallVec::new();
        let mut strides = SmallVec::new();

        for &d in &dims {
            sizes.push(self.sizes[d as usize]);
            strides.push(self.strides[d as usize]);
        }

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
        }
        .into()
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let ndim = self.ndim();
        let dim = if dim > ndim { ndim } else { dim };

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();
        sizes.insert(dim, 1);
        strides.insert(dim, if dim == ndim { 1 } else { self.strides[dim] });

        let input_tensor = Tensor::new(self.clone());

        let mut tensor = TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
        };

        if self.requires_grad() {
            let backward = Arc::new(autograd::UnsqueezeBackward::new(input_tensor, dim));
            let meta = AutogradMeta {
                grad: None,
                grad_fn: Some(backward.clone()),
                requires_grad: true,
                is_leaf: false,
            };
            tensor.autograd_meta = Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        tensor.into()
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        match dim {
            Some(d) => {
                if d >= self.ndim() || self.sizes[d] != 1 {
                    return self.clone().into();
                }
                let mut sizes = self.sizes.clone();
                let mut strides = self.strides.clone();
                sizes.remove(d);
                strides.remove(d);

                TensorImpl {
                    storage: Arc::clone(&self.storage),
                    sizes,
                    strides,
                    storage_offset: self.storage_offset,
                    dtype: self.dtype,
                    device: self.device,
                    version_counter: Arc::clone(&self.version_counter),
                    autograd_meta: self.autograd_meta.clone(),
                }
                .into()
            }
            None => {
                let mut sizes = SmallVec::new();
                let mut strides = SmallVec::new();
                for (s, &st) in self.sizes.iter().zip(self.strides.iter()) {
                    if *s != 1 {
                        sizes.push(*s);
                        strides.push(st);
                    }
                }

                if sizes.is_empty() {
                    sizes.push(1);
                    strides.push(1);
                }

                TensorImpl {
                    storage: Arc::clone(&self.storage),
                    sizes,
                    strides,
                    storage_offset: self.storage_offset,
                    dtype: self.dtype,
                    device: self.device,
                    version_counter: Arc::clone(&self.version_counter),
                    autograd_meta: self.autograd_meta.clone(),
                }
                .into()
            }
        }
    }

    pub fn expand(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        if sizes.len() < self.ndim() {
            panic!("expand: not enough dimensions");
        }

        let new_sizes = sizes.clone();
        let offset = sizes.len() - self.ndim();

        for i in 0..self.ndim() {
            let target = new_sizes[offset + i];
            let source = self.sizes[i];
            if target != source && source != 1 {
                panic!(
                    "expand: cannot expand dimension {} from {} to {}",
                    i, source, target
                );
            }
        }

        let mut new_strides: SmallVec<[i64; 8]> = smallvec![0; sizes.len()];
        for i in 0..self.ndim() {
            new_strides[offset + i] = if self.sizes[i] == 1 {
                0
            } else {
                self.strides[i]
            };
        }

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
        }
        .into()
    }

    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        if dim >= self.ndim() {
            panic!("slice: dimension out of range");
        }

        let size = self.sizes[dim];
        let start = if start < 0 { size + start } else { start };
        let end = if end < 0 { size + end } else { end };
        let start = start.max(0) as usize;
        let end = (end.min(size)) as usize;

        if start > end {
            panic!("slice: invalid range");
        }

        let mut sizes = self.sizes.clone();
        let numel = ((end as i64 - start as i64 + step - 1) / step) as usize;
        sizes[dim] = numel as i64;

        let storage_offset = self.storage_offset + (start as i64) * self.strides[dim];

        let mut strides = self.strides.clone();
        strides[dim] *= step;

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: None,
        }
        .into()
    }

    pub fn requires_grad(&self) -> bool {
        self.autograd_meta
            .as_ref()
            .and_then(|m| m.lock().ok())
            .map(|m| m.requires_grad)
            .unwrap_or(false)
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.autograd_meta.is_none() {
            self.autograd_meta = Some(Arc::new(std::sync::Mutex::new(AutogradMeta::new(
                requires_grad,
            ))));
        } else if let Some(meta) = &mut self.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.requires_grad = requires_grad;
            }
        }
    }

    pub fn requires_grad_(mut self, requires_grad: bool) -> Tensor {
        if let Some(meta) = &mut self.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.requires_grad = requires_grad;
            }
        }
        self.into()
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.autograd_meta.as_ref()?.lock().ok()?.grad.clone()
    }

    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        if let Some(meta) = &mut self.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = grad;
            }
        }
    }

    pub fn set_grad_for_tensor(tensor: &Tensor, grad: Option<Tensor>) {
        if let Some(meta) = &tensor.inner.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = grad;
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.autograd_meta
            .as_ref()
            .and_then(|m| m.lock().ok())
            .map(|m| m.is_leaf)
            .unwrap_or(true)
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn autograd::Node>> {
        self.autograd_meta.as_ref()?.lock().ok()?.grad_fn.clone()
    }

    pub fn detach(&self) -> Tensor {
        let mut new = self.clone();
        new.autograd_meta = None;
        new.into()
    }

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if dtype == self.dtype {
            return self.clone().into();
        }
        // TODO: implement proper type conversion
        self.clone().into()
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        if device == self.device {
            return self.clone().into();
        }
        // TODO: implement device transfer
        self.clone().into()
    }

    pub fn increment_version(&self) {
        self.version_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn version(&self) -> u64 {
        self.version_counter.load(Ordering::Relaxed)
    }

    #[track_caller]
    pub fn data_ptr(&self) -> *const u8 {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr(),
            Storage::Wgpu(_) => {
                let location = std::panic::Location::caller();
                panic!(
                    "Cannot get CPU pointer from GPU storage. Use .to_cpu() first.\nCalled from: {}:{}",
                    location.file(),
                    location.line()
                );
            }
        }
    }

    #[track_caller]
    pub fn data_ptr_f32(&self) -> *const f32 {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                let ptr = cpu.data.as_ref().as_ptr();
                // storage_offset is in elements, cast to f32 pointer first
                let f32_ptr = ptr as *const f32;
                unsafe { f32_ptr.add(self.storage_offset as usize) }
            }
            Storage::Wgpu(_) => {
                let location = std::panic::Location::caller();
                panic!(
                    "Cannot get CPU pointer from GPU storage. Device: {:?}, Storage: {:?}\nLocation: {}:{}",
                    self.device,
                    self.storage.as_ref(),
                    location.file(),
                    location.line()
                );
            }
        }
    }

    pub fn data_ptr_f32_mut(&mut self) -> *mut f32 {
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Unsafe: caller must ensure exclusive ownership of storage
                // This is guaranteed by &mut self if Arc is not shared
                let ptr = cpu.data.as_ref().as_ptr() as *mut f32;
                unsafe { ptr.add(self.storage_offset as usize) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Unsafe: caller must ensure exclusive ownership of storage
                // This is guaranteed by &mut self if Arc is not shared
                let ptr = cpu.data.as_ref().as_ptr() as *mut u8;
                let elem_size = match self.dtype {
                    DType::F32 | DType::I32 | DType::Bool => 4,
                    DType::F64 | DType::I64 => 8,
                    DType::F16 | DType::BF16 => 2,
                };
                unsafe { ptr.add(self.storage_offset as usize * elem_size) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => unsafe {
                let ptr = cpu.data.as_ref().as_ptr() as *const f32;
                let ptr = ptr.add(self.storage_offset as usize);
                let numel = self.numel() as usize;
                // Unconditional bounds validation to prevent UB in release builds
                let storage_len = cpu.data.len() / std::mem::size_of::<f32>();
                assert!(
                    self.storage_offset as usize + numel <= storage_len,
                    "as_f32_slice: offset + numel exceeds storage bounds. \
                     offset={}, numel={}, storage_len={}",
                    self.storage_offset,
                    numel,
                    storage_len
                );
                std::slice::from_raw_parts(ptr, numel)
            },
            Storage::Wgpu(_) => {
                panic!("Cannot slice GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        let ptr = self.data_ptr_f32_mut();
        let numel = self.numel() as usize;
        unsafe {
            // Unconditional bounds validation to prevent UB in release builds
            if let Storage::Cpu(cpu) = self.storage.as_ref() {
                let storage_len = cpu.data.len() / std::mem::size_of::<f32>();
                assert!(
                    self.storage_offset as usize + numel <= storage_len,
                    "as_f32_slice_mut: offset + numel exceeds storage bounds. \
                     offset={}, numel={}, storage_len={}",
                    self.storage_offset,
                    numel,
                    storage_len
                );
            }
            std::slice::from_raw_parts_mut(ptr, numel)
        }
    }

    /// Check if storage is on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Wgpu(_))
    }

    /// Check if storage is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    /// Get GPU buffer reference if on GPU
    pub fn gpu_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        match self.storage.as_ref() {
            Storage::Wgpu(gpu) => Some(gpu.buffer.clone()),
            Storage::Cpu(cpu) => {
                // Check cached GPU buffers
                let cache = cpu.gpu_buffer_cache.read();
                // Return any cached buffer (just use first one for now)
                cache.values().next().cloned()
            }
        }
    }

    /// Get GPU buffer for specific device, creating if needed
    pub fn get_or_create_gpu_buffer(&self, device_id: usize) -> Option<Arc<wgpu::Buffer>> {
        match self.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => Some(gpu.buffer.clone()),
            Storage::Wgpu(_) => None, // Wrong device
            Storage::Cpu(_) => {
                // Check cache first
                if let Some(buffer) = self.storage.get_or_create_gpu_buffer(device_id) {
                    return Some(buffer);
                }
                None
            }
        }
    }

    /// Cache a GPU buffer for this tensor
    pub fn cache_gpu_buffer(&self, device_id: usize, buffer: Arc<wgpu::Buffer>) {
        self.storage.cache_gpu_buffer(device_id, buffer);
    }

    /// Get CPU data if available
    pub fn cpu_data(&self) -> Option<&[u8]> {
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => Some(&cpu.data),
            _ => None,
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

fn compute_strides(sizes: &[i64]) -> SmallVec<[i64; 8]> {
    let mut strides: SmallVec<[i64; 8]> = smallvec![0; sizes.len()];
    if sizes.is_empty() {
        return strides;
    }

    let mut stride = 1i64;
    for i in (0..sizes.len()).rev() {
        strides[i] = stride;
        stride *= sizes[i];
    }
    strides
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

    pub fn from_scalar(value: f32) -> Self {
        let mut storage = Arc::new(Storage::new_cpu(DType::F32, 4));
        let storage_mut = Arc::make_mut(&mut storage);
        let Storage::Cpu(cpu_storage) = storage_mut else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu_storage.data);
        let ptr = data.as_mut_ptr() as *mut f32;
        unsafe {
            *ptr = value;
        }
        let sizes: SmallVec<[i64; 8]> = smallvec![];
        Tensor::new(TensorImpl::new(storage, sizes, DType::F32))
    }

    pub fn from_vec(values: Vec<f32>, shape: Vec<i64>) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let storage = Arc::new(Storage::from_vec(values, DType::F32, Device::Cpu));
        Tensor::new(TensorImpl::new(storage, sizes, DType::F32))
    }

    pub fn from_vec_with_device(values: Vec<f32>, shape: Vec<i64>, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        // Create CPU storage first (GPU buffer creation requires GpuContext)
        let storage = Arc::new(Storage::from_vec(values, DType::F32, Device::Cpu));
        // Create tensor with requested device (not storage device)
        Tensor::new(TensorImpl::new_with_device(
            storage,
            sizes,
            device,
            DType::F32,
        ))
    }

    pub fn id(&self) -> usize {
        self.inner.id()
    }

    pub fn zeros(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let numel: i64 = sizes.iter().product();
        let nbytes = (numel * dtype.size() as i64) as usize;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire(nbytes, device),
            Device::Wgpu(device_id) => {
                use crate::kernels::gpu::get_context;
                let ctx = get_context(device_id);
                let buffer = ctx.create_buffer(nbytes, "zeros");
                Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes,
                    device_id,
                    staging: RwLock::new(None),
                }))
            }
        };

        let strides = compute_strides(&sizes);
        Tensor::new(TensorImpl {
            storage,
            sizes,
            strides,
            storage_offset: 0,
            dtype,
            device,
            version_counter: Arc::new(AtomicU64::new(0)),
            autograd_meta: None,
        })
    }

    pub fn empty(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let numel: i64 = sizes.iter().product();
        let nbytes = (numel * dtype.size() as i64) as usize;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire(nbytes, device),
            Device::Wgpu(device_id) => {
                // For GPU empty, we create a new buffer (uninitialized)
                // Note: GPU buffers are not zeroed by default
                use crate::kernels::gpu::get_context;
                let ctx = get_context(device_id);
                let buffer = ctx.create_buffer(nbytes, "empty");
                Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes,
                    device_id,
                    staging: RwLock::new(None),
                }))
            }
        };

        match device {
            Device::Cpu => Tensor::new(TensorImpl::new(storage, sizes, dtype)),
            Device::Wgpu(_) => {
                Tensor::new(TensorImpl::new_with_device(storage, sizes, device, dtype))
            }
        }
    }

    pub fn ones(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        match device {
            Device::Cpu => {
                let mut t = Self::zeros(shape, dtype, device);
                let numel = t.inner.numel() as usize;
                let inner = Arc::make_mut(&mut t.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    panic!("Expected CPU storage for ones()");
                };
                let data = Arc::make_mut(&mut cpu_storage.data);
                match dtype {
                    DType::F32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, numel)
                        };
                        slice.fill(1.0);
                    }
                    DType::F64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, numel)
                        };
                        slice.fill(1.0);
                    }
                    DType::I32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut i32, numel)
                        };
                        slice.fill(1);
                    }
                    DType::BF16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut half::bf16,
                                numel,
                            )
                        };
                        slice.fill(half::bf16::from_f32(1.0));
                    }
                    DType::F16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut half::f16,
                                numel,
                            )
                        };
                        slice.fill(half::f16::from_f32(1.0));
                    }
                    DType::I64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut i64, numel)
                        };
                        slice.fill(1);
                    }
                    DType::Bool => {
                        let slice =
                            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr(), numel) };
                        slice.fill(1);
                    }
                }
                t
            }
            Device::Wgpu(device_id) => {
                let cpu_ones = Self::ones(shape.clone(), dtype, Device::Cpu);
                cpu_ones.to_gpu(device_id)
            }
        }
    }

    pub fn full(shape: Vec<i64>, value: f32, dtype: DType, device: Device) -> Self {
        let mut t = Self::zeros(shape, dtype, device);

        match device {
            Device::Cpu => {
                let numel = t.inner.numel() as usize;
                let inner = Arc::make_mut(&mut t.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    panic!("Expected CPU storage for full()");
                };
                let ptr = Arc::make_mut(&mut cpu_storage.data).as_mut_ptr();

                match dtype {
                    DType::F32 => {
                        let f32_ptr = ptr as *mut f32;
                        for i in 0..numel {
                            unsafe {
                                *f32_ptr.add(i) = value;
                            }
                        }
                    }
                    DType::F64 => {
                        let f64_ptr = ptr as *mut f64;
                        for i in 0..numel {
                            unsafe {
                                *f64_ptr.add(i) = value as f64;
                            }
                        }
                    }
                    DType::I32 => {
                        let i32_ptr = ptr as *mut i32;
                        for i in 0..numel {
                            unsafe {
                                *i32_ptr.add(i) = value as i32;
                            }
                        }
                    }
                    DType::BF16 => {
                        let bf16_ptr = ptr as *mut half::bf16;
                        for i in 0..numel {
                            unsafe {
                                *bf16_ptr.add(i) = half::bf16::from_f32(value);
                            }
                        }
                    }
                    DType::F16 => {
                        let f16_ptr = ptr as *mut half::f16;
                        for i in 0..numel {
                            unsafe {
                                *f16_ptr.add(i) = half::f16::from_f32(value);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Device::Wgpu(device_id) => {
                // For GPU tensors, use a kernel to fill with value
                // Create a scalar tensor with the value and broadcast it
                use crate::kernels::gpu::get_context;
                let ctx = get_context(device_id);

                // Create a buffer with the value
                let numel = t.inner.numel() as usize;
                let data = vec![value; numel];
                let buffer = ctx.create_gpu_buffer_from_data(&data, "full");

                // Update the tensor's storage
                let inner = Arc::make_mut(&mut t.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                *storage = Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: numel * 4,
                    device_id,
                    staging: RwLock::new(None),
                });
            }
        }
        t
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn shape(&self) -> Vec<i64> {
        self.inner.sizes.to_vec()
    }

    pub fn strides(&self) -> Vec<i64> {
        self.inner.strides.to_vec()
    }

    pub fn numel(&self) -> i64 {
        self.inner.numel()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    pub fn contiguous(&self) -> Tensor {
        self.inner.contiguous()
    }

    pub fn view(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output: Tensor = self.inner.view(sizes).into();

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ViewBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn reshape(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output = self.inner.reshape(sizes);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ViewBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    /// Fused reshape and permute operation to reduce intermediate allocations
    /// This is useful for attention mechanisms where reshape+permute is common
    pub fn reshape_permute(&self, shape: Vec<i64>, perm: Vec<i64>) -> Tensor {
        // First reshape
        let reshaped = self.reshape(shape);
        // Then permute
        let output = reshaped.permute(perm);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ViewBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        self.inner.transpose(dim0, dim1)
    }

    pub fn permute(&self, dims: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = dims.into();
        self.inner.permute(sizes)
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let output = self.inner.squeeze(dim);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ViewBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        self.inner.unsqueeze(dim)
    }

    pub fn expand(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        self.inner.expand(sizes)
    }

    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        let output = self.inner.slice(dim, start, end, step);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SliceBackward::new(self.clone(), dim, start, end, step, edges);
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        self.inner.to_dtype(dtype)
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        self.inner.to_device(device)
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn requires_grad_(&self, requires_grad: bool) -> Tensor {
        let mut inner = self.inner.clone();
        Arc::make_mut(&mut inner).set_requires_grad(requires_grad);
        Tensor::new(inner.as_ref().clone())
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad()
    }

    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(meta) = &self.inner.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = grad;
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn autograd::Node>> {
        self.inner.grad_fn()
    }

    pub fn detach(&self) -> Tensor {
        self.inner.detach()
    }

    pub fn item(&self) -> f32 {
        if self.inner.numel() != 1 {
            panic!("item() can only be called on tensors with one element");
        }

        let ptr = match self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr(),
            Storage::Wgpu(_) => {
                panic!("Cannot call item() on GPU tensor. Use .cpu() first.");
            }
        };
        match self.inner.dtype {
            DType::F32 => {
                let f32_ptr = ptr as *const f32;
                unsafe { *f32_ptr.add(self.inner.storage_offset as usize) }
            }
            DType::F64 => {
                let f64_ptr = ptr as *const f64;
                unsafe { *f64_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::I32 => {
                let i32_ptr = ptr as *const i32;
                unsafe { *i32_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::I64 => {
                let i64_ptr = ptr as *const i64;
                unsafe { *i64_ptr.add(self.inner.storage_offset as usize) as f32 }
            }
            DType::BF16 => {
                let bf16_ptr = ptr as *const half::bf16;
                unsafe { f32::from(*bf16_ptr.add(self.inner.storage_offset as usize)) }
            }
            DType::F16 => {
                let f16_ptr = ptr as *const half::f16;
                unsafe { f32::from(*f16_ptr.add(self.inner.storage_offset as usize)) }
            }
            _ => panic!("Unsupported dtype for item()"),
        }
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        match &self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Fast path: contiguous F32 tensor - direct memory copy
                if self.inner.dtype == DType::F32
                    && self.inner.is_contiguous()
                    && self.inner.storage_offset == 0
                {
                    let data = cpu.data.as_ref().as_ptr() as *const f32;
                    let numel = self.inner.numel() as usize;
                    return unsafe { std::slice::from_raw_parts(data, numel) }.to_vec();
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
                                linear_idx += strides[dim];
                                if indices[dim] < sizes[dim] {
                                    break;
                                }
                                // Wrap: subtract size * stride, reset to 0
                                linear_idx -= sizes[dim] * strides[dim];
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
                    _ => panic!("Unsupported dtype for to_numpy"),
                }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot convert GPU tensor to numpy directly. Use .cpu() first.");
            }
        }
    }

    #[track_caller]
    pub fn data_ptr(&self) -> *const u8 {
        self.inner.data_ptr()
    }

    #[track_caller]
    pub fn data_ptr_f32(&self) -> *const f32 {
        self.inner.data_ptr_f32()
    }

    pub fn data_ptr_f32_mut(&mut self) -> *mut f32 {
        let inner = &self.inner;
        match inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Unsafe: caller must ensure exclusive ownership of storage
                // This is guaranteed by &mut self if Arc is not shared
                let ptr = cpu.data.as_ref().as_ptr() as *mut f32;
                unsafe { ptr.add(inner.storage_offset as usize) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    /// Get a raw byte pointer to the tensor data (for arbitrary dtypes)
    /// Note: storage_offset is in elements, so we need to multiply by element size
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        let inner = &self.inner;
        match inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                // Unsafe: caller must ensure exclusive ownership of storage
                // This is guaranteed by &mut self if Arc is not shared
                let ptr = cpu.data.as_ref().as_ptr() as *mut u8;
                let elem_size = match inner.dtype {
                    DType::F32 | DType::I32 | DType::Bool => 4,
                    DType::F64 | DType::I64 => 8,
                    DType::F16 | DType::BF16 => 2,
                };
                unsafe { ptr.add(inner.storage_offset as usize * elem_size) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        // For BF16/F16 types, we need to convert to F32
        match self.inner.dtype {
            DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
                self.inner.as_f32_slice()
            }
            DType::BF16 | DType::F16 => {
                // For half-precision types, we need to convert to F32
                // This requires creating a new tensor with F32 dtype
                // For now, we'll panic as this is not yet fully implemented
                panic!("BF16/F16 to f32 slice conversion not yet implemented. Use dtype-specific operations instead.");
            }
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        // For BF16/F16 types, we cannot directly get a mutable f32 slice
        // The data is stored in half-precision format
        // For operations that need f32, use the dtype-specific operations
        match self.inner.dtype {
            DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
                // Use Arc::make_mut to ensure exclusive ownership
                // This will clone the TensorImpl if there are multiple Arc owners,
                // ensuring we have unique access before getting a mutable slice
                let inner = Arc::make_mut(&mut self.inner);
                inner.as_f32_slice_mut()
            }
            DType::BF16 | DType::F16 => {
                panic!("Cannot get mutable f32 slice for BF16/F16 tensor. Use dtype-specific operations.");
            }
        }
    }

    pub fn increment_version(&self) {
        self.inner.increment_version();
    }

    pub fn version(&self) -> u64 {
        self.inner.version()
    }

    /// Move tensor to CPU memory
    pub fn to_cpu(&self) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Cpu(_) => {
                // Storage is already CPU, but device might be GPU (lazy GPU tensor)
                // Return a clone with CPU device
                Tensor::new(TensorImpl {
                    storage: self.inner.storage.clone(),
                    sizes: self.inner.sizes.clone(),
                    strides: self.inner.strides.clone(),
                    storage_offset: self.inner.storage_offset,
                    dtype: self.inner.dtype,
                    device: Device::Cpu,
                    version_counter: Arc::new(AtomicU64::new(0)),
                    autograd_meta: self.inner.autograd_meta.clone(),
                })
            }
            Storage::Wgpu(gpu) => {
                // Read GPU buffer to CPU
                use crate::kernels::gpu::get_context;
                let ctx = get_context(gpu.device_id);
                let data = ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let storage = Arc::new(Storage::Cpu(CpuStorage {
                    data: Arc::new(bytemuck::cast_slice(&data).to_vec()),
                    nbytes: gpu.nbytes,
                    gpu_buffer_cache: RwLock::new(HashMap::new()),
                }));
                Tensor::new(TensorImpl {
                    storage,
                    sizes: self.inner.sizes.clone(),
                    strides: self.inner.strides.clone(),
                    storage_offset: self.inner.storage_offset,
                    dtype: self.inner.dtype,
                    device: Device::Cpu,
                    version_counter: Arc::new(AtomicU64::new(0)),
                    autograd_meta: self.inner.autograd_meta.clone(),
                })
            }
        }
    }

    /// Move tensor to GPU memory
    pub fn to_gpu(&self, device_id: usize) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => self.clone(),
            Storage::Wgpu(gpu) => {
                // Tensor is on a different GPU device, need to move it
                use crate::kernels::gpu::get_context;
                let src_ctx = get_context(gpu.device_id);
                let f32_data = src_ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let byte_data = bytemuck::cast_slice(&f32_data);

                let Some(dst_ctx) = crate::kernels::gpu::try_get_context(device_id) else {
                    // Target GPU unavailable, return data on CPU
                    let cpu_data: Vec<f32> = bytemuck::cast_slice(&f32_data).to_vec();
                    return Tensor::from_vec(cpu_data, self.shape().to_vec());
                };
                let buffer = dst_ctx.create_gpu_buffer_from_bytes(byte_data, "to_gpu");

                let storage = Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: gpu.nbytes,
                    device_id,
                    staging: RwLock::new(None),
                }));

                Tensor::new(TensorImpl {
                    storage,
                    sizes: self.inner.sizes.clone(),
                    strides: self.inner.strides.clone(),
                    storage_offset: self.inner.storage_offset,
                    dtype: self.inner.dtype,
                    device: Device::Wgpu(device_id),
                    version_counter: Arc::new(AtomicU64::new(0)),
                    autograd_meta: self.inner.autograd_meta.clone(),
                })
            }
            _ => {
                // CPU storage or "lazy GPU" tensor - get CPU data
                let cpu_data = self.as_f32_slice().to_vec();
                let dtype = self.inner.dtype;

                // Try to create GPU tensor; fall back to CPU if unavailable
                let Some(ctx) = crate::kernels::gpu::try_get_context(device_id) else {
                    return self.clone();
                };
                let buffer = ctx.create_gpu_buffer_from_data(&cpu_data, "to_gpu");

                let storage = Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: cpu_data.len() * 4,
                    device_id,
                    staging: RwLock::new(None),
                }));

                Tensor::new(TensorImpl {
                    storage,
                    sizes: self.inner.sizes.clone(),
                    strides: self.inner.strides.clone(),
                    storage_offset: self.inner.storage_offset,
                    dtype,
                    device: Device::Wgpu(device_id),
                    version_counter: Arc::new(AtomicU64::new(0)),
                    autograd_meta: self.inner.autograd_meta.clone(),
                })
            }
        }
    }

    fn broadcast_shapes(a: &[i64], b: &[i64]) -> Result<Vec<i64>, String> {
        let ndim = a.len().max(b.len());
        let mut out = vec![1i64; ndim];
        for i in 0..ndim {
            let a_dim = if i < ndim - a.len() {
                1
            } else {
                a[i - (ndim - a.len())]
            };
            let b_dim = if i < ndim - b.len() {
                1
            } else {
                b[i - (ndim - b.len())]
            };
            if a_dim == b_dim {
                out[i] = a_dim;
            } else if a_dim == 1 {
                out[i] = b_dim;
            } else if b_dim == 1 {
                out[i] = a_dim;
            } else {
                return Err(format!(
                    "shapes {:?} and {:?} are not broadcast-compatible",
                    a, b
                ));
            }
        }
        Ok(out)
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let _ = Self::broadcast_shapes(&self.shape(), &other.shape())
            .unwrap_or_else(|e| panic!("{}", e));
        // Fast path: CPU contiguous same-shape add, skip dispatch overhead
        if self.device() == Device::Cpu
            && other.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && other.inner.dtype == DType::F32
            && self.is_contiguous()
            && other.is_contiguous()
            && self.inner.sizes == other.inner.sizes
        {
            let numel = self.inner.numel() as usize;
            let mut output = Tensor::zeros(self.shape(), DType::F32, Device::Cpu);
            {
                let inner = Arc::make_mut(&mut output.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    unreachable!("add fast path only runs when both tensors are on CPU")
                };
                let out_data = Arc::make_mut(&mut cpu_storage.data);
                let a_ptr = self.data_ptr_f32();
                let b_ptr = other.data_ptr_f32();
                let out_ptr = out_data.as_mut_ptr() as *mut f32;

                // Single-threaded AVX2 SIMD - faster than rayon for memory-bound ops
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let av = _mm256_loadu_ps(a_ptr.add(i));
                                let bv = _mm256_loadu_ps(b_ptr.add(i));
                                _mm256_storeu_ps(out_ptr.add(i), _mm256_add_ps(av, bv));
                                i += 8;
                            }
                            for j in i..numel {
                                *out_ptr.add(j) = *a_ptr.add(j) + *b_ptr.add(j);
                            }
                        }
                    } else {
                        for i in 0..numel {
                            unsafe {
                                *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for i in 0..numel {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        }
                    }
                }
            }
            // Attach autograd if needed
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let edges = {
                    let mut edges = autograd::make_edge(self);
                    edges.extend(autograd::make_edge(other));
                    edges
                };
                let backward = autograd::AddBackward::new(vec![self.clone(), other.clone()], edges);
                let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                Arc::make_mut(&mut output.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(meta)));
            }
            return output;
        }

        // General path: dispatch
        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("add", dispatch_key, &[self, other]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::AddBackward::new(vec![self.clone(), other.clone()], edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    /// In-place addition for gradient accumulation
    /// This is used internally by the autograd engine to accumulate gradients
    pub fn add_(&mut self, other: &Tensor) -> &mut Self {
        // For GPU tensors or non-contiguous tensors, use dispatch-based addition
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).add(other);
            *self = result;
            return self;
        }

        // CPU path: direct memory manipulation (requires contiguous self)
        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;

        // If other is broadcast (e.g., expanded scalar), use general path
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).add(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        // Debug: print first few values
        // println!("add_ called: numel={}, self_ptr={:p}, other_ptr={:p}", numel, self_ptr, other_ptr);
        // println!("  self[0]={}, other[0]={}", unsafe { *self_ptr }, unsafe { *other_ptr });

        match dtype {
            DType::F32 => {
                // SIMD + parallel path for F32 gradient accumulation (hot path)
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        use rayon::prelude::*;
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        let self_usize = self_ptr as usize;
                        let other_usize = other_ptr as usize;
                        (0..num_chunks).into_par_iter().for_each(|chunk| {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);
                            let s_ptr = self_usize as *mut f32;
                            let o_ptr = other_usize as *const f32;

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(s_ptr.add(i));
                                            let ov = _mm256_loadu_ps(o_ptr.add(i));
                                            let r = _mm256_add_ps(sv, ov);
                                            _mm256_storeu_ps(s_ptr.add(i), r);
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *s_ptr.add(j) += *o_ptr.add(j);
                                        }
                                        return;
                                    }
                                }
                            }
                            // Scalar fallback for this chunk
                            for i in start..end {
                                unsafe {
                                    *s_ptr.add(i) += *o_ptr.add(i);
                                }
                            }
                        });
                        return self;
                    }
                }
                // Small tensor or non-parallel: SIMD inline or scalar
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                let r = _mm256_add_ps(sv, ov);
                                _mm256_storeu_ps(self_ptr.add(i), r);
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) += *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                // Scalar fallback
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                // For BF16, we need to upcast to F32 for computation
                // Use data_ptr_mut() to get correct byte offset for 2-byte types
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val + other_val);
                    }
                }
            }
            DType::F16 => {
                // For F16, we need to upcast to F32 for computation
                // Use data_ptr_mut() to get correct byte offset for 2-byte types
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val + other_val);
                    }
                }
            }
            _ => unimplemented!("add_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place multiplication
    pub fn mul_(&mut self, other: &Tensor) -> &mut Self {
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).mul(other);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).mul(other);
            *self = result;
            return self;
        }

        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        match dtype {
            DType::F32 => {
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        use rayon::prelude::*;
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        let self_usize = self_ptr as usize;
                        let other_usize = other_ptr as usize;
                        (0..num_chunks).into_par_iter().for_each(|chunk| {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);
                            let s_p = self_usize as *mut f32;
                            let o_p = other_usize as *const f32;
                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(s_p.add(i));
                                            let ov = _mm256_loadu_ps(o_p.add(i));
                                            _mm256_storeu_ps(s_p.add(i), _mm256_mul_ps(sv, ov));
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *s_p.add(j) *= *o_p.add(j);
                                        }
                                        return;
                                    }
                                }
                            }
                            for j in start..end {
                                unsafe {
                                    *s_p.add(j) *= *o_p.add(j);
                                }
                            }
                        });
                        return self;
                    }
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                _mm256_storeu_ps(self_ptr.add(i), _mm256_mul_ps(sv, ov));
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) *= *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                // For BF16, we need to upcast to F32 for computation
                // Use data_ptr_mut() to get correct byte offset for 2-byte types
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val * other_val);
                    }
                }
            }
            DType::F16 => {
                // For F16, we need to upcast to F32 for computation
                // Use data_ptr_mut() to get correct byte offset for 2-byte types
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val * other_val);
                    }
                }
            }
            _ => unimplemented!("mul_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place scalar multiplication: self *= scalar
    /// Avoids allocating a scalar tensor for optimizer hot paths.
    pub fn mul_scalar_(&mut self, scalar: f32) -> &mut Self {
        if self.inner.is_gpu() {
            let scalar_t = Tensor::from_scalar(scalar);
            let result = (self as &Tensor).mul(&scalar_t);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let self_ptr = self.data_ptr_f32_mut();
        for i in 0..numel {
            unsafe {
                *self_ptr.add(i) *= scalar;
            }
        }
        self
    }

    /// Non-in-place scalar multiplication without creating a scalar tensor.
    /// Avoids the 5-heap-alloc overhead of Tensor::from_scalar().
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.clone();
        result.mul_scalar_(scalar);
        result
    }

    /// In-place scalar addition: self += scalar
    pub fn add_scalar_(&mut self, scalar: f32) -> &mut Self {
        if self.inner.is_gpu() {
            let scalar_t = Tensor::from_scalar(scalar);
            let result = (self as &Tensor).add(&scalar_t);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let self_ptr = self.data_ptr_f32_mut();
        for i in 0..numel {
            unsafe {
                *self_ptr.add(i) += scalar;
            }
        }
        self
    }

    /// In-place subtraction: self -= other
    pub fn sub_(&mut self, other: &Tensor) -> &mut Self {
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).sub(other);
            *self = result;
            return self;
        }
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).sub(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && numel >= 8 {
                unsafe {
                    let mut i = 0;
                    while i + 8 <= numel {
                        let sv = _mm256_loadu_ps(self_ptr.add(i));
                        let ov = _mm256_loadu_ps(other_ptr.add(i));
                        _mm256_storeu_ps(self_ptr.add(i), _mm256_sub_ps(sv, ov));
                        i += 8;
                    }
                    for j in i..numel {
                        *self_ptr.add(j) -= *other_ptr.add(j);
                    }
                    return self;
                }
            }
        }
        for i in 0..numel {
            unsafe {
                *self_ptr.add(i) -= *other_ptr.add(i);
            }
        }
        self
    }

    /// In-place division: self /= other
    pub fn div_(&mut self, other: &Tensor) -> &mut Self {
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).div(other);
            *self = result;
            return self;
        }
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).div(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && numel >= 8 {
                unsafe {
                    let mut i = 0;
                    while i + 8 <= numel {
                        let sv = _mm256_loadu_ps(self_ptr.add(i));
                        let ov = _mm256_loadu_ps(other_ptr.add(i));
                        _mm256_storeu_ps(self_ptr.add(i), _mm256_div_ps(sv, ov));
                        i += 8;
                    }
                    for j in i..numel {
                        *self_ptr.add(j) /= *other_ptr.add(j);
                    }
                    return self;
                }
            }
        }
        for i in 0..numel {
            unsafe {
                *self_ptr.add(i) /= *other_ptr.add(i);
            }
        }
        self
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("sub", dispatch_key, &[self, other]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::SubBackward::new(vec![self.clone(), other.clone()], edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        // Fast path: CPU contiguous same-shape mul, skip dispatch overhead
        if self.device() == Device::Cpu
            && other.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && other.inner.dtype == DType::F32
            && self.is_contiguous()
            && other.is_contiguous()
            && self.inner.sizes == other.inner.sizes
        {
            let numel = self.inner.numel() as usize;
            let mut output = Tensor::zeros(self.shape(), DType::F32, Device::Cpu);
            {
                let inner = Arc::make_mut(&mut output.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    unreachable!("mul fast path only runs when both tensors are on CPU")
                };
                let out_data = Arc::make_mut(&mut cpu_storage.data);
                let a_ptr = self.data_ptr_f32();
                let b_ptr = other.data_ptr_f32();
                let out_ptr = out_data.as_mut_ptr() as *mut f32;

                // Single-threaded AVX2 SIMD - faster than rayon for memory-bound ops
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let av = _mm256_loadu_ps(a_ptr.add(i));
                                let bv = _mm256_loadu_ps(b_ptr.add(i));
                                _mm256_storeu_ps(out_ptr.add(i), _mm256_mul_ps(av, bv));
                                i += 8;
                            }
                            for j in i..numel {
                                *out_ptr.add(j) = *a_ptr.add(j) * *b_ptr.add(j);
                            }
                        }
                    } else {
                        for i in 0..numel {
                            unsafe {
                                *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for i in 0..numel {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        }
                    }
                }
            }
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let edges = {
                    let mut edges = autograd::make_edge(self);
                    edges.extend(autograd::make_edge(other));
                    edges
                };
                let backward = autograd::MulBackward::new(vec![self.clone(), other.clone()], edges);
                let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                Arc::make_mut(&mut output.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(meta)));
            }
            return output;
        }

        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("mul", dispatch_key, &[self, other]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::MulBackward::new(vec![self.clone(), other.clone()], edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("div", dispatch_key, &[self, other]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::DivBackward::new(vec![self.clone(), other.clone()], edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Fast path: 2D CPU matmul, skip dispatch overhead
        if self.inner.ndim() == 2
            && other.inner.ndim() == 2
            && self.device() == Device::Cpu
            && other.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && other.inner.dtype == DType::F32
            && self.is_contiguous()
            && other.is_contiguous()
        {
            return self.matmul_fast_2d(other);
        }

        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("matmul", dispatch_key, &[self, other]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::MatmulBackward::new(self.clone(), other.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    /// Fast 2D contiguous F32 matmul bypassing dispatch
    fn matmul_fast_2d(&self, other: &Tensor) -> Tensor {
        let a_shape = &self.inner.sizes;
        let b_shape = &other.inner.sizes;
        let m = a_shape[0] as usize;
        let k = a_shape[1] as usize;
        let n = b_shape[1] as usize;

        // Allocate output directly
        let mut output = Tensor::zeros(vec![m as i64, n as i64], DType::F32, Device::Cpu);
        {
            let output_inner = Arc::make_mut(&mut output.inner);
            let output_storage = Arc::make_mut(&mut output_inner.storage);
            let Storage::Cpu(cpu_storage) = output_storage else {
                unreachable!("matmul_fast_2d always allocates CPU output")
            };
            let out_data = Arc::make_mut(&mut cpu_storage.data);

            let a_ptr = self.data_ptr_f32();
            let b_ptr = other.data_ptr_f32();
            let out_ptr = out_data.as_mut_ptr() as *mut f32;

            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, m * n) };

            crate::kernels::blas::matmul_blas_into(a_slice, b_slice, out_slice, m, k, n);
        }

        // Attach autograd if needed
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = {
                let mut edges = autograd::make_edge(self);
                edges.extend(autograd::make_edge(other));
                edges
            };
            let backward = autograd::MatmulBackward::new(self.clone(), other.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }
        output
    }

    pub fn neg(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("neg", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::NegBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn relu(&self) -> Tensor {
        // Fast path: contiguous CPU F32, skip dispatch
        if self.device() == Device::Cpu && self.inner.dtype == DType::F32 && self.is_contiguous() {
            let numel = self.inner.numel() as usize;
            let mut output = Tensor::zeros(self.shape(), DType::F32, Device::Cpu);
            {
                let inner = Arc::make_mut(&mut output.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    panic!()
                };
                let out_data = Arc::make_mut(&mut cpu_storage.data);
                let a_ptr = self.data_ptr_f32();
                let out_ptr = out_data.as_mut_ptr() as *mut f32;
                for i in 0..numel {
                    unsafe {
                        *out_ptr.add(i) = (*a_ptr.add(i)).max(0.0);
                    }
                }
            }
            if autograd::is_grad_enabled() && self.requires_grad() {
                let edges = autograd::make_edge(self);
                let backward = autograd::ReluBackward::new(self.clone(), edges);
                let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                Arc::make_mut(&mut output.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(meta)));
            }
            return output;
        }

        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("relu", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ReluBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn exp(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("exp", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ExpBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn ln(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("log", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::LogBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("sigmoid", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SigmoidBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn tanh(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("tanh", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::TanhBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn silu(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("silu", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SiLUBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn gelu(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("gelu", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::GeluBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let slope_tensor = Tensor::from_scalar(negative_slope);
        let result = dispatch("leaky_relu", dispatch_key, &[self, &slope_tensor]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::LeakyReLUBackward::new(self.clone(), negative_slope, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn softplus(&self, beta: f32, threshold: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let beta_t = Tensor::from_scalar(beta);
        let threshold_t = Tensor::from_scalar(threshold);
        let result = dispatch("softplus", dispatch_key, &[self, &beta_t, &threshold_t]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SoftplusBackward::new(self.clone(), beta, threshold, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn hardswish(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("hardswish", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::HardswishBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn elu(&self, alpha: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let alpha_tensor = Tensor::from_scalar(alpha);
        let result = dispatch("elu", dispatch_key, &[self, &alpha_tensor]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ELUBackward::new(self.clone(), alpha, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn softmax(&self, dim: i32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("softmax", dispatch_key, &[self, &dim_scalar(dim)]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SoftmaxBackward::new(output.clone(), dim as usize, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("sqrt", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SqrtBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn fused_linear_gelu(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let device = match (self.device(), weight.device()) {
            (Device::Wgpu(id), _) => Device::Wgpu(id),
            (_, Device::Wgpu(id)) => Device::Wgpu(id),
            _ => {
                if let Some(b) = bias {
                    b.device()
                } else {
                    Device::Cpu
                }
            }
        };
        let dispatch_key = device_to_dispatch_key(device);
        let args: Vec<&Tensor> = match bias {
            Some(b) => vec![self, weight, b],
            None => vec![self, weight],
        };
        let result = dispatch("fused_linear_gelu", dispatch_key, &args);
        result[0].clone()
    }

    pub fn clamp(&self, min_val: f32, max_val: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "clamp",
            dispatch_key,
            &[
                self,
                &Tensor::from_scalar(min_val),
                &Tensor::from_scalar(max_val),
            ],
        );
        result[0].clone()
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("pow", dispatch_key, &[self, &Tensor::from_scalar(exponent)]);
        result[0].clone()
    }

    pub fn abs(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("abs", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::AbsBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn sum(&self, dim: i32, keepdim: bool) -> Tensor {
        // Fast path: contiguous CPU F32, last-dim sum, no keepdim, 2D tensors only
        if self.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && self.is_contiguous()
            && !keepdim
            && self.inner.ndim() == 2
        {
            let ndim = self.inner.ndim() as i32;
            let dim_normalized = if dim < 0 { ndim + dim } else { dim } as usize;
            if dim_normalized == ndim as usize - 1 {
                let shape = self.shape();
                let dim_size = shape[dim_normalized] as usize;
                let num_rows = shape[0] as usize;
                let output = crate::kernels::cpu::sum_last_dim_contiguous(self, dim_size, num_rows);
                if autograd::is_grad_enabled() && self.requires_grad() {
                    let mut output = output;
                    let edges = autograd::make_edge(self);
                    let backward =
                        autograd::SumBackward::new(self.clone(), dim_normalized, keepdim, edges);
                    let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                    meta.grad_fn = Some(std::sync::Arc::new(backward));
                    Arc::make_mut(&mut output.inner).autograd_meta =
                        Some(Arc::new(std::sync::Mutex::new(meta)));
                    return output;
                }
                return output;
            }
        }

        // Fallback: dispatch
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "sum",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        );
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::SumBackward::new(self.clone(), dim as usize, keepdim, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn max(&self, dim: i32, keepdim: bool) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "max",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        );
        result[0].clone()
    }

    pub fn mean(&self, dim: i32, keepdim: bool) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "mean",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        );
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let dim_size = self.shape()[dim as usize];
            let edges = autograd::make_edge(self);
            let backward =
                autograd::MeanBackward::new(self.clone(), dim as usize, keepdim, dim_size, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    pub fn flip(&self, dim: i32) -> Tensor {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if dim >= self.ndim() {
            panic!("flip: dimension out of range");
        }
        let size = self.inner.sizes[dim];
        let mut result_data = vec![0.0f32; self.numel() as usize];
        let src = self.to_cpu();
        let src_data = src.as_f32_slice();
        let ndim = self.ndim();
        let strides = &self.inner.strides;
        let sizes = &self.inner.sizes;
        let mut indices = vec![0i64; ndim];
        for out_idx in 0..self.numel() as usize {
            let mut lin = 0i64;
            for d in 0..ndim {
                let idx = if d == dim {
                    size - 1 - indices[d]
                } else {
                    indices[d]
                };
                lin += idx * strides[d];
            }
            result_data[out_idx] = src_data[lin as usize];
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < sizes[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Tensor::from_vec(result_data, self.inner.sizes.clone().into_vec())
    }

    pub fn maximum(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("maximum", dispatch_key, &[self, other]);
        result[0].clone()
    }

    pub fn log_softmax(&self, dim: i32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "log_softmax",
            dispatch_key,
            &[self, &Tensor::from_scalar(dim as f32)],
        );
        result[0].clone()
    }

    pub fn as_i64_slice(&self) -> Vec<i64> {
        let src = self.to_cpu();
        let data = src.as_f32_slice();
        data.iter().map(|&v| v as i64).collect()
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        self.add(other)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn add(self, other: Tensor) -> Tensor {
        (&self).add(&other)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        self.sub(other)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn sub(self, other: Tensor) -> Tensor {
        (&self).sub(&other)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul(other)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn mul(self, other: Tensor) -> Tensor {
        (&self).mul(&other)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        self.div(other)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn div(self, other: Tensor) -> Tensor {
        (&self).div(&other)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow, unconditional_recursion)]
    fn neg(self) -> Tensor {
        (&self).neg()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.neg()
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

impl Tensor {
    pub fn gt_scalar(&self, threshold: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "gt_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(threshold)],
        );
        result[0].clone()
    }

    pub fn sign(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("sign", dispatch_key, &[self]);
        result[0].clone()
    }

    pub fn lt_scalar(&self, threshold: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "lt_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(threshold)],
        );
        result[0].clone()
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "add_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(scalar)],
        );
        result[0].clone()
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "div_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(scalar)],
        );
        result[0].clone()
    }

    pub fn logical_not(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("logical_not", dispatch_key, &[self]);
        result[0].clone()
    }

    pub fn cat(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("cat: need at least one tensor");
        }
        let ndim = tensors[0].ndim();
        let dim = if dim < 0 { ndim as i32 + dim } else { dim } as usize;
        if dim >= ndim {
            panic!("cat: dimension out of range");
        }
        let mut output_shape: SmallVec<[i64; 8]> = tensors[0].inner.sizes.clone();
        let mut total_dim_size: i64 = 0;
        for t in tensors {
            if t.ndim() != ndim {
                panic!("cat: tensors must have same number of dimensions");
            }
            total_dim_size += t.inner.sizes[dim];
            for d in 0..ndim {
                if d != dim && output_shape[d] != t.inner.sizes[d] {
                    panic!("cat: tensor sizes mismatch at dimension {}", d);
                }
            }
        }
        output_shape[dim] = total_dim_size;
        let numel: i64 = output_shape.iter().product();
        let mut output_data = vec![0.0f32; numel as usize];
        let mut out_strides: SmallVec<[i64; 8]> = SmallVec::new();
        let mut s = 1i64;
        for d in (0..ndim).rev() {
            out_strides.push(s);
            s *= output_shape[d];
        }
        out_strides.reverse();
        let mut out_offset = 0i64;
        for t in tensors {
            let t_data = t.as_f32_slice();
            let t_strides = &t.inner.strides;
            let t_sizes = &t.inner.sizes;
            let mut indices = vec![0i64; ndim];
            let dim_size = t.inner.sizes[dim];
            for _ in 0..t.numel() as usize {
                let mut t_lin = t.inner.storage_offset;
                let mut o_lin = out_offset;
                for d in 0..ndim {
                    t_lin += indices[d] * t_strides[d];
                    o_lin += indices[d] * out_strides[d];
                }
                output_data[o_lin as usize] = t_data[t_lin as usize];
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < t_sizes[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            out_offset += dim_size * out_strides[dim];
        }
        Tensor::from_vec(output_data, output_shape.into_vec())
    }

    pub fn repeat(&self, repeats: &[i64]) -> Tensor {
        if repeats.len() < self.ndim() {
            panic!("repeat: number of repeats must be >= number of dimensions");
        }
        let mut new_shape: SmallVec<[i64; 8]> = SmallVec::new();
        let mut total: i64 = 1;
        for (i, &r) in repeats.iter().enumerate() {
            let orig = if i < repeats.len() - self.ndim() {
                1
            } else {
                self.inner.sizes[i - (repeats.len() - self.ndim())]
            };
            new_shape.push(orig * r);
            total *= orig * r;
        }
        let mut output_data = vec![0.0f32; total as usize];
        let src = self.to_cpu();
        let src_data = src.as_f32_slice();
        let src_strides = &self.inner.strides;
        let src_sizes = &self.inner.sizes;
        let ndim = self.ndim();
        let offset = repeats.len() - ndim;
        let mut src_indices = vec![0i64; ndim];
        for out_idx in 0..total as usize {
            let mut rep_indices: Vec<i64> = Vec::with_capacity(repeats.len());
            let mut remaining = out_idx as i64;
            for d in (0..repeats.len()).rev() {
                let dim_size = new_shape[d];
                rep_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            rep_indices.reverse();
            for d in 0..ndim {
                src_indices[d] = rep_indices[d + offset] % src_sizes[d];
            }
            let mut src_lin = self.inner.storage_offset;
            for d in 0..ndim {
                src_lin += src_indices[d] * src_strides[d];
            }
            output_data[out_idx] = src_data[src_lin as usize];
        }
        Tensor::from_vec(output_data, new_shape.into_vec())
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_data = self.as_f32_slice();
        let cond_data = condition.as_f32_slice();
        let other_data = other.as_f32_slice();
        let mut output_data = vec![0.0f32; numel];
        for i in 0..numel {
            output_data[i] = if cond_data[i] != 0.0 {
                self_data[i]
            } else {
                other_data[i]
            };
        }
        Tensor::from_vec(output_data, self.shape())
    }
}

pub fn einsum(equation: &str, tensors: &[Tensor]) -> Tensor {
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

    // Find contracted dims (appear in both inputs but not in output)
    let mut contracted: std::collections::HashSet<char> = std::collections::HashSet::new();
    for &c in &a_chars {
        if b_chars.contains(&c) && !out_chars.contains(&c) {
            contracted.insert(c);
        }
    }

    // Build permutation for a: move contracted dims to the end
    let mut a_perm: Vec<i64> = Vec::new();
    for &c in &a_chars {
        if !contracted.contains(&c) {
            a_perm.push(a_chars.iter().position(|&x| x == c).unwrap() as i64);
        }
    }
    for &c in &a_chars {
        if contracted.contains(&c) {
            a_perm.push(a_chars.iter().position(|&x| x == c).unwrap() as i64);
        }
    }
    let a_permuted = if a_perm.iter().enumerate().all(|(i, &p)| p as usize == i) {
        a.clone()
    } else {
        a.permute(a_perm)
    };

    // Build permutation for b: move contracted dims to the front
    let mut b_perm: Vec<i64> = Vec::new();
    for &c in &b_chars {
        if contracted.contains(&c) {
            b_perm.push(b_chars.iter().position(|&x| x == c).unwrap() as i64);
        }
    }
    for &c in &b_chars {
        if !contracted.contains(&c) {
            b_perm.push(b_chars.iter().position(|&x| x == c).unwrap() as i64);
        }
    }
    let b_permuted = if b_perm.iter().enumerate().all(|(i, &p)| p as usize == i) {
        b.clone()
    } else {
        b.permute(b_perm)
    };

    // Reshape to 2D for matmul
    let a_batch: i64 = a_permuted.shape()[..a_permuted.ndim() - contracted.len()]
        .iter()
        .product();
    let a_contract: i64 = a_permuted.shape()[a_permuted.ndim() - contracted.len()..]
        .iter()
        .product();
    let b_contract: i64 = b_permuted.shape()[..contracted.len()].iter().product();
    let b_rest: i64 = b_permuted.shape()[contracted.len()..].iter().product();

    if a_contract != b_contract {
        panic!(
            "einsum: contracted dimensions don't match: {} vs {}",
            a_contract, b_contract
        );
    }

    let a_2d = a_permuted.reshape(vec![a_batch, a_contract]);
    let b_2d = b_permuted.reshape(vec![b_contract, b_rest]);
    let result_2d = a_2d.matmul(&b_2d);

    // Reshape to output shape
    let mut out_shape: Vec<i64> = Vec::new();
    for &c in &out_chars {
        if let Some(pos) = a_chars.iter().position(|&x| x == c) {
            out_shape.push(a.inner.sizes[pos]);
        } else if let Some(pos) = b_chars.iter().position(|&x| x == c) {
            out_shape.push(b.inner.sizes[pos]);
        }
    }
    if out_shape.is_empty() {
        result_2d.reshape(vec![])
    } else {
        result_2d.reshape(out_shape)
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
        assert!((result.item() - 32.0).abs() < 1e-5);
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
        assert!((data[0] - 0.693147).abs() < 1e-3);
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
        let y = x.flip(0);
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

pub fn clip_grad_norm_(tensors: &[Tensor], max_norm: f32, norm_type: f32) -> f32 {
    let mut total_norm = 0.0f32;
    for t in tensors {
        if let Some(g) = t.grad() {
            let g_data = g.as_f32_slice();
            let param_norm = if norm_type == 2.0 {
                g_data.iter().map(|x| x * x).sum::<f32>().sqrt()
            } else {
                g_data
                    .iter()
                    .map(|x| x.abs().powf(norm_type))
                    .sum::<f32>()
                    .powf(1.0 / norm_type)
            };
            total_norm += param_norm * param_norm;
        }
    }
    total_norm = total_norm.sqrt();
    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef < 1.0 {
        for t in tensors {
            if let Some(g) = t.grad() {
                let g_data = g.as_f32_slice();
                let clipped: Vec<f32> = g_data.iter().map(|x| x * clip_coef).collect();
                let clipped_tensor = Tensor::from_vec(clipped, g.shape());
                t.set_grad(Some(clipped_tensor));
            }
        }
    }
    total_norm
}

pub fn clip_grad_value_(tensors: &[Tensor], clip_value: f32) {
    for t in tensors {
        if let Some(g) = t.grad() {
            let g_data = g.as_f32_slice();
            let clipped: Vec<f32> = g_data
                .iter()
                .map(|x| x.max(-clip_value).min(clip_value))
                .collect();
            let clipped_tensor = Tensor::from_vec(clipped, g.shape());
            t.set_grad(Some(clipped_tensor));
        }
    }
}
