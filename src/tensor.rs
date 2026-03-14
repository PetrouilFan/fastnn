use crate::autograd::{self, AutogradMeta};
use crate::dispatcher::{device_to_dispatch_key, dispatch};
use crate::storage::{CpuStorage, DType, Device, GpuStorage, Storage};
use parking_lot::RwLock;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct TensorImpl {
    pub storage: Arc<Storage>,
    pub sizes: SmallVec<[i64; 8]>,
    pub strides: SmallVec<[i64; 8]>,
    pub storage_offset: i64,
    pub dtype: DType,
    pub device: Device,
    pub version_counter: Arc<AtomicU64>,
    pub autograd_meta: Option<AutogradMeta>,
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
        self.view(self.sizes.clone()).into()
    }

    pub fn view(&self, sizes: SmallVec<[i64; 8]>) -> TensorImpl {
        let numel: i64 = sizes.iter().product();
        if numel != self.numel() {
            panic!(
                "size mismatch: view size {} != numel {}",
                numel,
                self.numel()
            );
        }

        let mut new = TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes: sizes.clone(),
            strides: compute_strides(&sizes),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: None,
        };

        if autograd::is_grad_enabled() {
            if let Some(meta) = &self.autograd_meta {
                if meta.requires_grad {
                    new.set_requires_grad(true);
                }
            }
        }

        new
    }

    pub fn reshape(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        let numel: i64 = sizes.iter().product();
        if numel != self.numel() {
            panic!(
                "size mismatch: reshape size {} != numel {}",
                numel,
                self.numel()
            );
        }

        let mut new_sizes = sizes.clone();
        let mut product: i64 = 1;
        let mut minus_one_idx = None;

        for (i, s) in new_sizes.iter().enumerate() {
            if *s == -1 {
                if minus_one_idx.is_some() {
                    panic!("can only specify one unknown dimension");
                }
                minus_one_idx = Some(i);
            } else {
                product *= s;
            }
        }

        if let Some(idx) = minus_one_idx {
            let known: i64 = self.numel() / product;
            new_sizes[idx] = known;
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
            autograd_meta: None,
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
            autograd_meta: None,
        }
        .into()
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let ndim = self.ndim();
        let dim = if dim > ndim { ndim } else { dim };

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();
        sizes.insert(dim, 1);
        strides.insert(dim, if dim == ndim { self.numel() } else { 0 });

        TensorImpl {
            storage: Arc::clone(&self.storage),
            sizes,
            strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: None,
        }
        .into()
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
                    autograd_meta: None,
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
                    autograd_meta: None,
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
            sizes: new_sizes,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device,
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: None,
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
        let end = (end.min(size) - 1) as usize;

        if start > end {
            panic!("slice: invalid range");
        }

        let mut sizes = self.sizes.clone();
        let numel = ((end - start) / step as usize) + 1;
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
            .map(|m| m.requires_grad)
            .unwrap_or(false)
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.autograd_meta.is_none() {
            self.autograd_meta = Some(AutogradMeta::new(requires_grad));
        } else if let Some(meta) = &mut self.autograd_meta {
            meta.requires_grad = requires_grad;
        }
    }

    pub fn requires_grad_(&self, requires_grad: bool) -> Tensor {
        let self_ptr = self as *const TensorImpl as *mut TensorImpl;
        unsafe {
            (*self_ptr).set_requires_grad(requires_grad);
        }
        Tensor::new(self.clone())
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.autograd_meta.as_ref()?.grad.clone()
    }

    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        if let Some(meta) = &mut self.autograd_meta {
            meta.grad = grad;
        }
    }

    pub fn set_grad_for_tensor(tensor: &Tensor, grad: Option<Tensor>) {
        let ptr = Arc::as_ptr(&tensor.inner) as *mut TensorImpl;
        unsafe {
            if let Some(meta) = (*ptr).autograd_meta.as_mut() {
                meta.grad = grad;
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.autograd_meta
            .as_ref()
            .map(|m| m.is_leaf)
            .unwrap_or(true)
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn autograd::Node>> {
        self.autograd_meta.as_ref()?.grad_fn.clone()
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
        self.version_counter.fetch_add(1, Ordering::SeqCst);
    }

    pub fn version(&self) -> u64 {
        self.version_counter.load(Ordering::SeqCst)
    }

    pub fn data_ptr(&self) -> *const u8 {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => cpu.data.as_ptr(),
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn data_ptr_f32(&self) -> *const f32 {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                let ptr = cpu.data.as_ptr() as *const f32;
                unsafe { ptr.add(self.storage_offset as usize) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn data_ptr_f32_mut(&self) -> *mut f32 {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => {
                let ptr = cpu.data.as_ptr() as *mut f32;
                unsafe { ptr.add(self.storage_offset as usize) }
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => unsafe {
                let ptr = cpu.data.as_ptr() as *const f32;
                let numel = self.numel() as usize;
                std::slice::from_raw_parts(ptr, numel)
            },
            Storage::Wgpu(_) => {
                panic!("Cannot slice GPU storage. Use .to_cpu() first.");
            }
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        match &self.storage.as_ref() {
            Storage::Cpu(cpu) => unsafe {
                let ptr = cpu.data.as_ptr() as *mut f32;
                let numel = self.numel() as usize;
                std::slice::from_raw_parts_mut(ptr, numel)
            },
            Storage::Wgpu(_) => {
                panic!("Cannot slice GPU storage. Use .to_cpu() first.");
            }
        }
    }

    /// Check if storage is on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.storage.as_ref(), Storage::Wgpu(_))
    }

    /// Check if storage is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self.storage.as_ref(), Storage::Cpu(_))
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
        let data = cpu_storage.data.as_mut_ptr() as *mut f32;
        unsafe {
            *data = value;
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
        // Create CPU storage first (GPU buffers are created on-demand)
        let storage = Arc::new(Storage::new_cpu(dtype, nbytes));
        // Use new_with_device for GPU tensors to track the target device
        match device {
            Device::Cpu => Tensor::new(TensorImpl::new(storage, sizes, dtype)),
            Device::Wgpu(_) => {
                Tensor::new(TensorImpl::new_with_device(storage, sizes, device, dtype))
            }
        }
    }

    pub fn empty(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let numel: i64 = sizes.iter().product();
        let nbytes = (numel * dtype.size() as i64) as usize;
        // Create uninitialized storage
        let mut storage = Storage::new_cpu(dtype, nbytes);
        let storage = Arc::new(storage);
        // Use new_with_device for GPU tensors to track the target device
        match device {
            Device::Cpu => Tensor::new(TensorImpl::new(storage, sizes, dtype)),
            Device::Wgpu(_) => {
                Tensor::new(TensorImpl::new_with_device(storage, sizes, device, dtype))
            }
        }
    }

    pub fn ones(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let mut t = Self::zeros(shape, dtype, device);
        let numel = t.inner.numel() as usize;
        let inner = Arc::make_mut(&mut t.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        let Storage::Cpu(cpu_storage) = storage else {
            panic!("Expected CPU storage for ones()");
        };
        let ptr = cpu_storage.data.as_mut_ptr();
        match dtype {
            DType::F32 => {
                let f32_ptr = ptr as *mut f32;
                for i in 0..numel {
                    unsafe {
                        *f32_ptr.add(i) = 1.0;
                    }
                }
            }
            DType::F64 => {
                let f64_ptr = ptr as *mut f64;
                for i in 0..numel {
                    unsafe {
                        *f64_ptr.add(i) = 1.0;
                    }
                }
            }
            DType::I32 => {
                let i32_ptr = ptr as *mut i32;
                for i in 0..numel {
                    unsafe {
                        *i32_ptr.add(i) = 1;
                    }
                }
            }
            _ => {}
        }
        t
    }

    pub fn full(shape: Vec<i64>, value: f32, dtype: DType, device: Device) -> Self {
        let mut t = Self::zeros(shape, dtype, device);
        let numel = t.inner.numel() as usize;
        let inner = Arc::make_mut(&mut t.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        let Storage::Cpu(cpu_storage) = storage else {
            panic!("Expected CPU storage for full()");
        };
        let ptr = cpu_storage.data.as_mut_ptr();

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
            _ => {}
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
        self.inner.view(sizes).into()
    }

    pub fn reshape(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        self.inner.reshape(sizes)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        self.inner.transpose(dim0, dim1)
    }

    pub fn permute(&self, dims: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = dims.into();
        self.inner.permute(sizes)
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        self.inner.squeeze(dim)
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        self.inner.unsqueeze(dim)
    }

    pub fn expand(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        self.inner.expand(sizes)
    }

    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        self.inner.slice(dim, start, end, step)
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
        self.inner.requires_grad_(requires_grad)
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad()
    }

    pub fn set_grad(&self, _grad: Option<Tensor>) {
        // Need mutable access - clone and modify
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
            Storage::Cpu(cpu) => cpu.data.as_ptr(),
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
            _ => panic!("Unsupported dtype for item()"),
        }
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        match &self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                match self.inner.dtype {
                    DType::F32 => {
                        // Use stride-aware indexing
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ptr() as *const f32;

                        // Create an iterator over all elements using strides
                        let mut indices = vec![0i64; self.inner.ndim()];

                        for _ in 0..self.inner.numel() {
                            // Calculate linear index from multi-dimensional indices using strides
                            let mut linear_idx = self.inner.storage_offset;
                            for (i, &stride) in self.inner.strides.iter().enumerate() {
                                linear_idx += indices[i] * stride;
                            }

                            unsafe {
                                result.push(*data.add(linear_idx as usize));
                            }

                            // Increment indices
                            for dim in (0..self.inner.ndim()).rev() {
                                indices[dim] += 1;
                                if indices[dim] < self.inner.sizes[dim] {
                                    break;
                                }
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::F64 => {
                        // Similar for F64
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ptr() as *const f64;

                        let mut indices = vec![0i64; self.inner.ndim()];

                        for _ in 0..self.inner.numel() {
                            let mut linear_idx = self.inner.storage_offset;
                            for (i, &stride) in self.inner.strides.iter().enumerate() {
                                linear_idx += indices[i] * stride;
                            }

                            unsafe {
                                result.push(*data.add(linear_idx as usize) as f32);
                            }

                            for dim in (0..self.inner.ndim()).rev() {
                                indices[dim] += 1;
                                if indices[dim] < self.inner.sizes[dim] {
                                    break;
                                }
                                indices[dim] = 0;
                            }
                        }
                        result
                    }
                    DType::I32 => {
                        // Similar for I32
                        let mut result = Vec::with_capacity(self.inner.numel() as usize);
                        let data = cpu.data.as_ptr() as *const i32;

                        let mut indices = vec![0i64; self.inner.ndim()];

                        for _ in 0..self.inner.numel() {
                            let mut linear_idx = self.inner.storage_offset;
                            for (i, &stride) in self.inner.strides.iter().enumerate() {
                                linear_idx += indices[i] * stride;
                            }

                            unsafe {
                                result.push(*data.add(linear_idx as usize) as f32);
                            }

                            for dim in (0..self.inner.ndim()).rev() {
                                indices[dim] += 1;
                                if indices[dim] < self.inner.sizes[dim] {
                                    break;
                                }
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

    pub fn data_ptr(&self) -> *const u8 {
        self.inner.data_ptr()
    }

    pub fn data_ptr_f32(&self) -> *const f32 {
        self.inner.data_ptr_f32()
    }

    pub fn data_ptr_f32_mut(&mut self) -> *mut f32 {
        Arc::make_mut(&mut self.inner).data_ptr_f32_mut()
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        self.inner.as_f32_slice()
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.inner).as_f32_slice_mut()
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
            Storage::Cpu(_) => self.clone(),
            Storage::Wgpu(gpu) => {
                // Read GPU buffer to CPU
                use crate::kernels::gpu::get_context;
                let ctx = get_context(gpu.device_id);
                let data = ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let storage = Arc::new(Storage::Cpu(CpuStorage {
                    data: bytemuck::cast_slice(&data).to_vec(),
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
                    autograd_meta: None,
                })
            }
        }
    }

    /// Move tensor to GPU memory
    pub fn to_gpu(&self, device_id: usize) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => self.clone(),
            _ => {
                // Get CPU data
                let cpu_data = self.as_f32_slice().to_vec();
                let shape = self.shape();
                let dtype = self.inner.dtype;

                // Create GPU tensor
                use crate::kernels::gpu::get_context;
                let ctx = get_context(device_id);
                let buffer = ctx.create_gpu_buffer_from_data(&cpu_data, "to_gpu");

                let storage = Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: cpu_data.len() * 4,
                    device_id,
                    staging: RwLock::new(None),
                }));

                let sizes: SmallVec<[i64; 8]> = shape.iter().copied().collect();
                Tensor::new(TensorImpl {
                    storage,
                    sizes,
                    strides: self.inner.strides.clone(),
                    storage_offset: self.inner.storage_offset,
                    dtype,
                    device: Device::Wgpu(device_id),
                    version_counter: Arc::new(AtomicU64::new(0)),
                    autograd_meta: None,
                })
            }
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    /// In-place addition for gradient accumulation
    /// This is used internally by the autograd engine to accumulate gradients
    pub fn add_(&mut self, other: &Tensor) -> &mut Self {
        let self_inner = Arc::make_mut(&mut self.inner);
        let dtype = self_inner.dtype;
        let numel = self_inner.numel() as usize;
        let self_storage = Arc::make_mut(&mut self_inner.storage);

        match (self_storage, other.inner.storage.as_ref()) {
            (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => match dtype {
                DType::F32 => {
                    let self_ptr = cpu_self.data.as_mut_ptr() as *mut f32;
                    let other_ptr = cpu_other.data.as_ptr() as *const f32;
                    for i in 0..numel {
                        unsafe {
                            *self_ptr.add(i) += *other_ptr.add(i);
                        }
                    }
                }
                DType::F64 => {
                    let self_ptr = cpu_self.data.as_mut_ptr() as *mut f64;
                    let other_ptr = cpu_other.data.as_ptr() as *const f64;
                    for i in 0..numel {
                        unsafe {
                            *self_ptr.add(i) += *other_ptr.add(i);
                        }
                    }
                }
                DType::I32 => {
                    let self_ptr = cpu_self.data.as_mut_ptr() as *mut i32;
                    let other_ptr = cpu_other.data.as_ptr() as *const i32;
                    for i in 0..numel {
                        unsafe {
                            *self_ptr.add(i) += *other_ptr.add(i);
                        }
                    }
                }
                _ => unimplemented!("add_ for dtype {:?}", dtype),
            },
            _ => unimplemented!("add_ for non-CPU storage"),
        }
        self
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    pub fn relu(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("relu", dispatch_key, &[self]);
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = autograd::ReluBackward::new(self.clone(), edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
    }

    pub fn sum(&self, dim: i32, keepdim: bool) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "sum",
            dispatch_key,
            &[
                self,
                &Tensor::from_scalar(dim as f32),
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
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
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
                &Tensor::from_scalar(dim as f32),
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
                &Tensor::from_scalar(dim as f32),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        );
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let numel: i64 = self.shape().iter().product();
            let edges = autograd::make_edge(self);
            let backward =
                autograd::MeanBackward::new(self.clone(), dim as usize, keepdim, numel, edges);
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta = Some(meta);
            output
        } else {
            output
        }
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
}
