use crate::autograd::{self, AutogradMeta};
use crate::storage::{DType, Device, Storage};
use crate::storage_pool::get_storage_pool;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use self::factories::simd_copy_f32;
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
use self::factories::memcpy_f32;
use self::shape::compute_strides;

mod device;
mod factories;
mod indexing;
mod ops;
mod reductions;
mod shape;

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

    /// Create a view sharing storage, version_counter, and autograd_meta from self.
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
            version_counter: Arc::clone(&self.version_counter),
            autograd_meta: self.autograd_meta.clone(),
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
        }
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

    pub(crate) fn attach_grad_fn(mut output: Tensor, backward: Arc<dyn autograd::Node + 'static>) -> Tensor {
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(backward);
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(std::sync::Mutex::new(meta)));
        output
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
                    return result;
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

    /// Get a direct byte slice view of the tensor data for efficient serialization.
    /// Only works for contiguous CPU tensors.
    /// Returns None for GPU tensors or non-contiguous tensors.
    pub fn as_byte_slice(&self) -> Option<&[u8]> {
        match &self.inner.storage.as_ref() {
            Storage::Cpu(cpu) => {
                if !self.is_contiguous() {
                    return None;
                }
                let data = cpu.data.as_ref();
                let numel = self.inner.numel() as usize;
                let elem_size = self.inner.dtype.size();
                let byte_len = numel * elem_size;
                let start = self.inner.storage_offset as usize * elem_size;
                // Ensure we don't read past the storage
                if start + byte_len > data.len() {
                    return None;
                }
                Some(&data[start..][..byte_len])
            }
            Storage::Wgpu(_) => None,
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
