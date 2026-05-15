// Tensor factory/constructor methods

use crate::backend::wgpu::context::get_wgpu_context;
use crate::storage::{DType, Device, GpuStorage, Storage};
use crate::storage_pool::get_storage_pool;
use parking_lot::RwLock;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::shape::compute_strides;
use super::{Tensor, TensorImpl};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

/// SIMD-accelerated copy for f32 slices (AVX2)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) fn simd_copy_f32(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: `src` and `dst` are valid pointers to `len` elements, and `i` is
        // within bounds (i + 8 <= len). The SIMD load/store intrinsics require valid,
        // aligned-or-guaranteed-ok pointers (loadu/storeu tolerate misalignment).
        unsafe {
            let v = _mm256_loadu_ps(src.add(i));
            _mm256_storeu_ps(dst.add(i), v);
        }
        i += 8;
    }
    while i < len {
        // SAFETY: `src` and `dst` are valid pointers to `len` elements, and `i` is
        // within bounds (i < len).
        unsafe {
            *dst.add(i) = *src.add(i);
        }
        i += 1;
    }
}

/// Fallback memcpy for f32
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(super) fn memcpy_f32(src: *const f32, dst: *mut f32, len: usize) {
    // SAFETY: The caller guarantees `src` and `dst` are valid pointers to `len`
    // elements of `f32` and that the two memory regions do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

/// Cached scalar tensors for common dimension values (0-7)
/// Avoids heap allocation on every softmax/sum/mean/max call
#[allow(dead_code)]
pub(super) fn dim_scalar(dim: i32) -> Tensor {
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

impl Tensor {
    pub fn from_scalar(value: f32) -> Self {
        let mut storage = get_storage_pool().acquire_uninit(4, Device::Cpu);
        let storage_mut = Arc::make_mut(&mut storage);
        let Storage::Cpu(cpu_storage) = storage_mut else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu_storage.data);
        let ptr = data.as_mut_ptr() as *mut f32;
        // SAFETY: The pointer `ptr` is derived from a uniquely owned `Vec<u8>` via
        // `Arc::make_mut`, is properly aligned for `f32`, and points to a buffer of
        // at least 4 bytes (allocated by `Storage::new_cpu(DType::F32, 4)`).
        unsafe {
            *ptr = value;
        }
        let sizes: SmallVec<[i64; 8]> = smallvec![];
        Tensor::new(TensorImpl::new(storage, sizes, DType::F32))
    }

    pub fn from_vec(values: Vec<f32>, shape: Vec<i64>) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let nbytes = values.len() * 4;
        let mut storage = get_storage_pool().acquire_uninit(nbytes, Device::Cpu);
        let Storage::Cpu(cpu) = Arc::make_mut(&mut storage) else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu.data);
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                data.as_mut_ptr() as *mut f32,
                values.len(),
            );
        }
        Tensor::new(TensorImpl::new(storage, sizes, DType::F32))
    }

    pub fn from_vec_with_device(values: Vec<f32>, shape: Vec<i64>, _device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let storage = Arc::new(Storage::from_vec(values, DType::F32, Device::Cpu));
        Tensor::new(TensorImpl::new(storage, sizes, DType::F32))
    }

    pub fn zeros(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let numel: i64 = sizes.iter().product();
        let nbytes = (numel * dtype.size() as i64) as usize;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire_zeroed(nbytes, device),
            Device::Wgpu(device_id) => {
                let ctx = get_wgpu_context(device_id);
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
            requires_grad: false,
        })
    }

    pub fn empty(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let numel: i64 = sizes.iter().product();
        let nbytes = (numel * dtype.size() as i64) as usize;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire_uninit(nbytes, device),
            Device::Wgpu(device_id) => {
                let ctx = get_wgpu_context(device_id);
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
                let sizes: SmallVec<[i64; 8]> = shape.into();
                let numel: i64 = sizes.iter().product();
                let nbytes = (numel * dtype.size() as i64) as usize;
                let numel = numel as usize;

                let mut storage = get_storage_pool().acquire_uninit(nbytes, device);
                let inner = Arc::make_mut(&mut storage);
                let Storage::Cpu(cpu_storage) = inner else {
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
                    DType::U4 | DType::U8 => {
                        panic!("ones(): packed U4/U8 tensors are not supported. Use the IR quantization pass to create packed weights.")
                    }
                }
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
            requires_grad: false,
                })
            }
            Device::Wgpu(device_id) => {
                let cpu_ones = Self::ones(shape.clone(), dtype, Device::Cpu);
                cpu_ones.to_gpu(device_id)
            }
        }
    }

    pub fn full(shape: Vec<i64>, value: f32, dtype: DType, device: Device) -> Self {
        if value == 0.0 {
            return Self::zeros(shape, dtype, device);
        }

        match device {
            Device::Cpu => {
                let sizes: SmallVec<[i64; 8]> = shape.into();
                let numel: i64 = sizes.iter().product();
                let nbytes = (numel * dtype.size() as i64) as usize;
                let numel = numel as usize;

                let mut storage = get_storage_pool().acquire_uninit(nbytes, device);
                let inner = Arc::make_mut(&mut storage);
                let Storage::Cpu(cpu_storage) = inner else {
                    panic!("Expected CPU storage for full()");
                };
                let data = Arc::make_mut(&mut cpu_storage.data);
                match dtype {
                    DType::F32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, numel)
                        };
                        slice.fill(value);
                    }
                    DType::F64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, numel)
                        };
                        slice.fill(value as f64);
                    }
                    DType::I32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut i32, numel)
                        };
                        slice.fill(value as i32);
                    }
                    DType::BF16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut half::bf16,
                                numel,
                            )
                        };
                        slice.fill(half::bf16::from_f32(value));
                    }
                    DType::F16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut half::f16,
                                numel,
                            )
                        };
                        slice.fill(half::f16::from_f32(value));
                    }
                    _ => {}
                }
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
            requires_grad: false,
                })
            }
            Device::Wgpu(device_id) => {
                let sizes: SmallVec<[i64; 8]> = shape.into();
                let numel: i64 = sizes.iter().product();
                let ctx = get_wgpu_context(device_id);
                let data = vec![value; numel as usize];
                let buffer = ctx.create_gpu_buffer_from_data(&data, "full");
                let storage = Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: numel as usize * 4,
                    device_id,
                    staging: RwLock::new(None),
                }));
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
            requires_grad: false,
                })
            }
        }
    }
}
