// Tensor factory/constructor methods

use crate::kernels::gpu::get_context;
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
        unsafe {
            let v = _mm256_loadu_ps(src.add(i));
            _mm256_storeu_ps(dst.add(i), v);
        }
        i += 8;
    }
    while i < len {
        unsafe {
            *dst.add(i) = *src.add(i);
        }
        i += 1;
    }
}

/// Fallback memcpy for f32
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(super) fn memcpy_f32(src: *const f32, dst: *mut f32, len: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

/// Cached scalar tensors for common dimension values (0-7)
/// Avoids heap allocation on every softmax/sum/mean/max call
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
            Device::Cpu => get_storage_pool().acquire_uninit(nbytes, device),
            Device::Wgpu(device_id) => {
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
                let ctx = get_context(device_id);
                let numel = t.inner.numel() as usize;
                let data = vec![value; numel];
                let buffer = ctx.create_gpu_buffer_from_data(&data, "full");

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
}
