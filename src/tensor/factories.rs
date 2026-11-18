// Tensor factory/constructor methods

use crate::storage::{DType, Device, Storage};
use crate::storage_pool::get_storage_pool;
use crate::{FastnnError, FastnnResult};
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::atomic::AtomicI8;
use std::sync::Arc;

use super::shape::compute_strides;
use super::{validate_tensor_shape, Tensor, TensorImpl};

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
        let Storage::Cpu(cpu_storage) = storage_mut;
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
        Self::try_from_vec(values, shape).expect("Tensor::from_vec failed")
    }

    pub fn try_from_vec(values: Vec<f32>, shape: Vec<i64>) -> FastnnResult<Self> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let (numel, nbytes) = validate_tensor_shape(&sizes, DType::F32)?;
        if values.len() != numel {
            return Err(FastnnError::shape(format!(
                "shape {sizes:?} requires {numel} values, but {} were provided",
                values.len()
            )));
        }
        let mut storage = get_storage_pool().acquire_uninit(nbytes, Device::Cpu);
        let Storage::Cpu(cpu) = Arc::make_mut(&mut storage);
        let data = Arc::make_mut(&mut cpu.data);
        // Copy bytes so empty storage does not require an f32-aligned pointer.
        data.copy_from_slice(bytemuck::cast_slice(&values));
        Ok(Tensor::new(TensorImpl::try_new(
            storage,
            sizes,
            DType::F32,
        )?))
    }

    pub fn from_vec_with_device(values: Vec<f32>, shape: Vec<i64>, device: Device) -> Self {
        Self::try_from_vec_with_device(values, shape, device)
            .expect("Tensor::from_vec_with_device failed")
    }

    pub fn try_from_vec_with_device(
        values: Vec<f32>,
        shape: Vec<i64>,
        device: Device,
    ) -> FastnnResult<Self> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let (numel, _nbytes) = validate_tensor_shape(&sizes, DType::F32)?;
        if values.len() != numel {
            return Err(FastnnError::shape(format!(
                "shape {sizes:?} requires {numel} values, but {} were provided",
                values.len()
            )));
        }
        let _ = _nbytes;
        match device {
            Device::Cpu => Self::try_from_vec(values, sizes.into_vec()),
        }
    }

    pub fn zeros(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        Self::try_zeros(shape, dtype, device).expect("Tensor::zeros failed")
    }

    pub fn try_zeros(shape: Vec<i64>, dtype: DType, device: Device) -> FastnnResult<Self> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let (_, nbytes) = validate_tensor_shape(&sizes, dtype)?;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire_zeroed(nbytes, device),
        };

        let inner = match device {
            Device::Cpu => TensorImpl::try_new(storage, sizes, dtype)?,
        };
        Ok(Tensor::new(inner))
    }

    pub fn empty(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        Self::try_empty(shape, dtype, device).expect("Tensor::empty failed")
    }

    pub fn try_empty(shape: Vec<i64>, dtype: DType, device: Device) -> FastnnResult<Self> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let (_, nbytes) = validate_tensor_shape(&sizes, dtype)?;

        let storage = match device {
            Device::Cpu => get_storage_pool().acquire_uninit(nbytes, device),
        };

        let inner = match device {
            Device::Cpu => TensorImpl::try_new(storage, sizes, dtype)?,
        };
        Ok(Tensor::new(inner))
    }

    pub fn ones(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        Self::try_ones(shape, dtype, device).expect("Tensor::ones failed")
    }

    pub fn try_ones(shape: Vec<i64>, dtype: DType, device: Device) -> FastnnResult<Self> {
        validate_tensor_shape(&shape, dtype)?;
        if dtype.scalar_byte_width().is_none() {
            return Err(FastnnError::dtype(
                "ones does not support packed dtypes; create packed weights through quantization",
            ));
        }
        Ok(Self::ones_validated(shape, dtype, device))
    }

    fn ones_validated(shape: Vec<i64>, dtype: DType, device: Device) -> Self {
        match device {
            Device::Cpu => {
                let sizes: SmallVec<[i64; 8]> = shape.into();
                let numel: i64 = sizes.iter().product();
                let nbytes = dtype.storage_bytes(numel as usize);
                let numel = numel as usize;

                let mut storage = get_storage_pool().acquire_uninit(nbytes, device);
                let inner = Arc::make_mut(&mut storage);
                let Storage::Cpu(cpu_storage) = inner;
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
                    DType::I4
                    | DType::I8Scaled
                    | DType::U4Scaled
                    | DType::U8Scaled
                    | DType::F8
                    | DType::F8R
                    | DType::F4 => {
                        panic!("ones(): packed/FP tensors are not supported. Use the IR quantization pass to create packed weights.")
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
                    contiguous_cache: AtomicI8::new(1),
                })
            }
        }
    }

    pub fn full(shape: Vec<i64>, value: f32, dtype: DType, device: Device) -> Self {
        Self::try_full(shape, value, dtype, device).expect("Tensor::full failed")
    }

    pub fn try_full(
        shape: Vec<i64>,
        value: f32,
        dtype: DType,
        device: Device,
    ) -> FastnnResult<Self> {
        validate_tensor_shape(&shape, dtype)?;
        if value != 0.0 && dtype.scalar_byte_width().is_none() {
            return Err(FastnnError::dtype(
                "full with a nonzero value does not support packed dtypes",
            ));
        }
        Ok(Self::full_validated(shape, value, dtype, device))
    }

    fn full_validated(shape: Vec<i64>, value: f32, dtype: DType, device: Device) -> Self {
        if value == 0.0 {
            return Self::zeros(shape, dtype, device);
        }

        match device {
            Device::Cpu => {
                let sizes: SmallVec<[i64; 8]> = shape.into();
                let numel: i64 = sizes.iter().product();
                let nbytes = dtype.storage_bytes(numel as usize);
                let numel = numel as usize;

                let mut storage = get_storage_pool().acquire_uninit(nbytes, device);
                let inner = Arc::make_mut(&mut storage);
                let Storage::Cpu(cpu_storage) = inner;
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
                    DType::I64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut i64, numel)
                        };
                        slice.fill(value as i64);
                    }
                    DType::Bool => {
                        let slice =
                            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr(), numel) };
                        slice.fill(if value != 0.0 { 1u8 } else { 0u8 });
                    }
                    DType::I4
                    | DType::I8Scaled
                    | DType::U4Scaled
                    | DType::U8Scaled
                    | DType::F8
                    | DType::F8R
                    | DType::F4 => {
                        panic!("full() does not support packed U4/U8 dtypes. Use zeros() or from_vec() with packed data.");
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
                    contiguous_cache: AtomicI8::new(1),
                })
            }
        }
    }
}
