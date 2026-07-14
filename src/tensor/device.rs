use crate::error::{FastnnError, FastnnResult};
use crate::storage::{CpuStorage, DType, Device, Storage};
use std::sync::Arc;

use super::{Tensor, TensorImpl};
#[cfg(feature = "gpu")]
use crate::storage::GpuStorage;
#[cfg(feature = "gpu")]
use parking_lot::RwLock;

impl TensorImpl {
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        self.try_to_dtype(dtype)
            .expect("TensorImpl::to_dtype failed")
    }

    pub fn try_to_dtype(&self, dtype: DType) -> FastnnResult<Tensor> {
        if dtype == self.dtype {
            return Ok(self.clone().into());
        }
        if !self.is_contiguous() {
            return Err(FastnnError::shape(
                "dtype conversion requires a contiguous tensor",
            ));
        }
        if self.dtype.scalar_byte_width().is_none() || dtype.scalar_byte_width().is_none() {
            return Err(FastnnError::dtype(
                "packed tensor dtype conversion requires explicit quantize/dequantize operations",
            ));
        }
        if !matches!(self.storage.as_ref(), Storage::Cpu(_)) {
            return Err(FastnnError::device("dtype conversion requires CPU storage"));
        }
        self.to_dtype_validated(dtype)
    }

    fn to_dtype_validated(&self, dtype: DType) -> FastnnResult<Tensor> {
        #[cfg_attr(not(feature = "gpu"), allow(irrefutable_let_patterns))]
        let Storage::Cpu(cpu) = self.storage.as_ref() else {
            return Err(FastnnError::device("dtype conversion requires CPU storage"));
        };
        let numel = usize::try_from(self.numel())
            .map_err(|_| FastnnError::Overflow("dtype conversion element count overflow".into()))?;
        let source_width = self.dtype.scalar_byte_width().ok_or_else(|| {
            FastnnError::dtype("dtype conversion requires plain scalar source storage")
        })?;
        let source_offset = usize::try_from(self.storage_offset)
            .map_err(|_| FastnnError::shape("dtype conversion storage offset is negative"))?
            .checked_mul(source_width)
            .ok_or_else(|| {
                FastnnError::Overflow("dtype conversion source offset overflow".into())
            })?;
        let source_len = numel
            .checked_mul(source_width)
            .ok_or_else(|| FastnnError::Overflow("dtype conversion source size overflow".into()))?;
        let source_end = source_offset.checked_add(source_len).ok_or_else(|| {
            FastnnError::Overflow("dtype conversion source range overflow".into())
        })?;
        let source = cpu
            .data
            .get(source_offset..source_end)
            .ok_or_else(|| FastnnError::shape("dtype conversion source range exceeds storage"))?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(numel)
            .map_err(|error| FastnnError::Allocation(error.to_string()))?;
        for chunk in source.chunks_exact(source_width) {
            let value = match self.dtype {
                DType::F32 => {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(chunk);
                    f32::from_le_bytes(bytes)
                }
                DType::F64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(chunk);
                    f64::from_le_bytes(bytes) as f32
                }
                DType::I32 => {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(chunk);
                    i32::from_le_bytes(bytes) as f32
                }
                DType::I64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(chunk);
                    i64::from_le_bytes(bytes) as f32
                }
                DType::F16 => {
                    let mut bytes = [0u8; 2];
                    bytes.copy_from_slice(chunk);
                    half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
                }
                DType::BF16 => {
                    let mut bytes = [0u8; 2];
                    bytes.copy_from_slice(chunk);
                    half::bf16::from_bits(u16::from_le_bytes(bytes)).to_f32()
                }
                DType::Bool => f32::from(chunk[0] != 0),
                _ => {
                    return Err(FastnnError::dtype(
                        "dtype conversion requires plain scalar source storage",
                    ));
                }
            };
            values.push(value);
        }

        let nbytes = dtype.try_storage_bytes(numel)?;
        let mut new_bytes = Vec::new();
        new_bytes
            .try_reserve_exact(nbytes)
            .map_err(|error| FastnnError::Allocation(error.to_string()))?;
        for value in values {
            match dtype {
                DType::F32 => new_bytes.extend_from_slice(&value.to_le_bytes()),
                DType::F64 => new_bytes.extend_from_slice(&(value as f64).to_le_bytes()),
                DType::I32 => new_bytes.extend_from_slice(&(value as i32).to_le_bytes()),
                DType::I64 => new_bytes.extend_from_slice(&(value as i64).to_le_bytes()),
                DType::F16 => {
                    new_bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                }
                DType::BF16 => new_bytes
                    .extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes()),
                DType::Bool => new_bytes.push(u8::from(value != 0.0)),
                _ => {
                    return Err(FastnnError::dtype(
                        "dtype conversion requires plain scalar target storage",
                    ));
                }
            }
        }
        let new_storage = Arc::new(Storage::Cpu(CpuStorage::from_vec(new_bytes, nbytes)));
        Ok(TensorImpl::try_new(new_storage, self.sizes.clone(), dtype)?.into())
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        if device == self.device {
            return self.clone().into();
        }
        let tensor: Tensor = self.clone().into();
        match device {
            Device::Cpu => tensor.to_cpu(),
            #[cfg(feature = "gpu")]
            Device::Wgpu(device_id) => tensor.to_gpu(device_id),
        }
    }

    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        if matches!(self.device, Device::Wgpu(_)) {
            return true;
        }
        false
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    #[cfg(feature = "gpu")]
    pub fn gpu_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        match self.storage.as_ref() {
            Storage::Wgpu(gpu) => Some(gpu.buffer.clone()),
            Storage::Cpu(cpu) => {
                let cache = cpu.gpu_buffer_cache.read();
                cache.values().next().cloned()
            }
        }
    }

    #[cfg(feature = "gpu")]
    pub fn get_or_create_gpu_buffer(&self, device_id: usize) -> Option<Arc<wgpu::Buffer>> {
        match self.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => Some(gpu.buffer.clone()),
            Storage::Wgpu(_) => None,
            Storage::Cpu(_) => {
                if let Some(buffer) = self.storage.get_or_create_gpu_buffer(device_id) {
                    return Some(buffer);
                }
                None
            }
        }
    }

    #[cfg(feature = "gpu")]
    pub fn cache_gpu_buffer(&self, device_id: usize, buffer: Arc<wgpu::Buffer>) {
        self.storage.cache_gpu_buffer(device_id, buffer);
    }

    pub fn cpu_data(&self) -> Option<&[u8]> {
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => Some(&cpu.data),
            #[cfg_attr(not(feature = "gpu"), allow(unreachable_patterns))]
            _ => None,
        }
    }
}

impl Tensor {
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        self.try_to_dtype(dtype).expect("Tensor::to_dtype failed")
    }

    pub fn try_to_dtype(&self, dtype: DType) -> FastnnResult<Tensor> {
        self.inner.try_to_dtype(dtype)
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        self.inner.to_device(device)
    }

    pub fn to_cpu(&self) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Cpu(_) => Tensor::new(
                self.inner
                    .new_on_device(self.inner.storage.clone(), Device::Cpu),
            ),
            #[cfg(feature = "gpu")]
            Storage::Wgpu(gpu) => {
                use crate::backend::wgpu::context::get_wgpu_context;
                let ctx = get_wgpu_context(gpu.device_id);
                let data = ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let byte_data = bytemuck::cast_slice(&data).to_vec();
                let storage = Arc::new(Storage::Cpu(CpuStorage::from_vec(byte_data, gpu.nbytes)));
                Tensor::new(self.inner.new_on_device(storage, Device::Cpu))
            }
        }
    }

    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, device_id: usize) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => self.clone(),
            Storage::Wgpu(gpu) => {
                use crate::backend::wgpu::context::get_wgpu_context;
                let src_ctx = get_wgpu_context(gpu.device_id);
                let f32_data = src_ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let byte_data = bytemuck::cast_slice(&f32_data);

                let Some(dst_ctx) = crate::backend::wgpu::context::try_get_wgpu_context(device_id)
                else {
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

                Tensor::new(self.inner.new_on_device(storage, Device::Wgpu(device_id)))
            }
            _ => {
                let cpu_data = self.as_f32_slice().to_vec();
                let Some(ctx) = crate::backend::wgpu::context::try_get_wgpu_context(device_id)
                else {
                    return self.clone();
                };
                let buffer = ctx.create_gpu_buffer_from_data(&cpu_data, "to_gpu");

                let storage = Arc::new(Storage::Wgpu(GpuStorage {
                    buffer: buffer.buffer,
                    nbytes: cpu_data.len() * 4,
                    device_id,
                    staging: RwLock::new(None),
                }));

                Tensor::new(self.inner.new_on_device(storage, Device::Wgpu(device_id)))
            }
        }
    }
}
