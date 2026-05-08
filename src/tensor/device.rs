use crate::storage::{CpuStorage, DType, Device, GpuStorage, Storage};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use super::{Tensor, TensorImpl};

impl TensorImpl {
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

    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Wgpu(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    pub fn gpu_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        match self.storage.as_ref() {
            Storage::Wgpu(gpu) => Some(gpu.buffer.clone()),
            Storage::Cpu(cpu) => {
                let cache = cpu.gpu_buffer_cache.read();
                cache.values().next().cloned()
            }
        }
    }

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

    pub fn cache_gpu_buffer(&self, device_id: usize, buffer: Arc<wgpu::Buffer>) {
        self.storage.cache_gpu_buffer(device_id, buffer);
    }

    pub fn cpu_data(&self) -> Option<&[u8]> {
        match self.storage.as_ref() {
            Storage::Cpu(cpu) => Some(&cpu.data),
            _ => None,
        }
    }
}

impl Tensor {
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        self.inner.to_dtype(dtype)
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        self.inner.to_device(device)
    }

    pub fn to_cpu(&self) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Cpu(_) => {
                Tensor::new(self.inner.new_on_device(self.inner.storage.clone(), Device::Cpu))
            }
            Storage::Wgpu(gpu) => {
                use crate::kernels::gpu::get_context;
                let ctx = get_context(gpu.device_id);
                let data = ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let storage = Arc::new(Storage::Cpu(CpuStorage {
                    data: Arc::new(bytemuck::cast_slice(&data).to_vec()),
                    nbytes: gpu.nbytes,
                    gpu_buffer_cache: RwLock::new(HashMap::new()),
                }));
                Tensor::new(self.inner.new_on_device(storage, Device::Cpu))
            }
        }
    }

    pub fn to_gpu(&self, device_id: usize) -> Tensor {
        match self.inner.storage.as_ref() {
            Storage::Wgpu(gpu) if gpu.device_id == device_id => self.clone(),
            Storage::Wgpu(gpu) => {
                use crate::kernels::gpu::get_context;
                let src_ctx = get_context(gpu.device_id);
                let f32_data = src_ctx.read_buffer_from_arc(&gpu.buffer, gpu.nbytes);
                let byte_data = bytemuck::cast_slice(&f32_data);

                let Some(dst_ctx) = crate::kernels::gpu::try_get_context(device_id) else {
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

                Tensor::new(self.inner.new_on_device(storage, Device::Wgpu(device_id)))
            }
        }
    }
}
