use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::storage::{DType, Device, Storage};

pub struct StoragePool {
    // Key: (nbytes, device)
    // We only pool CPU storage for now to ensure correctness (zeroing GPU buffers is complex)
    buffers: RwLock<HashMap<usize, Vec<Arc<Storage>>>>,
}

impl StoragePool {
    pub fn new() -> Self {
        Self {
            buffers: RwLock::new(HashMap::new()),
        }
    }

    pub fn acquire(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                let key = nbytes;
                let mut buffers = self.buffers.write();

                if let Some(storages) = buffers.get_mut(&key) {
                    // Pop storage and check if we have exclusive ownership
                    while let Some(storage) = storages.pop() {
                        match Arc::try_unwrap(storage) {
                            Ok(mut owned_storage) => {
                                match &mut owned_storage {
                                    Storage::Cpu(cpu) => {
                                        // Get mutable access to the Vec<u8> and zero it
                                        let data = Arc::make_mut(&mut cpu.data);
                                        data.fill(0);
                                    }
                                    Storage::Wgpu(_) => {
                                        // GPU storage - can't zero here without a kernel
                                    }
                                }
                                return Arc::new(owned_storage);
                            }
                            Err(storage) => {
                                // Arc still shared by someone else, discard and try next
                                // Don't push back - it would be a stale reference
                                drop(storage);
                            }
                        }
                    }
                }

                // Pool miss, allocate new
                Arc::new(Storage::new_cpu(DType::F32, nbytes))
            }
            Device::Wgpu(_) => {
                // For GPU, we don't pool yet because we can't easily zero the buffer
                // without a kernel dispatch.
                // Fallback to direct allocation (which currently creates a new buffer)
                // Note: Tensor::zeros for GPU currently calls ctx.create_buffer directly
                // so this path might not be hit if Tensor::zeros is updated correctly.
                // But for consistency, we return a new storage.
                // Since we don't have access to GpuContext here easily (circular dep),
                // we rely on the caller to handle GPU.
                // Actually, `Tensor::zeros` should handle GPU directly.
                // This path is just a fallback.
                panic!("GPU pooling not implemented in StoragePool::acquire");
            }
        }
    }

    pub fn release(&self, storage: Arc<Storage>) {
        match storage.device() {
            Device::Cpu => {
                let nbytes = storage.nbytes();
                let key = nbytes;

                let mut buffers = self.buffers.write();
                let storages = buffers.entry(key).or_default();

                // Limit the pool size to prevent unbounded growth
                if storages.len() < 64 {
                    storages.push(storage);
                }
                // Otherwise, drop the storage (Arc count goes to 0, memory freed)
            }
            Device::Wgpu(_) => {
                // GPU storage is not pooled, let it drop
            }
        }
    }
}

// Use OnceLock to initialize the static pool
use std::sync::OnceLock;
static STORAGE_POOL_INSTANCE: OnceLock<StoragePool> = OnceLock::new();

pub fn get_storage_pool() -> &'static StoragePool {
    STORAGE_POOL_INSTANCE.get_or_init(StoragePool::new)
}

impl StoragePool {
    pub fn new() -> Self {
        Self {
            buffers: RwLock::new(HashMap::new()),
        }
    }

    pub fn acquire(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                let key = nbytes;
                let mut buffers = self.buffers.write();

                if let Some(storages) = buffers.get_mut(&key) {
                    // Pop storage and check if we have exclusive ownership
                    while let Some(storage) = storages.pop() {
                        match Arc::try_unwrap(storage) {
                            Ok(mut owned_storage) => {
                                match &mut owned_storage {
                                    Storage::Cpu(cpu) => {
                                        // Get mutable access to the Vec<u8> and zero it
                                        let data = Arc::make_mut(&mut cpu.data);
                                        data.fill(0);
                                    }
                                    Storage::Wgpu(_) => {
                                        // GPU storage - can't zero here without a kernel
                                    }
                                }
                                eprintln!(
                                    "[storage_pool] Acquired buffer nbytes={} (pool hit)",
                                    nbytes
                                );
                                return Arc::new(owned_storage);
                            }
                            Err(storage) => {
                                // Arc still shared by someone else, discard and try next
                                // Don't push back - it would be a stale reference
                                eprintln!(
                                    "[storage_pool] Discarding stale buffer nbytes={}, refcount={}",
                                    nbytes,
                                    Arc::strong_count(&storage)
                                );
                                // Continue to try next buffer
                            }
                        }
                    }
                }

                // Pool miss, allocate new
                eprintln!(
                    "[storage_pool] Allocated new buffer nbytes={} (pool miss)",
                    nbytes
                );
                Arc::new(Storage::new_cpu(DType::F32, nbytes))
            }
            Device::Wgpu(_) => {
                // For GPU, we don't pool yet because we can't easily zero the buffer
                // without a kernel dispatch.
                // Fallback to direct allocation (which currently creates a new buffer)
                // Note: Tensor::zeros for GPU currently calls ctx.create_buffer directly
                // so this path might not be hit if Tensor::zeros is updated correctly.
                // But for consistency, we return a new storage.
                // Since we don't have access to GpuContext here easily (circular dep),
                // we rely on the caller to handle GPU.
                // Actually, `Tensor::zeros` should handle GPU directly.
                // This path is just a fallback.
                panic!("GPU pooling not implemented in StoragePool::acquire");
            }
        }
    }

    pub fn release(&self, storage: Arc<Storage>) {
        match storage.device() {
            Device::Cpu => {
                let nbytes = storage.nbytes();
                let key = nbytes;
                let refcount = Arc::strong_count(&storage);

                let mut buffers = self.buffers.write();
                let storages = buffers.entry(key).or_default();

                // Limit the pool size to prevent unbounded growth
                if storages.len() < 64 {
                    eprintln!(
                        "[storage_pool] Releasing buffer nbytes={}, refcount={}, pool_size={}",
                        nbytes,
                        refcount,
                        storages.len()
                    );
                    storages.push(storage);
                } else {
                    eprintln!(
                        "[storage_pool] Pool full, dropping buffer nbytes={}, refcount={}",
                        nbytes, refcount
                    );
                }
                // Otherwise, drop the storage (Arc count goes to 0, memory freed)
            }
            Device::Wgpu(_) => {
                // GPU storage is not pooled, let it drop
            }
        }
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> (usize, usize) {
        let buffers = self.buffers.read();
        let total_buffers: usize = buffers.values().map(|v| v.len()).sum();
        (buffers.len(), total_buffers)
    }
}

// Use OnceLock to initialize the static pool
use std::sync::OnceLock;
static STORAGE_POOL_INSTANCE: OnceLock<StoragePool> = OnceLock::new();

pub fn get_storage_pool() -> &'static StoragePool {
    STORAGE_POOL_INSTANCE.get_or_init(StoragePool::new)
}
