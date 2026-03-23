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
                    if let Some(storage) = storages.pop() {
                        // Try to get exclusive ownership via Arc::try_unwrap
                        // This avoids cloning and keeps the actual buffer alive for reuse
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
                                // Arc still shared by someone else, push back and allocate fresh
                                storages.push(storage);
                            }
                        }
                    }
                }

                // Pool miss, allocate new
                // DType is not stored in Storage, so we can use any.
                Arc::new(Storage::new_cpu(DType::F32, nbytes))
            }
            Device::Wgpu(_) => {
                // GPU storage is not pooled - allocate fresh storage.
                // The caller is responsible for managing GPU buffer lifecycle.
                Arc::new(Storage::new_cpu(DType::F32, nbytes))
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
