use dashmap::DashMap;
use std::sync::Arc;

use crate::storage::{DType, Device, Storage};

pub struct StoragePool {
    buffers: DashMap<usize, Vec<Arc<Storage>>>,
}

impl StoragePool {
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
        }
    }

    pub fn acquire(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                let key = nbytes;

                if let Some(mut storages) = self.buffers.get_mut(&key) {
                    if let Some(storage) = storages.pop() {
                        match Arc::try_unwrap(storage) {
                            Ok(mut owned_storage) => {
                                match &mut owned_storage {
                                    Storage::Cpu(cpu) => {
                                        let data = Arc::make_mut(&mut cpu.data);
                                        data.fill(0);
                                    }
                                    Storage::Wgpu(_) => {}
                                }
                                return Arc::new(owned_storage);
                            }
                            Err(storage) => {
                                storages.push(storage);
                            }
                        }
                    }
                }

                Arc::new(Storage::new_cpu(DType::F32, nbytes))
            }
            Device::Wgpu(_) => Arc::new(Storage::new_cpu(DType::F32, nbytes)),
        }
    }

    pub fn release(&self, storage: Arc<Storage>) {
        match storage.device() {
            Device::Cpu => {
                let nbytes = storage.nbytes();
                let key = nbytes;

                if let Some(mut storages) = self.buffers.get_mut(&key) {
                    if storages.len() < 64 {
                        storages.push(storage);
                    }
                } else {
                    self.buffers.insert(key, vec![storage]);
                }
            }
            Device::Wgpu(_) => {}
        }
    }
}

// Use OnceLock to initialize the static pool
use std::sync::OnceLock;
static STORAGE_POOL_INSTANCE: OnceLock<StoragePool> = OnceLock::new();

pub fn get_storage_pool() -> &'static StoragePool {
    STORAGE_POOL_INSTANCE.get_or_init(StoragePool::new)
}
