use dashmap::DashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use crate::storage::{DType, Device, Storage};

pub struct StoragePool {
    buffers: DashMap<usize, Vec<Arc<Storage>>>,
    max_buffers_per_size: usize,
}

impl StoragePool {
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
            max_buffers_per_size: 100, // Limit to prevent unbounded growth
        }
    }

    /// Acquire a storage buffer of at least nbytes from the pool
    /// If no suitable buffer exists, allocate a new one
    pub fn acquire(&self, nbytes: usize, device: Device, dtype: DType) -> Arc<Storage> {
        // Only CPU storage can be pooled (GPU buffers managed separately)
        if let Device::Cpu = device {
            // Look for exact size match first
            if let Some(mut vec) = self.buffers.get_mut(&nbytes) {
                if let Some(buffer) = vec.pop() {
                    // If vec is now empty, remove the entry to keep DashMap clean
                    if vec.is_empty() {
                        self.buffers.remove(&nbytes);
                    }
                    return buffer;
                }
            }
            
            // Try to find a larger buffer that we can split (simplified - just allocate new)
            // In a more sophisticated implementation, we could split larger buffers
        }
        
        // Allocate new storage if nothing suitable in pool
        Storage::new_cpu(dtype, nbytes)
    }

    /// Release a storage buffer back to the pool for reuse
    pub fn release(&self, storage: Arc<Storage>) {
        // Only CPU storage can be pooled
        if let Storage::Cpu(_) = &storage {
            if let Device::Cpu = storage.device() {
                let nbytes = storage.nbytes();
                
                // Only pool if under limit for this size
                if let Some(vec) = self.buffers.get(&nbytes) {
                    if vec.len() < self.max_buffers_per_size {
                        if let Some(mut vec) = self.buffers.get_mut(&nbytes) {
                            vec.push(storage);
                        }
                    }
                } else {
                    // Create new entry for this size
                    self.buffers.insert(nbytes, vec![storage]);
                }
            }
        }
        // If not CPU storage or not on CPU device, just drop it (storage will be freed)
    }

    pub fn clear(&self) {
        self.buffers.clear();
    }
}

// Use OnceLock to initialize the static pool
use std::sync::OnceLock;
static STORAGE_POOL_INSTANCE: OnceLock<StoragePool> = OnceLock::new();

pub fn get_storage_pool() -> &'static StoragePool {
    STORAGE_POOL_INSTANCE.get_or_init(StoragePool::new)
}