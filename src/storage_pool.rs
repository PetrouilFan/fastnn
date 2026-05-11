use dashmap::DashMap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::storage::{DType, Device, Storage};

/// Threshold below which we use thread-local cache.
/// For very small tensors, the DashMap overhead dominates the allocation cost.
const SMALL_TENSOR_THRESHOLD: usize = 1024;

/// Maximum number of entries in the thread-local cache per thread.
const MAX_SMALL_CACHE: usize = 32;

thread_local! {
    /// Thread-local cache for small storage buffers.
    /// Maps buffer size in bytes to a list of available storages of that size.
    static SMALL_CACHE: RefCell<HashMap<usize, Vec<Arc<Storage>>>> = RefCell::new(HashMap::new());
}

pub struct StoragePool {
    buffers: DashMap<usize, Vec<Arc<Storage>>>,
}

impl StoragePool {
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
        }
    }

    /// Acquire a buffer without zeroing (caller MUST write every byte).
    pub fn acquire_uninit(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                // For small tensors, try thread-local cache first
                if nbytes < SMALL_TENSOR_THRESHOLD {
                    let cached = SMALL_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        if let Some(storages) = cache.get_mut(&nbytes) {
                            if let Some(storage) = storages.pop() {
                                if storages.is_empty() {
                                    cache.remove(&nbytes);
                                }
                                // Skip zero-fill — caller promises to write all bytes
                                match Arc::try_unwrap(storage) {
                                    Ok(s) => return Some(Arc::new(s)),
                                    Err(_) => return None,
                                }
                            }
                        }
                        None
                    });
                    if let Some(storage) = cached {
                        return storage;
                    }
                }

                let key = nbytes;
                if let Some(mut storages) = self.buffers.get_mut(&key) {
                    if let Some(storage) = storages.pop() {
                        match Arc::try_unwrap(storage) {
                            Ok(owned_storage) => return Arc::new(owned_storage),
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

    /// Acquire a zeroed buffer (for ops requiring zero-init).
    pub fn acquire_zeroed(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                // For small tensors, try thread-local cache first
                if nbytes < SMALL_TENSOR_THRESHOLD {
                    let cached = SMALL_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        if let Some(storages) = cache.get_mut(&nbytes) {
                            if let Some(storage) = storages.pop() {
                                if storages.is_empty() {
                                    cache.remove(&nbytes);
                                }
                                // Zero the storage before returning
                                match Arc::try_unwrap(storage) {
                                    Ok(mut s) => {
                                        match &mut s {
                                            Storage::Cpu(cpu) => {
                                                let data = Arc::make_mut(&mut cpu.data);
                                                data.fill(0);
                                            }
                                            Storage::Wgpu(_) => {}
                                        }
                                        return Some(Arc::new(s));
                                    }
                                    Err(_) => {
                                        // try_unwrap failed, storage is shared - skip cache
                                        return None;
                                    }
                                }
                            }
                        }
                        None
                    });
                    if let Some(storage) = cached {
                        return storage;
                    }
                }

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

    /// Backward-compatible: zero-fills by default.
    pub fn acquire(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        self.acquire_zeroed(nbytes, device)
    }

    pub fn release(&self, storage: Arc<Storage>) {
        match storage.device() {
            Device::Cpu => {
                let nbytes = storage.nbytes();

                // For small tensors, cache in thread-local to avoid DashMap overhead
                if nbytes < SMALL_TENSOR_THRESHOLD {
                    let can_cache = SMALL_CACHE.with(|cache| {
                        let cache = cache.borrow();
                        let total_storages: usize = cache.values().map(|v| v.len()).sum();
                        total_storages < MAX_SMALL_CACHE
                    });

                    if can_cache {
                        SMALL_CACHE.with(|cache| {
                            let mut cache = cache.borrow_mut();
                            cache.entry(nbytes).or_insert_with(Vec::new).push(storage);
                        });
                        return;
                    }
                    // Cache is full, fall through to global pool
                }

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

    pub fn clear(&self) {
        self.buffers.clear();
    }
}

impl Default for StoragePool {
    fn default() -> Self {
        Self::new()
    }
}

// Use OnceLock to initialize the static pool
use std::sync::OnceLock;
static STORAGE_POOL_INSTANCE: OnceLock<StoragePool> = OnceLock::new();

pub fn get_storage_pool() -> &'static StoragePool {
    STORAGE_POOL_INSTANCE.get_or_init(StoragePool::new)
}
