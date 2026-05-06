use dashmap::DashMap;
use std::cell::RefCell;
use std::sync::Arc;

use crate::storage::{DType, Device, Storage};

/// Threshold below which we use thread-local cache.
/// For very small tensors, the DashMap overhead dominates the allocation cost.
const SMALL_TENSOR_THRESHOLD: usize = 1024;

/// Maximum number of entries in the thread-local cache per thread.
const MAX_SMALL_CACHE: usize = 32;

thread_local! {
    /// Thread-local cache for small storage buffers.
    static SMALL_CACHE: RefCell<Vec<(usize, Arc<Storage>)>> = const { RefCell::new(Vec::new()) };
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

    /// Acquire a buffer without zeroing (caller will initialize).
    pub fn acquire_uninit(&self, nbytes: usize, device: Device) -> Arc<Storage> {
        match device {
            Device::Cpu => {
                // For small tensors, try thread-local cache first to avoid DashMap overhead
                if nbytes < SMALL_TENSOR_THRESHOLD {
                    let cached = SMALL_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        // Find a matching size entry
                        let mut found_idx = None;
                        for (i, entry) in cache.iter().enumerate() {
                            if entry.0 == nbytes {
                                found_idx = Some(i);
                                break;
                            }
                        }
                        found_idx.map(|idx| {
                            let (_, storage) = cache.remove(idx);
                            storage
                        })
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
                            Err(storage) => { storages.push(storage); }
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
                        let mut found_idx = None;
                        for (i, entry) in cache.iter().enumerate() {
                            if entry.0 == nbytes {
                                found_idx = Some(i);
                                break;
                            }
                        }
                        if let Some(idx) = found_idx {
                            let (_, storage) = cache.remove(idx);
                            // Zero the storage before returning
                            if let Ok(mut s) = Arc::try_unwrap(storage) {
                                match &mut s {
                                    Storage::Cpu(cpu) => {
                                        let data = Arc::make_mut(&mut cpu.data);
                                        data.fill(0);
                                    }
                                    Storage::Wgpu(_) => {}
                                }
                                return Some(Arc::new(s));
                            }
                            // If try_unwrap fails, storage is shared - skip cache
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
                            Err(storage) => { storages.push(storage); }
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
                        cache.borrow().len() < MAX_SMALL_CACHE
                    });
                    
                    if can_cache {
                        SMALL_CACHE.with(|cache| {
                            cache.borrow_mut().push((nbytes, storage));
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
