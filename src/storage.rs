use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static STORAGE_COUNTER: AtomicU64 = AtomicU64::new(0);
static ALLOC_STATS: std::sync::OnceLock<AllocStats> = std::sync::OnceLock::new();

#[allow(dead_code)]
const MAX_CACHED_BLOCKS: usize = 32;
const POOL_THRESHOLD: usize = 10 * 1024 * 1024;

struct MemoryPool {
    free_blocks: HashMap<usize, Vec<Vec<u8>>>,
    total_cached: usize,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            free_blocks: HashMap::new(),
            total_cached: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Vec<u8> {
        if size == 0 {
            return vec![];
        }

        if size > POOL_THRESHOLD {
            return vec![0u8; size];
        }

        let rounded_size = (size + 63) & !63;

        if let Some(blocks) = self.free_blocks.get_mut(&rounded_size) {
            if let Some(block) = blocks.pop() {
                self.total_cached = self.total_cached.saturating_sub(rounded_size);
                return block;
            }
        }

        vec![0u8; rounded_size]
    }

    #[allow(dead_code)]
    fn deallocate(&mut self, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }

        let rounded_size = (data.len() + 63) & !63;

        if rounded_size > POOL_THRESHOLD || self.total_cached > POOL_THRESHOLD {
            return;
        }

        let current_size = self.total_cached;
        if current_size + rounded_size > POOL_THRESHOLD {
            return;
        }

        if let Some(blocks) = self.free_blocks.get_mut(&rounded_size) {
            if blocks.len() < MAX_CACHED_BLOCKS {
                self.total_cached += rounded_size;
                blocks.push(data);
            }
        } else {
            self.total_cached += rounded_size;
            self.free_blocks.insert(rounded_size, vec![data]);
        }
    }
}

static MEMORY_POOL: std::sync::OnceLock<RwLock<MemoryPool>> = std::sync::OnceLock::new();

fn get_memory_pool() -> &'static RwLock<MemoryPool> {
    MEMORY_POOL.get_or_init(|| RwLock::new(MemoryPool::new()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    F16,
    BF16,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::Bool => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" | "float32" => Some(DType::F32),
            "f64" | "float64" => Some(DType::F64),
            "i32" | "int32" => Some(DType::I32),
            "i64" | "int64" => Some(DType::I64),
            "bool" => Some(DType::Bool),
            "f16" | "float16" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    // Cuda(u32),    // TODO(gpu): add CUDA device
    // Metal,        // TODO(gpu): add Metal device
    // Wgpu,         // TODO(gpu): add WebGPU device
}

impl Device {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cpu" | "CPU" => Some(Device::Cpu),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Storage {
    pub data: Vec<u8>,
    pub nbytes: usize,
    pub dtype: DType,
    pub device: Device,
    pub id: u64,
}

impl Storage {
    pub fn new(dtype: DType, device: Device, nbytes: usize) -> Self {
        let id = STORAGE_COUNTER.fetch_add(1, Ordering::SeqCst);

        let data = if nbytes > 0 {
            let pool = get_memory_pool();
            pool.write().allocate(nbytes)
        } else {
            vec![]
        };

        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        Storage {
            data,
            nbytes,
            dtype,
            device,
            id,
        }
    }

    pub fn from_vec<T: bytemuck::Pod>(data: Vec<T>, dtype: DType, device: Device) -> Self {
        let nbytes = std::mem::size_of::<T>() * data.len();
        let mut storage = Storage::new(dtype, device, nbytes);
        let ptr = storage.data.as_mut_ptr() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        storage
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_freed(self.nbytes);
        }
    }
}

#[derive(Debug, Default)]
pub struct AllocStats {
    total_allocated: AtomicUsize,
    total_freed: AtomicUsize,
    num_allocs: AtomicU64,
}

impl AllocStats {
    fn new() -> Self {
        Self::default()
    }

    pub fn get() -> &'static Self {
        ALLOC_STATS.get_or_init(AllocStats::new)
    }

    fn add_alloc(&self, nbytes: usize) {
        self.total_allocated.fetch_add(nbytes, Ordering::SeqCst);
        self.num_allocs.fetch_add(1, Ordering::SeqCst);
    }

    fn add_freed(&self, nbytes: usize) {
        self.total_freed.fetch_add(nbytes, Ordering::SeqCst);
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst)
    }

    pub fn total_freed(&self) -> usize {
        self.total_freed.load(Ordering::SeqCst)
    }

    pub fn current_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst) - self.total_freed.load(Ordering::SeqCst)
    }

    pub fn num_allocs(&self) -> u64 {
        self.num_allocs.load(Ordering::SeqCst)
    }
}

pub fn allocator_stats() -> String {
    let stats = AllocStats::get();
    format!("{{\"total_allocated\": {}, \"total_freed\": {}, \"current_bytes\": {}, \"num_allocs\": {}}}",
        stats.total_allocated(),
        stats.total_freed(),
        stats.current_bytes(),
        stats.num_allocs())
}
