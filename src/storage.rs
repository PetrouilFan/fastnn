use mimalloc::MiMalloc;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;

static STORAGE_COUNTER: AtomicU64 = AtomicU64::new(0);
static ALLOC_STATS: std::sync::OnceLock<AllocStats> = std::sync::OnceLock::new();

// Forward declaration for GPU buffer type
pub struct GpuBufferHandle {
    pub buffer: Arc<wgpu::Buffer>,
    pub device_id: usize,
}

#[allow(dead_code)]
const MAX_CACHED_BLOCKS: usize = 64;
const POOL_THRESHOLD: usize = 8 * 1024 * 1024;

const NUM_SIZE_CLASSES: usize = 16;
const SIZE_CLASSES: &[usize] = &[
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
    1048576, 2097152,
];

fn get_size_class(size: usize) -> usize {
    for &class in SIZE_CLASSES.iter() {
        if size <= class {
            return class;
        }
    }
    SIZE_CLASSES[NUM_SIZE_CLASSES - 1]
}

struct MemoryPool {
    free_blocks: Vec<Vec<u8>>,
    total_cached: usize,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            free_blocks: Vec::with_capacity(NUM_SIZE_CLASSES),
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

        let size_class = get_size_class(size);

        for i in 0..self.free_blocks.len() {
            if self.free_blocks[i].len() >= size_class {
                let block = self.free_blocks.swap_remove(i);
                self.total_cached = self.total_cached.saturating_sub(block.len());
                return block;
            }
        }

        let rounded_size = (size + 63) & !63;
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

        if self.free_blocks.len() < MAX_CACHED_BLOCKS {
            self.total_cached += rounded_size;
            self.free_blocks.push(data);
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
    Wgpu(usize),
}

impl Device {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cpu" | "CPU" => Some(Device::Cpu),
            "gpu" | "GPU" | "wgpu" | "Wgpu" => Some(Device::Wgpu(0)),
            s if s.starts_with("gpu:") || s.starts_with("wgpu:") => {
                let idx: usize = s[4..].parse().ok()?;
                Some(Device::Wgpu(idx))
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Wgpu(_) => "wgpu",
        }
    }
}

// CPU storage variant - with optional lazy GPU buffer cache
#[derive(Debug)]
pub struct CpuStorage {
    pub data: Vec<u8>,
    pub nbytes: usize,
    // Lazy GPU buffer cache: maps device_id -> GPU buffer
    // This avoids repeated CPU->GPU transfers for tensors used in multiple GPU ops
    pub gpu_buffer_cache: RwLock<HashMap<usize, Arc<wgpu::Buffer>>>,
}

// GPU storage variant - keeps data on GPU
#[derive(Debug)]
pub struct GpuStorage {
    pub buffer: Arc<wgpu::Buffer>,
    pub nbytes: usize,
    pub device_id: usize,
    // Staging buffer for async readback (filled on demand)
    pub staging: RwLock<Option<Vec<u8>>>,
}

// Main storage enum - can be CPU or GPU resident
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Wgpu(GpuStorage),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(cpu) => Storage::Cpu(CpuStorage {
                data: cpu.data.clone(),
                nbytes: cpu.nbytes,
                gpu_buffer_cache: RwLock::new(HashMap::new()),
            }),
            Storage::Wgpu(gpu) => Storage::Wgpu(GpuStorage {
                buffer: gpu.buffer.clone(),
                nbytes: gpu.nbytes,
                device_id: gpu.device_id,
                staging: RwLock::new(None),
            }),
        }
    }
}

impl Storage {
    // Create CPU storage
    pub fn new_cpu(dtype: DType, nbytes: usize) -> Self {
        let data = if nbytes > 0 {
            let pool = get_memory_pool();
            pool.write().allocate(nbytes)
        } else {
            vec![]
        };

        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        Storage::Cpu(CpuStorage {
            data,
            nbytes,
            gpu_buffer_cache: RwLock::new(HashMap::new()),
        })
    }

    // Create GPU storage (buffer must be created separately)
    pub fn new_gpu(buffer: Arc<wgpu::Buffer>, nbytes: usize, device_id: usize) -> Self {
        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        Storage::Wgpu(GpuStorage {
            buffer,
            nbytes,
            device_id,
            staging: RwLock::new(None),
        })
    }

    pub fn from_vec<T: bytemuck::Pod>(data: Vec<T>, dtype: DType, device: Device) -> Self {
        let nbytes = std::mem::size_of::<T>() * data.len();
        match device {
            Device::Cpu => {
                let mut storage = Storage::new_cpu(dtype, nbytes);
                if let Storage::Cpu(cpu) = &mut storage {
                    let ptr = cpu.data.as_mut_ptr() as *mut T;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                    }
                }
                storage
            }
            Device::Wgpu(device_id) => {
                // For GPU, create CPU storage first
                // The actual GPU buffer will be created on-demand when needed
                let mut cpu_storage = Storage::new_cpu(dtype, nbytes);
                if let Storage::Cpu(cpu) = &mut cpu_storage {
                    let ptr = cpu.data.as_mut_ptr() as *mut T;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                    }
                }
                cpu_storage
            }
        }
    }

    pub fn as_ptr<T>(&self) -> *const T {
        match self {
            Storage::Cpu(cpu) => cpu.data.as_ptr() as *const T,
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use to_cpu() first.")
            }
        }
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        match self {
            Storage::Cpu(cpu) => cpu.data.as_mut_ptr() as *mut T,
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use to_cpu() first.")
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Storage::Cpu(cpu) => cpu.nbytes,
            Storage::Wgpu(gpu) => gpu.nbytes,
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Wgpu(gpu) => Device::Wgpu(gpu.device_id),
        }
    }

    /// Get or create a cached GPU buffer for this CPU storage
    /// This enables lazy GPU buffer creation - only transfer to GPU when needed
    pub fn get_or_create_gpu_buffer(&self, device_id: usize) -> Option<Arc<wgpu::Buffer>> {
        match self {
            Storage::Cpu(cpu) => {
                let mut cache = cpu.gpu_buffer_cache.write();
                if let Some(buffer) = cache.get(&device_id) {
                    return Some(buffer.clone());
                }
                None
            }
            Storage::Wgpu(gpu) => Some(gpu.buffer.clone()),
        }
    }

    /// Cache a GPU buffer for this CPU storage
    pub fn cache_gpu_buffer(&self, device_id: usize, buffer: Arc<wgpu::Buffer>) {
        if let Storage::Cpu(cpu) = self {
            let mut cache = cpu.gpu_buffer_cache.write();
            cache.insert(device_id, buffer);
        }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_freed(self.nbytes());
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
