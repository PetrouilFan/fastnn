use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[allow(dead_code)]
static STORAGE_COUNTER: AtomicU64 = AtomicU64::new(0);
static ALLOC_STATS: std::sync::OnceLock<AllocStats> = std::sync::OnceLock::new();

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
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::Bool => 1,
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
            s if s.starts_with("gpu:") => {
                let idx: usize = s[4..].parse().ok()?;
                Some(Device::Wgpu(idx))
            }
            s if s.starts_with("wgpu:") => {
                let idx: usize = s[5..].parse().ok()?;
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
    pub data: Arc<Vec<u8>>,
    pub nbytes: usize,
    // Lazy GPU buffer cache: maps device_id -> GPU buffer
    // This avoids repeated CPU->GPU transfers for tensors used in multiple GPU ops
    // Wrapped in Arc so clones of Storage share the same cache
    pub gpu_buffer_cache: Arc<RwLock<HashMap<usize, Arc<wgpu::Buffer>>>>,
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
                // Share the GPU buffer cache across clones to avoid redundant allocations
                gpu_buffer_cache: cpu.gpu_buffer_cache.clone(),
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
    pub fn new_cpu(_dtype: DType, nbytes: usize) -> Self {
        let data = if nbytes > 0 {
            Arc::new(vec![0u8; nbytes])
        } else {
            Arc::new(vec![])
        };

        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        Storage::Cpu(CpuStorage {
            data,
            nbytes,
            gpu_buffer_cache: Arc::new(RwLock::new(HashMap::new())),
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
        // Convert the Vec<T> into a Vec<u8> without copying the data
        // SAFETY: T: Pod guarantees that the in-memory representation of T is
        // equivalent to a [u8; size_of::<T>()], and that the layout is safe to reinterpret.
        let (ptr, len, cap) = data.into_raw_parts();
        let byte_len = len * std::mem::size_of::<T>();
        let byte_cap = cap * std::mem::size_of::<T>();
        let data = unsafe { Arc::new(Vec::from_raw_parts(ptr as *mut u8, byte_len, byte_cap)) };

        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        match device {
            Device::Cpu => Storage::Cpu(CpuStorage {
                data,
                nbytes,
                gpu_buffer_cache: Arc::new(RwLock::new(HashMap::new())),
            }),
            Device::Wgpu(_device_id) => {
                // For GPU, create CPU storage first; actual GPU buffer will be created on-demand
                Storage::Cpu(CpuStorage {
                    data,
                    nbytes,
                    gpu_buffer_cache: Arc::new(RwLock::new(HashMap::new())),
                })
            }
        }
    }

    pub fn as_ptr<T>(&self) -> *const T {
        match self {
            Storage::Cpu(cpu) => {
                let ptr = cpu.data.as_ref().as_ptr() as *const T;
                debug_assert!(ptr as usize % std::mem::align_of::<T>() == 0,
                    "Misaligned pointer: expected alignment of {} for type {}, got {:p}",
                    std::mem::align_of::<T>(), std::any::type_name::<T>(), ptr);
                ptr
            }
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use to_cpu() first.")
            }
        }
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        match self {
            Storage::Cpu(cpu) => {
                let data = Arc::make_mut(&mut cpu.data);
                let ptr = data.as_mut_ptr() as *mut T;
                debug_assert!(ptr as usize % std::mem::align_of::<T>() == 0,
                    "Misaligned pointer: expected alignment of {} for type {}, got {:p}",
                    std::mem::align_of::<T>(), std::any::type_name::<T>(), ptr);
                ptr
            }
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
                let cache = cpu.gpu_buffer_cache.write();
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
        self.total_allocated.fetch_add(nbytes, Ordering::Relaxed);
        self.num_allocs.fetch_add(1, Ordering::Relaxed);
    }

    fn add_freed(&self, nbytes: usize) {
        self.total_freed.fetch_add(nbytes, Ordering::Relaxed);
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    pub fn total_freed(&self) -> usize {
        self.total_freed.load(Ordering::Relaxed)
    }

    pub fn current_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed) - self.total_freed.load(Ordering::Relaxed)
    }

    pub fn num_allocs(&self) -> u64 {
        self.num_allocs.load(Ordering::Relaxed)
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
