use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

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

    pub fn to_str(&self) -> &'static str {
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

    pub fn to_str(&self) -> &'static str {
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
        let data = vec![0u8; nbytes];

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
        ALLOC_STATS.get_or_init(|| AllocStats::new())
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
