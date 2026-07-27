use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "gpu")]
use parking_lot::RwLock;
#[cfg(feature = "gpu")]
use std::collections::HashMap;
use std::sync::Arc;

const CACHE_LINE_ALIGN: usize = 64;

/// A byte buffer guaranteed to be 64-byte (cache-line) aligned, suitable for
/// direct SIMD load/store without alignment checks.
///
/// Provides [`Deref<Target=[u8]>`] and [`DerefMut`] so existing code that treats
/// `&AlignedVec` as `&[u8]` works transparently.
#[derive(Debug)]
pub struct AlignedVec {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: AlignedVec owns uniquely-identified memory.  Access is governed by
// Rust's borrow rules (shared through `&self` / `Arc`, exclusive through
// `&mut self` / `Arc::make_mut`).
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    /// Allocate a zero-initialized buffer whose start address is 64-byte aligned.
    /// Returns an empty (no-allocation) buffer when `nbytes == 0`.
    pub fn new_zeroed(nbytes: usize) -> Self {
        if nbytes == 0 {
            return AlignedVec {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let layout = Layout::from_size_align(nbytes, CACHE_LINE_ALIGN).expect("AlignedVec layout");
        // SAFETY: `layout` has non-zero size and valid 64-byte alignment. The
        // returned allocation is owned exclusively by this `AlignedVec` and is
        // deallocated with the identical layout in `Drop`.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        AlignedVec { ptr, len: nbytes }
    }

    /// Build an aligned buffer by copying the contents of `vec`.
    /// The original `Vec` is dropped, freeing its (potentially unaligned) memory.
    pub fn from_vec(vec: Vec<u8>) -> Self {
        let nbytes = vec.len();
        if nbytes == 0 {
            drop(vec);
            return AlignedVec::new_zeroed(0);
        }
        let mut buf = AlignedVec::new_zeroed(nbytes);
        buf.copy_from_slice(&vec);
        buf
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        let mut buf = AlignedVec::new_zeroed(self.len);
        if self.len > 0 {
            buf.copy_from_slice(self);
        }
        buf
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.len > 0 {
            // Layout must match the one used in new_zeroed.
            let layout = Layout::from_size_align(self.len, CACHE_LINE_ALIGN)
                .expect("AlignedVec dealloc layout");
            // SAFETY: non-empty `AlignedVec` values own a live allocation made
            // by `alloc_zeroed` with this exact size and alignment.
            unsafe { dealloc(self.ptr, layout) };
        }
    }
}

impl Deref for AlignedVec {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` is non-null, 64-byte aligned, initialized for
            // `self.len` bytes, and remains alive for the returned borrow.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: `&mut self` guarantees exclusive access; `self.ptr` is
            // non-null, aligned, initialized, and valid for `self.len` bytes.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

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
    /// Packed 4-bit (I4x8): 8 values per u32 word.
    /// Per-channel scales/zero_points are stored in the IR node metadata.
    I4,
    /// Packed 8-bit (I8x4): 4 values per u32 word.
    /// Per-channel scales/zero_points are stored in the IR node metadata.
    /// Formerly named `U8` (misleading — this was always signed I8x4).
    I8Scaled,
    /// FP8 E4M3: 4 values per u32 word.
    F8,
    /// FP8 E5M2 (range variant): 4 values per u32 word.
    F8R,
    /// FP4 E2M1 (NVFP4-style): 8 values per u32 word, 256-entry LUT dot product.
    F4,
    /// Unsigned packed 4-bit (U4x8): 8 values per u32 word.
    U4Scaled,
    /// Unsigned packed 8-bit (U8x4): 4 values per u32 word.
    U8Scaled,
}

impl DType {
    /// Canonical scalar stored by this legacy dtype label.
    pub const fn scalar_type(self) -> crate::types::ScalarType {
        use crate::types::ScalarType;
        match self {
            DType::F32 => ScalarType::F32,
            DType::F64 => ScalarType::F64,
            DType::I32 => ScalarType::I32,
            DType::I64 => ScalarType::I64,
            DType::Bool => ScalarType::Bool,
            DType::F16 => ScalarType::F16,
            DType::BF16 => ScalarType::BF16,
            DType::I4 => ScalarType::I4,
            DType::I8Scaled => ScalarType::I8,
            DType::F8 => ScalarType::Fp8E4M3,
            DType::F8R => ScalarType::Fp8E5M2,
            DType::F4 => ScalarType::Fp4E2M1,
            DType::U4Scaled => ScalarType::U4,
            DType::U8Scaled => ScalarType::U8,
        }
    }

    /// Canonical physical encoding used by this legacy dtype label.
    pub const fn storage_encoding(self) -> crate::types::StorageEncoding {
        use crate::types::StorageEncoding;
        match self {
            DType::I4 | DType::F4 | DType::U4Scaled => StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            DType::I8Scaled | DType::F8 | DType::F8R | DType::U8Scaled => StorageEncoding::Packed {
                word_bits: 32,
                lanes: 4,
            },
            _ => StorageEncoding::Plain,
        }
    }

    /// Canonical logical/storage/transform contract represented by this runtime dtype.
    pub fn value_representation(self) -> crate::types::ValueRepresentation {
        use crate::types::{RepresentationTransform, ScalarType, ValueRepresentation};
        let storage = self.scalar_type();
        let transform = match self {
            DType::I4 | DType::I8Scaled | DType::U4Scaled | DType::U8Scaled => {
                RepresentationTransform::RuntimeAffineQuantization
            }
            DType::F4 | DType::F8 | DType::F8R => RepresentationTransform::RuntimeScaledAffine,
            _ => RepresentationTransform::None,
        };
        ValueRepresentation {
            logical: if matches!(transform, RepresentationTransform::None) {
                storage
            } else {
                ScalarType::F32
            },
            storage,
            encoding: self.storage_encoding(),
            transform,
        }
    }

    /// Logical width of one encoded value.
    pub fn logical_bit_width(self) -> usize {
        usize::from(self.scalar_type().bit_width())
    }

    /// Byte width for plain scalar storage. Packed representations return `None`.
    pub fn scalar_byte_width(self) -> Option<usize> {
        match self.storage_encoding() {
            crate::types::StorageEncoding::Plain => self.scalar_type().plain_byte_width(),
            crate::types::StorageEncoding::Packed { .. } => None,
        }
    }

    /// Exact bytes required for `numel` values, including packed-word rounding.
    pub fn storage_bytes(self, numel: usize) -> usize {
        self.try_storage_bytes(numel)
            .expect("dtype storage size overflow")
    }

    pub fn try_storage_bytes(self, numel: usize) -> crate::FastnnResult<usize> {
        self.value_representation().storage_bytes(numel)
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
            DType::I4 => "i4",
            DType::I8Scaled => "i8",
            DType::F8 => "f8",
            DType::F8R => "f8r",
            DType::F4 => "f4",
            DType::U4Scaled => "u4",
            DType::U8Scaled => "u8",
        }
    }

    pub fn from_str_label(s: &str) -> Option<Self> {
        match s {
            "f32" | "float32" => Some(DType::F32),
            "f64" | "float64" => Some(DType::F64),
            "i32" | "int32" => Some(DType::I32),
            "i64" | "int64" => Some(DType::I64),
            "bool" => Some(DType::Bool),
            "f16" | "float16" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            "int4" | "i4" => Some(DType::I4),
            "u4" | "uint4" => Some(DType::U4Scaled),
            "i8" | "int8" => Some(DType::I8Scaled),
            "u8" | "uint8" => Some(DType::U8Scaled),
            "f8" | "fp8" | "e4m3" => Some(DType::F8),
            "f8r" | "fp8r" | "e5m2" => Some(DType::F8R),
            "f4" | "fp4" | "e2m1" => Some(DType::F4),
            "u4scaled" | "uint4x8" => Some(DType::U4Scaled),
            "u8scaled" | "uint8x4" => Some(DType::U8Scaled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    #[cfg(feature = "gpu")]
    Wgpu(usize),
}

impl Device {
    pub fn from_str_label(s: &str) -> Option<Self> {
        match s {
            "cpu" | "CPU" => Some(Device::Cpu),
            #[cfg(feature = "gpu")]
            "gpu" | "GPU" | "wgpu" | "Wgpu" => Some(Device::Wgpu(0)),
            #[cfg(feature = "gpu")]
            s if s.starts_with("gpu:") => {
                let idx: usize = s[4..].parse().ok()?;
                Some(Device::Wgpu(idx))
            }
            #[cfg(feature = "gpu")]
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
            #[cfg(feature = "gpu")]
            Device::Wgpu(_) => "wgpu",
        }
    }
}

/// Helper to construct a `CpuStorage` from a `Vec<u8>` (copied into aligned memory).
impl CpuStorage {
    pub fn from_vec(data: Vec<u8>, nbytes: usize) -> Self {
        CpuStorage {
            data: Arc::new(AlignedVec::from_vec(data)),
            nbytes,
            #[cfg(feature = "gpu")]
            gpu_buffer_cache: RwLock::new(HashMap::new()),
        }
    }
}

// CPU storage variant - with optional lazy GPU buffer cache
#[derive(Debug)]
pub struct CpuStorage {
    pub data: Arc<AlignedVec>,
    pub nbytes: usize,
    // Lazy GPU buffer cache: maps device_id -> GPU buffer
    // This avoids repeated CPU->GPU transfers for tensors used in multiple GPU ops
    #[cfg(feature = "gpu")]
    pub gpu_buffer_cache: RwLock<HashMap<usize, Arc<wgpu::Buffer>>>,
}

// GPU storage variant - keeps data on GPU
#[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
    Wgpu(GpuStorage),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(cpu) => Storage::Cpu(CpuStorage {
                data: cpu.data.clone(),
                nbytes: cpu.nbytes,
                // Share GPU buffer cache via Arc so cloned tensors don't discard cached mappings
                #[cfg(feature = "gpu")]
                gpu_buffer_cache: RwLock::new(cpu.gpu_buffer_cache.read().clone()),
            }),
            #[cfg(feature = "gpu")]
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
    // Create CPU storage with 64-byte aligned buffer
    pub fn new_cpu(_dtype: DType, nbytes: usize) -> Self {
        let data = Arc::new(AlignedVec::new_zeroed(nbytes));

        if let Some(stats) = ALLOC_STATS.get() {
            stats.add_alloc(nbytes);
        }

        Storage::Cpu(CpuStorage {
            data,
            nbytes,
            #[cfg(feature = "gpu")]
            gpu_buffer_cache: RwLock::new(HashMap::new()),
        })
    }

    // Create GPU storage (buffer must be created separately)
    #[cfg(feature = "gpu")]
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

    pub fn from_vec_owned<T: bytemuck::Pod>(data: Vec<T>, _dtype: DType, _device: Device) -> Self {
        let nbytes = std::mem::size_of::<T>() * data.len();
        // Copy into 64-byte aligned storage (safe, no unsafe needed)
        let src_bytes = bytemuck::cast_slice(&data);
        let mut aligned = AlignedVec::new_zeroed(nbytes);
        if nbytes > 0 {
            aligned.copy_from_slice(src_bytes);
        }
        drop(data);
        Storage::Cpu(CpuStorage {
            data: Arc::new(aligned),
            nbytes,
            #[cfg(feature = "gpu")]
            gpu_buffer_cache: Default::default(),
        })
    }

    pub fn from_vec<T: bytemuck::Pod>(data: Vec<T>, dtype: DType, device: Device) -> Self {
        Self::from_vec_owned(data, dtype, device)
    }

    pub fn as_ptr<T>(&self) -> *const T {
        match self {
            Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr() as *const T,
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use to_cpu() first.")
            }
        }
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        match self {
            Storage::Cpu(cpu) => {
                let data = Arc::make_mut(&mut cpu.data);
                data.as_mut_ptr() as *mut T
            }
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => {
                panic!("Cannot get CPU pointer from GPU storage. Use to_cpu() first.")
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Storage::Cpu(cpu) => cpu.nbytes,
            #[cfg(feature = "gpu")]
            Storage::Wgpu(gpu) => gpu.nbytes,
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "gpu")]
            Storage::Wgpu(gpu) => Device::Wgpu(gpu.device_id),
        }
    }

    /// Get or create a cached GPU buffer for this CPU storage
    /// This enables lazy GPU buffer creation - only transfer to GPU when needed
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
    pub fn cache_gpu_buffer(&self, device_id: usize, buffer: Arc<wgpu::Buffer>) {
        if let Storage::Cpu(cpu) = self {
            let mut cache = cpu.gpu_buffer_cache.write();
            cache.insert(device_id, buffer);
        }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        let is_last = match self {
            Storage::Cpu(cpu) => Arc::strong_count(&cpu.data) == 1,
            #[cfg(feature = "gpu")]
            Storage::Wgpu(gpu) => Arc::strong_count(&gpu.buffer) == 1,
        };
        if is_last {
            if let Some(stats) = ALLOC_STATS.get() {
                stats.add_freed(self.nbytes());
            }
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
        self.total_allocated.fetch_add(nbytes, Ordering::AcqRel);
        self.num_allocs.fetch_add(1, Ordering::Relaxed);
    }

    fn add_freed(&self, nbytes: usize) {
        self.total_freed.fetch_add(nbytes, Ordering::AcqRel);
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Acquire)
    }

    pub fn total_freed(&self) -> usize {
        self.total_freed.load(Ordering::Acquire)
    }

    pub fn current_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::Acquire) - self.total_freed.load(Ordering::Acquire)
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

#[cfg(test)]
mod dtype_tests {
    use super::{AlignedVec, DType, CACHE_LINE_ALIGN};
    use crate::types::{RepresentationTransform, ScalarType, StorageEncoding};

    #[test]
    fn aligned_vec_is_zeroed_and_cache_line_aligned() {
        let mut storage = AlignedVec::new_zeroed(65);
        assert_eq!((storage.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert!(storage.iter().all(|byte| *byte == 0));
        storage[64] = 7;
        assert_eq!(storage[64], 7);
    }

    #[test]
    fn separates_logical_width_from_scalar_storage_width() {
        assert_eq!(DType::F32.logical_bit_width(), 32);
        assert_eq!(DType::F32.scalar_byte_width(), Some(4));
        assert_eq!(DType::U4Scaled.logical_bit_width(), 4);
        assert_eq!(DType::U4Scaled.scalar_byte_width(), None);
    }

    #[test]
    fn packed_storage_size_rounds_to_complete_words() {
        assert_eq!(DType::U4Scaled.storage_bytes(1), 4);
        assert_eq!(DType::U4Scaled.storage_bytes(8), 4);
        assert_eq!(DType::U4Scaled.storage_bytes(9), 8);
        assert_eq!(DType::U8Scaled.storage_bytes(5), 8);
    }

    #[test]
    fn legacy_dtype_storage_contract_maps_to_canonical_types() {
        assert_eq!(DType::F32.scalar_type(), ScalarType::F32);
        assert_eq!(DType::F32.storage_encoding(), StorageEncoding::Plain);
        assert_eq!(DType::U4Scaled.scalar_type(), ScalarType::U4);
        assert_eq!(
            DType::U4Scaled.storage_encoding(),
            StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            }
        );
        assert_eq!(
            DType::U4Scaled.try_storage_bytes(9).unwrap(),
            DType::U4Scaled
                .storage_encoding()
                .storage_bytes(DType::U4Scaled.scalar_type(), 9)
                .unwrap()
        );
        assert!(matches!(
            DType::U4Scaled.value_representation().transform,
            RepresentationTransform::RuntimeAffineQuantization
        ));
        assert!(matches!(
            DType::F4.value_representation().transform,
            RepresentationTransform::RuntimeScaledAffine
        ));
    }
}
