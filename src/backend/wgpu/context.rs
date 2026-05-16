use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::dtypes::PackedWord;
use crate::error::{FastnnError, FastnnResult};
use crate::storage::DType;
use parking_lot::{Mutex, RwLock};

/// Global wgpu context — lazily initialized.
static WGPU_CONTEXT: OnceLock<Arc<Mutex<WgpuContext>>> = OnceLock::new();

/// wgpu device and queue for GPU compute.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Pipeline cache keyed by type name
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Bind group layout (reused across pipelines)
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Cached staging buffer for GPU readback (avoids re-allocation per call)
    pub staging_buffer: Option<wgpu::Buffer>,
    /// Size of the cached staging buffer in bytes
    pub staging_buffer_size: u64,
}

/// Handle returned by [`read_buffer_async`]; consumed by [`read_buffer_sync`].
/// The staging buffer is cloned so it remains valid across lock boundaries.
pub struct ReadbackHandle {
    pub staging: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl WgpuContext {
    /// Create a WgpuContext from an existing device and queue.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("packed gemv layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        WgpuContext {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_layout,
            staging_buffer: None,
            staging_buffer_size: 0,
        }
    }

    /// Get or create a compute pipeline for the given PackedWord type.
    pub fn get_or_build_pipeline<T: PackedWord>(&mut self) {
        let key = std::any::type_name::<T>().to_string();
        if self.pipelines.contains_key(&key) {
            return;
        }

        let shader_src = build_shader_source::<T>();
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("packed_{}_shader", T::BIT_WIDTH)),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("packed_{}_layout", T::BIT_WIDTH)),
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("packed_{}_gemv", T::BIT_WIDTH)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.pipelines.insert(key, pipeline);
    }

    /// Create a GPU buffer from raw bytes.
    pub fn create_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create a uniform buffer.
    pub fn create_uniform_buffer<T: bytemuck::Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Read a GPU buffer back to a `Vec<u8>` (convenience: async + sync).
    pub fn read_buffer(&mut self, buffer: &wgpu::Buffer, size: usize) -> Vec<u8> {
        let handle = self.read_buffer_async(buffer, size);
        self.read_buffer_sync(&handle)
    }

    /// Start a GPU readback: copies buffer content to a staging buffer and
    /// submits the copy command.  Returns a [`ReadbackHandle`] – call
    /// `read_buffer_sync` later to map and read.
    ///
    /// The `with_wgpu_context` lock is released between this call and
    /// `read_buffer_sync`, allowing multiple async copies to be issued before
    /// a single sync point.
    pub fn read_buffer_async(&mut self, buffer: &wgpu::Buffer, size: usize) -> ReadbackHandle {
        let staging_size = size as u64;
        if self.staging_buffer_size < staging_size {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_buffer = Some(buf);
            self.staging_buffer_size = staging_size;
        }
        let staging = self.staging_buffer.as_ref().unwrap().clone();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, staging_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        ReadbackHandle {
            staging,
            offset: 0,
            size: staging_size,
        }
    }

    /// Complete a GPU readback: polls the device, maps the staging buffer,
    /// and returns the data.
    pub fn read_buffer_sync(&self, handle: &ReadbackHandle) -> Vec<u8> {
        let buffer_slice = handle
            .staging
            .slice(handle.offset..handle.offset + handle.size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let result = receiver
            .recv()
            .expect("GPU buffer mapping channel closed unexpectedly");
        result.expect("Failed to map GPU buffer for read");

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        handle.staging.unmap();
        result
    }

    /// Submit an empty command buffer and poll the device to ensure all
    /// previously submitted work has completed.  Holds the lock only briefly.
    pub fn flush_queue(&self) {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flush"),
            });
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Read multiple buffers in a single sync point using a single cached
    /// staging buffer that grows as needed.
    /// Avoids per-dispatch `device.poll(Maintain::Wait)` overhead.
    pub fn read_buffers(&mut self, buffers: &[(&wgpu::Buffer, usize)]) -> Vec<Vec<u8>> {
        let total_size: u64 = buffers.iter().map(|(_, s)| *s as u64).sum();
        if total_size == 0 {
            return vec![Vec::new(); buffers.len()];
        }

        // Ensure cached staging buffer is large enough
        if self.staging_buffer_size < total_size {
            let new_size = total_size.next_power_of_two();
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batched_staging"),
                size: new_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_buffer = Some(buf);
            self.staging_buffer_size = new_size;
        }
        let staging = self.staging_buffer.as_ref().unwrap();

        // Copy all buffers into the single staging buffer at different offsets
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched_readback"),
            });

        let mut offset = 0u64;
        for &(buffer, size) in buffers {
            encoder.copy_buffer_to_buffer(buffer, 0, staging, offset, size as u64);
            offset += size as u64;
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map once, read all slices
        let slice = staging.slice(..total_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let result = receiver
            .recv()
            .expect("GPU buffer mapping channel closed unexpectedly");
        result.expect("Failed to map GPU buffer for read");

        let mapped = slice.get_mapped_range();
        let mut results = Vec::with_capacity(buffers.len());
        offset = 0;
        for &(_, size) in buffers {
            results.push(mapped[offset as usize..offset as usize + size].to_vec());
            offset += size as u64;
        }
        drop(mapped);
        staging.unmap();

        results
    }
}

// ============================================================================
// GPU context
// ============================================================================

static GPU_CONTEXTS: std::sync::OnceLock<RwLock<HashMap<usize, Arc<GpuContext>>>> =
    std::sync::OnceLock::new();

fn get_gpu_contexts() -> &'static RwLock<HashMap<usize, Arc<GpuContext>>> {
    GPU_CONTEXTS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn get_wgpu_context(device_id: usize) -> Arc<GpuContext> {
    {
        let contexts = get_gpu_contexts().read();
        if let Some(ctx) = contexts.get(&device_id) {
            return Arc::clone(ctx);
        }
    }
    let ctx = GpuContext::new(device_id).unwrap_or_else(|e| {
        panic!(
            "Failed to initialize GPU device {}: {}. \
             Ensure a compatible GPU driver (Vulkan/Metal/DX12) is installed. \
             Use Device::Cpu for CPU-only inference.",
            device_id, e
        )
    });
    let ctx = Arc::new(ctx);
    get_gpu_contexts()
        .write()
        .insert(device_id, Arc::clone(&ctx));
    ctx
}

/// Try to get a GPU context, returning None if GPU is unavailable.
pub fn try_get_wgpu_context(device_id: usize) -> Option<Arc<GpuContext>> {
    {
        let contexts = get_gpu_contexts().read();
        if let Some(ctx) = contexts.get(&device_id) {
            return Some(Arc::clone(ctx));
        }
    }
    let ctx = Arc::new(GpuContext::new(device_id).ok()?);
    get_gpu_contexts()
        .write()
        .insert(device_id, Arc::clone(&ctx));
    Some(ctx)
}

/// GPU execution is asynchronous by default.
///
/// Kernel launch helpers should submit command buffers and return GPU-backed
/// tensors without polling the device. Synchronization belongs at explicit
/// host readback boundaries such as `read_buffer`, `Tensor::to_cpu`, Python
/// `numpy()`, and test-only barriers.
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub device_id: usize,
    shader_modules: RwLock<HashMap<String, wgpu::ShaderModule>>,
    pipelines: RwLock<HashMap<String, wgpu::ComputePipeline>>,
    buffer_id_counter: AtomicUsize,
    buffer_pool: RwLock<HashMap<u32, Vec<wgpu::Buffer>>>,
    staging_buffer_cpu_to_gpu: RwLock<Option<wgpu::Buffer>>,
    staging_buffer_gpu_to_cpu: RwLock<Option<wgpu::Buffer>>,
    staging_buffer_size: AtomicUsize,
    shader_cache_dir: PathBuf,
    bind_group_cache: RwLock<HashMap<u64, wgpu::BindGroup>>,
    bind_group_cache_max_size: usize,
}

impl Clone for GpuContext {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            device_id: self.device_id,
            shader_modules: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            buffer_id_counter: AtomicUsize::new(0),
            buffer_pool: RwLock::new(HashMap::new()),
            staging_buffer_cpu_to_gpu: RwLock::new(None),
            staging_buffer_gpu_to_cpu: RwLock::new(None),
            staging_buffer_size: AtomicUsize::new(0),
            shader_cache_dir: self.shader_cache_dir.clone(),
            bind_group_cache: RwLock::new(HashMap::new()),
            bind_group_cache_max_size: self.bind_group_cache_max_size,
        }
    }
}

#[allow(dead_code)]
impl GpuContext {
    fn new(device_id: usize) -> FastnnResult<Self> {
        let instance = wgpu::Instance::default();
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());

        let adapter = if device_id < adapters.len() {
            adapters.swap_remove(device_id)
        } else {
            return Err(FastnnError::Cuda(format!(
                "No GPU adapter found for device_id {}. Available adapters: {}",
                device_id,
                adapters.len()
            )));
        };

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some(&format!("fastnn-gpu-device-{}", device_id)),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )) {
            Ok(result) => result,
            Err(e) => {
                return Err(FastnnError::Cuda(format!(
                    "Failed to request GPU device: {}. Ensure a valid WGPU backend (Vulkan/Metal/DX12) is available.",
                    e
                )));
            }
        };

        let shader_cache_dir = Self::get_shader_cache_dir();
        std::fs::create_dir_all(&shader_cache_dir).ok();

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_id,
            shader_modules: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            buffer_id_counter: AtomicUsize::new(0),
            buffer_pool: RwLock::new(HashMap::new()),
            staging_buffer_cpu_to_gpu: RwLock::new(None),
            staging_buffer_gpu_to_cpu: RwLock::new(None),
            staging_buffer_size: AtomicUsize::new(0),
            shader_cache_dir,
            bind_group_cache: RwLock::new(HashMap::new()),
            bind_group_cache_max_size: 256,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn get_bucket_index(size: usize) -> u32 {
        const MIN_ALIGNMENT: usize = 256;
        let aligned_size = size.max(MIN_ALIGNMENT).next_power_of_two();
        aligned_size.trailing_zeros()
    }

    pub fn acquire_buffer(&self, size: usize) -> wgpu::Buffer {
        let bucket = Self::get_bucket_index(size);
        let aligned_size = 1 << bucket;

        {
            let mut pool = self.buffer_pool.write();
            if let Some(buffers) = pool.get_mut(&bucket) {
                if let Some(buffer) = buffers.pop() {
                    return buffer;
                }
            }
        }

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: aligned_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn release_buffer(&self, buffer: wgpu::Buffer, size: usize) {
        let bucket = Self::get_bucket_index(size);
        let aligned_size = 1 << bucket;
        if buffer.size() != aligned_size as u64 {
            return;
        }
        let mut pool = self.buffer_pool.write();
        let buffers = pool.entry(bucket).or_default();
        const MAX_BUFFERS_PER_BUCKET: usize = 16;
        if buffers.len() < MAX_BUFFERS_PER_BUCKET {
            buffers.push(buffer);
        }
    }

    pub fn clear_buffer_pool(&self) {
        let mut pool = self.buffer_pool.write();
        pool.clear();
    }

    fn create_bind_group_hash(pipeline_name: &str, entries: &[wgpu::BindGroupEntry<'_>]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        pipeline_name.hash(&mut hasher);
        for entry in entries {
            if let wgpu::BindingResource::Buffer(buf) = &entry.resource {
                let ptr = buf.buffer as *const _ as u64;
                ptr.hash(&mut hasher);
                buf.offset.hash(&mut hasher);
                buf.size.hash(&mut hasher);
            }
            entry.binding.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn get_or_create_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        pipeline_name: &str,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        let hash_key = Self::create_bind_group_hash(pipeline_name, entries);

        {
            let cache = self.bind_group_cache.read();
            if let Some(bind_group) = cache.get(&hash_key) {
                return bind_group.clone();
            }
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(pipeline_name),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });

        {
            let mut cache = self.bind_group_cache.write();
            if cache.len() >= self.bind_group_cache_max_size {
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(hash_key, bind_group.clone());
        }

        bind_group
    }

    pub fn clear_bind_group_cache(&self) {
        let mut cache = self.bind_group_cache.write();
        cache.clear();
    }

    fn ensure_staging_buffer_cpu_to_gpu(&self, size: usize) -> wgpu::Buffer {
        let mut staging_guard = self.staging_buffer_cpu_to_gpu.write();

        if let Some(buffer) = &*staging_guard {
            if buffer.size() >= size as u64 {
                return buffer.clone();
            }
        }

        let new_size = size.next_power_of_two();
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_cpu_to_gpu"),
            size: new_size as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        *staging_guard = Some(new_buffer.clone());
        self.staging_buffer_size.store(new_size, Ordering::Relaxed);
        new_buffer
    }

    pub fn ensure_staging_buffer_gpu_to_cpu(&self, size: usize) -> wgpu::Buffer {
        let mut staging_guard = self.staging_buffer_gpu_to_cpu.write();

        if let Some(buffer) = &*staging_guard {
            if buffer.size() >= size as u64 {
                return buffer.clone();
            }
        }

        let new_size = size.next_power_of_two();
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_gpu_to_cpu"),
            size: new_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        *staging_guard = Some(new_buffer.clone());
        self.staging_buffer_size.store(new_size, Ordering::Relaxed);
        new_buffer
    }

    pub fn get_or_create_gpu_buffer(&self, cpu_data: &[f32], _label: &str) -> wgpu::Buffer {
        let size = std::mem::size_of_val(cpu_data);
        let buffer = self.acquire_buffer(size);
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(cpu_data));
        buffer
    }

    pub fn create_gpu_buffer_from_data(&self, data: &[f32], label: &str) -> GpuBuffer {
        let size = std::mem::size_of_val(data);
        let buffer = self.get_or_create_gpu_buffer(data, label);

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn create_gpu_buffer_from_bytes(&self, data: &[u8], _label: &str) -> GpuBuffer {
        let size = data.len();
        let buffer = self.acquire_buffer(size);
        self.queue.write_buffer(&buffer, 0, data);

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn write_bytes_to_buffer(&self, data: &[u8], dest: &wgpu::Buffer) {
        self.queue.write_buffer(dest, 0, data);
    }

    pub fn create_buffer(&self, size: usize, _label: &str) -> GpuBuffer {
        self.create_buffer_with_usage(
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            _label,
        )
    }

    pub fn create_buffer_with_usage(
        &self,
        size: usize,
        usage: wgpu::BufferUsages,
        _label: &str,
    ) -> GpuBuffer {
        let aligned_size = size.next_power_of_two();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: aligned_size as u64,
            usage,
            mapped_at_creation: false,
        });
        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn create_uniform_buffer_from_pod<T: bytemuck::Pod>(
        &self,
        data: &T,
        label: &str,
    ) -> GpuBuffer {
        use wgpu::util::DeviceExt;
        let bytes = bytemuck::bytes_of(data);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size: bytes.len(),
            device_id: self.device_id,
        }
    }

    pub fn create_buffer_from_data(&self, data: &[f32], label: &str) -> GpuBuffer {
        let size = std::mem::size_of_val(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        buffer.unmap();

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn create_uniform_buffer(&self, data: &[f32], label: &str) -> GpuBuffer {
        let size = std::mem::size_of_val(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        buffer.unmap();

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn create_uniform_buffer_u32(&self, data: &[u32], label: &str) -> GpuBuffer {
        let size = std::mem::size_of_val(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        buffer.unmap();

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn read_buffer(&self, buffer: &GpuBuffer) -> Vec<f32> {
        self.read_buffer_from_arc(&buffer.buffer, buffer.size)
    }

    pub fn read_buffer_from_arc(&self, buffer: &Arc<wgpu::Buffer>, size: usize) -> Vec<f32> {
        let staging = self.ensure_staging_buffer_gpu_to_cpu(size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..size as u64);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let result = receiver
            .recv()
            .expect("GPU staging buffer mapping channel closed unexpectedly");
        result.expect("Failed to map staging buffer for read");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn get_shader_cache_dir() -> PathBuf {
        let home_dir = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home_dir)
            .join(".cache")
            .join("fastnn")
            .join("shaders")
    }

    fn get_cache_key(&self, op_name: &str, dtype: DType, wgsl: &str) -> String {
        use fnv::FnvHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = FnvHasher::default();
        op_name.hash(&mut hasher);
        dtype.hash(&mut hasher);
        wgsl.hash(&mut hasher);
        let hash = hasher.finish();

        format!("{}_{}_{}.spv", op_name, dtype.as_str(), hash)
    }

    fn load_pipeline_data_from_cache(&self, cache_key: &str) -> Option<Vec<u8>> {
        let cache_path = self.shader_cache_dir.join(cache_key);
        std::fs::read(&cache_path).ok()
    }

    fn save_pipeline_data_to_cache(&self, cache_key: &str, data: &[u8]) {
        let cache_path = self.shader_cache_dir.join(cache_key);
        let _ = std::fs::write(&cache_path, data);
    }

    pub fn get_or_create_shader(&self, name: &str, wgsl: &str) -> wgpu::ShaderModule {
        {
            let modules = self.shader_modules.read();
            if let Some(module) = modules.get(name) {
                return module.clone();
            }
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        self.shader_modules
            .write()
            .insert(name.to_string(), module.clone());
        module
    }

    pub fn create_pipeline(&self, name: &str, wgsl: &str, dtype: DType) -> wgpu::ComputePipeline {
        {
            let pipelines = self.pipelines.read();
            if let Some(pipeline) = pipelines.get(name) {
                return pipeline.clone();
            }
        }

        let cache_key = self.get_cache_key(name, dtype, wgsl);
        let existing_data = self.load_pipeline_data_from_cache(&cache_key);

        let pipeline_cache =
            unsafe {
                self.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some(&cache_key),
                    data: existing_data.as_deref(),
                    fallback: true,
                })
            };

        let shader = self.get_or_create_shader(name, wgsl);
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: Some(&pipeline_cache),
                compilation_options: Default::default(),
            });

        if let Some(new_data) = pipeline_cache.get_data() {
            if !new_data.is_empty() {
                self.save_pipeline_data_to_cache(&cache_key, &new_data);
            }
        }

        self.pipelines
            .write()
            .insert(name.to_string(), pipeline.clone());
        pipeline
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct GpuBuffer {
    pub id: usize,
    pub buffer: Arc<wgpu::Buffer>,
    pub size: usize,
    pub device_id: usize,
}

// ---------------------------------------------------------------------------

/// Initialize the global wgpu context (lazy).
pub fn ensure_wgpu_context() {
    WGPU_CONTEXT.get_or_init(|| {
        let gpu_ctx = get_wgpu_context(0);
        let device = gpu_ctx.device().clone();
        let queue = gpu_ctx.queue().clone();
        let ctx = WgpuContext::from_device(device, queue);
        Arc::new(Mutex::new(ctx))
    });
}

/// Get a reference to the global wgpu context.
pub fn with_wgpu_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut WgpuContext) -> R,
{
    ensure_wgpu_context();
    let ctx = WGPU_CONTEXT.get().unwrap();
    let mut guard = ctx.lock();
    f(&mut guard)
}

/// Build the WGSL shader source for a given PackedWord type.
fn build_shader_source<T: PackedWord>() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read>       weights:     array<u32>;
@group(0) @binding(1) var<storage, read>       activations: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:      array<f32>;

struct Params {{
    scale: f32,
    zero: f32,
    k_packed: u32,
    k: u32,
    m: u32,
}}
@group(0) @binding(3) var<uniform> params: Params;

const ITEMS: u32 = {items}u;

fn unpack_word(packed: u32) -> {return_type} {{
{unpack_body}
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.m) {{ return; }}

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.k_packed; k = k + 1u) {{
        let unpacked = unpack_word(weights[row * params.k_packed + k]);
        {dot_logic}
    }}
    output[row] = acc * params.scale + params.zero;
}}
"#,
        items = T::ITEMS,
        return_type = T::wgsl_return_type(),
        unpack_body = T::wgsl_unpack_body(),
        dot_logic = generate_dot_logic::<T>(),
    )
}

/// Generate the dot product logic for the shader based on the packed type.
fn generate_dot_logic<T: PackedWord>() -> String {
    match T::BIT_WIDTH {
        4 => concat!(
            "        let act_base = k * ITEMS;\n",
            "        let act0 = vec4<f32>(\n",
            "            activations[act_base],\n",
            "            activations[act_base + 1u],\n",
            "            activations[act_base + 2u],\n",
            "            activations[act_base + 3u],\n",
            "        );\n",
            "        let act1 = vec4<f32>(\n",
            "            activations[act_base + 4u],\n",
            "            activations[act_base + 5u],\n",
            "            activations[act_base + 6u],\n",
            "            activations[act_base + 7u],\n",
            "        );\n",
            "        acc += dot(unpacked[0], act0) + dot(unpacked[1], act1);\n",
        )
        .to_string(),
        8 => concat!(
            "        let act_base = k * ITEMS;\n",
            "        let act0 = vec4<f32>(\n",
            "            activations[act_base],\n",
            "            activations[act_base + 1u],\n",
            "            activations[act_base + 2u],\n",
            "            activations[act_base + 3u],\n",
            "        );\n",
            "        acc += dot(unpacked, act0);\n",
        )
        .to_string(),
        16 => concat!(
            "        let act_base = k * ITEMS;\n",
            "        let act0 = vec2<f32>(\n",
            "            activations[act_base],\n",
            "            activations[act_base + 1u],\n",
            "        );\n",
            "        acc += dot(unpacked, act0);\n",
        )
        .to_string(),
        _ => "        acc += unpacked * activations[k];\n".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U4x8};

    #[test]
    fn test_build_shader_f32x1() {
        let shader = build_shader_source::<F32x1>();
        assert!(shader.contains("bitcast<f32>(packed)"));
        assert!(shader.contains("f32"));
    }

    #[test]
    fn test_build_shader_u4x8() {
        let shader = build_shader_source::<U4x8>();
        assert!(shader.contains("mat2x4<f32>"));
        assert!(shader.contains("unpack_word"));
    }
}
