use crate::storage::{CpuStorage, DType, Device as TensorDevice, GpuStorage, Storage};
use crate::tensor::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wgpu::{Buffer, ComputePipeline, Device, Queue, ShaderModule};

// Import GPU operations module to register kernels
mod ops;

static GPU_CONTEXTS: std::sync::OnceLock<RwLock<HashMap<usize, GpuContext>>> =
    std::sync::OnceLock::new();

fn get_gpu_contexts() -> &'static RwLock<HashMap<usize, GpuContext>> {
    GPU_CONTEXTS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn get_context(device_id: usize) -> Arc<GpuContext> {
    let mut contexts = get_gpu_contexts().write();
    if let Some(ctx) = contexts.get(&device_id) {
        return Arc::new(ctx.clone());
    }

    let ctx = GpuContext::new(device_id);
    contexts.insert(device_id, ctx.clone());
    Arc::new(ctx)
}

pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub device_id: usize,
    shader_modules: RwLock<HashMap<String, ShaderModule>>,
    pipelines: RwLock<HashMap<String, ComputePipeline>>,
    buffer_id_counter: AtomicUsize,
    // Buffer pool for reusing GPU buffers by size
    buffer_pool: RwLock<HashMap<usize, Vec<wgpu::Buffer>>>,
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
        }
    }
}

impl GpuContext {
    fn new(device_id: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("No GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fastnn-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to request GPU device");

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_id,
            shader_modules: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            buffer_id_counter: AtomicUsize::new(0),
            buffer_pool: RwLock::new(HashMap::new()),
        }
    }

    /// Acquire a buffer - create a new one (simplified for now)
    pub fn acquire_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for reuse
    pub fn release_buffer(&self, _buffer: wgpu::Buffer, _size: usize) {
        // Simplified: don't pool for now
    }

    /// Get or create a buffer from CPU data
    pub fn get_or_create_gpu_buffer(&self, cpu_data: &[f32], label: &str) -> wgpu::Buffer {
        let size = cpu_data.len() * std::mem::size_of::<f32>();
        let buffer = self.acquire_buffer(size);

        // Write data to buffer
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_staging", label)),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });

        {
            let mut range = staging.slice(..).get_mapped_range_mut();
            range.copy_from_slice(bytemuck::cast_slice(cpu_data));
        }

        staging.unmap();

        // Copy to actual buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}_copy", label)),
            });
        encoder.copy_buffer_to_buffer(&staging, 0, &buffer, 0, size as u64);
        self.queue.submit(Some(encoder.finish()));

        buffer
    }

    /// Create GPU buffer from CPU data (like create_buffer_from_data)
    pub fn create_gpu_buffer_from_data(&self, data: &[f32], label: &str) -> GpuBuffer {
        let size = data.len() * std::mem::size_of::<f32>();
        let buffer = self.get_or_create_gpu_buffer(data, label);

        let id = self.buffer_id_counter.fetch_add(1, Ordering::SeqCst);
        GpuBuffer {
            id,
            buffer: Arc::new(buffer),
            size,
            device_id: self.device_id,
        }
    }

    pub fn create_buffer(&self, size: usize, label: &str) -> GpuBuffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
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

    pub fn create_buffer_from_data(&self, data: &[f32], label: &str) -> GpuBuffer {
        let size = data.len() * std::mem::size_of::<f32>();
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
        let size = data.len() * std::mem::size_of::<f32>();
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

    pub fn read_buffer_from_arc(&self, buffer: &Arc<Buffer>, size: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range().to_vec();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(slice);
        staging.unmap();
        result
    }

    pub fn get_or_create_shader(&self, name: &str, wgsl: &str) -> ShaderModule {
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

    pub fn create_pipeline(&self, name: &str, wgsl: &str) -> ComputePipeline {
        {
            let pipelines = self.pipelines.read();
            if let Some(pipeline) = pipelines.get(name) {
                return pipeline.clone();
            }
        }

        let shader = self.get_or_create_shader(name, wgsl);
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
        self.pipelines
            .write()
            .insert(name.to_string(), pipeline.clone());
        pipeline
    }
}

#[derive(Clone)]
pub struct GpuBuffer {
    pub id: usize,
    pub buffer: Arc<Buffer>,
    pub size: usize,
    pub device_id: usize,
}

fn get_tensor_data(tensor: &Tensor) -> Vec<f32> {
    let numel = tensor.inner.numel() as usize;
    let ptr = tensor.data_ptr_f32();
    unsafe { std::slice::from_raw_parts(ptr, numel).to_vec() }
}

fn create_output_tensor(data: Vec<f32>, shape: Vec<i64>, device: TensorDevice) -> Tensor {
    let storage = Arc::new(Storage::from_vec(data, DType::F32, device));
    Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        shape.iter().copied().collect(),
    ))
}

const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] + b[idx];
}
"#;

const SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] - b[idx];
}
"#;

const MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] * b[idx];
}
"#;

const DIV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] / b[idx];
}
"#;

const NEG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = -input[idx];
}
"#;

const ABS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = abs(input[idx]);
}
"#;

const EXP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = exp(input[idx]);
}
"#;

const LOG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = log(input[idx]);
}
"#;

const SQRT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = sqrt(input[idx]);
}
"#;

const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = max(input[idx], 0.0);
}
"#;

const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    let x = input[idx];
    let x3 = x * x * x;
    let t = (0.7978846 * (x + 0.044715 * x3)).tan();
    output[idx] = 0.5 * x * (1.0 + t);
}
"#;

const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    let x = input[idx];
    output[idx] = 1.0 / (1.0 + exp(-x));
}
"#;

const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = input[idx].tanh();
}
"#;

const SILU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}
"#;

const FUSED_ADD_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    let sum = a[idx] + b[idx];
    output[idx] = max(sum, 0.0);
}
"#;

const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    let m = params.x;
    let n = params.y;
    let k = params.z;

    if (row >= m || col >= n) { return; }

    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        let a_idx = row * k + i;
        let b_idx = i * n + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let out_idx = row * n + col;
    output[out_idx] = sum;
}
"#;

fn run_unary_kernel(input: &Tensor, shader: &str, name: &str, device_id: usize) -> Tensor {
    let ctx = get_context(device_id);
    let shape = input.shape().to_vec();
    let numel = shape.iter().product::<i64>() as usize;
    let input_buffer = input.inner.gpu_buffer().unwrap_or_else(|| {
        // Input is on CPU, need to copy to GPU
        let input_data = get_tensor_data(input);
        ctx.create_gpu_buffer_from_data(&input_data, "input").buffer
    });

    let gpu_output = ctx.create_buffer(numel * 4, "output");

    let pipeline = ctx.create_pipeline(name, shader);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_output.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(((numel as u64 + 255) / 256) as u32, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.device.poll(wgpu::Maintain::Wait);

    // Create tensor with GPU storage (no CPU copy!)
    let storage = Arc::new(Storage::Wgpu(GpuStorage {
        buffer: gpu_output.buffer,
        nbytes: numel * 4,
        device_id,
        staging: RwLock::new(None),
    }));
    Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        shape.iter().copied().collect(),
    ))
}

fn run_binary_kernel(a: &Tensor, b: &Tensor, shader: &str, name: &str, device_id: usize) -> Tensor {
    let ctx = get_context(device_id);
    let shape = a.shape().to_vec();
    let numel = shape.iter().product::<i64>() as usize;

    // Get or create GPU buffers - avoid copying if already on GPU
    let a_buffer = a.inner.gpu_buffer().unwrap_or_else(|| {
        let a_data = get_tensor_data(a);
        ctx.create_gpu_buffer_from_data(&a_data, "a").buffer
    });

    let b_buffer = b.inner.gpu_buffer().unwrap_or_else(|| {
        let b_data = get_tensor_data(b);
        ctx.create_gpu_buffer_from_data(&b_data, "b").buffer
    });

    let gpu_output = ctx.create_buffer(numel * 4, "output");

    let pipeline = ctx.create_pipeline(name, shader);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_output.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(((numel as u64 + 255) / 256) as u32, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.device.poll(wgpu::Maintain::Wait);

    // Create tensor with GPU storage (no CPU copy!)
    let storage = Arc::new(Storage::Wgpu(GpuStorage {
        buffer: gpu_output.buffer,
        nbytes: numel * 4,
        device_id,
        staging: RwLock::new(None),
    }));
    Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        shape.iter().copied().collect(),
    ))
}

pub fn gpu_add(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_binary_kernel(a, b, ADD_SHADER, "add", device_id)]
}

pub fn gpu_sub(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_binary_kernel(a, b, SUB_SHADER, "sub", device_id)]
}

pub fn gpu_mul(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_binary_kernel(a, b, MUL_SHADER, "mul", device_id)]
}

pub fn gpu_div(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_binary_kernel(a, b, DIV_SHADER, "div", device_id)]
}

pub fn gpu_neg(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, NEG_SHADER, "neg", device_id)]
}

pub fn gpu_abs(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, ABS_SHADER, "abs", device_id)]
}

pub fn gpu_exp(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, EXP_SHADER, "exp", device_id)]
}

pub fn gpu_log(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, LOG_SHADER, "log", device_id)]
}

pub fn gpu_sqrt(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, SQRT_SHADER, "sqrt", device_id)]
}

pub fn gpu_relu(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, RELU_SHADER, "relu", device_id)]
}

pub fn gpu_gelu(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, GELU_SHADER, "gelu", device_id)]
}

pub fn gpu_sigmoid(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, SIGMOID_SHADER, "sigmoid", device_id)]
}

pub fn gpu_tanh(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, TANH_SHADER, "tanh", device_id)]
}

pub fn gpu_silu(a: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_unary_kernel(a, SILU_SHADER, "silu", device_id)]
}

pub fn gpu_fused_add_relu(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    vec![run_binary_kernel(
        a,
        b,
        FUSED_ADD_RELU_SHADER,
        "fused_add_relu",
        device_id,
    )]
}

pub fn gpu_matmul(a: &Tensor, b: &Tensor, device_id: usize) -> Vec<Tensor> {
    let ctx = get_context(device_id);
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[a_shape.len() - 2] as usize;
    let k = a_shape[a_shape.len() - 1] as usize;
    let n = b_shape[b_shape.len() - 1] as usize;

    let mut output_shape: Vec<i64> = vec![];
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            output_shape.push(a_shape[i]);
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    // Get or create GPU buffers - avoid copying if already on GPU
    let a_buffer = a.inner.gpu_buffer().unwrap_or_else(|| {
        let a_data = get_tensor_data(a);
        ctx.create_gpu_buffer_from_data(&a_data, "matmul_a").buffer
    });

    let b_buffer = b.inner.gpu_buffer().unwrap_or_else(|| {
        let b_data = get_tensor_data(b);
        ctx.create_gpu_buffer_from_data(&b_data, "matmul_b").buffer
    });

    let gpu_out = ctx.create_buffer(m * n * 4, "matmul_output");

    let params_data_f32: Vec<f32> = vec![m as f32, n as f32, k as f32, 0.0];
    let params_buffer = ctx.create_uniform_buffer(&params_data_f32, "params");

    let pipeline = ctx.create_pipeline("matmul", MATMUL_SHADER);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let x_groups = (n + 7) / 8;
        let y_groups = (m + 7) / 8;
        compute_pass.dispatch_workgroups(x_groups as u32, y_groups as u32, 1);
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.device.poll(wgpu::Maintain::Wait);

    // Create tensor with GPU storage (no CPU copy!)
    let storage = Arc::new(Storage::Wgpu(GpuStorage {
        buffer: gpu_out.buffer,
        nbytes: m * n * 4,
        device_id,
        staging: RwLock::new(None),
    }));
    vec![Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ))]
}
