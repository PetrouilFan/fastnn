use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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

    /// Read a GPU buffer back to a `Vec<u8>`.
    pub fn read_buffer(&mut self, buffer: &wgpu::Buffer, size: usize) -> Vec<u8> {
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
        let staging = self.staging_buffer.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, staging_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..staging_size);
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
        staging.unmap();
        result
    }
}

/// Initialize the global wgpu context (lazy).
pub fn ensure_wgpu_context() {
    WGPU_CONTEXT.get_or_init(|| {
        let gpu_ctx = crate::kernels::gpu::get_context(0);
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
    let mut guard = ctx.lock().unwrap();
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
        4 => {
            concat!(
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
            .to_string()
        }
        8 => {
            concat!(
                "        let act_base = k * ITEMS;\n",
                "        let act0 = vec4<f32>(\n",
                "            activations[act_base],\n",
                "            activations[act_base + 1u],\n",
                "            activations[act_base + 2u],\n",
                "            activations[act_base + 3u],\n",
                "        );\n",
                "        acc += dot(unpacked, act0);\n",
            )
            .to_string()
        }
        16 => {
            concat!(
                "        let act_base = k * ITEMS;\n",
                "        let act0 = vec2<f32>(\n",
                "            activations[act_base],\n",
                "            activations[act_base + 1u],\n",
                "        );\n",
                "        acc += dot(unpacked, act0);\n",
            )
            .to_string()
        }
        _ => {
            "        acc += unpacked * activations[k];\n".to_string()
        }
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
