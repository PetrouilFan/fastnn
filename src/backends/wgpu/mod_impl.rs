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
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Bind group layout (reused across pipelines)
    bind_group_layout: wgpu::BindGroupLayout,
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
        // Use the GpuContext's device to ensure all buffers share the same device
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

/// Uniform buffer parameters for the GEMV kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvParams {
    scale: f32,
    zero: f32,
    k_packed: u32,
    k: u32,
    m: u32,
}

/// GPU GEMV: matrix × vector using packed weights.
pub fn gemv_wgpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32]) -> Vec<f32> {
    // Per-row quantized tensors require per-row scale/zero which the GPU shader
    // does not support. Fall back to CPU tiled path for per-channel quantization.
    if weights.is_per_channel() {
        let shape = weights.shape();
        let m = shape[0];
        let mut output = vec![0.0f32; m];
        crate::backends::packed_blas::gemv_packed_tiled(weights, activation, &mut output);
        return output;
    }

    let shape = weights.shape();
    assert!(shape.len() >= 2);
    let m = shape[0] as u32;
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS) as u32;

    with_wgpu_context(|ctx| {
        ctx.get_or_build_pipeline::<T>();
        let pipeline = ctx.pipelines.get(std::any::type_name::<T>()).unwrap();

        // Upload weights
        let weight_buffer = ctx.create_buffer(weights.as_bytes(), "weights");

        // Upload activations (as f32)
        let act_bytes = bytemuck::cast_slice(activation);
        let act_buffer = ctx.create_buffer(act_bytes, "activations");

        // Create output buffer
        let output_size = (m as usize) * std::mem::size_of::<f32>();
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload params
        let params = GemvParams {
            scale: weights.scale(),
            zero: weights.zero(),
            k_packed,
            k: k as u32,
            m,
        };
        let params_buffer = ctx.create_uniform_buffer(&params, "params");

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv_bind_group"),
            layout: &ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: act_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroup_count = m.div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemv_encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let raw = ctx.read_buffer(&output_buffer, output_size);
        let f32_data: &[f32] = bytemuck::cast_slice(&raw);
        f32_data.to_vec()
    })
}

/// GPU GEMV with persistent weight buffer — avoids re-uploading weights every call.
#[allow(clippy::too_many_arguments)]
pub fn gemv_wgpu_persistent<T: PackedWord>(
    bind_group_cache: &std::sync::Arc<std::sync::Mutex<Option<wgpu::BindGroup>>>,
    weight_buf: std::sync::Arc<wgpu::Buffer>,
    output_buf: std::sync::Arc<wgpu::Buffer>,
    params_buf: std::sync::Arc<wgpu::Buffer>,
    activation_buf: std::sync::Arc<wgpu::Buffer>,
    activation: &[f32],
    m: u32,
    k: u32,
    kpacked: u32,
    scale: f32,
    zero: f32,
) -> Vec<f32> {
    with_wgpu_context(|wctx| {
        wctx.get_or_build_pipeline::<T>();
        let pipeline = wctx.pipelines.get(std::any::type_name::<T>()).unwrap();

        // Write activation data into the cached activation buffer
        let act_bytes: &[u8] = bytemuck::cast_slice(activation);
        wctx.queue.write_buffer(&activation_buf, 0, act_bytes);

        // Write params to the cached params buffer
        let params = GemvParams {
            scale,
            zero,
            k_packed: kpacked,
            k,
            m,
        };
        let params_bytes: &[u8] = bytemuck::bytes_of(&params);
        wctx.queue.write_buffer(&params_buf, 0, params_bytes);

        // Get or create cached bind group
        let mut bg_guard = bind_group_cache.lock().unwrap();
        if bg_guard.is_none() {
            *bg_guard = Some(wctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_persistent_bindgroup"),
                layout: &wctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: activation_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            }));
        }
        let bind_group = bg_guard.as_ref().unwrap();

        let workgroup_count = m.div_ceil(256);

        // Reusable staging buffer for readback
        let output_size = m as usize * std::mem::size_of::<f32>();
        let staging_size = output_size as u64;
        if wctx.staging_buffer_size < staging_size {
            let buf = wctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            wctx.staging_buffer = Some(buf);
            wctx.staging_buffer_size = staging_size;
        }
        let staging = wctx.staging_buffer.as_ref().unwrap();

        // Single encoder: compute + copy-to-staging
        let mut encoder = wctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemv_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging in the same encoder
        encoder.copy_buffer_to_buffer(&output_buf, 0, staging, 0, output_size as u64);

        // Single submission for both compute + copy
        wctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read staging buffer
        let slice = staging.slice(..output_size as u64);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        wctx.device.poll(wgpu::Maintain::Wait);
        let result = receiver
            .recv()
            .expect("GPU staging buffer mapping channel closed unexpectedly");
        result.expect("Failed to map staging buffer");
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    })
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
            // U4x8: mat2x4<f32>, need to process 8 activation values per word
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
            // U8x4: vec4<f32>
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
            // F16x2: vec2<f32>
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
            // F32x1: scalar
            "        acc += unpacked * activations[k];\n".to_string()
        }
    }
}

/// Uniform buffer parameters for packed convolution kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    items_per_word: u32,
    _pad: u32,
}

/// GPU packed convolution forward pass.
///
/// Dispatches a compute shader that performs im2col + packed GEMM in one kernel.
/// Works with per-channel quantized packed weights (U4/U8).
///
/// TODO: The shader currently hardcodes U4x8 packing. Generalize to U8x4, F16x2, F32x1.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_packed_wgpu<T: PackedWord>(
    packed_weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> Vec<f32> {
    let output_size = n * out_c * out_h * out_w;
    let mut output = vec![0.0f32; output_size];

    with_wgpu_context(|wctx| {
        // Build a compute pipeline for packed conv
        let pipeline_key = format!("conv_packed_{}", T::BIT_WIDTH);
        if !wctx.pipelines.contains_key(&pipeline_key) {
            let shader_src = include_str!("shaders/conv_packed.wgsl");
            let shader = wctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&pipeline_key),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            let conv_bind_group_layout =
                wctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("conv_packed_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
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
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let pipeline_layout =
                wctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{}_layout", pipeline_key)),
                        bind_group_layouts: &[&conv_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline = wctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&pipeline_key),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            wctx.pipelines.insert(pipeline_key.clone(), pipeline);
        }

        let pipeline = wctx.pipelines.get(&pipeline_key).unwrap();

        // Upload packed weights
        let weight_buffer = wctx.create_buffer(packed_weight.as_bytes(), "conv_weights");

        // Upload input activations
        let input_bytes = bytemuck::cast_slice(input);
        let input_buffer = wctx.create_buffer(input_bytes, "conv_input");

        // Create output buffer
        let output_size_bytes = output_size * std::mem::size_of::<f32>();
        let output_buffer = wctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_output"),
            size: output_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload per-channel scales
        let scales = &packed_weight.scales;
        let scales_bytes = bytemuck::cast_slice(scales.as_slice());
        let scales_buffer = wctx.create_buffer(scales_bytes, "conv_scales");

        // Upload bias (or zeros)
        let bias_data: Vec<f32> = bias.map(|b| b.to_vec()).unwrap_or_else(|| vec![0.0; out_c]);
        let bias_bytes = bytemuck::cast_slice(&bias_data);
        let bias_buffer = wctx.create_buffer(bias_bytes, "conv_bias");

        // Upload params
        let params = ConvParams {
            n: n as u32,
            c: c as u32,
            h: h as u32,
            w: w as u32,
            out_c: out_c as u32,
            out_h: out_h as u32,
            out_w: out_w as u32,
            kh: kh as u32,
            kw: kw as u32,
            stride: stride as u32,
            pad: pad as u32,
            dilation: dilation as u32,
            items_per_word: T::ITEMS as u32,
            _pad: 0,
        };
        let params_buffer = wctx.create_uniform_buffer(&params, "conv_params");

        // Create bind group
        let bind_group = wctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch: 1 thread per (oc, oh, ow)
        let workgroup_x = out_c.div_ceil(8) as u32;
        let workgroup_y = out_h.div_ceil(8) as u32;
        let workgroup_z = out_w as u32;

        let mut encoder = wctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x.max(1), workgroup_y.max(1), workgroup_z.max(1));
        }

        wctx.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let raw = wctx.read_buffer(&output_buffer, output_size_bytes);
        let f32_data: &[f32] = bytemuck::cast_slice(&raw);
        output.copy_from_slice(f32_data);
    });

    output
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
