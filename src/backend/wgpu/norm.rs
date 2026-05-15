use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

const NORM_WORKGROUP_SIZE: u32 = 256;

pub(super) fn dispatch_norm_gpu(
    arena: &super::WgpuBuffer,
    kernel_name: &str,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    resolved_params: &[usize],
) -> Result<(), BackendError> {
    let read_slice = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let input_data = read_slice(0);
    let weight_data = read_slice(1);
    let bias_data = read_slice(2);

    let hidden_dim = weight_data.len();
    if hidden_dim == 0 {
        return Err(BackendError::Dispatch("norm: hidden_dim is zero".into()));
    }

    let num_rows = input_data.len() / hidden_dim;
    let eps_bits = match kernel_name {
        "rms_norm" => resolved_params
            .first()
            .copied()
            .unwrap_or(f32::to_bits(1e-5) as usize),
        _ => resolved_params
            .first()
            .copied()
            .unwrap_or(f32::to_bits(1e-5) as usize),
    };

    let is_batch_norm = if kernel_name == "norm_f32" {
        resolved_params.get(1).copied().unwrap_or(0)
    } else {
        0
    };

    if is_batch_norm != 0 {
        return Err(BackendError::UnsupportedOp(format!(
            "{}_batch_norm",
            kernel_name
        )));
    }

    let is_rms: u32 = if kernel_name == "rms_norm" { 1 } else { 0 };

    with_wgpu_context(|ctx| -> Result<(), BackendError> {
        let shader = build_norm_shader();
        let pipeline_key = format!("wgpu_backend_{}", kernel_name);
        ensure_norm_pipeline(ctx, &pipeline_key, &shader).map_err(BackendError::Dispatch)?;

        let buf_input = ctx.create_buffer(bytemuck::cast_slice(&input_data), "norm_input");
        let buf_weight = ctx.create_buffer(bytemuck::cast_slice(&weight_data), "norm_weight");
        let buf_bias = ctx.create_buffer(bytemuck::cast_slice(&bias_data), "norm_bias");

        let output_size = (output_slice.size) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norm_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct NormParams {
            num_rows: u32,
            hidden_dim: u32,
            eps: u32,
            is_rms: u32,
        }
        let params = NormParams {
            num_rows: num_rows as u32,
            hidden_dim: hidden_dim as u32,
            eps: eps_bits as u32,
            is_rms,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "norm_params");

        let pipeline = &ctx.pipelines[&pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        let wgc = (num_rows as u32).div_ceil(1);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("norm_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("norm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        let result: &[f32] = bytemuck::cast_slice(&raw);

        let out = arena.data_mut();
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
            &mut out[output_slice.offset..output_slice.offset + output_slice.size],
        );
        let len = out_f32.len().min(result.len());
        out_f32[..len].copy_from_slice(&result[..len]);

        Ok(())
    })
}

fn ensure_norm_pipeline(
    ctx: &mut super::context::WgpuContext,
    pipeline_key: &str,
    wgsl_source: &str,
) -> Result<(), String> {
    if ctx.pipelines.contains_key(pipeline_key) {
        return Ok(());
    }

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(pipeline_key),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}_layout", pipeline_key)),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_pl_layout", pipeline_key)),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(pipeline_key),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    ctx.pipelines.insert(pipeline_key.to_string(), pipeline);
    Ok(())
}

fn build_norm_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read>     weight: array<f32>;
@group(0) @binding(2) var<storage, read>     bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct NormParams {
    num_rows: u32,
    hidden_dim: u32,
    eps: u32,
    is_rms: u32,
}
@group(0) @binding(4) var<uniform> params: NormParams;

var<workgroup> wg_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.num_rows) { return; }

    let N = params.hidden_dim;
    let tid = lid.x;
    let WGS = 256u;
    let row_start = row * N;

    let items_per_thread = (N + WGS - 1u) / WGS;
    let start = tid * items_per_thread;
    let end = min(start + items_per_thread, N);

    // Pass 1: accumulate sum
    var sum: f32 = 0.0;
    if (params.is_rms == 0u) {
        for (var j: u32 = start; j < end; j = j + 1u) {
            sum = sum + input[row_start + j];
        }
    } else {
        for (var j: u32 = start; j < end; j = j + 1u) {
            let x = input[row_start + j];
            sum = sum + x * x;
        }
    }
    wg_buf[tid] = sum;
    workgroupBarrier();

    var active = WGS;
    while (active > 1u) {
        active = (active + 1u) / 2u;
        if (tid < active) {
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + active];
        }
        workgroupBarrier();
    }
    let total = wg_buf[0];

    if (params.is_rms == 0u) {
        // LayerNorm
        let mean = total / f32(N);

        // Pass 2: variance
        var var_sum: f32 = 0.0;
        for (var j: u32 = start; j < end; j = j + 1u) {
            let diff = input[row_start + j] - mean;
            var_sum = var_sum + diff * diff;
        }
        wg_buf[tid] = var_sum;
        workgroupBarrier();

        active = WGS;
        while (active > 1u) {
            active = (active + 1u) / 2u;
            if (tid < active) {
                wg_buf[tid] = wg_buf[tid] + wg_buf[tid + active];
            }
            workgroupBarrier();
        }
        let total_var = wg_buf[0];
        let variance = total_var / f32(N);
        let inv_std = 1.0 / sqrt(variance + bitcast<f32>(params.eps));

        // Pass 3: normalize
        for (var j: u32 = start; j < end; j = j + 1u) {
            let idx = row_start + j;
            output[idx] = (input[idx] - mean) * inv_std * weight[j] + bias[j];
        }
    } else {
        // RMSNorm
        let ms = total / f32(N);
        let inv_rms = 1.0 / sqrt(ms + bitcast<f32>(params.eps));

        for (var j: u32 = start; j < end; j = j + 1u) {
            let idx = row_start + j;
            output[idx] = input[idx] * inv_rms * weight[j] + bias[j];
        }
    }
}
"#
    .to_string()
}
