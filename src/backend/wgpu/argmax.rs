use crate::backend::wgpu::context::WgpuContext;
use crate::backend::{BackendError, BufferSlice};
use super::PendingRead;

fn ensure_argmax_pipeline(
    ctx: &mut WgpuContext,
    key: &str,
    wgsl_source: &str,
) -> Result<(), String> {
    let pipeline_key = format!("wgpu_backend_{}", key);
    if ctx.pipelines.contains_key(&pipeline_key) {
        return Ok(());
    }

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&pipeline_key),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some(&pipeline_key),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    ctx.pipelines.insert(pipeline_key, pipeline);
    Ok(())
}

pub(super) fn dispatch_argmax_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
) -> Result<(), BackendError> {
    let out_start = output_slice.offset;

    let input = {
        let d = arena.data_mut();
        if let Some(s) = input_slices.first() {
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            return Err(BackendError::Dispatch("argmax: no input".to_string()));
        }
    };

    let numel = input.len();
    if numel == 0 {
        return Ok(());
    }

    let num_level1_wgs = (numel + 255) / 256;

    // Level 1 shader: each workgroup finds local max among up to 256 elements
    let shader_l1 = build_argmax_level1_shader();
    ensure_argmax_pipeline(ctx, "argmax_l1", &shader_l1).map_err(BackendError::Dispatch)?;

    // Level 2 shader: single workgroup reduces level-1 outputs
    let shader_l2 = build_argmax_level2_shader();
    ensure_argmax_pipeline(ctx, "argmax_l2", &shader_l2).map_err(BackendError::Dispatch)?;

    let buf_in = ctx.create_buffer(bytemuck::cast_slice(&input), "am_input");

    // Intermediate buffer: level 1 results, packed as vec2<f32> (x=value, y=index as f32)
    let intermediate_size = (num_level1_wgs * 8) as u64;
    let buf_intermediate = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("am_intermediate"),
        size: intermediate_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let output_size = 8u64;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("am_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Level 1 dispatch ──
    {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct AmParamsL1 {
            numel: u32,
            n_wgs: u32,
        }
        let params = AmParamsL1 {
            numel: numel as u32,
            n_wgs: num_level1_wgs as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "am_params_l1");

        let pipeline_key = "wgpu_backend_argmax_l1";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("am_bg_l1"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_intermediate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("am_pass_l1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_level1_wgs as u32, 1, 1);
    }

    // ── Level 2 dispatch ──
    {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct AmParamsL2 {
            n_wgs: u32,
            _pad: u32,
        }
        let params = AmParamsL2 {
            n_wgs: num_level1_wgs as u32,
            _pad: 0,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "am_params_l2");

        let pipeline_key = "wgpu_backend_argmax_l2";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("am_bg_l2"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_intermediate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("am_pass_l2"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset: out_start,
        size: output_size as usize,
    });
    Ok(())
}

fn build_argmax_level1_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> intermediate: array<vec2<f32>>;

struct AmParamsL1 {
    numel: u32,
    n_wgs: u32,
}
@group(0) @binding(2) var<uniform> params: AmParamsL1;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let wg = gid.x;
    if (wg >= params.n_wgs) { return; }

    let base = wg * 256u;
    let remaining = params.numel - base;

    var local_max_val = -3.402823e+38;
    var local_max_idx = 0u;
    var j = lid.x;
    while j < remaining {
        let idx = base + j;
        let v = input[idx];
        if v > local_max_val {
            local_max_val = v;
            local_max_idx = idx;
        }
        j = j + 256u;
    }

    shared_val[lid.x] = local_max_val;
    shared_idx[lid.x] = local_max_idx;
    workgroupBarrier();

    var stride: u32 = 128u;
    while stride > 0u {
        if lid.x < stride {
            if shared_val[lid.x + stride] > shared_val[lid.x] {
                shared_val[lid.x] = shared_val[lid.x + stride];
                shared_idx[lid.x] = shared_idx[lid.x + stride];
            }
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }

    if lid.x == 0u {
        intermediate[wg] = vec2<f32>(shared_val[0], f32(shared_idx[0]));
    }
}
"#
    .to_string()
}

fn build_argmax_level2_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read> intermediate: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<u64>;

struct AmParamsL2 {
    n_wgs: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: AmParamsL2;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var local_max_val = -3.402823e+38;
    var local_max_idx = 0u;
    var j = lid.x;
    while j < params.n_wgs {
        let v = intermediate[j].x;
        if v > local_max_val {
            local_max_val = v;
            local_max_idx = u32(intermediate[j].y);
        }
        j = j + 256u;
    }

    shared_val[lid.x] = local_max_val;
    shared_idx[lid.x] = local_max_idx;
    workgroupBarrier();

    var stride: u32 = 128u;
    while stride > 0u {
        if lid.x < stride {
            if shared_val[lid.x + stride] > shared_val[lid.x] {
                shared_val[lid.x] = shared_val[lid.x + stride];
                shared_idx[lid.x] = shared_idx[lid.x + stride];
            }
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }

    if lid.x == 0u {
        output[0] = u64(shared_idx[0]);
    }
}
"#
    .to_string()
}
