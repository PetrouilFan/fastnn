use crate::backend::wgpu::context::WgpuContext;
use crate::backend::BackendError;
use super::PendingRead;

pub(super) fn dispatch_matmul_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    resolved_params: &[usize],
    _shape_env: &crate::ir::node::ShapeEnv,
) -> Result<(), BackendError> {
    let m = resolved_params.first().copied().unwrap_or(1);
    let k = resolved_params.get(1).copied().unwrap_or(1);
    let n = resolved_params.get(2).copied().unwrap_or(1);

    let read_slice = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let a_data = read_slice(0);
    let b_data = read_slice(1);

    let shader = build_matmul_shader();
    super::pipeline::ensure_compute_pipeline(ctx, "matmul", &shader)
        .map_err(BackendError::Dispatch)?;

    let buf_a = ctx.create_buffer(bytemuck::cast_slice(&a_data), "mm_a");
    let buf_b = ctx.create_buffer(bytemuck::cast_slice(&b_data), "mm_b");

    let output_size = (m * n * 4) as u64;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mm_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    #[allow(non_snake_case)]
    struct MatMulParams {
        M: u32,
        K: u32,
        N: u32,
    }
    let params = MatMulParams {
        M: m as u32,
        K: k as u32,
        N: n as u32,
    };
    let buf_params = ctx.create_uniform_buffer(&params, "mm_params");

    let pipeline_key = "wgpu_backend_matmul";
    let pipeline = &ctx.pipelines[pipeline_key];
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mm_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let wgc_x = (m as u32).div_ceil(16);
    let wgc_y = (n as u32).div_ceil(16);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset: output_slice.offset,
        size: output_size as usize,
    });
    Ok(())
}

pub(super) fn dispatch_matmul_activation_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    resolved_params: &[usize],
    _shape_env: &crate::ir::node::ShapeEnv,
    activation: &str,
    has_bias: bool,
) -> Result<(), BackendError> {
    let m = resolved_params.first().copied().unwrap_or(1);
    let k = resolved_params.get(1).copied().unwrap_or(1);
    let n = resolved_params.get(2).copied().unwrap_or(1);

    let activation_opcode: u32 = match activation {
        "relu" => 1,
        "gelu" => 2,
        "silu" => 3,
        _ => 0,
    };

    let read_slice = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let a_data = read_slice(0);
    let b_data = read_slice(1);
    let bias_data: Vec<f32> = if has_bias {
        read_slice(2)
    } else {
        vec![0.0]
    };
    let bias_len: u32 = if has_bias { n as u32 } else { 0 };

    let shader = build_matmul_activation_shader();
    super::pipeline::ensure_matmul_activation_pipeline(ctx, "matmul_activation", &shader)
        .map_err(BackendError::Dispatch)?;

    let buf_a = ctx.create_buffer(bytemuck::cast_slice(&a_data), "mm_a");
    let buf_b = ctx.create_buffer(bytemuck::cast_slice(&b_data), "mm_b");

    let output_size = (m * n * 4) as u64;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mm_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    #[allow(non_snake_case)]
    struct MatMulActivationParams {
        M: u32,
        K: u32,
        N: u32,
        activation: u32,
        bias_len: u32,
    }
    let params = MatMulActivationParams {
        M: m as u32,
        K: k as u32,
        N: n as u32,
        activation: activation_opcode,
        bias_len,
    };
    let buf_params = ctx.create_uniform_buffer(&params, "mm_activation_params");

    let buf_bias = ctx.create_buffer(bytemuck::cast_slice(&bias_data), "mm_bias");

    let pipeline_key = "wgpu_backend_matmul_activation";
    let pipeline = &ctx.pipelines[pipeline_key];
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mm_activation_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_bias.as_entire_binding(),
            },
        ],
    });

    let wgc_x = (m as u32).div_ceil(16);
    let wgc_y = (n as u32).div_ceil(16);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mm_activation_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset: output_slice.offset,
        size: output_size as usize,
    });
    Ok(())
}

fn build_matmul_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>         a:      array<f32>;
@group(0) @binding(1) var<storage, read>         b:      array<f32>;
@group(0) @binding(2) var<storage, read_write>   output: array<f32>;

struct MatMulParams {
    M: u32,
    K: u32,
    N: u32,
}
@group(0) @binding(3) var<uniform> params: MatMulParams;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>,
) {
    let bx = wgid.x;
    let by = wgid.y;
    let lx = lid.x;
    let ly = lid.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    var sum: f32 = 0.0;

    let num_tiles = (K + 15u) / 16u;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_row = bx * 16u + lx;
        let a_col = t * 16u + ly;
        if (a_row < M && a_col < K) {
            tileA[lx][ly] = a[a_row * K + a_col];
        } else {
            tileA[lx][ly] = 0.0;
        }

        let b_row = t * 16u + lx;
        let b_col = by * 16u + ly;
        if (b_row < K && b_col < N) {
            tileB[lx][ly] = b[b_row * N + b_col];
        } else {
            tileB[lx][ly] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            sum = sum + tileA[lx][i] * tileB[i][ly];
        }

        workgroupBarrier();
    }

    let row = gid.x;
    let col = gid.y;
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
"#
    .to_string()
}

fn build_matmul_activation_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>         a:      array<f32>;
@group(0) @binding(1) var<storage, read>         b:      array<f32>;
@group(0) @binding(2) var<storage, read_write>   output: array<f32>;

struct MatMulActivationParams {
    M: u32,
    K: u32,
    N: u32,
    activation: u32,
    bias_len: u32,
}
@group(0) @binding(3) var<uniform> params: MatMulActivationParams;

@group(0) @binding(4) var<storage, read>         bias:   array<f32>;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

fn gelu_impl(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + tanh(tanh_arg));
}

fn silu_impl(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn apply_activation(x: f32, act: u32) -> f32 {
    switch act {
        case 0u: { return x; }           // none
        case 1u: { return max(x, 0.0); } // relu
        case 2u: { return gelu_impl(x); }
        case 3u: { return silu_impl(x); }
        default: { return x; }
    }
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>,
) {
    let bx = wgid.x;
    let by = wgid.y;
    let lx = lid.x;
    let ly = lid.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    var sum: f32 = 0.0;

    let num_tiles = (K + 15u) / 16u;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_row = bx * 16u + lx;
        let a_col = t * 16u + ly;
        if (a_row < M && a_col < K) {
            tileA[lx][ly] = a[a_row * K + a_col];
        } else {
            tileA[lx][ly] = 0.0;
        }

        let b_row = t * 16u + lx;
        let b_col = by * 16u + ly;
        if (b_row < K && b_col < N) {
            tileB[lx][ly] = b[b_row * N + b_col];
        } else {
            tileB[lx][ly] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            sum = sum + tileA[lx][i] * tileB[i][ly];
        }

        workgroupBarrier();
    }

    let row = gid.x;
    let col = gid.y;
    if (row < M && col < N) {
        let idx = row * N + col;
        var val = sum;
        if (params.bias_len > 0u) {
            val = val + bias[idx % params.bias_len];
        }
        output[idx] = apply_activation(val, params.activation);
    }
}
"#
    .to_string()
}
