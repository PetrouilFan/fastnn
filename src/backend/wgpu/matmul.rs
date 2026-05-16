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

    let wgc_x = (m as u32).div_ceil(32);
    let wgc_y = (n as u32).div_ceil(32);
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

    let wgc_x = (m as u32).div_ceil(32);
    let wgc_y = (n as u32).div_ceil(32);
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

const TILE_DIM: u32 = 32u;
const TILE_A_STRIDE: u32 = 33u;

var<workgroup> tileA: array<array<f32, TILE_A_STRIDE>, TILE_DIM>;
var<workgroup> tileB: array<array<f32, TILE_DIM>, TILE_DIM>;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>,
) {
    let bx = wgid.x;
    let by = wgid.y;
    let tx = lid.x;
    let ty = lid.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    let row_base = bx * TILE_DIM + tx * 4u;
    let col_base = by * TILE_DIM + ty * 4u;

    var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
    var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
    var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
    var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

    let num_tiles = (K + TILE_DIM - 1u) / TILE_DIM;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tileA: each thread loads 4 rows x 4 K-cols = 16 elements as 4 vec4 loads
        let a_k_base = t * TILE_DIM + ty * 4u;
        for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let a_row = row_base + r;
            if (a_row < M && a_k_base < K) {
                let a_addr = a_row * K + a_k_base;
                let loaded = vec4<f32>(a[a_addr], a[a_addr + 1u], a[a_addr + 2u], a[a_addr + 3u]);
                tileA[tx * 4u + r][ty * 4u] = loaded[0];
                tileA[tx * 4u + r][ty * 4u + 1u] = loaded[1];
                tileA[tx * 4u + r][ty * 4u + 2u] = loaded[2];
                tileA[tx * 4u + r][ty * 4u + 3u] = loaded[3];
            } else {
                tileA[tx * 4u + r][ty * 4u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 1u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 2u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 3u] = 0.0;
            }
        }

        // Load tileB: each thread loads 4 K-rows x 4 cols = 16 elements as 4 vec4 loads
        let b_n_base = by * TILE_DIM + ty * 4u;
        for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let b_k_row = t * TILE_DIM + tx * 4u + r;
            if (b_k_row < K && b_n_base < N) {
                let b_addr = b_k_row * N + b_n_base;
                let loaded = vec4<f32>(b[b_addr], b[b_addr + 1u], b[b_addr + 2u], b[b_addr + 3u]);
                tileB[tx * 4u + r][ty * 4u] = loaded[0];
                tileB[tx * 4u + r][ty * 4u + 1u] = loaded[1];
                tileB[tx * 4u + r][ty * 4u + 2u] = loaded[2];
                tileB[tx * 4u + r][ty * 4u + 3u] = loaded[3];
            } else {
                tileB[tx * 4u + r][ty * 4u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 1u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 2u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 3u] = 0.0;
            }
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_DIM; i = i + 1u) {
            let a0 = tileA[tx * 4u + 0u][i];
            let a1 = tileA[tx * 4u + 1u][i];
            let a2 = tileA[tx * 4u + 2u][i];
            let a3 = tileA[tx * 4u + 3u][i];
            let b0 = tileB[i][ty * 4u + 0u];
            let b1 = tileB[i][ty * 4u + 1u];
            let b2 = tileB[i][ty * 4u + 2u];
            let b3 = tileB[i][ty * 4u + 3u];
            sum00 = sum00 + a0 * b0;
            sum01 = sum01 + a0 * b1;
            sum02 = sum02 + a0 * b2;
            sum03 = sum03 + a0 * b3;
            sum10 = sum10 + a1 * b0;
            sum11 = sum11 + a1 * b1;
            sum12 = sum12 + a1 * b2;
            sum13 = sum13 + a1 * b3;
            sum20 = sum20 + a2 * b0;
            sum21 = sum21 + a2 * b1;
            sum22 = sum22 + a2 * b2;
            sum23 = sum23 + a2 * b3;
            sum30 = sum30 + a3 * b0;
            sum31 = sum31 + a3 * b1;
            sum32 = sum32 + a3 * b2;
            sum33 = sum33 + a3 * b3;
        }

        workgroupBarrier();
    }

    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let row = row_base + r;
        if (row >= M) { break; }
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let col = col_base + c;
            if (col >= N) { break; }
            let idx = row * N + col;
            switch r * 4u + c {
                case 0u: { output[idx] = sum00; }
                case 1u: { output[idx] = sum01; }
                case 2u: { output[idx] = sum02; }
                case 3u: { output[idx] = sum03; }
                case 4u: { output[idx] = sum10; }
                case 5u: { output[idx] = sum11; }
                case 6u: { output[idx] = sum12; }
                case 7u: { output[idx] = sum13; }
                case 8u: { output[idx] = sum20; }
                case 9u: { output[idx] = sum21; }
                case 10u: { output[idx] = sum22; }
                case 11u: { output[idx] = sum23; }
                case 12u: { output[idx] = sum30; }
                case 13u: { output[idx] = sum31; }
                case 14u: { output[idx] = sum32; }
                case 15u: { output[idx] = sum33; }
                default: { }
            }
        }
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

const TILE_DIM: u32 = 32u;
const TILE_A_STRIDE: u32 = 33u;

var<workgroup> tileA: array<array<f32, TILE_A_STRIDE>, TILE_DIM>;
var<workgroup> tileB: array<array<f32, TILE_DIM>, TILE_DIM>;

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

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>,
) {
    let bx = wgid.x;
    let by = wgid.y;
    let tx = lid.x;
    let ty = lid.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    let row_base = bx * TILE_DIM + tx * 4u;
    let col_base = by * TILE_DIM + ty * 4u;

    var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
    var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
    var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
    var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

    let num_tiles = (K + TILE_DIM - 1u) / TILE_DIM;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_k_base = t * TILE_DIM + ty * 4u;
        for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let a_row = row_base + r;
            if (a_row < M && a_k_base < K) {
                let a_addr = a_row * K + a_k_base;
                let loaded = vec4<f32>(a[a_addr], a[a_addr + 1u], a[a_addr + 2u], a[a_addr + 3u]);
                tileA[tx * 4u + r][ty * 4u] = loaded[0];
                tileA[tx * 4u + r][ty * 4u + 1u] = loaded[1];
                tileA[tx * 4u + r][ty * 4u + 2u] = loaded[2];
                tileA[tx * 4u + r][ty * 4u + 3u] = loaded[3];
            } else {
                tileA[tx * 4u + r][ty * 4u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 1u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 2u] = 0.0;
                tileA[tx * 4u + r][ty * 4u + 3u] = 0.0;
            }
        }

        let b_n_base = by * TILE_DIM + ty * 4u;
        for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let b_k_row = t * TILE_DIM + tx * 4u + r;
            if (b_k_row < K && b_n_base < N) {
                let b_addr = b_k_row * N + b_n_base;
                let loaded = vec4<f32>(b[b_addr], b[b_addr + 1u], b[b_addr + 2u], b[b_addr + 3u]);
                tileB[tx * 4u + r][ty * 4u] = loaded[0];
                tileB[tx * 4u + r][ty * 4u + 1u] = loaded[1];
                tileB[tx * 4u + r][ty * 4u + 2u] = loaded[2];
                tileB[tx * 4u + r][ty * 4u + 3u] = loaded[3];
            } else {
                tileB[tx * 4u + r][ty * 4u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 1u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 2u] = 0.0;
                tileB[tx * 4u + r][ty * 4u + 3u] = 0.0;
            }
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_DIM; i = i + 1u) {
            let a0 = tileA[tx * 4u + 0u][i];
            let a1 = tileA[tx * 4u + 1u][i];
            let a2 = tileA[tx * 4u + 2u][i];
            let a3 = tileA[tx * 4u + 3u][i];
            let b0 = tileB[i][ty * 4u + 0u];
            let b1 = tileB[i][ty * 4u + 1u];
            let b2 = tileB[i][ty * 4u + 2u];
            let b3 = tileB[i][ty * 4u + 3u];
            sum00 = sum00 + a0 * b0;
            sum01 = sum01 + a0 * b1;
            sum02 = sum02 + a0 * b2;
            sum03 = sum03 + a0 * b3;
            sum10 = sum10 + a1 * b0;
            sum11 = sum11 + a1 * b1;
            sum12 = sum12 + a1 * b2;
            sum13 = sum13 + a1 * b3;
            sum20 = sum20 + a2 * b0;
            sum21 = sum21 + a2 * b1;
            sum22 = sum22 + a2 * b2;
            sum23 = sum23 + a2 * b3;
            sum30 = sum30 + a3 * b0;
            sum31 = sum31 + a3 * b1;
            sum32 = sum32 + a3 * b2;
            sum33 = sum33 + a3 * b3;
        }

        workgroupBarrier();
    }

    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let row = row_base + r;
        if (row >= M) { break; }
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let col = col_base + c;
            if (col >= N) { break; }
            let idx = row * N + col;
            var val: f32;
            switch r * 4u + c {
                case 0u: { val = sum00; }
                case 1u: { val = sum01; }
                case 2u: { val = sum02; }
                case 3u: { val = sum03; }
                case 4u: { val = sum10; }
                case 5u: { val = sum11; }
                case 6u: { val = sum12; }
                case 7u: { val = sum13; }
                case 8u: { val = sum20; }
                case 9u: { val = sum21; }
                case 10u: { val = sum22; }
                case 11u: { val = sum23; }
                case 12u: { val = sum30; }
                case 13u: { val = sum31; }
                case 14u: { val = sum32; }
                case 15u: { val = sum33; }
                default: { val = 0.0; }
            }
            if (params.bias_len > 0u) {
                val = val + bias[idx % params.bias_len];
            }
            output[idx] = apply_activation(val, params.activation);
        }
    }
}
"#
    .to_string()
}
