use super::PendingRead;
use crate::backend::wgpu::context::WgpuContext;
use crate::backend::BackendError;
use crate::dispatch_gpu_compute;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RdParams {
    num_groups: u32,
    group_size: u32,
    is_mean: u32,
    _pad: u32,
}

dispatch_gpu_compute!(
    dispatch_reduce_gpu,
    build_reduce_shader(),
    "reduce",
    input,
    arg1,
    arg2,
    {
        let ng = input.len().checked_div(arg1).unwrap_or(1);
        (ng * 4) as u64
    },
    {
        let ng = input.len().checked_div(arg1).unwrap_or(1);
        RdParams {
            num_groups: ng as u32,
            group_size: arg1 as u32,
            is_mean: arg2 as u32,
            _pad: 0,
        }
    },
    {
        let ng = input.len().checked_div(arg1).unwrap_or(1);
        (ng as u32).div_ceil(256)
    },
);

fn build_reduce_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct RdParams {
    num_groups: u32,
    group_size: u32,
    is_mean: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: RdParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= params.num_groups) { return; }
    let start = g * params.group_size;
    let end = min(start + params.group_size, arrayLength(&input));

    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + input[j];
    }
    if (params.is_mean == 1u) {
        sum = sum / f32(end - start);
    }
    output[g] = sum;
}
"#
    .to_string()
}
