use crate::dispatch_gpu_compute;
use std::sync::OnceLock;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TpParams {
    m: u32,
    n: u32,
}

/// Cached transpose shader source — built once, reused for every dispatch.
pub(crate) fn cached_transpose_shader() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| {
        super::pipeline::record_shader_miss();
        build_transpose_shader_inner()
    })
}

dispatch_gpu_compute!(
    dispatch_transpose_gpu,
    cached_transpose_shader(),
    "transpose",
    input,
    arg1,
    arg2,
    (arg1 * arg2 * 4) as u64,
    TpParams {
        m: arg1 as u32,
        n: arg2 as u32
    },
    (arg1 as u32).div_ceil(16).max(1),
    (arg2 as u32).div_ceil(16).max(1),
    1u32,
);

fn build_transpose_shader_inner() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TpParams {
    m: u32,
    n: u32,
}
@group(0) @binding(2) var<uniform> params: TpParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= params.m || j >= params.n) { return; }
    output[j * params.m + i] = input[i * params.n + j];
}
"#
    .to_string()
}
