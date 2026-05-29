use crate::dispatch_gpu_compute;
use std::sync::OnceLock;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SfParams {
    numel: u32,
    row_size: u32,
}

/// Cached softmax shader source — built once, reused for every dispatch.
pub(crate) fn cached_softmax_shader() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    if let Some(shader) = S.get() {
        super::pipeline::record_shader_hit();
        shader
    } else {
        S.get_or_init(|| {
            super::pipeline::record_shader_miss();
            build_softmax_shader_inner()
        })
    }
}

dispatch_gpu_compute!(
    dispatch_softmax_gpu,
    cached_softmax_shader(),
    "softmax",
    input,
    arg1,
    arg2,
    (arg1 * 4) as u64,
    {
        let rs = if arg2 > 0 { arg2 } else { 1 };
        SfParams {
            numel: arg1 as u32,
            row_size: rs as u32,
        }
    },
    ((arg1 as u32) / (if arg2 > 0 { arg2 as u32 } else { 1u32 })).div_ceil(256),
);

fn build_softmax_shader_inner() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SfParams {
    numel: u32,
    row_size: u32,
}
@group(0) @binding(2) var<uniform> params: SfParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row * params.row_size >= params.numel) { return; }
    let start = row * params.row_size;
    let end = min(start + params.row_size, params.numel);

    // Find max
    var max_val: f32 = -3.402823e+38;
    for (var j: u32 = start; j < end; j = j + 1u) {
        max_val = max(max_val, input[j]);
    }

    // Compute sum of exp
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + exp(input[j] - max_val);
    }

    // Normalize
    for (var j: u32 = start; j < end; j = j + 1u) {
        output[j] = exp(input[j] - max_val) / sum;
    }
}
"#
    .to_string()
}
