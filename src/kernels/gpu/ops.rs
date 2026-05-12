//! GPU kernel ops for v2.0.0 WGPU backend.
//! Called by the WGPU backend's compile() step.

#![allow(dead_code)]

use crate::kernels::gpu::GpuContext;
use crate::storage::DType;
use wgpu::Buffer;

// ============================================================================
// WGSL Shaders
// ============================================================================

const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var a_vec = vec4<f32>(a[vec_idx], a[vec_idx + 1u], a[vec_idx + 2u], a[vec_idx + 3u]);
        var b_vec = vec4<f32>(b[vec_idx], b[vec_idx + 1u], b[vec_idx + 2u], b[vec_idx + 3u]);
        var out_vec = a_vec + b_vec;
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = a[idx] + b[idx];
    }
}
"#;

const SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var a_vec = vec4<f32>(a[vec_idx], a[vec_idx + 1u], a[vec_idx + 2u], a[vec_idx + 3u]);
        var b_vec = vec4<f32>(b[vec_idx], b[vec_idx + 1u], b[vec_idx + 2u], b[vec_idx + 3u]);
        var out_vec = a_vec - b_vec;
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = a[idx] - b[idx];
    }
}
"#;

const MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var a_vec = vec4<f32>(a[vec_idx], a[vec_idx + 1u], a[vec_idx + 2u], a[vec_idx + 3u]);
        var b_vec = vec4<f32>(b[vec_idx], b[vec_idx + 1u], b[vec_idx + 2u], b[vec_idx + 3u]);
        var out_vec = a_vec * b_vec;
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = a[idx] * b[idx];
    }
}
"#;

const DIV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var a_vec = vec4<f32>(a[vec_idx], a[vec_idx + 1u], a[vec_idx + 2u], a[vec_idx + 3u]);
        var b_vec = vec4<f32>(b[vec_idx], b[vec_idx + 1u], b[vec_idx + 2u], b[vec_idx + 3u]);
        var out_vec = a_vec / b_vec;
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = a[idx] / b[idx];
    }
}
"#;

const NEG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = -in_vec;
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = -input[idx];
    }
}
"#;

const ABS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = abs(in_vec);
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = abs(input[idx]);
    }
}
"#;

const EXP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = vec4<f32>(exp(in_vec.x), exp(in_vec.y), exp(in_vec.z), exp(in_vec.w));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = exp(input[idx]);
    }
}
"#;

const LOG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = vec4<f32>(log(in_vec.x), log(in_vec.y), log(in_vec.z), log(in_vec.w));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = log(input[idx]);
    }
}
"#;

const SQRT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = sqrt(in_vec);
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = sqrt(input[idx]);
    }
}
"#;

const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = max(in_vec, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = max(input[idx], 0.0);
    }
}
"#;

const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var x_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var x3_vec = x_vec * x_vec * x_vec;
        var in_arg_vec = vec4<f32>(0.7978846, 0.7978846, 0.7978846, 0.7978846) * (x_vec + vec4<f32>(0.044715, 0.044715, 0.044715, 0.044715) * x3_vec);
        var t_vec = tanh(in_arg_vec);
        var out_vec = vec4<f32>(0.5, 0.5, 0.5, 0.5) * x_vec * (vec4<f32>(1.0, 1.0, 1.0, 1.0) + t_vec);
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        let x = input[idx];
        let x3 = x * x * x;
        let in_arg = 0.7978846 * (x + 0.044715 * x3);
        let t = tanh(in_arg);
        output[idx] = 0.5 * x * (1.0 + t);
    }
}
"#;

const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = 1.0 / (1.0 + exp(-in_vec));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        let x = input[idx];
        output[idx] = 1.0 / (1.0 + exp(-x));
    }
}
"#;

const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = tanh(in_vec);
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        output[idx] = tanh(input[idx]);
    }
}
"#;

const SILU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var in_vec = vec4<f32>(input[vec_idx], input[vec_idx + 1u], input[vec_idx + 2u], input[vec_idx + 3u]);
        var out_vec = in_vec / (1.0 + exp(-in_vec));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        let x = input[idx];
        output[idx] = x / (1.0 + exp(-x));
    }
}
"#;

const FUSED_ADD_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_idx = idx * 4u;
    
    if (vec_idx + 3u < arrayLength(&output)) {
        var a_vec = vec4<f32>(a[vec_idx], a[vec_idx + 1u], a[vec_idx + 2u], a[vec_idx + 3u]);
        var b_vec = vec4<f32>(b[vec_idx], b[vec_idx + 1u], b[vec_idx + 2u], b[vec_idx + 3u]);
        var sum_vec = a_vec + b_vec;
        var out_vec = max(sum_vec, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        output[vec_idx] = out_vec.x;
        output[vec_idx + 1u] = out_vec.y;
        output[vec_idx + 2u] = out_vec.z;
        output[vec_idx + 3u] = out_vec.w;
    } else if (idx < arrayLength(&output)) {
        let sum = a[idx] + b[idx];
        output[idx] = max(sum, 0.0);
    }
}
"#;

const MATMUL_SHADER: &str = r#"
struct Params {
    m: u32,
    n: u32,
    k: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    
    let tile_size = 16u;

    if (row >= m || col >= n) { return; }

    var sum = 0.0;
    
    // Loop over tiles
    for (var tile = 0u; tile < k; tile = tile + tile_size) {
        // Load tile from A into shared memory
        let a_row = row;
        let a_col = tile + local_id.x;
        if (a_row < m && a_col < k) {
            tileA[local_id.y][local_id.x] = a[a_row * k + a_col];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }
        
        // Load tile from B into shared memory
        let b_row = tile + local_id.y;
        let b_col = col;
        if (b_row < k && b_col < n) {
            tileB[local_id.y][local_id.x] = b[b_row * n + b_col];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute dot product from shared memory
        for (var i = 0u; i < tile_size; i = i + 1u) {
            sum += tileA[local_id.y][i] * tileB[i][local_id.x];
        }
        
        workgroupBarrier();
    }

    let out_idx = row * n + col;
    output[out_idx] = sum;
}
"#;

const SUM_REDUCE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    // Each workgroup computes one output element
    // For now, assume 2D tensor (m, n) and reduce along dim 1 (sum columns)
    // This is a simplified version for the benchmark
}
"#;

const MEAN_REDUCE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
}
"#;

const MAX_REDUCE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
}
"#;

const MIN_REDUCE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
}
"#;

const TRANSPOSE_SHADER: &str = r#"
struct Params {
    ndim: u32,
    numel: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> strides: array<u32>;
@group(0) @binding(3) var<storage, read> new_strides: array<u32>;
@group(0) @binding(4) var<storage, read> dims: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }

    var remaining = idx;
    var out_idx: u32 = 0u;
    for (var d = 0u; d < params.ndim; d = d + 1u) {
        let dim_size = dims[d];
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        out_idx = out_idx + coord * new_strides[d];
    }

    output[idx] = input[out_idx];
}
"#;

const GT_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    output[idx] = select(0.0, 1.0, input[idx] > params.scalar);
}
"#;

const LT_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    output[idx] = select(0.0, 1.0, input[idx] < params.scalar);
}
"#;

const LOGICAL_NOT_SHADER: &str = r#"
struct Params {
    numel: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    output[idx] = select(1.0, 0.0, input[idx] != 0.0);
}
"#;

const MUL_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    let vec_idx = idx * 4u;
    if (vec_idx + 3u < params.numel) {
        let s = vec4<f32>(params.scalar, params.scalar, params.scalar, params.scalar);
        let v = vec4<f32>(input[vec_idx], input[vec_idx+1u], input[vec_idx+2u], input[vec_idx+3u]);
        let out = v * s;
        output[vec_idx] = out.x;
        output[vec_idx+1u] = out.y;
        output[vec_idx+2u] = out.z;
        output[vec_idx+3u] = out.w;
    } else if (idx < params.numel) {
        output[idx] = input[idx] * params.scalar;
    }
}
"#;

const ADD_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    let vec_idx = idx * 4u;
    if (vec_idx + 3u < params.numel) {
        let s = vec4<f32>(params.scalar, params.scalar, params.scalar, params.scalar);
        let v = vec4<f32>(input[vec_idx], input[vec_idx+1u], input[vec_idx+2u], input[vec_idx+3u]);
        let out = v + s;
        output[vec_idx] = out.x;
        output[vec_idx+1u] = out.y;
        output[vec_idx+2u] = out.z;
        output[vec_idx+3u] = out.w;
    } else if (idx < params.numel) {
        output[idx] = input[idx] + params.scalar;
    }
}
"#;

const SUB_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    let vec_idx = idx * 4u;
    if (vec_idx + 3u < params.numel) {
        let s = vec4<f32>(params.scalar, params.scalar, params.scalar, params.scalar);
        let v = vec4<f32>(input[vec_idx], input[vec_idx+1u], input[vec_idx+2u], input[vec_idx+3u]);
        let out = v - s;
        output[vec_idx] = out.x;
        output[vec_idx+1u] = out.y;
        output[vec_idx+2u] = out.z;
        output[vec_idx+3u] = out.w;
    } else if (idx < params.numel) {
        output[idx] = input[idx] - params.scalar;
    }
}
"#;

const DIV_SCALAR_SHADER: &str = r#"
struct Params {
    numel: u32,
    scalar: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) { return; }
    let vec_idx = idx * 4u;
    if (vec_idx + 3u < params.numel) {
        let s = vec4<f32>(params.scalar, params.scalar, params.scalar, params.scalar);
        let v = vec4<f32>(input[vec_idx], input[vec_idx+1u], input[vec_idx+2u], input[vec_idx+3u]);
        let out = v / s;
        output[vec_idx] = out.x;
        output[vec_idx+1u] = out.y;
        output[vec_idx+2u] = out.z;
        output[vec_idx+3u] = out.w;
    } else if (idx < params.numel) {
        output[idx] = input[idx] / params.scalar;
    }
}
"#;

const EMBEDDING_SHADER: &str = r#"
struct Params {
    numel: u32,
    embedding_dim: u32,
    num_embeddings: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let flat_idx = global_id.x;
    if (flat_idx >= params.numel) { return; }

    let embedding_dim = params.embedding_dim;
    let num_embeddings = params.num_embeddings;
    let col = flat_idx % embedding_dim;
    let row = flat_idx / embedding_dim;
    let idx = u32(indices[row]);
    if (idx < num_embeddings) {
        output[flat_idx] = weight[idx * embedding_dim + col];
    } else {
        output[flat_idx] = 0.0;
    }
}
"#;

// ============================================================================
// Internal dispatch helpers
// ============================================================================

fn alloc_output(ctx: &GpuContext, numel: usize) -> Buffer {
    ctx.acquire_buffer(numel * 4)
}

fn dispatch_unary(ctx: &GpuContext, input: &Buffer, output: &Buffer, numel: usize, shader: &str, name: &str) {
    let pipeline = ctx.create_pipeline(name, shader, DType::F32);
    let bind_group = ctx.get_or_create_bind_group(
        &pipeline,
        name,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    );
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (numel as u64).div_ceil(256) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);
}

fn dispatch_binary(ctx: &GpuContext, a: &Buffer, b: &Buffer, output: &Buffer, numel: usize, shader: &str, name: &str) {
    let pipeline = ctx.create_pipeline(name, shader, DType::F32);
    let bind_group = ctx.get_or_create_bind_group(
        &pipeline,
        name,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    );
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (numel as u64).div_ceil(256) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);
}

fn dispatch_scalar(ctx: &GpuContext, input: &Buffer, output: &Buffer, scalar: f32, numel: usize, shader: &str, name: &str) {
    let params_data: Vec<u32> = vec![numel as u32, scalar.to_bits(), 0, 0];
    let params_buffer = ctx.create_uniform_buffer_u32(&params_data, "params");

    let pipeline = ctx.create_pipeline(name, shader, DType::F32);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (numel as u64).div_ceil(256) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);
}

/// Helper to compute output numel after reduction along a dimension.
fn reduction_output_numel(shape: &[usize], dim: usize, keepdim: bool) -> usize {
    let dim = if dim >= shape.len() { shape.len() - 1 } else { dim };
    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    out_shape.iter().product()
}

/// Generate and dispatch a reduction shader for 2D input (reduce along dim 1).
fn dispatch_reduction_2d(ctx: &GpuContext, input: &Buffer, output: &Buffer, m: usize, n: usize, op: &str, name: &str) {
    let shader = match op {
        "sum" => format!(
            r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let row = global_id.x;
                if (row >= {}) {{ return; }}
                
                var sum = 0.0;
                for (var col = 0u; col < {}; col = col + 1u) {{
                    let idx = row * {} + col;
                    sum += input[idx];
                }}
                output[row] = sum;
            }}
            "#,
            m, n, n
        ),
        "mean" => format!(
            r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let row = global_id.x;
                if (row >= {}) {{ return; }}
                
                var sum = 0.0;
                for (var col = 0u; col < {}; col = col + 1u) {{
                    let idx = row * {} + col;
                    sum += input[idx];
                }}
                output[row] = sum / {};
            }}
            "#,
            m, n, n, n as f32
        ),
        "max" => format!(
            r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let row = global_id.x;
                if (row >= {}) {{ return; }}
                
                var max_val = -3.4028235e38;
                for (var col = 0u; col < {}; col = col + 1u) {{
                    let idx = row * {} + col;
                    let val = input[idx];
                    if (val > max_val) {{
                        max_val = val;
                    }}
                }}
                output[row] = max_val;
            }}
            "#,
            m, n, n
        ),
        "min" => format!(
            r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let row = global_id.x;
                if (row >= {}) {{ return; }}
                
                var min_val = 3.4028235e38;
                for (var col = 0u; col < {}; col = col + 1u) {{
                    let idx = row * {} + col;
                    let val = input[idx];
                    if (val < min_val) {{
                        min_val = val;
                    }}
                }}
                output[row] = min_val;
            }}
            "#,
            m, n, n
        ),
        _ => panic!("Unknown reduction op: {}", op),
    };

    let pipeline = ctx.create_pipeline(name, &shader, DType::F32);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (m as u64).div_ceil(256) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);
}

/// Generate and dispatch an N-dimensional reduction shader.
#[allow(clippy::too_many_arguments)]
fn dispatch_reduction_nd(
    ctx: &GpuContext,
    input: &Buffer,
    output: &Buffer,
    shape: &[usize],
    dim: usize,
    op: &str,
    name: &str,
) {
    let output_numel: usize = shape[..dim].iter().product::<usize>() * shape[dim + 1..].iter().product::<usize>();
    let reduce_size: usize = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let reduce_stride: usize = inner_size;

    let shader = match op {
        "sum" => format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let out_idx = global_id.x;
    if (out_idx >= {output_numel}) {{ return; }}

    let outer_idx = out_idx / {inner_size};
    let inner_idx = out_idx % {inner_size};
    let base_idx = outer_idx * {reduce_size} * {reduce_stride} + inner_idx;

    var result = 0.0;
    for (var i = 0u; i < {reduce_size}; i = i + 1u) {{
        let idx = base_idx + i * {reduce_stride};
        result += input[idx];
    }}
    output[out_idx] = result;
}}
"#,
            output_numel = output_numel,
            reduce_size = reduce_size,
            reduce_stride = reduce_stride,
            inner_size = inner_size,
        ),
        "mean" => format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let out_idx = global_id.x;
    if (out_idx >= {output_numel}) {{ return; }}

    let outer_idx = out_idx / {inner_size};
    let inner_idx = out_idx % {inner_size};
    let base_idx = outer_idx * {reduce_size} * {reduce_stride} + inner_idx;

    var result = 0.0;
    for (var i = 0u; i < {reduce_size}; i = i + 1u) {{
        let idx = base_idx + i * {reduce_stride};
        result += input[idx];
    }}
    output[out_idx] = result / {reduce_size_f};
}}
"#,
            output_numel = output_numel,
            reduce_size = reduce_size,
            reduce_stride = reduce_stride,
            inner_size = inner_size,
            reduce_size_f = reduce_size as f32,
        ),
        "max" => format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let out_idx = global_id.x;
    if (out_idx >= {output_numel}) {{ return; }}

    let outer_idx = out_idx / {inner_size};
    let inner_idx = out_idx % {inner_size};
    let base_idx = outer_idx * {reduce_size} * {reduce_stride} + inner_idx;

    var result = -3.4028235e38;
    for (var i = 0u; i < {reduce_size}; i = i + 1u) {{
        let idx = base_idx + i * {reduce_stride};
        let val = input[idx];
        if (val > result) {{
            result = val;
        }}
    }}
    output[out_idx] = result;
}}
"#,
            output_numel = output_numel,
            reduce_size = reduce_size,
            reduce_stride = reduce_stride,
            inner_size = inner_size,
        ),
        "min" => format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let out_idx = global_id.x;
    if (out_idx >= {output_numel}) {{ return; }}

    let outer_idx = out_idx / {inner_size};
    let inner_idx = out_idx % {inner_size};
    let base_idx = outer_idx * {reduce_size} * {reduce_stride} + inner_idx;

    var result = 3.4028235e38;
    for (var i = 0u; i < {reduce_size}; i = i + 1u) {{
        let idx = base_idx + i * {reduce_stride};
        let val = input[idx];
        if (val < result) {{
            result = val;
        }}
    }}
    output[out_idx] = result;
}}
"#,
            output_numel = output_numel,
            reduce_size = reduce_size,
            reduce_stride = reduce_stride,
            inner_size = inner_size,
        ),
        _ => panic!("Unknown reduction op: {}", op),
    };

    let pipeline = ctx.create_pipeline(name, &shader, DType::F32);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (output_numel as u64).div_ceil(256) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);
}

// ============================================================================
// Public API: Element-wise binary ops
// ============================================================================

pub fn gpu_add(ctx: &GpuContext, a: &Buffer, b: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_binary(ctx, a, b, &output, numel, ADD_SHADER, "add");
    output
}

pub fn gpu_sub(ctx: &GpuContext, a: &Buffer, b: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_binary(ctx, a, b, &output, numel, SUB_SHADER, "sub");
    output
}

pub fn gpu_mul(ctx: &GpuContext, a: &Buffer, b: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_binary(ctx, a, b, &output, numel, MUL_SHADER, "mul");
    output
}

pub fn gpu_div(ctx: &GpuContext, a: &Buffer, b: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_binary(ctx, a, b, &output, numel, DIV_SHADER, "div");
    output
}

// ============================================================================
// Public API: Element-wise unary ops
// ============================================================================

pub fn gpu_neg(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, NEG_SHADER, "neg");
    output
}

pub fn gpu_abs(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, ABS_SHADER, "abs");
    output
}

pub fn gpu_exp(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, EXP_SHADER, "exp");
    output
}

pub fn gpu_log(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, LOG_SHADER, "log");
    output
}

pub fn gpu_sqrt(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, SQRT_SHADER, "sqrt");
    output
}

pub fn gpu_relu(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, RELU_SHADER, "relu");
    output
}

pub fn gpu_gelu(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, GELU_SHADER, "gelu");
    output
}

pub fn gpu_sigmoid(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, SIGMOID_SHADER, "sigmoid");
    output
}

pub fn gpu_tanh(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, TANH_SHADER, "tanh");
    output
}

pub fn gpu_silu(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_unary(ctx, input, &output, numel, SILU_SHADER, "silu");
    output
}

// ============================================================================
// Public API: Fused ops
// ============================================================================

pub fn gpu_fused_add_relu(ctx: &GpuContext, a: &Buffer, b: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_binary(ctx, a, b, &output, numel, FUSED_ADD_RELU_SHADER, "fused_add_relu");
    output
}

// ============================================================================
// Public API: Matmul
// ============================================================================

pub fn gpu_matmul(ctx: &GpuContext, a: &Buffer, b: &Buffer, m: usize, k: usize, n: usize) -> Buffer {
    let output = alloc_output(ctx, m * n);

    let params_data: Vec<u32> = vec![m as u32, n as u32, k as u32, 0];
    let params_buffer = ctx.create_uniform_buffer_u32(&params_data, "matmul_params");

    let pipeline = ctx.create_pipeline("matmul", MATMUL_SHADER, DType::F32);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
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
        let x_groups = n.div_ceil(8).min(65535);
        let y_groups = m.div_ceil(8).min(65535);
        compute_pass.dispatch_workgroups(x_groups as u32, y_groups as u32, 1);
    }
    ctx.queue().submit([encoder.finish()]);

    output
}

// ============================================================================
// Public API: Reduction ops (sum, mean, max, min)
// ============================================================================

pub fn gpu_sum(ctx: &GpuContext, input: &Buffer, shape: &[usize], dim: usize, keepdim: bool) -> Buffer {
    let output_numel = reduction_output_numel(shape, dim, keepdim);
    let output = alloc_output(ctx, output_numel);

    let dim = if dim >= shape.len() { shape.len() - 1 } else { dim };

    if shape.len() == 2 && dim == 1 {
        dispatch_reduction_2d(ctx, input, &output, shape[0], shape[1], "sum", "sum_reduce");
    } else {
        dispatch_reduction_nd(ctx, input, &output, shape, dim, "sum", "sum_reduce");
    }

    output
}

pub fn gpu_mean(ctx: &GpuContext, input: &Buffer, shape: &[usize], dim: usize, keepdim: bool) -> Buffer {
    let output_numel = reduction_output_numel(shape, dim, keepdim);
    let output = alloc_output(ctx, output_numel);

    let dim = if dim >= shape.len() { shape.len() - 1 } else { dim };

    if shape.len() == 2 && dim == 1 {
        dispatch_reduction_2d(ctx, input, &output, shape[0], shape[1], "mean", "mean_reduce");
    } else {
        dispatch_reduction_nd(ctx, input, &output, shape, dim, "mean", "mean_reduce");
    }

    output
}

pub fn gpu_max(ctx: &GpuContext, input: &Buffer, shape: &[usize], dim: usize, keepdim: bool) -> Buffer {
    let output_numel = reduction_output_numel(shape, dim, keepdim);
    let output = alloc_output(ctx, output_numel);

    let dim = if dim >= shape.len() { shape.len() - 1 } else { dim };

    if shape.len() == 2 && dim == 1 {
        dispatch_reduction_2d(ctx, input, &output, shape[0], shape[1], "max", "max_reduce");
    } else {
        dispatch_reduction_nd(ctx, input, &output, shape, dim, "max", "max_reduce");
    }

    output
}

pub fn gpu_min(ctx: &GpuContext, input: &Buffer, shape: &[usize], dim: usize, keepdim: bool) -> Buffer {
    let output_numel = reduction_output_numel(shape, dim, keepdim);
    let output = alloc_output(ctx, output_numel);

    let dim = if dim >= shape.len() { shape.len() - 1 } else { dim };

    if shape.len() == 2 && dim == 1 {
        dispatch_reduction_2d(ctx, input, &output, shape[0], shape[1], "min", "min_reduce");
    } else {
        dispatch_reduction_nd(ctx, input, &output, shape, dim, "min", "min_reduce");
    }

    output
}

// ============================================================================
// Public API: Scalar comparison / arithmetic ops
// ============================================================================

pub fn gpu_gt_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, GT_SCALAR_SHADER, "gt_scalar");
    output
}

pub fn gpu_lt_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, LT_SCALAR_SHADER, "lt_scalar");
    output
}

pub fn gpu_logical_not(ctx: &GpuContext, input: &Buffer, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    // LOGICAL_NOT_SHADER uses params.numel but ignores scalar field
    dispatch_scalar(ctx, input, &output, 0.0, numel, LOGICAL_NOT_SHADER, "logical_not");
    output
}

pub fn gpu_mul_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, MUL_SCALAR_SHADER, "mul_scalar");
    output
}

pub fn gpu_add_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, ADD_SCALAR_SHADER, "add_scalar");
    output
}

pub fn gpu_sub_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, SUB_SCALAR_SHADER, "sub_scalar");
    output
}

pub fn gpu_div_scalar(ctx: &GpuContext, input: &Buffer, scalar: f32, numel: usize) -> Buffer {
    let output = alloc_output(ctx, numel);
    dispatch_scalar(ctx, input, &output, scalar, numel, DIV_SCALAR_SHADER, "div_scalar");
    output
}

// ============================================================================
// Public API: Transpose
// ============================================================================

pub fn gpu_transpose(
    ctx: &GpuContext,
    input: &Buffer,
    strides_buf: &Buffer,
    new_strides_buf: &Buffer,
    dims_buf: &Buffer,
    ndim: u32,
    numel: u32,
) -> Buffer {
    let output = alloc_output(ctx, numel as usize);

    let params_data: Vec<u32> = vec![ndim, numel, 0, 0];
    let params_buffer = ctx.create_uniform_buffer_u32(&params_data, "transpose_params");

    let pipeline = ctx.create_pipeline("transpose", TRANSPOSE_SHADER, DType::F32);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("transpose"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: strides_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: new_strides_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transpose"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transpose"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (numel as u64).div_ceil(64) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);

    output
}

// ============================================================================
// Public API: Embedding
// ============================================================================

pub fn gpu_embedding(
    ctx: &GpuContext,
    weight: &Buffer,
    indices: &Buffer,
    num_embeddings: u32,
    embedding_dim: u32,
    numel: usize,
) -> Buffer {
    let output = alloc_output(ctx, numel);

    let params_data: Vec<u32> = vec![numel as u32, embedding_dim, num_embeddings, 0];
    let params_buffer = ctx.create_uniform_buffer_u32(&params_data, "embedding_params");

    let pipeline = ctx.create_pipeline("embedding", EMBEDDING_SHADER, DType::F32);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embedding"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embedding"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embedding"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let num_workgroups = (numel as u64).div_ceil(64) as u32;
        let x_groups = num_workgroups.min(65535);
        compute_pass.dispatch_workgroups(x_groups, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);

    output
}
