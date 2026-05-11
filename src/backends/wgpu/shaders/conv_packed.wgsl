// WGSL compute shader for packed convolution (U4/U8 version)
// Each invocation computes one output element: (oc, oh, ow)
// Packed weights are stored as u32 with per-channel scale.

struct Params {
    n: u32, c: u32, h: u32, w: u32,
    out_c: u32, out_h: u32, out_w: u32,
    kh: u32, kw: u32,
    stride: u32, pad: u32, dilation: u32,
    items_per_word: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;

const ITEMS_PER_WORD: u32 = 8u;

fn unpack_s4(val: u32) -> f32 {
    // Sign-extend 4-bit to f32
    var s = i32(val);
    if (s >= 8) { s = s - 16; }
    return f32(s);
}

fn unpack_s8(val: u32) -> f32 {
    // Sign-extend 8-bit to f32
    var s = i32(val);
    if (s >= 128) { s = s - 256; }
    return f32(s);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let oc = id.x;
    let oh = id.y;
    let ow = id.z;

    if (oc >= params.out_c || oh >= params.out_h || ow >= params.out_w) {
        return;
    }

    let oc_weights_base = oc * (params.c * params.kh * params.kw);
    let channel_scale = scales[oc];
    let channel_bias = bias[oc];

    var sum = 0.0f;
    for (var ic = 0u; ic < params.c; ic++) {
        for (var ky = 0u; ky < params.kh; ky++) {
            for (var kx = 0u; kx < params.kw; kx++) {
                let in_y = i32(oh * params.stride + ky * params.dilation) - i32(params.pad);
                let in_x = i32(ow * params.stride + kx * params.dilation) - i32(params.pad);

                if (in_y >= 0 && in_y < i32(params.h) && in_x >= 0 && in_x < i32(params.w)) {
                    let flat_idx = oc_weights_base + ic * (params.kh * params.kw) + ky * params.kw + kx;
                    let word_idx = flat_idx / ITEMS_PER_WORD;
                    let sub_idx = flat_idx % ITEMS_PER_WORD;
                    let packed_word = packed_weights[word_idx];

                    let shift = sub_idx * 4u;
                    let nibble = (packed_word >> shift) & 0xFu;
                    let q_val = unpack_s4(nibble);

                    let a = input[ic * (params.h * params.w) + u32(in_y) * params.w + u32(in_x)];
                    sum += q_val * a;
                }
            }
        }
    }

    let out_idx = oc * (params.out_h * params.out_w) + oh * params.out_w + ow;
    output[out_idx] = sum * channel_scale + channel_bias;
}
