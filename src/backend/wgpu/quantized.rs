//! Quantized matmul and conv2d GPU dispatch for packed types.
//!
//! Supports I4x8, I8x4, F4x8 (E2M1), F8x4 (E4M3), and F8x4R (E5M2).
//!
//! Each GPU invocation computes one output element `(m, n)`:
//!   1. Reads the packed weight word for output channel `n`
//!   2. Unpacks to f32 (signed nibble/byte or FP4/FP8 decode)
//!   3. Dequantizes per-channel (scale ± zero_point for integer, scale-only for float)
//!   4. Dot products with the corresponding f32 activation segment
//!   5. Writes the result to the output
//!
//! The conv2d path performs im2col on the CPU and then reuses the same
//! quantized-matmul GPU pipeline.

use super::PendingRead;
use crate::backend::wgpu::context::WgpuContext;
use crate::backend::BackendError;
use std::sync::OnceLock;

// ─============================================================================
// Quantized matmul dispatch (GPU)
// ─============================================================================

/// Dispatch quantized matmul on the GPU.
///
/// `dtype_tag` selects the packed format: `"i4"`, `"i8"`, `"f4"`, `"f8"`, `"f8r"`.
/// `params = [M, K, N]`
/// `input_slices[0]` — f32 activations, shape `[M, K]`
/// `input_slices[1]` — packed weights, shape `[N, K]` (packed row-major)
pub(super) fn dispatch_quantized_matmul_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    params: &[usize],
    dtype_tag: &str,
    scales: &[f32],
    zero_points: &[f32],
) -> Result<(), BackendError> {
    let m = params.first().copied().unwrap_or(1);
    let k = params.get(1).copied().unwrap_or(1);
    let n = params.get(2).copied().unwrap_or(1);
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let items = if matches!(dtype_tag, "i4" | "f4") {
        8usize
    } else {
        4usize
    };
    let padded_k = k.div_ceil(items) * items;

    let read_f32 = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let activations = read_f32(0);
    if activations.len() < m * k {
        return Err(BackendError::Dispatch(format!(
            "quantized_matmul: activations len {} < M*K {}",
            activations.len(),
            m * k
        )));
    }

    let mut act_padded = activations;
    act_padded.resize(m * padded_k, 0.0f32);

    let packed_bytes = if 1 < input_slices.len() {
        let s = &input_slices[1];
        arena.data_mut()[s.offset..s.offset + s.size].to_vec()
    } else {
        return Err(BackendError::Dispatch(
            "quantized_matmul: missing packed weight input".into(),
        ));
    };

    let output_size = m * n * 4;
    dispatch_quantized_gemm_gpu(
        ctx,
        encoder,
        pending_reads,
        &act_padded,
        &packed_bytes,
        m,
        padded_k,
        n,
        items,
        dtype_tag,
        scales,
        zero_points,
        output_slice.offset,
        output_size,
    )
}

// ─============================================================================
// Quantized conv2d dispatch (im2col on CPU + GPU quantized matmul)
// ─============================================================================

/// Dispatch quantized conv2d via CPU im2col + GPU quantized matmul.
///
/// `dtype_tag` selects the packed format: `"i4"`, `"i8"`, `"f4"`.
/// `params = [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]`
/// `input_slices[0]` — f32 activations, shape `[N, C, H, W]`
/// `input_slices[1]` — packed weights, shape `[OC, C*KH*KW]` (packed row-major)
pub(super) fn dispatch_quantized_conv_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    params: &[usize],
    dtype_tag: &str,
    scales: &[f32],
    zero_points: &[f32],
) -> Result<(), BackendError> {
    if params.len() < 9 {
        return Err(BackendError::Dispatch(
            "quantized_conv: expected params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into(),
        ));
    }
    let stride = params[0];
    let padding = params[1];
    let dilation = params[2];
    let groups = params[3].max(1);
    let input_c = params[4];
    let input_h = params[5];
    let input_w = params[6];
    let kernel_h = params[7];
    let kernel_w = params[8];

    let read_f32 = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let input_data = read_f32(0);
    if input_data.is_empty() {
        return Err(BackendError::Dispatch(
            "quantized_conv: empty activation input".into(),
        ));
    }

    let packed_bytes = if 1 < input_slices.len() {
        let s = &input_slices[1];
        let raw = &arena.data_mut()[s.offset..s.offset + s.size];
        let mut aligned = vec![0u8; raw.len().div_ceil(4) * 4];
        aligned[..raw.len()].copy_from_slice(raw);
        aligned
    } else {
        return Err(BackendError::Dispatch(
            "quantized_conv: missing packed weight input".into(),
        ));
    };

    let c = input_c;
    let h = input_h;
    let w = input_w;
    let n_batch = input_data.len() / (c * h * w).max(1);
    let c_per_group = c / groups;

    let dk_h = (kernel_h - 1) * dilation + 1;
    let dk_w = (kernel_w - 1) * dilation + 1;
    let output_h = if h + 2 * padding >= dk_h {
        (h + 2 * padding - dk_h) / stride + 1
    } else {
        0
    };
    let output_w = if w + 2 * padding >= dk_w {
        (w + 2 * padding - dk_w) / stride + 1
    } else {
        0
    };
    if output_h == 0 || output_w == 0 {
        return Ok(());
    }

    let items = if matches!(dtype_tag, "i4" | "f4") {
        8usize
    } else {
        4usize
    };
    let col_w = c * kernel_h * kernel_w;
    let col_h = n_batch * output_h * output_w;
    let bit_width = if matches!(dtype_tag, "i4" | "f4") {
        4
    } else {
        8
    };
    let f = packed_bytes.len() * 8 / (col_w * bit_width).max(1);
    if f == 0 {
        return Err(BackendError::Dispatch(
            "quantized_conv: computed zero output channels".into(),
        ));
    }

    // CPU im2col: [N, C, H, W] → [N*H_out*W_out, C*KH*KW]
    let mut col_matrix = vec![0.0f32; col_h * col_w];
    for nn in 0..n_batch {
        for hh in 0..output_h {
            for ww in 0..output_w {
                let row = (nn * output_h + hh) * output_w + ww;
                for g in 0..groups {
                    for cc in 0..c_per_group {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let h_in = (hh * stride + kh * dilation) as i64 - padding as i64;
                                let w_in = (ww * stride + kw * dilation) as i64 - padding as i64;
                                if h_in >= 0 && h_in < h as i64 && w_in >= 0 && w_in < w as i64 {
                                    let src = nn * (c * h * w)
                                        + (g * c_per_group + cc) * (h * w)
                                        + h_in as usize * w
                                        + w_in as usize;
                                    let dst = row * col_w
                                        + (g * c_per_group + cc) * (kernel_h * kernel_w)
                                        + kh * kernel_w
                                        + kw;
                                    if src < input_data.len() && dst < col_matrix.len() {
                                        col_matrix[dst] = input_data[src];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let padded_k = col_w.div_ceil(items) * items;
    let output_size = col_h * f * 4;
    dispatch_quantized_gemm_gpu(
        ctx,
        encoder,
        pending_reads,
        &col_matrix,
        &packed_bytes,
        col_h,
        padded_k,
        f,
        items,
        dtype_tag,
        scales,
        zero_points,
        output_slice.offset,
        output_size,
    )
}

// ─============================================================================
// Cached quantized shader sources
// ─============================================================================

macro_rules! define_cached_shader {
    ($name:ident, $items:expr, $dtype:expr) => {
        fn $name() -> &'static str {
            static S: OnceLock<String> = OnceLock::new();
            if let Some(s) = S.get() {
                super::pipeline::record_shader_hit();
                s
            } else {
                S.get_or_init(|| {
                    super::pipeline::record_shader_miss();
                    build_quantized_matmul_shader_inner($items, $dtype)
                })
            }
        }
    };
}

define_cached_shader!(cached_quantized_i4_shader, 8, "i4");
define_cached_shader!(cached_quantized_i8_shader, 4, "i8");
define_cached_shader!(cached_quantized_f4_shader, 8, "f4");
define_cached_shader!(cached_quantized_f8_shader, 4, "f8");
define_cached_shader!(cached_quantized_f8r_shader, 4, "f8r");

/// Return the cached shader for the given dtype tag.
fn cached_quantized_shader(dtype_tag: &str) -> &'static str {
    match dtype_tag {
        "i4" => cached_quantized_i4_shader(),
        "i8" => cached_quantized_i8_shader(),
        "f4" => cached_quantized_f4_shader(),
        "f8" => cached_quantized_f8_shader(),
        "f8r" => cached_quantized_f8r_shader(),
        _ => cached_quantized_i8_shader(),
    }
}

// ─============================================================================
// Shared GPU quantized GEMM dispatch
// ─============================================================================

/// Internal GPU dispatch for quantized GEMM.
///
/// `dtype_tag` selects the unpack/dequantize formula:
///   `"i4"`, `"i8"` — signed integer with zero_point
///   `"f4"`, `"f8"`, `"f8r"` — float decode, scale-only (no zero_point)
///
/// Uploads data to the GPU, runs the quantized matmul shader,
/// and pushes a deferred readback into `pending_reads`.
#[allow(clippy::too_many_arguments)]
fn dispatch_quantized_gemm_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    activations: &[f32],
    packed_weights: &[u8],
    m: usize,
    k_padded: usize,
    n: usize,
    items: usize,
    dtype_tag: &str,
    scales: &[f32],
    zero_points: &[f32],
    cpu_offset: usize,
    output_bytes: usize,
) -> Result<(), BackendError> {
    let k_packed = k_padded / items;
    let is_float = matches!(dtype_tag, "f4" | "f8" | "f8r");

    let scale_data: Vec<f32> = if scales.is_empty() {
        vec![1.0f32; n]
    } else {
        let mut sd = vec![1.0f32; n];
        let copy_len = n.min(scales.len());
        sd[..copy_len].copy_from_slice(&scales[..copy_len]);
        sd
    };

    // For float types, zero_point is unused but the shader still reads the buffer.
    // Use all-zeros (harmless: `w * a * s + 0 * ...`).
    let zp_data: Vec<f32> = if zero_points.is_empty() || is_float {
        vec![0.0f32; n]
    } else {
        let mut zd = vec![0.0f32; n];
        let copy_len = n.min(zero_points.len());
        zd[..copy_len].copy_from_slice(&zero_points[..copy_len]);
        zd
    };

    let shader_src = cached_quantized_shader(dtype_tag);
    let short_key = format!("quantized_matmul_{}", dtype_tag);
    super::pipeline::ensure_quantized_compute_pipeline(ctx, &short_key, shader_src)
        .map_err(BackendError::Dispatch)?;
    let pipeline_key = format!("wgpu_backend_quantized_{}", short_key);
    let pipeline = &ctx.pipelines[&pipeline_key];

    let buf_act = ctx.create_buffer(bytemuck::cast_slice(activations), "qgemm_act");
    let buf_weight = ctx.create_buffer(packed_weights, "qgemm_weight");
    let buf_scale = ctx.create_buffer(bytemuck::cast_slice(&scale_data), "qgemm_scale");
    let buf_zp = ctx.create_buffer(bytemuck::cast_slice(&zp_data), "qgemm_zp");

    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qgemm_output"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    #[allow(non_snake_case)]
    struct QuantParams {
        M: u32,
        K: u32,
        N: u32,
        K_packed: u32,
    }
    let qparams = QuantParams {
        M: m as u32,
        K: k_padded as u32,
        N: n as u32,
        K_packed: k_packed as u32,
    };
    let buf_params = ctx.create_uniform_buffer(&qparams, "qgemm_params");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qgemm_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_act.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_scale.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_zp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let wgc_x = (m as u32).div_ceil(16);
    let wgc_y = (n as u32).div_ceil(16);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qgemm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset,
        size: output_bytes,
    });
    Ok(())
}

// ─============================================================================
// Quantized GEMM WGSL shader
// ─============================================================================

/// Build the quantized matmul WGSL shader with per-channel dequantization.
///
/// `items` = 8 for 4-bit types (I4x8, F4x8), 4 for 8-bit types (I8x4, F8x4, F8x4R).
/// `dtype_tag` selects the unpack function and dequantization formula:
///   `"i4"` — signed 4-bit nibble unpack, `(w - zp) * a * s` formula
///   `"i8"` — signed 8-bit byte unpack, `(w - zp) * a * s` formula
///   `"f4"` — FP4 E2M1 decode, `w * a * s` formula (no zp)
///   `"f8"` — FP8 E4M3 decode, `w * a * s` formula
///   `"f8r"` — FP8 E5M2 decode, `w * a * s` formula
///
/// Workgroup size 16×16 — each invocation computes one `(m, n)` output element.
/// Bindings:
///   0: packed weights (array<u32>)
///   1: activations   (array<f32>)
///   2: scales        (array<f32>)
///   3: zero_points   (array<f32>)
///   4: output        (array<f32>)
///   5: params        (uniform QuantParams)
fn build_quantized_matmul_shader_inner(items: usize, dtype_tag: &str) -> String {
    let unpack_fn = match dtype_tag {
        "i4" => {
            r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 4u;
    let nibble = (word >> shift) & 0xFu;
    let is_neg = nibble >= 8u;
    let val = select(nibble, nibble - 16u, is_neg);
    return f32(val);
}
"#
        }
        "i8" => {
            r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    let is_neg = byte >= 128u;
    let val = select(byte, byte - 256u, is_neg);
    return f32(val);
}
"#
        }
        "f4" => {
            r#"
const FP4_MAG: array<f32, 8> = array<f32, 8>(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0);
fn unpack_fp4(byte: u32) -> f32 {
    let nib = byte & 0xFu;
    let mag_idx = nib & 7u;
    let sign = nib >> 3u;
    let mag = FP4_MAG[mag_idx];
    return select(mag, -mag, sign != 0u && nib != 0u);
}
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 4u;
    let byte = (word >> shift) & 0xFFu;
    return unpack_fp4(byte);
}
"#
        }
        "f8" => {
            r#"
fn unpack_fp8_e4m3(byte: u32) -> f32 {
    let sign = byte & 0x80u;
    let exp = (byte >> 3u) & 0xFu;
    let mant = byte & 0x7u;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign != 0u);
    }
    if (exp == 0xFu) {
        if (mant == 0u) {
            return select(1.0 / 0.0, -1.0 / 0.0, sign != 0u);
        }
        return 0.0 / 0.0;
    }
    if (exp == 0u) {
        let v = 0.015625 * f32(mant) / 8.0;
        return select(v, -v, sign != 0u);
    }
    let v = exp2(f32(i32(exp) - 7)) * (1.0 + f32(mant) / 8.0);
    return select(v, -v, sign != 0u);
}
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    return unpack_fp8_e4m3(byte);
}
"#
        }
        "f8r" => {
            r#"
fn unpack_fp8_e5m2(byte: u32) -> f32 {
    let sign = byte & 0x80u;
    let exp = (byte >> 2u) & 0x1Fu;
    let mant = byte & 0x3u;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign != 0u);
    }
    if (exp == 0x1Fu) {
        if (mant == 0u) {
            return select(1.0 / 0.0, -1.0 / 0.0, sign != 0u);
        }
        return 0.0 / 0.0;
    }
    if (exp == 0u) {
        let v = exp2(-14.0) * f32(mant) / 4.0;
        return select(v, -v, sign != 0u);
    }
    let v = exp2(f32(i32(exp) - 15)) * (1.0 + f32(mant) / 4.0);
    return select(v, -v, sign != 0u);
}
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    return unpack_fp8_e5m2(byte);
}
"#
        }
        _ => {
            r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    let is_neg = byte >= 128u;
    let val = select(byte, byte - 256u, is_neg);
    return f32(val);
}
"#
        }
    };

    let is_float = matches!(dtype_tag, "f4" | "f8" | "f8r");
    let dequant_formula = if is_float {
        "acc += w_val * a_val * scale;"
    } else {
        "acc += (w_val - zp) * a_val * scale;"
    };

    format!(
        r#"
struct QuantParams {{
    M: u32,
    K: u32,
    N: u32,
    K_packed: u32,
}}

@group(0) @binding(0) var<storage, read>       weights:     array<u32>;
@group(0) @binding(1) var<storage, read>       activations: array<f32>;
@group(0) @binding(2) var<storage, read>       scales:      array<f32>;
@group(0) @binding(3) var<storage, read>       zero_points: array<f32>;
@group(0) @binding(4) var<storage, read_write> output:      array<f32>;
@group(0) @binding(5) var<uniform>             params:      QuantParams;

const ITEMS: u32 = {items}u;

{unpack_fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let m = gid.x;
    let n = gid.y;
    if (m >= params.M || n >= params.N) {{
        return;
    }}

    let scale = scales[n];
    let zp = zero_points[n];

    var acc: f32 = 0.0;
    for (var k_word: u32 = 0u; k_word < params.K_packed; k_word = k_word + 1u) {{
        let weight_word = weights[n * params.K_packed + k_word];
        for (var i: u32 = 0u; i < ITEMS; i = i + 1u) {{
            let k_idx = k_word * ITEMS + i;
            if (k_idx >= params.K) {{
                break;
            }}
            let w_val = unpack_word(weight_word, i);
            let a_val = activations[m * params.K + k_idx];
            {dequant_formula}
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        items = items,
        unpack_fn = unpack_fn,
        dequant_formula = dequant_formula,
    )
}

// ─============================================================================
// Gradient quantization / dequantization shaders (F8x4R)
// ─============================================================================

/// Cached quantize-gradient WGSL shader: f32 → F8x4R (E5M2).
fn cached_quantize_gradient_shader() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    if let Some(s) = S.get() {
        s
    } else {
        S.get_or_init(|| {
            r#"
struct GradQuantParams {
    numel: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform>             params: GradQuantParams;

fn f32_to_e5m2(val: f32) -> u32 {
    if (val == 0.0) {
        return select(0x00u, 0x80u, val < 0.0);
    }
    let abs_val = abs(val);
    if (abs_val >= 57344.0) {
        let inf_bits = 0x7Cu;
        return select(inf_bits, inf_bits | 0x80u, val < 0.0);
    }
    let exp_raw = i32(log2(abs_val));
    let exp_biased = exp_raw + 15;
    if (exp_biased <= 0) {
        let mant = u32(abs_val * 67108864.0);
        return select(mant & 0x3u, (mant & 0x3u) | 0x80u, val < 0.0);
    }
    let mant_val = abs_val / exp2(f32(exp_biased - 15)) - 1.0;
    let mant_bits = u32(mant_val * 4.0) & 0x3u;
    let exp_bits = u32(exp_biased) & 0x1Fu;
    let byte = (exp_bits << 2u) | mant_bits;
    return select(byte, byte | 0x80u, val < 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * 4u;
    if (idx >= params.numel) { return; }
    var v: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = idx + i;
        v[i] = select(0.0, input[j], j < params.numel);
    }
    var packed: u32 = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        packed = packed | (f32_to_e5m2(v[i]) << (i * 8u));
    }
    output[gid.x] = packed;
}
"#
            .to_string()
        })
    }
}

/// Cached dequantize-gradient WGSL shader: F8x4R → f32.
fn cached_dequantize_gradient_shader() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    if let Some(s) = S.get() {
        s
    } else {
        S.get_or_init(|| {
            r#"
struct GradQuantParams {
    numel: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: GradQuantParams;

fn e5m2_to_f32(byte: u32) -> f32 {
    let sign = byte & 0x80u;
    let exp = (byte >> 2u) & 0x1Fu;
    let mant = byte & 0x3u;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign != 0u);
    }
    if (exp == 0x1Fu) {
        if (mant == 0u) {
            return select(1.0 / 0.0, -1.0 / 0.0, sign != 0u);
        }
        return 0.0 / 0.0;
    }
    if (exp == 0u) {
        let v = exp2(-14.0) * f32(mant) / 4.0;
        return select(v, -v, sign != 0u);
    }
    let v = exp2(f32(i32(exp) - 15)) * (1.0 + f32(mant) / 4.0);
    return select(v, -v, sign != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    let word = input[word_idx];
    for (var i = 0u; i < 4u; i = i + 1u) {
        let byte = (word >> (i * 8u)) & 0xFFu;
        let idx = word_idx * 4u + i;
        if (idx < params.numel) {
            output[idx] = e5m2_to_f32(byte);
        }
    }
}
"#
            .to_string()
        })
    }
}

/// Dispatch F8x4R gradient quantization (f32 → packed) on GPU.
pub(super) fn dispatch_quantize_gradient_gpu(
    ctx: &mut super::WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<super::PendingRead>,
    arena: &super::WgpuBuffer,
    input_slice: crate::backend::BufferSlice,
    output_slice: crate::backend::BufferSlice,
    numel: usize,
) -> Result<(), BackendError> {
    let shader_src = cached_quantize_gradient_shader();
    let short_key = "gradient_quantize_f8x4r";
    super::pipeline::ensure_simple_compute_pipeline(ctx, short_key, shader_src)
        .map_err(BackendError::Dispatch)?;
    let pipeline_key = format!("wgpu_simple_{}", short_key);
    let pipeline = &ctx.pipelines[&pipeline_key];

    let buf_input = {
        let d = arena.data_mut();
        let data = &d[input_slice.offset..input_slice.offset + input_slice.size];
        ctx.create_buffer(data, "gq_input")
    };
    let buf_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gq_output"),
        size: output_slice.size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GradQuantParams {
        numel: u32,
        _pad: u32,
    }
    let qp = GradQuantParams {
        numel: numel as u32,
        _pad: 0,
    };
    let buf_params = ctx.create_uniform_buffer(&qp, "gq_params");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gq_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let num_workgroups = (numel.div_ceil(4) as u32).div_ceil(64);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gq_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(num_workgroups.max(1), 1, 1);

    pending_reads.push(super::PendingRead {
        buffer: buf_output,
        cpu_offset: output_slice.offset,
        size: output_slice.size,
    });
    Ok(())
}

/// Dispatch F8x4R gradient dequantization (packed → f32) on GPU.
pub(super) fn dispatch_dequantize_gradient_gpu(
    ctx: &mut super::WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<super::PendingRead>,
    arena: &super::WgpuBuffer,
    input_slice: crate::backend::BufferSlice,
    output_slice: crate::backend::BufferSlice,
    numel: usize,
) -> Result<(), BackendError> {
    let shader_src = cached_dequantize_gradient_shader();
    let short_key = "gradient_dequantize_f8x4r";
    super::pipeline::ensure_simple_compute_pipeline(ctx, short_key, shader_src)
        .map_err(BackendError::Dispatch)?;
    let pipeline_key = format!("wgpu_simple_{}", short_key);
    let pipeline = &ctx.pipelines[&pipeline_key];

    let buf_input = {
        let d = arena.data_mut();
        let data = &d[input_slice.offset..input_slice.offset + input_slice.size];
        ctx.create_buffer(data, "gdq_input")
    };
    let buf_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gdq_output"),
        size: output_slice.size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GradQuantParams {
        numel: u32,
        _pad: u32,
    }
    let qp = GradQuantParams {
        numel: numel as u32,
        _pad: 0,
    };
    let buf_params = ctx.create_uniform_buffer(&qp, "gdq_params");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gdq_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let num_workgroups = (numel.div_ceil(4) as u32).div_ceil(64);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gdq_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(num_workgroups.max(1), 1, 1);

    pending_reads.push(super::PendingRead {
        buffer: buf_output,
        cpu_offset: output_slice.offset,
        size: output_slice.size,
    });
    Ok(())
}
