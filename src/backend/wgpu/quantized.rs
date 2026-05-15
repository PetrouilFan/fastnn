#![allow(dead_code)]
//! Quantized matmul and conv2d GPU dispatch for U4x8 / U8x4 packed types.
//!
//! Each GPU invocation computes one output element `(m, n)`:
//!   1. Reads the packed weight word for output channel `n`
//!   2. Unpacks to f32 (4- or 8-bit nibbles/bytes)
//!   3. Dequantizes per-channel (scale, zero_point)
//!   4. Dot products with the corresponding f32 activation segment
//!   5. Writes the result to the output
//!
//! The conv2d path performs im2col on the CPU and then reuses the same
//! quantized-matmul GPU pipeline.

use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

// ─============================================================================
// Quantized matmul dispatch (GPU)
// ─============================================================================

/// Dispatch `matmul_u4` or `matmul_u8` on the GPU.
///
/// `params = [M, K, N]`
/// `input_slices[0]` — f32 activations, shape `[M, K]`
/// `input_slices[1]` — packed u4/u8 weights, shape `[N, K]` (packed row-major)
pub(super) fn dispatch_quantized_matmul_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    params: &[usize],
    bit_width: usize,
    scales: &[f32],
    zero_points: &[f32],
) -> Result<(), BackendError> {
    let m = params.first().copied().unwrap_or(1);
    let k = params.get(1).copied().unwrap_or(1);
    let n = params.get(2).copied().unwrap_or(1);
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let items = if bit_width == 4 { 8usize } else { 4usize };
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

    let result = dispatch_quantized_gemm_gpu(
        &act_padded,
        &packed_bytes,
        m,
        padded_k,
        n,
        items,
        bit_width,
        scales,
        zero_points,
    )?;

    let out = arena.data_mut();
    let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
        &mut out[output_slice.offset..output_slice.offset + output_slice.size],
    );
    let len = out_f32.len().min(result.len());
    out_f32[..len].copy_from_slice(&result[..len]);

    Ok(())
}

// ─============================================================================
// Quantized conv2d dispatch (im2col on CPU + GPU quantized matmul)
// ─============================================================================

/// Dispatch `conv2d_u4` or `conv2d_u8` via CPU im2col + GPU quantized matmul.
///
/// `params = [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]`
/// `input_slices[0]` — f32 activations, shape `[N, C, H, W]`
/// `input_slices[1]` — packed u4/u8 weights, shape `[OC, C*KH*KW]` (packed row-major)
pub(super) fn dispatch_quantized_conv_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    params: &[usize],
    bit_width: usize,
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

    let items = if bit_width == 4 { 8usize } else { 4usize };
    let col_w = c * kernel_h * kernel_w;
    let col_h = n_batch * output_h * output_w;
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
    let result = dispatch_quantized_gemm_gpu(
        &col_matrix,
        &packed_bytes,
        col_h,
        padded_k,
        f,
        items,
        bit_width,
        scales,
        zero_points,
    )?;

    let out = arena.data_mut();
    let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
        &mut out[output_slice.offset..output_slice.offset + output_slice.size],
    );
    let len = out_f32.len().min(result.len());
    out_f32[..len].copy_from_slice(&result[..len]);

    Ok(())
}

// ─============================================================================
// Shared GPU quantized GEMM dispatch
// ─============================================================================

/// Internal GPU dispatch for quantized GEMM.
///
/// Uploads the given data to the GPU, runs the quantized matmul shader,
/// and returns the f32 output as a `Vec<f32>`.
fn dispatch_quantized_gemm_gpu(
    activations: &[f32],   // [M, K_padded]
    packed_weights: &[u8], // packed weight bytes [N * K_packed * 4]
    m: usize,
    k_padded: usize,
    n: usize,
    items: usize,
    bit_width: usize,
    scales: &[f32],
    zero_points: &[f32],
) -> Result<Vec<f32>, BackendError> {
    let k_packed = k_padded / items;
    let output_bytes = m * n * 4;

    let scale_data: Vec<f32> = if scales.is_empty() {
        vec![1.0f32; n]
    } else {
        let mut sd = vec![1.0f32; n];
        let copy_len = n.min(scales.len());
        sd[..copy_len].copy_from_slice(&scales[..copy_len]);
        sd
    };

    let zp_data: Vec<f32> = if zero_points.is_empty() {
        vec![0.0f32; n]
    } else {
        let mut zd = vec![0.0f32; n];
        let copy_len = n.min(zero_points.len());
        zd[..copy_len].copy_from_slice(&zero_points[..copy_len]);
        zd
    };

    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader_src = build_quantized_matmul_shader(items);
        let short_key = format!("quantized_matmul_u{}", bit_width);
        super::pipeline::ensure_quantized_compute_pipeline(ctx, &short_key, &shader_src)
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
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qgemm_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qgemm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_bytes);
        let result: &[f32] = bytemuck::cast_slice(&raw);
        Ok(result.to_vec())
    })
}

// ─============================================================================
// Quantized GEMM WGSL shader
// ─============================================================================

/// Build the quantized matmul WGSL shader with per-channel dequantization.
///
/// `items` = 8 for U4x8, 4 for U8x4.
///
/// Workgroup size 16×16 — each invocation computes one `(m, n)` output element.
/// Bindings:
///   0: packed weights (array<u32>)
///   1: activations   (array<f32>)
///   2: scales        (array<f32>)
///   3: zero_points   (array<f32>)
///   4: output        (array<f32>)
///   5: params        (uniform QuantParams)
fn build_quantized_matmul_shader(items: usize) -> String {
    let unpack_fn = if items == 8 {
        r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 4u;
    let nibble = (word >> shift) & 0xFu;
    let is_neg = nibble >= 8u;
    let val = select(nibble, nibble - 16u, is_neg);
    return f32(val);
}
"#
    } else {
        r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    let is_neg = byte >= 128u;
    let val = select(byte, byte - 256u, is_neg);
    return f32(val);
}
"#
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
            acc += (w_val - zp) * a_val * scale;
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        items = items,
        unpack_fn = unpack_fn,
    )
}
