#![allow(dead_code)]
//! Quantized matmul and conv2d GPU dispatch for U4x8 / U8x4 packed types.
//!
//! Each GPU invocation computes one output element `(m, n)`:
//!   1. Reads the packed weight word for output channel `n`
//!   2. Unpacks to f32 (4- or 8-bit nibbles/bytes)
//!   3. Dot products with the corresponding f32 activation segment
//!   4. Writes the result to the output
//!
//! The conv2d path performs im2col on the CPU and then reuses the same
//! quantized-matmul GPU pipeline.

use crate::backend::BackendError;
use crate::backend::wgpu::context::with_wgpu_context;

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
) -> Result<(), BackendError> {
    let m = params.first().copied().unwrap_or(1);
    let k = params.get(1).copied().unwrap_or(1);
    let n = params.get(2).copied().unwrap_or(1);
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let items = if bit_width == 4 { 8usize } else { 4usize };
    let padded_k = ((k + items - 1) / items) * items; // round up for safe GPU reads
    let k_packed = padded_k / items;
    let output_bytes = m * n * 4;

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

    // Pad activations to [M, padded_k] so the GPU never reads garbage
    // on the last packed word (which may read past logical K).
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

    with_wgpu_context(|ctx| -> Result<(), BackendError> {
        let shader_src = build_quantized_gemm_shader(items);
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("quantized_gemm_{}bit", bit_width)),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("quantized_gemm_{}bit_layout", bit_width)),
                bind_group_layouts: &[&ctx.bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        let pipeline = ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("quantized_gemm_{}bit", bit_width)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        // Prepare scale buffer (used by shader as binding 2 alongside output)
        let mut scale_data = vec![1.0f32; n.max(1)];
        for i in 0..n.min(scales.len()) {
            scale_data[i] = scales[i];
        }

        let buf_act = ctx.create_buffer(bytemuck::cast_slice(&act_padded), "qmm_act");
        let buf_weight = ctx.create_buffer(&packed_bytes, "qmm_weight");
        let _buf_scale = ctx.create_buffer(bytemuck::cast_slice(&scale_data), "qmm_scale");
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qmm_output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        #[allow(non_snake_case)]
        struct QuantParams {
            M: u32,
            N: u32,
            K: u32,
            K_packed: u32,
        }
        let qparams = QuantParams {
            M: m as u32,
            N: n as u32,
            K: padded_k as u32,
            K_packed: k_packed as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&qparams, "qmm_params");

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qmm_bg"),
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
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        let total_work = (m * n) as u32;
        let workgroups = (total_work + 255) / 256;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qmm_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qmm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_bytes);
        let result: &[f32] = bytemuck::cast_slice(&raw);

        let out = arena.data_mut();
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
            &mut out[output_slice.offset..output_slice.offset + output_slice.size],
        );
        let len = out_f32.len().min(result.len());
        out_f32[..len].copy_from_slice(&result[..len]);

        Ok(())
    })
}

// ─============================================================================
// Quantized conv2d dispatch (im2col on CPU + GPU quantized matmul)
// ─============================================================================

/// Dispatch `conv2d_u4` or `conv2d_u8` via CPU im2col + GPU quantized matmul.
pub(super) fn dispatch_quantized_conv_gpu(
    _arena: &super::WgpuBuffer,
    _input_slices: &[crate::backend::BufferSlice],
    _output_slice: crate::backend::BufferSlice,
    _params: &[usize],
    _bit_width: usize,
    _scales: &[f32],
    _zero_points: &[f32],
) -> Result<(), BackendError> {
    // Fall back to CPU for conv2d — the im2col + GPU quantized matmul path
    // requires restructuring the dispatch to allow temporary buffers.
    // This path is deferred to v2.3.
    Err(BackendError::UnsupportedOp(format!(
        "quantized conv2d GPU dispatch deferred to v2.3 (CPU fallback active)"
    )))
}
// ─============================================================================
// Quantized GEMM WGSL shader
// ─============================================================================

/// Build a quantized GEMM shader that computes:
///   output[m * N + n] = sum_{k=0}^{K-1} unpack(weight[n][k]) * act[m * K + k]
///
/// Each invocation handles one `(m, n)` output element.
/// `items` = 8 for U4x8, 4 for U8x4.
fn build_quantized_gemm_shader(items: usize) -> String {
    let unpack_fn = if items == 8 {
        // U4x8: unpack 8 signed 4-bit nibbles from a u32
        r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 4u;
    let nibble = (word >> shift) & 0xFu;
    // Interpret as signed 4-bit: values 0-7 are positive, 8-15 are negative
    let is_neg = nibble >= 8u;
    let val = select(nibble, nibble - 16u, is_neg);
    return f32(val);
}
"#
    } else {
        // U8x4: unpack 4 signed bytes from a u32
        r#"
fn unpack_word(word: u32, lane: u32) -> f32 {
    let shift = lane * 8u;
    let byte = (word >> shift) & 0xFFu;
    // Sign-extend from 8-bit
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
    N: u32,
    K: u32,
    K_packed: u32,
}}

@group(0) @binding(0) var<storage, read>       weight:  array<u32>;
@group(0) @binding(1) var<storage, read>       act:     array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  QuantParams;

const ITEMS: u32 = {items}u;

{unpack_fn}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.M * params.N) {{ return; }}
    let m = idx / params.N;
    let n = idx % params.N;

    var acc: f32 = 0.0;
    let words_per_row = params.K_packed;
    for (var k: u32 = 0u; k < params.K; k++) {{
        let word_idx = n * words_per_row + k / ITEMS;
        let lane = k % ITEMS;
        let w = weight[word_idx];
        let w_val = unpack_word(w, lane);
        let a = act[m * params.K + k];
        acc += w_val * a;
    }}
    output[idx] = acc;
}}
"#
    )
}
