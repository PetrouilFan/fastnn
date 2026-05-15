use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

pub(super) fn dispatch_pool_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    resolved_params: &[usize],
) -> Result<(), BackendError> {
    let kernel_size = resolved_params.first().copied().unwrap_or(2);
    let stride = resolved_params.get(1).copied().unwrap_or(2);
    let padding = resolved_params.get(2).copied().unwrap_or(0);
    let is_max = resolved_params.get(3).copied().unwrap_or(0);

    let read_input = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let input_data = read_input(0);
    let input_len = input_data.len();
    let output_len = output_slice.size / 4;

    let (n, c, h, w) = infer_pool_dims(input_len, output_len, kernel_size, stride, padding)
        .ok_or_else(|| BackendError::Dispatch("pool: could not infer dimensions".into()))?;

    let h_out = (h + 2 * padding).saturating_sub(kernel_size) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(kernel_size) / stride + 1;

    let expected_output_len = n * c * h_out * w_out;
    if expected_output_len != output_len {
        return Err(BackendError::Dispatch(format!(
            "pool: output size mismatch: expected {}, got {}",
            expected_output_len, output_len
        )));
    }

    with_wgpu_context(|ctx| -> Result<(), BackendError> {
        let shader = build_pool_shader();
        super::pipeline::ensure_compute_pipeline(ctx, "pool", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_input = ctx.create_buffer(bytemuck::cast_slice(&input_data), "pool_input");

        let output_size = (output_len * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        #[allow(non_snake_case)]
        struct PoolParams {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            H_out: u32,
            W_out: u32,
            kernel_size: u32,
            stride: u32,
            padding: u32,
            is_max: u32,
        }
        let params = PoolParams {
            N: n as u32,
            C: c as u32,
            H: h as u32,
            W: w as u32,
            H_out: h_out as u32,
            W_out: w_out as u32,
            kernel_size: kernel_size as u32,
            stride: stride as u32,
            padding: padding as u32,
            is_max: is_max as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "pool_params");

        let pipeline_key = "wgpu_backend_pool";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_input.as_entire_binding(),
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

        let wgc_x = h_out.div_ceil(8).max(1) as u32;
        let wgc_y = w_out.div_ceil(8).max(1) as u32;
        let wgc_z = (c * n).max(1) as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pool_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pool_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x, wgc_y, wgc_z);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
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

fn infer_pool_dims(
    input_len: usize,
    output_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Option<(usize, usize, usize, usize)> {
    for &n in &[1, 2, 4, 8, 16, 32] {
        if !input_len.is_multiple_of(n) || !output_len.is_multiple_of(n) {
            continue;
        }
        let nc_hw = input_len / n;
        let nc_hout_wout = output_len / n;
        for &c in &[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            if !nc_hw.is_multiple_of(c) || !nc_hout_wout.is_multiple_of(c) {
                continue;
            }
            let hw = nc_hw / c;
            let hout_wout = nc_hout_wout / c;
            for h in 1..=hw {
                if !hw.is_multiple_of(h) {
                    continue;
                }
                let w = hw / h;
                let h_out = (h + 2 * padding).saturating_sub(kernel_size) / stride + 1;
                let w_out = (w + 2 * padding).saturating_sub(kernel_size) / stride + 1;
                if h_out * w_out == hout_wout && h_out > 0 && w_out > 0 {
                    return Some((n, c, h, w));
                }
            }
        }
    }
    None
}

fn build_pool_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       _dummy: array<f32>;
@group(0) @binding(2) var<storage, read_write>  output: array<f32>;

struct PoolParams {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    H_out: u32,
    W_out: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    is_max: u32,
}
@group(0) @binding(3) var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h_out = gid.x;
    let w_out = gid.y;
    let cf = gid.z;
    if (h_out >= params.H_out || w_out >= params.W_out) {
        return;
    }
    let c = cf % params.C;
    let n = cf / params.C;

    if (params.is_max == 1u) {
        var max_val: f32 = -3.402823e+38;
        for (var kh: u32 = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let h_in = h_out * params.stride + kh;
                let w_in = w_out * params.stride + kw;
                if (h_in >= params.padding && w_in >= params.padding) {
                    let h_in_s = h_in - params.padding;
                    let w_in_s = w_in - params.padding;
                    if (h_in_s < params.H && w_in_s < params.W) {
                        let idx = n * (params.C * params.H * params.W) + c * (params.H * params.W) + h_in_s * params.W + w_in_s;
                        max_val = max(max_val, input[idx]);
                    }
                }
            }
        }
        let out_idx = n * (params.C * params.H_out * params.W_out) + c * (params.H_out * params.W_out) + h_out * params.W_out + w_out;
        output[out_idx] = max_val;
    } else {
        var sum: f32 = 0.0;
        var count: u32 = 0u;
        for (var kh: u32 = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let h_in = h_out * params.stride + kh;
                let w_in = w_out * params.stride + kw;
                if (h_in >= params.padding && w_in >= params.padding) {
                    let h_in_s = h_in - params.padding;
                    let w_in_s = w_in - params.padding;
                    if (h_in_s < params.H && w_in_s < params.W) {
                        let idx = n * (params.C * params.H * params.W) + c * (params.H * params.W) + h_in_s * params.W + w_in_s;
                        sum = sum + input[idx];
                        count = count + 1u;
                    }
                }
            }
        }
        let out_idx = n * (params.C * params.H_out * params.W_out) + c * (params.H_out * params.W_out) + h_out * params.W_out + w_out;
        if (count > 0u) {
            output[out_idx] = sum / f32(count);
        } else {
            output[out_idx] = 0.0;
        }
    }
}
"#
    .to_string()
}
