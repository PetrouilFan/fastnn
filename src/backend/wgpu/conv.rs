use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

pub(super) fn dispatch_conv_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    resolved_params: &[usize],
    _shape_env: &crate::ir::node::ShapeEnv,
) -> Result<(), BackendError> {
    let stride = resolved_params.first().copied().unwrap_or(1);
    let padding = resolved_params.get(1).copied().unwrap_or(0);
    let dilation = resolved_params.get(2).copied().unwrap_or(1);
    let groups = resolved_params.get(3).copied().unwrap_or(1);

    let read_slice = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let input_data = read_slice(0);
    let weight_data = read_slice(1);

    let ni = input_data.len();
    let nw = weight_data.len();
    let no = output_slice.size / 4;

    let (n, oc, c, kh, kw, h, w) =
        infer_conv_dims(ni, nw, no, stride, padding, dilation, groups)
            .ok_or_else(|| BackendError::Dispatch("conv2d: could not infer dimensions".into()))?;

    let h_out = (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;

    with_wgpu_context(|ctx| -> Result<(), BackendError> {
        let shader = build_conv_shader();
        super::pipeline::ensure_compute_pipeline(ctx, "conv2d", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_input = ctx.create_buffer(bytemuck::cast_slice(&input_data), "conv_input");
        let buf_weight = ctx.create_buffer(bytemuck::cast_slice(&weight_data), "conv_weight");

        let output_size = (n * oc * h_out * w_out * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        #[allow(non_snake_case)]
        struct ConvParams {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            OC: u32,
            KH: u32,
            KW: u32,
            stride: u32,
            padding: u32,
            dilation: u32,
            groups: u32,
            H_out: u32,
            W_out: u32,
        }
        let params = ConvParams {
            N: n as u32,
            C: c as u32,
            H: h as u32,
            W: w as u32,
            OC: oc as u32,
            KH: kh as u32,
            KW: kw as u32,
            stride: stride as u32,
            padding: padding as u32,
            dilation: dilation as u32,
            groups: groups as u32,
            H_out: h_out as u32,
            W_out: w_out as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "conv_params");

        let pipeline_key = "wgpu_backend_conv2d";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_weight.as_entire_binding(),
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

        let wgc_x = h_out.div_ceil(8).max(1);
        let wgc_y = w_out.div_ceil(8).max(1);
        let wgc_z = (oc * n).max(1);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x as u32, wgc_y as u32, wgc_z as u32);
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

fn infer_conv_dims(
    ni: usize,
    nw: usize,
    no: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Option<(usize, usize, usize, usize, usize, usize, usize)> {
    for &n in &[1, 2, 4, 8, 16, 32, 64] {
        if !ni.is_multiple_of(n) || !no.is_multiple_of(n) {
            continue;
        }
        let c_hw = ni / n;
        let f_hout_wout = no / n;
        for cg in 1..=c_hw.min(4096) {
            let c = cg * groups;
            if !c_hw.is_multiple_of(c) {
                continue;
            }
            let hw = c_hw / c;
            if !nw.is_multiple_of(cg) {
                continue;
            }
            let f_kh_kw = nw / cg;
            for f in 1..=f_hout_wout.min(f_kh_kw) {
                if !f_hout_wout.is_multiple_of(f) || !f_kh_kw.is_multiple_of(f) {
                    continue;
                }
                let hout_wout = f_hout_wout / f;
                let kh_kw = f_kh_kw / f;
                for &kh in &[1, 3, 5, 7, 11] {
                    if kh > kh_kw || !kh_kw.is_multiple_of(kh) {
                        continue;
                    }
                    let kw = kh_kw / kh;
                    for h in 1..=hw {
                        if !hw.is_multiple_of(h) {
                            continue;
                        }
                        let w = hw / h;
                        let h_out =
                            (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
                        let w_out =
                            (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;
                        if h_out * w_out == hout_wout && h_out > 0 && w_out > 0 {
                            return Some((n, f, c, kh, kw, h, w));
                        }
                    }
                }
            }
        }
    }
    None
}

fn build_conv_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>         input:  array<f32>;
@group(0) @binding(1) var<storage, read>         weight: array<f32>;
@group(0) @binding(2) var<storage, read_write>   output: array<f32>;

struct ConvParams {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    OC: u32,
    KH: u32,
    KW: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    H_out: u32,
    W_out: u32,
}
@group(0) @binding(3) var<uniform> params: ConvParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h_out = gid.x;
    let w_out = gid.y;
    let of = gid.z;
    if (h_out >= params.H_out || w_out >= params.W_out) {
        return;
    }
    let oc = of % params.OC;
    let n = of / params.OC;

    let C_per_group = params.C / params.groups;
    let oc_per_group = params.OC / params.groups;
    let g = oc / oc_per_group;
    let c_base = g * C_per_group;

    var sum: f32 = 0.0;
    for (var cc: u32 = 0u; cc < C_per_group; cc = cc + 1u) {
        for (var kkh: u32 = 0u; kkh < params.KH; kkh = kkh + 1u) {
            for (var kkw: u32 = 0u; kkw < params.KW; kkw = kkw + 1u) {
                let h_in = h_out * params.stride + kkh * params.dilation;
                let w_in = w_out * params.stride + kkw * params.dilation;
                if (h_in >= params.padding && w_in >= params.padding) {
                    let h_in_s = h_in - params.padding;
                    let w_in_s = w_in - params.padding;
                    if (h_in_s < params.H && w_in_s < params.W) {
                        let input_c = c_base + cc;
                        let input_idx = n * (params.C * params.H * params.W) + input_c * (params.H * params.W) + h_in_s * params.W + w_in_s;
                        let weight_idx = oc * C_per_group * params.KH * params.KW + cc * params.KH * params.KW + kkh * params.KW + kkw;
                        sum = sum + input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    let out_idx = n * (params.OC * params.H_out * params.W_out) + oc * (params.H_out * params.W_out) + h_out * params.W_out + w_out;
    output[out_idx] = sum;
}
"#
    .to_string()
}
