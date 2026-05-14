use crate::backend::{BackendError, BufferSlice};
use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::wgpu::pipeline::ensure_compute_pipeline;

pub(super) fn dispatch_argmax_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
) -> Result<(), BackendError> {
    let out_start = output_slice.offset;
    let out_end = output_slice.offset + output_slice.size;

    let input = {
        let d = arena.data_mut();
        if let Some(s) = input_slices.first() {
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            return Err(BackendError::Dispatch("argmax: no input".to_string()));
        }
    };

    let numel = input.len();
    if numel == 0 {
        return Ok(());
    }

    let result = with_wgpu_context(|ctx| -> Result<Vec<u64>, BackendError> {
        let shader = build_argmax_shader();
        ensure_compute_pipeline(ctx, "argmax", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(&input), "am_input");
        let output_size = (numel as u64) * 8;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("am_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct AmParams { numel: u32, _pad: u32 }
        let params = AmParams { numel: numel as u32, _pad: 0 };
        let buf_params = ctx.create_uniform_buffer(&params, "am_params");

        let pipeline_key = "wgpu_backend_argmax";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("am_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc = 1u32;
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("am_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("am_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })?;

    let out = arena.data_mut();
    let out_u64 = bytemuck::cast_slice_mut::<_, u64>(&mut out[out_start..out_end]);
    let len = out_u64.len().min(result.len());
    out_u64[..len].copy_from_slice(&result[..len]);
    Ok(())
}

fn build_argmax_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u64>;

struct AmParams {
    numel: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: AmParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }

    if (i == 0u) {
        var max_val: f32 = input[0];
        var max_idx: u64 = 0u64;
        for (var j: u32 = 1u; j < params.numel; j = j + 1u) {
            if (input[j] > max_val) {
                max_val = input[j];
                max_idx = u64(j);
            }
        }
        output[0] = max_idx;
    }
}
"#
    .to_string()
}
