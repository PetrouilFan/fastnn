use crate::backend::BackendError;
use crate::backend::wgpu::context::with_wgpu_context;

pub(super) fn dispatch_transpose_gpu(
    input: &[f32],
    m: usize,
    n: usize,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader = build_transpose_shader();
        super::pipeline::ensure_compute_pipeline(ctx, "transpose", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "tp_input");
        let output_size = (m * n * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tp_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TpParams {
            m: u32,
            n: u32,
        }
        let params = TpParams {
            m: m as u32,
            n: n as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "tp_params");

        let pipeline_key = "wgpu_backend_transpose";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tp_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc_x = (m as u32).div_ceil(16);
        let wgc_y = (n as u32).div_ceil(16);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tp_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })
}

fn build_transpose_shader() -> String {
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
