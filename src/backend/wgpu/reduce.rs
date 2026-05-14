use crate::backend::BackendError;
use crate::backend::wgpu::context::with_wgpu_context;

pub(super) fn dispatch_reduce_gpu(
    input: &[f32],
    group_size: usize,
    is_mean: usize,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader = build_reduce_shader();
        super::pipeline::ensure_compute_pipeline(ctx, "reduce", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "rd_input");
        let num_groups = input.len().checked_div(group_size).unwrap_or(1);
        let output_size = (num_groups * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rd_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RdParams {
            num_groups: u32,
            group_size: u32,
            is_mean: u32,
            _pad: u32,
        }
        let params = RdParams {
            num_groups: num_groups as u32,
            group_size: group_size as u32,
            is_mean: is_mean as u32,
            _pad: 0,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "rd_params");

        let pipeline_key = "wgpu_backend_reduce";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc = (num_groups as u32).div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rd_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })
}

fn build_reduce_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct RdParams {
    num_groups: u32,
    group_size: u32,
    is_mean: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: RdParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= params.num_groups) { return; }
    let start = g * params.group_size;
    let end = min(start + params.group_size, arrayLength(&input));

    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + input[j];
    }
    if (params.is_mean == 1u) {
        sum = sum / f32(end - start);
    }
    output[g] = sum;
}
"#
    .to_string()
}
