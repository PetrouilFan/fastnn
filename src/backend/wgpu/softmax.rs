use crate::backend::wgpu::context::WgpuContext;
use crate::backend::BackendError;
use super::PendingRead;

pub(super) fn dispatch_softmax_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    input: &[f32],
    numel: usize,
    _axis: usize,
    cpu_offset: usize,
) -> Result<(), BackendError> {
    let shader = build_softmax_shader();
    super::pipeline::ensure_compute_pipeline(ctx, "softmax", &shader)
        .map_err(BackendError::Dispatch)?;

    let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "sf_input");
    let output_size = (numel * 4) as u64;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sf_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SfParams {
        numel: u32,
        row_size: u32,
    }
    let row_size = input.len().checked_div(numel).unwrap_or(1);
    let params = SfParams {
        numel: numel as u32,
        row_size: row_size as u32,
    };
    let buf_params = ctx.create_uniform_buffer(&params, "sf_params");

    let pipeline_key = "wgpu_backend_softmax";
    let pipeline = &ctx.pipelines[pipeline_key];
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sf_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_in.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let wgc = (numel as u32).div_ceil(256);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sf_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wgc, 1, 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset,
        size: output_size as usize,
    });
    Ok(())
}

fn build_softmax_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SfParams {
    numel: u32,
    row_size: u32,
}
@group(0) @binding(2) var<uniform> params: SfParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row * params.row_size >= params.numel) { return; }
    let start = row * params.row_size;
    let end = min(start + params.row_size, params.numel);

    // Find max
    var max_val: f32 = -3.402823e+38;
    for (var j: u32 = start; j < end; j = j + 1u) {
        max_val = max(max_val, input[j]);
    }

    // Compute sum of exp
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + exp(input[j] - max_val);
    }

    // Normalize
    for (var j: u32 = start; j < end; j = j + 1u) {
        output[j] = exp(input[j] - max_val) / sum;
    }
}
"#
    .to_string()
}
