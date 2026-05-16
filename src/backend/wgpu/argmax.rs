use crate::backend::wgpu::context::WgpuContext;
use crate::backend::wgpu::pipeline::ensure_compute_pipeline;
use crate::backend::{BackendError, BufferSlice};
use super::PendingRead;

pub(super) fn dispatch_argmax_gpu(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &super::WgpuBuffer,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
) -> Result<(), BackendError> {
    let out_start = output_slice.offset;

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

    let shader = build_argmax_shader();
    ensure_compute_pipeline(ctx, "argmax", &shader).map_err(BackendError::Dispatch)?;

    let buf_in = ctx.create_buffer(bytemuck::cast_slice(&input), "am_input");
    let output_size = 8u64;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("am_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AmParams {
        numel: u32,
        _pad: u32,
    }
    let params = AmParams {
        numel: numel as u32,
        _pad: 0,
    };
    let buf_params = ctx.create_uniform_buffer(&params, "am_params");

    let pipeline_key = "wgpu_backend_argmax";
    let pipeline = &ctx.pipelines[pipeline_key];
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("am_bg"),
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

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("am_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pending_reads.push(PendingRead {
        buffer: buf_out,
        cpu_offset: out_start,
        size: output_size as usize,
    });
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

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Each thread scans multiple elements, keeping its local max
    var local_max_val = -3.402823e+38;
    var local_max_idx = 0u;
    var j = lid.x;
    while j < params.numel {
        if input[j] > local_max_val {
            local_max_val = input[j];
            local_max_idx = j;
        }
        j = j + 256u;
    }

    // Write local max to shared memory
    shared_val[lid.x] = local_max_val;
    shared_idx[lid.x] = local_max_idx;
    workgroupBarrier();

    // Tree reduction in shared memory
    var stride: u32 = 128u;
    while stride > 0u {
        if lid.x < stride {
            if shared_val[lid.x + stride] > shared_val[lid.x] {
                shared_val[lid.x] = shared_val[lid.x + stride];
                shared_idx[lid.x] = shared_idx[lid.x + stride];
            }
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }

    // Thread 0 writes the global result
    if lid.x == 0u {
        output[0] = u64(shared_idx[0]);
    }
}
"#
    .to_string()
}
