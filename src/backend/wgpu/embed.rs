use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

pub(super) fn dispatch_embed_gpu(
    arena: &super::WgpuBuffer,
    input_slices: &[crate::backend::BufferSlice],
    output_slice: crate::backend::BufferSlice,
    _resolved_params: &[usize],
) -> Result<(), BackendError> {
    let indices_size = input_slices.first().map(|s| s.size).unwrap_or(0);
    if indices_size < 8 {
        return Ok(());
    }
    let num_indices = indices_size / 8;
    let numel_out = output_slice.size / 4;
    if numel_out < num_indices {
        return Err(BackendError::Dispatch(
            "embed: output buffer too small for indices".into(),
        ));
    }
    let embedding_dim = numel_out / num_indices;
    if embedding_dim == 0 {
        return Err(BackendError::Dispatch(
            "embed: embedding_dim is zero".into(),
        ));
    }

    let read_raw = |idx: usize| -> Vec<u8> {
        let s = &input_slices[idx];
        let d = arena.data_mut();
        d[s.offset..s.offset + s.size].to_vec()
    };

    let indices_raw = read_raw(0);
    let weight_raw = read_raw(1);

    with_wgpu_context(|ctx| -> Result<(), BackendError> {
        let shader = build_embed_shader();
        super::pipeline::ensure_compute_pipeline(ctx, "embed", &shader)
            .map_err(BackendError::Dispatch)?;

        let buf_indices = ctx.create_buffer(&indices_raw, "embed_indices");
        let buf_weight = ctx.create_buffer(&weight_raw, "embed_weight");

        let output_size = output_slice.size as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct EmbedParams {
            num_indices: u32,
            embedding_dim: u32,
            vocab_size: u32,
        }
        let vocab_size = weight_raw.len() / (embedding_dim * 4);
        let params = EmbedParams {
            num_indices: num_indices as u32,
            embedding_dim: embedding_dim as u32,
            vocab_size: vocab_size as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "embed_params");

        let pipeline_key = "wgpu_backend_embed";
        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_indices.as_entire_binding(),
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

        let wgc_x = (num_indices as u32).div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embed_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x.max(1), 1, 1);
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

fn build_embed_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     indices: array<i64>;
@group(0) @binding(1) var<storage, read>     weight:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct EmbedParams {
    num_indices: u32,
    embedding_dim: u32,
    vocab_size: u32,
}
@group(0) @binding(3) var<uniform> params: EmbedParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_indices) { return; }
    let idx = indices[i];
    let src_base = u32(idx) * params.embedding_dim;
    let dst_base = i * params.embedding_dim;
    for (var j = 0u; j < params.embedding_dim; j = j + 1u) {
        output[dst_base + j] = weight[src_base + j];
    }
}
"#
    .to_string()
}
