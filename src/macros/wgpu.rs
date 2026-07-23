//! Macros for WGPU backend boilerplate.
//! Used by src/backend/wgpu/ to eliminate ~300 lines of
//! duplicated compute shader dispatch code.

/// Generate a compute shader dispatch function.
///
/// Generates `pub(super) fn $fn_name(ctx, encoder, pending_reads, $input, $arg1, $arg2, cpu_offset)`
/// that builds the shader, ensures the pipeline, creates input / output / uniform buffers,
/// sets up the bind group (3 entries: input, output, params), records a compute pass,
/// and pushes a deferred readback.
///
/// The `$input`, `$arg1`, `$arg2` identifiers must be plain identifier tokens that match
/// the variable names used in `$output_size`, `$params`, and the workgroup expressions.
/// In practice all callers pass `input, arg1, arg2` — these are the function parameter names
/// and the names used in the caller's expression fragments.
///
/// Two forms:
/// - **9‑arg** (1‑D dispatch) — `$wg_y` / `$wg_z` default to `1u32`.
/// - **11‑arg** (3‑D dispatch) — all three dispatch dimensions supplied.
///
/// # Example
///
/// ```ignore
/// dispatch_gpu_compute!(
///     dispatch_softmax_gpu,
///     build_softmax_shader(),
///     "softmax",
///     input, arg1, arg2,
///     (arg1 * 4) as u64,                 // output_size
///     SfParams { numel: arg1 as u32, .. }, // params
///     wg_x_expr,
/// );
/// ```
#[macro_export]
macro_rules! dispatch_gpu_compute {
    // ── 9-arg form (1-D dispatch) ──────────────────────────────────────
    (
        $fn_name:ident,
        $shader_builder:expr,
        $shader_key:literal,
        $input:ident, $arg1:ident, $arg2:ident,
        $output_size:expr,
        $params:expr,
        $wg_x:expr,
    ) => {
        $crate::dispatch_gpu_compute!(
            $fn_name,
            $shader_builder,
            $shader_key,
            $input,
            $arg1,
            $arg2,
            $output_size,
            $params,
            $wg_x,
            1u32,
            1u32,
        );
    };
    // ── 11-arg form (3-D dispatch) ─────────────────────────────────────
    (
        $fn_name:ident,
        $shader_builder:expr,
        $shader_key:literal,
        $input:ident, $arg1:ident, $arg2:ident,
        $output_size:expr,
        $params:expr,
        $wg_x:expr,
        $wg_y:expr,
        $wg_z:expr,
    ) => {
        pub(super) fn $fn_name(
            ctx: &mut $crate::backend::wgpu::context::WgpuContext,
            encoder: &mut wgpu::CommandEncoder,
            pending_reads: &mut Vec<$crate::backend::wgpu::PendingRead>,
            $input: &[f32],
            $arg1: usize,
            $arg2: usize,
            cpu_offset: usize,
        ) -> Result<(), $crate::backend::BackendError> {
            let shader = $shader_builder;
            $crate::backend::wgpu::pipeline::ensure_simple_compute_pipeline(
                ctx,
                $shader_key,
                &shader,
            )
            .map_err($crate::backend::BackendError::Dispatch)?;

            let buf_in = ctx
                .create_pooled_buffer(bytemuck::cast_slice($input), concat!($shader_key, "_input"));
            let output_size: u64 = $output_size.max(1); // WGPU requires non-zero buffer size
            let buf_out = ctx.acquire_buffer_for_size(output_size as usize);

            let params = $params;
            let buf_params = ctx.create_uniform_buffer(&params, concat!($shader_key, "_params"));

            let pipeline_key = concat!("wgpu_simple_", $shader_key);
            let pipeline = &ctx.pipelines[pipeline_key];
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(concat!($shader_key, "_bg")),
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

            let wgc_x = $wg_x;
            let wgc_y = $wg_y;
            let wgc_z = $wg_z;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(concat!($shader_key, "_pass")),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wgc_x, wgc_y, wgc_z);
            }

            pending_reads.push($crate::backend::wgpu::PendingRead {
                buffer: buf_out,
                cpu_offset,
                size: output_size as usize,
            });
            Ok(())
        }
    };
}

/// Generate a compute‑pipeline builder function.
///
/// The generated function checks whether a pipeline with key `{prefix}{key}`
/// exists in `ctx.pipelines` and creates it if absent, using the supplied
/// bind‑group layout entries.
///
/// Pass the bindings **without** a leading `&` — the macro adds it:
///
/// ```ignore
/// build_pipeline!(ensure_compute_pipeline, "wgpu_backend_", [
///     wgpu::BindGroupLayoutEntry { binding: 0, .. },
///     wgpu::BindGroupLayoutEntry { binding: 1, .. },
///     wgpu::BindGroupLayoutEntry { binding: 2, .. },
///     wgpu::BindGroupLayoutEntry { binding: 3, .. },
/// ]);
/// ```
#[macro_export]
macro_rules! build_pipeline {
    ($fn_name:ident, $key_prefix:expr, $bindings:expr) => {
        pub fn $fn_name(
            ctx: &mut $crate::backend::wgpu::context::WgpuContext,
            key: &str,
            wgsl_source: &str,
        ) -> Result<(), String> {
            let pipeline_key = format!("{}{}", $key_prefix, key);
            if !ctx.pipelines.contains_key(&pipeline_key) {
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&pipeline_key),
                        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
                    });

                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(&format!("{}_layout", pipeline_key)),
                            entries: &$bindings,
                        });

                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(&format!("{}_pl_layout", pipeline_key)),
                            bind_group_layouts: &[&layout],
                            push_constant_ranges: &[],
                        });

                let pipeline =
                    ctx.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(&pipeline_key),
                            layout: Some(&pipeline_layout),
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });

                ctx.pipelines.insert(pipeline_key, pipeline);
            }
            Ok(())
        }
    };
}
