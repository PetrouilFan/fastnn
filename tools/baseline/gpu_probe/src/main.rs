//! Enumerate the exact wgpu adapters the FastNN `WgpuBackend` sees and verify a
//! minimal shader compiles and dispatches on each. Prints JSON for the baseline
//! report. Mirrors `src/backend/wgpu/context.rs::GpuContext::new` selection:
//! `Instance::default().enumerate_adapters(Backends::all())`, indexed by
//! `device_id`.

use serde_json::json;

fn main() {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    let mut records = Vec::new();
    for (index, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let limits = adapter.limits();

        let probe = pollster::block_on(async {
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("fastnn-gpu-probe"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .map_err(|error| format!("request_device: {error}"))?;

            // Same shape as the failing FastNN shaders: a local named `active`.
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
                    @compute @workgroup_size(1)
                    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                        let probed = 2.0;
                        out[gid.x] = probed;
                    }
                    "#
                    .into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("probe"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe-out"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe-read"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, 4);
            queue.submit([encoder.finish()]);

            let slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|error| format!("map channel: {error}"))?
                .map_err(|error| format!("map: {error}"))?;
            let mapped = slice.get_mapped_range();
            let value = f32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
            drop(mapped);
            staging.unmap();
            Ok::<f32, String>(value)
        });

        records.push(json!({
            "device_id": index,
            "name": info.name,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
            "driver": info.driver,
            "driver_info": info.driver_info,
            "vendor": info.vendor,
            "device": info.device,
            "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size,
            "max_compute_workgroup_size_x": limits.max_compute_workgroup_size_x,
            "compute_probe": match probe {
                Ok(value) => json!({"ok": true, "value": value}),
                Err(error) => json!({"ok": false, "error": error}),
            },
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "instance_backends": format!("{:?}", wgpu::Backends::all()),
            "adapter_count": records.len(),
            "adapters": records,
        }))
        .unwrap()
    );
}
