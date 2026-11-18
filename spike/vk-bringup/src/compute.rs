//! Standalone storage-buffer compute dispatch.
//!
//! Pipeline: device-local `a`, `b`, `c`, `out` buffers; one host-visible
//! staging buffer for upload; one persistently mapped host-visible buffer for
//! readback; one descriptor set with four storage buffers; one compute
//! pipeline; one command buffer and one fence, both reused for every
//! submission.
//!
//! The command pool is created with `RESET_COMMAND_BUFFER` and is owned by this
//! engine. Vulkan command pools are externally synchronised: one pool belongs
//! to one thread at a time. This spike is single-threaded; the production
//! backend must create one pool per worker thread (see NOTES-FN3-11.md).
//!
//! Completion is fence-based (`vkWaitForFences`) for every submission.
//! `vkDeviceWaitIdle` appears only in teardown.

use std::time::Instant;

use ash::vk;
use ash::vk::Handle;

use crate::device::DeviceContext;
use crate::error::{check, Error, Result};
use crate::ledger::Ledger;

const WORKGROUP_SIZE: u32 = 64;
const SHADER_SPV: &[u8] = include_bytes!("../shaders/fma.comp.spv");
const FENCE_TIMEOUT_NS: u64 = 60 * 1_000_000_000;
const SAMPLE_COUNT: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub count: u32,
    pub repeats: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchReport {
    pub count: u32,
    pub workgroups: u32,
    pub primary_ms: f64,
    pub max_abs_error: f32,
    pub exact_matches: u32,
    pub repeats: u32,
    pub sample_checks: u32,
    pub sample_mismatches: u32,
    pub final_max_abs_error: f32,
    pub final_exact: u32,
    pub repeat_min_ms: f64,
    pub repeat_mean_ms: f64,
    pub repeat_max_ms: f64,
    pub submits: u32,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
}

struct AllocatedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Size reported by `vkGetBufferMemoryRequirements` (what was allocated).
    allocation_size: u64,
}

pub struct Engine<'dev, 'inst> {
    ctx: &'dev DeviceContext<'inst>,
    cfg: RunConfig,
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    buffers: Vec<AllocatedBuffer>,
    upload: Option<AllocatedBuffer>,
    readback: Option<AllocatedBuffer>,
    readback_ptr: *mut u8,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    ledger: Ledger,
    submits: u32,
    destroyed: bool,
}

impl<'dev, 'inst> Engine<'dev, 'inst> {
    pub fn new(ctx: &'dev DeviceContext<'inst>, cfg: RunConfig) -> Result<Self> {
        validate_limits(ctx, &cfg)?;
        let mut engine = Self {
            ctx,
            cfg,
            shader: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            buffers: Vec::new(),
            upload: None,
            readback: None,
            readback_ptr: std::ptr::null_mut(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            ledger: Ledger::default(),
            submits: 0,
            destroyed: false,
        };
        if let Err(err) = engine.init() {
            engine.teardown();
            return Err(err);
        }
        Ok(engine)
    }

    fn init(&mut self) -> Result<()> {
        let device = &self.ctx.device;
        let count = self.cfg.count;
        let bytes = u64::from(count) * 4;

        let spv = shader_words(SHADER_SPV)?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        // SAFETY: `spv` is a live, 4-byte-aligned Vec<u32> containing validated
        // SPIR-V; the create-info borrows it only for this call.
        self.shader = unsafe { device.create_shader_module(&shader_info, None) }
            .map_err(|code| Error::vk("vkCreateShaderModule", code, "fma.comp.spv"))?;
        self.ledger.create("shader_module");

        let bindings: [vk::DescriptorSetLayoutBinding; 4] = std::array::from_fn(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `bindings` is a live array of four valid bindings for the
        // duration of the call.
        self.set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .map_err(|code| Error::vk("vkCreateDescriptorSetLayout", code, "4 storage buffers"))?;
        self.ledger.create("descriptor_set_layout");

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8); // uint n; float bias
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        // SAFETY: layout handles and range slice are live for the call.
        self.pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.map_err(
                |code| Error::vk("vkCreatePipelineLayout", code, "8-byte push constant"),
            )?;
        self.ledger.create("pipeline_layout");

        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout);
        // SAFETY: the shader module and pipeline layout are live; the stage
        // pointer is borrowed only for the call. `create_compute_pipelines`
        // returns the partially created pipelines alongside the error, which
        // we discard because none were requested beyond index 0.
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        match pipelines {
            Ok(pipelines) => {
                self.pipeline = pipelines[0];
                self.ledger.create("pipeline");
            }
            Err((pipelines, code)) => {
                for pipeline in pipelines {
                    if !pipeline.is_null() {
                        // SAFETY: partially created pipelines are owned by us.
                        unsafe { device.destroy_pipeline(pipeline, None) };
                    }
                }
                return Err(Error::vk(
                    "vkCreateComputePipelines",
                    code,
                    "fma.comp.spv entry point main",
                ));
            }
        }

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);
        // SAFETY: pool size and set count match the single allocation below.
        self.descriptor_pool =
            unsafe { device.create_descriptor_pool(&pool_info, None) }.map_err(|code| {
                Error::vk("vkCreateDescriptorPool", code, "4 storage buffers, 1 set")
            })?;
        self.ledger.create("descriptor_pool");

        let set_layouts = [self.set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool can satisfy exactly this one set with four storage
        // buffers, matching its declared sizes.
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|code| Error::vk("vkAllocateDescriptorSets", code, "one set"))?;
        self.descriptor_set = sets[0];

        let storage_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        for (name, usage) in [
            ("buffer_a", storage_usage),
            ("buffer_b", storage_usage),
            ("buffer_c", storage_usage),
            (
                "buffer_out",
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            ),
        ] {
            let buffer = create_buffer(
                self.ctx,
                &mut self.ledger,
                bytes,
                usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                name,
            )?;
            self.buffers.push(buffer);
        }

        let upload = create_buffer(
            self.ctx,
            &mut self.ledger,
            3 * bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            "staging_upload",
        )?;
        self.upload = Some(upload);

        self.readback = Some(create_buffer(
            self.ctx,
            &mut self.ledger,
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            "staging_readback",
        )?);
        let readback_memory = self
            .readback
            .as_ref()
            .ok_or_else(|| Error::internal("readback staging buffer missing"))?
            .memory;
        // SAFETY: readback memory is host-visible/coherent and not in use; the
        // mapping spans the whole allocation and stays valid until teardown.
        self.readback_ptr = unsafe {
            device.map_memory(
                readback_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|code| Error::vk("vkMapMemory", code, "readback staging buffer"))?
            as *mut u8;

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.ctx.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: family index came from the device's queue properties.
        self.command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
            .map_err(|code| Error::vk("vkCreateCommandPool", code, "compute queue family"))?;
        self.ledger.create("command_pool");

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was just created and is empty.
        let command_buffers = unsafe { device.allocate_command_buffers(&cmd_alloc) }
            .map_err(|code| Error::vk("vkAllocateCommandBuffers", code, "one primary buffer"))?;
        self.command_buffer = command_buffers[0];
        self.ledger.create("command_buffer");

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: no shared state; the fence starts unsignaled, matching the
        // first submission's use.
        self.fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|code| Error::vk("vkCreateFence", code, "submission fence"))?;
        self.ledger.create("fence");

        self.write_descriptors()?;
        Ok(())
    }

    fn write_descriptors(&self) -> Result<()> {
        let infos: [vk::DescriptorBufferInfo; 4] = std::array::from_fn(|i| {
            vk::DescriptorBufferInfo::default()
                .buffer(self.buffers[i].buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        });
        let writes: Vec<vk::WriteDescriptorSet> = (0..4)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
        // SAFETY: every write targets the allocated set, binding i exists in
        // the layout, and `infos` outlives the call.
        unsafe {
            self.ctx.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    pub fn run(mut self) -> Result<(DispatchReport, Ledger)> {
        let n = self.cfg.count as usize;
        let bytes = u64::from(self.cfg.count) * 4;

        let (a, b, c) = generate_inputs(self.cfg.count);

        // Upload all three inputs into one staging buffer.
        {
            let upload = self
                .upload
                .as_ref()
                .ok_or_else(|| Error::internal("upload staging buffer missing"))?;
            // SAFETY: the upload buffer is HOST_VISIBLE|HOST_COHERENT, is not
            // in flight, and the mapping spans the whole allocation, which is
            // exactly 3 * n * 4 bytes.
            let mapped = unsafe {
                self.ctx.device.map_memory(
                    upload.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|code| Error::vk("vkMapMemory", code, "upload staging buffer"))?
                as *mut u8;
            // SAFETY: `mapped` has room for 3 * n * 4 bytes; the three source
            // slices are distinct live Vecs of n f32 each and do not overlap
            // the mapping.
            unsafe {
                std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, mapped, a.len() * 4);
                std::ptr::copy_nonoverlapping(
                    b.as_ptr() as *const u8,
                    mapped.add(n * 4),
                    b.len() * 4,
                );
                std::ptr::copy_nonoverlapping(
                    c.as_ptr() as *const u8,
                    mapped.add(2 * n * 4),
                    c.len() * 4,
                );
            }
            // SAFETY: the mapping is no longer accessed and the copy above
            // completed; the buffer will be used by the GPU after submission.
            unsafe { self.ctx.device.unmap_memory(upload.memory) };
        }

        let workgroups = self.cfg.count.div_ceil(WORKGROUP_SIZE);
        let mut report = DispatchReport {
            count: self.cfg.count,
            workgroups,
            primary_ms: 0.0,
            max_abs_error: 0.0,
            exact_matches: 0,
            repeats: self.cfg.repeats,
            sample_checks: 0,
            sample_mismatches: 0,
            final_max_abs_error: 0.0,
            final_exact: 0,
            repeat_min_ms: f64::MAX,
            repeat_mean_ms: 0.0,
            repeat_max_ms: 0.0,
            submits: 0,
            upload_bytes: 3 * bytes,
            readback_bytes: bytes,
        };

        // --- Phase A: upload + one dispatch (bias 0) + full readback --------
        self.begin_recording()?;
        self.record_upload()?;
        self.record_dispatch(0.0);
        self.record_readback(bytes)?;
        self.end_recording()?;

        let started = Instant::now();
        self.submit_and_wait("initial upload+dispatch+readback")?;
        report.primary_ms = started.elapsed().as_secs_f64() * 1e3;

        let output = self.read_output(n);
        let (max_abs, exact, mismatch) = compare(output, &a, &b, &c, 0.0);
        report.max_abs_error = max_abs;
        report.exact_matches = exact;
        if let Some((index, got, want)) = mismatch {
            return Err(Error::internal(format!(
                "dispatch result mismatch at index {index}: GPU={got} CPU={want} \
                 (exact_matches={exact}/{n}, max_abs_error={max_abs})"
            )));
        }
        if exact != self.cfg.count {
            return Err(Error::internal(format!(
                "result assertion failed: {exact}/{} elements matched",
                self.cfg.count
            )));
        }

        // --- Phase B: repeated submissions, all handles reused --------------
        let mut repeat_total_ms = 0.0;
        for iteration in 1..=self.cfg.repeats {
            self.reset_for_recording()?;
            self.begin_recording()?;
            self.record_dispatch(iteration as f32);
            let sample_bytes = (SAMPLE_COUNT as u64 * 4).min(bytes);
            self.record_readback(sample_bytes)?;
            self.end_recording()?;

            let started = Instant::now();
            self.submit_and_wait("repeated dispatch")?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
            repeat_total_ms += elapsed_ms;
            report.repeat_min_ms = report.repeat_min_ms.min(elapsed_ms);
            report.repeat_max_ms = report.repeat_max_ms.max(elapsed_ms);

            let samples = self.read_output(SAMPLE_COUNT.min(n));
            for (index, got) in samples.iter().enumerate() {
                report.sample_checks += 1;
                let want = expected(&a, &b, &c, index, iteration as f32);
                if *got != want {
                    return Err(Error::internal(format!(
                        "repeated dispatch {iteration} sample mismatch at index {index}: \
                         GPU={got} CPU={want}"
                    )));
                }
            }
        }
        report.repeat_mean_ms = repeat_total_ms / f64::from(self.cfg.repeats);

        // --- Phase C: full readback of the final repeated result -------------
        self.reset_for_recording()?;
        self.begin_recording()?;
        self.record_readback(bytes)?;
        self.end_recording()?;
        self.submit_and_wait("final full readback")?;

        let output = self.read_output(n);
        let (max_abs, exact, mismatch) = compare(output, &a, &b, &c, self.cfg.repeats as f32);
        report.final_max_abs_error = max_abs;
        report.final_exact = exact;
        if let Some((index, got, want)) = mismatch {
            return Err(Error::internal(format!(
                "final result mismatch at index {index}: GPU={got} CPU={want} \
                 (bias={}, exact_matches={exact}/{n})",
                self.cfg.repeats
            )));
        }

        report.submits = self.submits;
        self.teardown();
        let ledger = std::mem::take(&mut self.ledger);
        Ok((report, ledger))
    }

    fn begin_recording(&self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default();
        // SAFETY: the buffer is in the initial state or was reset, and is not
        // in flight (the previous submission was waited on).
        check(
            "vkBeginCommandBuffer",
            unsafe {
                self.ctx
                    .device
                    .begin_command_buffer(self.command_buffer, &info)
            },
            "recording compute commands",
        )
    }

    fn end_recording(&self) -> Result<()> {
        // SAFETY: recording was begun on this buffer and no other thread uses it.
        check(
            "vkEndCommandBuffer",
            unsafe { self.ctx.device.end_command_buffer(self.command_buffer) },
            "finishing command recording",
        )
    }

    fn reset_for_recording(&mut self) -> Result<()> {
        // SAFETY: the previous submission completed (fence waited), so both the
        // fence and the command buffer are idle and may be reset.
        check(
            "vkResetFences",
            unsafe { self.ctx.device.reset_fences(&[self.fence]) },
            "reusing the submission fence",
        )?;
        check(
            "vkResetCommandBuffer",
            unsafe {
                self.ctx
                    .device
                    .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
            },
            "reusing the command buffer",
        )
    }

    fn submit_and_wait(&mut self, label: &'static str) -> Result<()> {
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        // SAFETY: the command buffer is recording-complete and owned by this
        // engine; the queue belongs to the same family used to create the pool;
        // the fence is unsignaled after `reset_for_recording` (or is fresh for
        // the first submission).
        check(
            "vkQueueSubmit",
            unsafe {
                self.ctx
                    .device
                    .queue_submit(self.ctx.queue, &[submit], self.fence)
            },
            label,
        )?;
        self.submits += 1;
        // SAFETY: waiting on our own fence; 60 s is a hang detector, not a
        // synchronisation shortcut.
        check(
            "vkWaitForFences",
            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS)
            },
            label,
        )
    }

    fn record_upload(&self) -> Result<()> {
        let bytes = u64::from(self.cfg.count) * 4;
        let upload = self
            .upload
            .as_ref()
            .ok_or_else(|| Error::internal("upload staging buffer missing"))?;
        let total = 3 * bytes;
        if bytes == 0 || upload.allocation_size < total {
            return Err(Error::internal(format!(
                "upload range check failed: need {total} bytes, staging allocation is {}",
                upload.allocation_size
            )));
        }
        let regions: [vk::BufferCopy; 3] = std::array::from_fn(|i| {
            vk::BufferCopy::default()
                .src_offset(i as u64 * bytes)
                .dst_offset(0)
                .size(bytes)
        });
        // SAFETY: the command buffer is recording; upload/buffers are live;
        // every region is within both buffers (3*n*4 staging, n*4 targets) and
        // every offset is 4-byte aligned.
        unsafe {
            for (i, region) in regions.iter().enumerate() {
                self.ctx.device.cmd_copy_buffer(
                    self.command_buffer,
                    upload.buffer,
                    self.buffers[i].buffer,
                    std::slice::from_ref(region),
                );
            }
        }
        for buffer in &self.buffers[..3] {
            self.buffer_barrier(
                buffer.buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
        }
        Ok(())
    }

    fn record_dispatch(&self, bias: f32) {
        // SAFETY: pipeline, descriptor set and buffers are live and compatible;
        // the push-constant range (8 bytes) matches the shader layout.
        unsafe {
            self.ctx.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.ctx.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            let mut push = [0u8; 8];
            push[0..4].copy_from_slice(&self.cfg.count.to_le_bytes());
            push[4..8].copy_from_slice(&bias.to_le_bytes());
            self.ctx.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            let groups = self.cfg.count.div_ceil(WORKGROUP_SIZE);
            self.ctx
                .device
                .cmd_dispatch(self.command_buffer, groups, 1, 1);
        }
    }

    fn record_readback(&self, bytes: u64) -> Result<()> {
        let total = u64::from(self.cfg.count) * 4;
        if bytes == 0 || bytes > total || !bytes.is_multiple_of(4) {
            return Err(Error::internal(format!(
                "readback range check failed: requested {bytes} bytes of {total} \
                 (must be a non-zero multiple of 4 within the output buffer)"
            )));
        }
        let readback = self
            .readback
            .as_ref()
            .ok_or_else(|| Error::internal("readback staging buffer missing"))?;
        if readback.allocation_size < bytes {
            return Err(Error::internal(format!(
                "readback range check failed: {bytes} bytes exceeds staging allocation {}",
                readback.allocation_size
            )));
        }
        self.buffer_barrier(
            self.buffers[3].buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(bytes);
        // SAFETY: both buffers are live; `bytes <= n*4` fits the source, the
        // destination is n*4, and offsets/sizes are 4-byte multiples.
        unsafe {
            self.ctx.device.cmd_copy_buffer(
                self.command_buffer,
                self.buffers[3].buffer,
                readback.buffer,
                std::slice::from_ref(&region),
            );
        }
        Ok(())
    }

    fn buffer_barrier(
        &self,
        buffer: vk::Buffer,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(self.ctx.family_index)
            .dst_queue_family_index(self.ctx.family_index)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the command buffer is recording; the barrier refers to a live
        // buffer owned by this engine and to the same queue family on both
        // sides (no ownership transfer).
        unsafe {
            self.ctx.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );
        }
    }

    fn read_output(&self, count: usize) -> &[f32] {
        debug_assert!(count as u64 * 4 <= self.cfg.count as u64 * 4);
        debug_assert!(!self.readback_ptr.is_null());
        // SAFETY: the readback mapping spans n*4 bytes and was written by a
        // command buffer whose fence has been waited on; `count <= n` and the
        // pointer is 4-byte aligned by the Vulkan implementation.
        unsafe { std::slice::from_raw_parts(self.readback_ptr as *const f32, count) }
    }

    pub fn teardown(&mut self) {
        if self.destroyed {
            return;
        }
        let device = &self.ctx.device;
        {
            // Teardown-only wait; every measured submission used a fence.
            // SAFETY: no other thread uses this device.
            let _ = unsafe { device.device_wait_idle() };

            if !self.readback_ptr.is_null() {
                if let Some(readback) = &self.readback {
                    // SAFETY: the pointer came from vkMapMemory on this
                    // allocation and is unmapped exactly once.
                    unsafe { device.unmap_memory(readback.memory) };
                }
                self.readback_ptr = std::ptr::null_mut();
            }
            if !self.fence.is_null() {
                // SAFETY: the fence is not in use after the idle wait.
                unsafe { device.destroy_fence(self.fence, None) };
                self.fence = vk::Fence::null();
                self.ledger.destroy("fence");
            }
            if !self.command_pool.is_null() {
                // SAFETY: the pool and its command buffers are idle; destroying
                // the pool frees the buffer allocated from it.
                unsafe { device.destroy_command_pool(self.command_pool, None) };
                self.command_pool = vk::CommandPool::null();
                self.command_buffer = vk::CommandBuffer::null();
                self.ledger.destroy("command_buffer");
                self.ledger.destroy("command_pool");
            }
            for buffer in self.buffers.drain(..) {
                destroy_buffer(device, buffer, &mut self.ledger);
            }
            if let Some(buffer) = self.upload.take() {
                destroy_buffer(device, buffer, &mut self.ledger);
            }
            if let Some(buffer) = self.readback.take() {
                destroy_buffer(device, buffer, &mut self.ledger);
            }
            if !self.descriptor_pool.is_null() {
                // SAFETY: all sets allocated from the pool are unused; the pool
                // frees them. No individual vkFreeDescriptorSets is required.
                unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
                self.descriptor_pool = vk::DescriptorPool::null();
                self.descriptor_set = vk::DescriptorSet::null();
                self.ledger.destroy("descriptor_pool");
            }
            if !self.pipeline.is_null() {
                // SAFETY: the pipeline is not bound to any pending work.
                unsafe { device.destroy_pipeline(self.pipeline, None) };
                self.pipeline = vk::Pipeline::null();
                self.ledger.destroy("pipeline");
            }
            if !self.pipeline_layout.is_null() {
                // SAFETY: no pipeline or command buffer refers to it anymore.
                unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
                self.pipeline_layout = vk::PipelineLayout::null();
                self.ledger.destroy("pipeline_layout");
            }
            if !self.set_layout.is_null() {
                // SAFETY: the pool was destroyed first, so no set refers to it.
                unsafe { device.destroy_descriptor_set_layout(self.set_layout, None) };
                self.set_layout = vk::DescriptorSetLayout::null();
                self.ledger.destroy("descriptor_set_layout");
            }
            if !self.shader.is_null() {
                // SAFETY: the pipeline that used it is destroyed.
                unsafe { device.destroy_shader_module(self.shader, None) };
                self.shader = vk::ShaderModule::null();
                self.ledger.destroy("shader_module");
            }
        }
        self.destroyed = true;
    }
}

impl Drop for Engine<'_, '_> {
    fn drop(&mut self) {
        self.teardown();
    }
}

fn validate_limits(ctx: &DeviceContext<'_>, cfg: &RunConfig) -> Result<()> {
    let limits = &ctx.info.limits;
    let bytes = u64::from(cfg.count) * 4;
    let workgroups = cfg.count.div_ceil(WORKGROUP_SIZE);

    if bytes == 0 {
        return Err(Error::Usage("--count must be at least 1".into()));
    }
    if bytes > u64::from(limits.max_storage_buffer_range) {
        return Err(Error::internal(format!(
            "count {} needs {bytes} bytes per buffer, above maxStorageBufferRange={} on \"{}\"",
            cfg.count, limits.max_storage_buffer_range, ctx.info.name
        )));
    }
    if workgroups > limits.max_compute_work_group_count[0] {
        return Err(Error::internal(format!(
            "count {} needs {workgroups} workgroups, above maxComputeWorkGroupCount[0]={}",
            cfg.count, limits.max_compute_work_group_count[0]
        )));
    }
    if WORKGROUP_SIZE > limits.max_compute_work_group_size[0]
        || WORKGROUP_SIZE > limits.max_compute_work_group_invocations
    {
        return Err(Error::internal(format!(
            "local_size_x={WORKGROUP_SIZE} exceeds limits (maxComputeWorkGroupSize={:?}, \
             maxComputeWorkGroupInvocations={})",
            limits.max_compute_work_group_size, limits.max_compute_work_group_invocations
        )));
    }
    if limits.max_push_constants_size < 8 {
        return Err(Error::internal(format!(
            "shader needs 8 push-constant bytes, device offers {}",
            limits.max_push_constants_size
        )));
    }
    if limits.max_descriptor_set_storage_buffers < 4 {
        return Err(Error::internal(format!(
            "shader needs 4 storage buffers in one set, device allows {}",
            limits.max_descriptor_set_storage_buffers
        )));
    }
    ctx.check_allocation_size(3 * bytes, "upload staging buffer")?;
    for name in ["buffer_a", "buffer_b", "buffer_c", "buffer_out"] {
        ctx.check_allocation_size(bytes, name)?;
    }
    Ok(())
}

fn create_buffer(
    ctx: &DeviceContext<'_>,
    ledger: &mut Ledger,
    size: u64,
    usage: vk::BufferUsageFlags,
    required_flags: vk::MemoryPropertyFlags,
    what: &str,
) -> Result<AllocatedBuffer> {
    let device = &ctx.device;
    ctx.check_allocation_size(size, what)?;

    let create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: plain buffer creation with a zeroed allocation-callback optional.
    let buffer = unsafe { device.create_buffer(&create_info, None) }
        .map_err(|code| Error::vk("vkCreateBuffer", code, what))?;

    // SAFETY: `buffer` was just created and is live.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    if let Err(err) = ctx.check_allocation_size(requirements.size, what) {
        // SAFETY: nothing was bound to the buffer yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(err);
    }
    let type_index = match ctx.find_memory_type(requirements.memory_type_bits, required_flags, what)
    {
        Ok(index) => index,
        Err(err) => {
            // SAFETY: nothing was bound to the buffer yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(err);
        }
    };

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    // SAFETY: the allocation size and type index come from the device's own
    // requirements query; no dedicated-allocation pNext is used.
    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(memory) => memory,
        Err(code) => {
            // SAFETY: nothing was bound to the buffer yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(Error::vk("vkAllocateMemory", code, what));
        }
    };

    // SAFETY: `buffer` was created without dedicated requirements and `memory`
    // is large enough for `requirements.size`; offset 0 is aligned.
    if let Err(err) = check(
        "vkBindBufferMemory",
        unsafe { device.bind_buffer_memory(buffer, memory, 0) },
        what,
    ) {
        // SAFETY: the failed bind leaves both objects owned by us; free in
        // reverse creation order.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(err);
    }

    ledger.create("buffer");
    ledger.create("device_memory");
    ledger.allocate_bytes("device_memory", requirements.size);

    Ok(AllocatedBuffer {
        buffer,
        memory,
        allocation_size: requirements.size,
    })
}

fn destroy_buffer(device: &ash::Device, buffer: AllocatedBuffer, ledger: &mut Ledger) {
    // SAFETY: the buffer is no longer used by any command buffer (the engine
    // waited for idle in teardown); its memory was bound by us.
    unsafe {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
    ledger.destroy("buffer");
    ledger.destroy("device_memory");
    ledger.free_bytes("device_memory", buffer.allocation_size);
}

fn shader_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::internal(format!(
            "embedded SPIR-V has invalid length {} (must be a non-zero multiple of 4)",
            bytes.len()
        )));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0 {
        words.push(u32::from_le_bytes(*chunk));
    }
    if words[0] != 0x0723_0203 {
        return Err(Error::internal(format!(
            "embedded SPIR-V has wrong magic 0x{:08x}",
            words[0]
        )));
    }
    Ok(words)
}

fn generate_inputs(count: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Small integer-valued data: every product and sum is exactly
    // representable in f32, so CPU and GPU must agree bit-for-bit and FMA
    // contraction cannot mask a real bug. The patterns are deliberately not
    // constants so a stale/partially copied buffer is visible.
    let mut a = Vec::with_capacity(count as usize);
    let mut b = Vec::with_capacity(count as usize);
    let mut c = Vec::with_capacity(count as usize);
    for i in 0..count {
        let i = u64::from(i);
        a.push(((i * 7 + 3) % 13) as f32 - 6.0);
        b.push(((i * 5 + 1) % 11) as f32 - 5.0);
        c.push(((i * 3 + 2) % 17) as f32 - 8.0);
    }
    (a, b, c)
}

fn expected(a: &[f32], b: &[f32], c: &[f32], index: usize, bias: f32) -> f32 {
    a[index] * b[index] + c[index] + bias
}

fn compare(
    output: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    bias: f32,
) -> (f32, u32, Option<(usize, f32, f32)>) {
    let mut max_abs = 0.0f32;
    let mut exact = 0u32;
    let mut first_mismatch = None;
    for (index, got) in output.iter().enumerate() {
        let want = expected(a, b, c, index, bias);
        let error = (got - want).abs();
        if error > max_abs {
            max_abs = error;
        }
        if *got == want {
            exact += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some((index, *got, want));
        }
    }
    (max_abs, exact, first_mismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_inputs_match_cpu_reference() {
        let (a, b, c) = generate_inputs(1000);
        for index in 0..a.len() {
            let want = a[index] * b[index] + c[index];
            assert_eq!(expected(&a, &b, &c, index, 0.0), want);
        }
        // The patterns must not be constant across the tail, otherwise a
        // partially copied buffer could pass unnoticed.
        assert_ne!(a[0], a[999]);
        assert_ne!(b[17], b[998]);
    }

    #[test]
    fn compare_detects_a_single_bad_element() {
        let (a, b, c) = generate_inputs(16);
        let mut output: Vec<f32> = (0..16).map(|i| expected(&a, &b, &c, i, 2.0)).collect();
        let (max_abs, exact, mismatch) = compare(&output, &a, &b, &c, 2.0);
        assert_eq!(max_abs, 0.0);
        assert_eq!(exact, 16);
        assert!(mismatch.is_none());

        output[7] += 1.0;
        let (max_abs, exact, mismatch) = compare(&output, &a, &b, &c, 2.0);
        assert_eq!(max_abs, 1.0);
        assert_eq!(exact, 15);
        assert_eq!(mismatch.map(|m| m.0), Some(7));
    }

    #[test]
    fn embedded_spirv_is_word_aligned_with_magic() {
        let words = shader_words(SHADER_SPV).expect("embedded SPIR-V is valid");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn spirv_rejects_bad_length_and_magic() {
        assert!(shader_words(&[]).is_err());
        assert!(shader_words(&[0, 1, 2]).is_err());
        assert!(shader_words(&[0u8; 8]).is_err());
    }
}
