//! Command recording, descriptor binding, submission and completion.
//!
//! # Per-thread command pools
//!
//! Vulkan command pools are *externally synchronised*: one pool may be used by
//! one thread at a time. [`Submitter`] owns exactly one pool (created with
//! `RESET_COMMAND_BUFFER`), one primary command buffer, one fence, one
//! transient descriptor pool and an optional timestamp query pool. The backend
//! keeps a `Submitter` per thread, keyed by `ThreadId`, and checks one out
//! exclusively for the duration of a dispatch; two threads never touch the same
//! pool. The `Mutex` around `vkQueueSubmit` is the only cross-thread
//! serialisation and is held for the submit call only, never across a wait.
//!
//! # Completion
//!
//! Every submission returns a [`Submission`] token. `wait` blocks on the
//! submitter's fence, reads the device timestamps and resets the fence and
//! command buffer. `vkQueueWaitIdle` is never used in the hot path; it appears
//! only in device teardown. Dropping a token without waiting still waits (and
//! poisons nothing) so a command buffer can never be reset while in flight.
//!
//! # Why not one fence per dispatch forever
//!
//! The token API is the seam for a fence ring: today one submitter records,
//! submits and waits before it is reused. Batching several dispatches into one
//! recording already amortises the submission; a ring of submitters (one per
//! in-flight slot) is the next step for #62/#72, not a correctness requirement
//! for the alpha.

use std::sync::Arc;
use std::time::Instant;

use ash::vk;

use super::context::{BindingArg, BindingRole, KernelInvocation};
use super::device::Device;
use super::error::{check, Error, Result};

/// Maximum descriptor sets one recording may allocate.
const DESCRIPTOR_SETS_PER_RECORDING: u32 = 64;
/// Storage-buffer descriptors per set reserved in the transient pool.
const STORAGE_DESCRIPTORS_PER_SET: u32 = 32;
/// Uniform-buffer descriptors per set reserved in the transient pool.
const UNIFORM_DESCRIPTORS_PER_SET: u32 = 8;

/// One buffer as it was bound in a dispatch, for residency records.
#[derive(Debug, Clone)]
pub struct BindingSnapshot {
    /// Descriptor binding number.
    pub binding: u32,
    /// Role in the dispatch.
    pub role: BindingRole,
    /// Process-unique buffer id.
    pub buffer_id: u64,
    /// Byte offset.
    pub offset: u64,
    /// Byte range.
    pub range: u64,
    /// `true` when the buffer lives in device-local memory.
    pub device_local: bool,
}

/// Result of one dispatch recorded into a submission.
#[derive(Debug, Clone)]
pub struct DispatchInfo {
    /// Workgroup counts.
    pub workgroups: [u32; 3],
    /// Bindings as recorded.
    pub bindings: Vec<BindingSnapshot>,
}

/// Result of waiting for a submission.
#[derive(Debug, Clone)]
pub struct SubmitReport {
    /// Monotonic submission serial of this submitter (1-based).
    pub serial: u64,
    /// Host wall time spent between submit and fence signal, in nanoseconds.
    pub host_ns: u64,
    /// Device time between the start and end timestamps, in nanoseconds, when
    /// the queue family supports timestamps.
    pub gpu_ns: Option<u64>,
    /// Raw `(start, end)` timestamp ticks.
    pub timestamps: Option<(u64, u64)>,
    /// Command buffers submitted (always 1 in this API).
    pub submissions: u32,
    /// Dispatches recorded.
    pub dispatches: u32,
    /// Buffer copies recorded.
    pub copies: u32,
    /// Bytes moved by buffer copies.
    pub copied_bytes: u64,
}

/// A per-thread recording context.
pub struct Submitter {
    device: Arc<Device>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    query_pool: Option<vk::QueryPool>,
    serial: u64,
    in_flight: bool,
    destroyed: bool,
}

impl Submitter {
    /// Create a command pool, command buffer, fence and transient descriptor
    /// pool for the calling thread.
    pub fn new(device: &Arc<Device>) -> Result<Submitter> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the family index came from the adapter's queue properties.
        let pool =
            unsafe { device.ash().create_command_pool(&pool_info, None) }.map_err(|code| {
                Error::vk(
                    "vkCreateCommandPool",
                    code,
                    format!(
                        "compute queue family {} for a per-thread submitter",
                        device.queue_family()
                    ),
                )
            })?;
        device.ledger().create("command_pool");

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was just created and is empty.
        let command_buffers = match unsafe { device.ash().allocate_command_buffers(&allocate_info) }
        {
            Ok(buffers) => buffers,
            Err(code) => {
                // SAFETY: the empty pool is destroyed exactly once.
                unsafe { device.ash().destroy_command_pool(pool, None) };
                device.ledger().destroy("command_pool");
                return Err(Error::vk(
                    "vkAllocateCommandBuffers",
                    code,
                    "one primary command buffer per submitter",
                ));
            }
        };
        device.ledger().create("command_buffer");

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: a plain fence with no shared state; it starts unsignalled,
        // matching the first submission, and is destroyed in `Drop`.
        let fence = match unsafe { device.ash().create_fence(&fence_info, None) } {
            Ok(fence) => fence,
            Err(code) => {
                // SAFETY: pool and buffer are unused; the pool frees the
                // command buffer.
                unsafe { device.ash().destroy_command_pool(pool, None) };
                device.ledger().destroy("command_buffer");
                device.ledger().destroy("command_pool");
                return Err(Error::vk(
                    "vkCreateFence",
                    code,
                    "per-thread submission fence",
                ));
            }
        };
        device.ledger().create("fence");

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(STORAGE_DESCRIPTORS_PER_SET * DESCRIPTOR_SETS_PER_RECORDING),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(UNIFORM_DESCRIPTORS_PER_SET * DESCRIPTOR_SETS_PER_RECORDING),
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(DESCRIPTOR_SETS_PER_RECORDING)
            .flags(vk::DescriptorPoolCreateFlags::empty());
        // SAFETY: the pool sizes bound every allocation this submitter makes in
        // one recording; the pool is reset only after its submission completed.
        let descriptor_pool = match unsafe {
            device
                .ash()
                .create_descriptor_pool(&descriptor_pool_info, None)
        } {
            Ok(pool) => pool,
            Err(code) => {
                // SAFETY: fence, pool and buffer are unused; free in reverse
                // creation order.
                unsafe {
                    device.ash().destroy_fence(fence, None);
                    device.ash().destroy_command_pool(pool, None);
                }
                device.ledger().destroy("fence");
                device.ledger().destroy("command_buffer");
                device.ledger().destroy("command_pool");
                return Err(Error::vk(
                    "vkCreateDescriptorPool",
                    code,
                    format!(
                        "{DESCRIPTOR_SETS_PER_RECORDING} transient sets with \
                         {STORAGE_DESCRIPTORS_PER_SET} storage and \
                         {UNIFORM_DESCRIPTORS_PER_SET} uniform descriptors each"
                    ),
                ));
            }
        };
        device.ledger().create("descriptor_pool");

        let timestamp_bits = device
            .adapter()
            .compute_family()
            .map(|family| family.timestamp_valid_bits)
            .unwrap_or(0);
        let query_pool = if timestamp_bits > 0 {
            let query_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            // SAFETY: a two-query timestamp pool; timestamps are valid on this
            // family. A failure is degraded (gpu timing absent), not fatal.
            match unsafe { device.ash().create_query_pool(&query_info, None) } {
                Ok(pool) => {
                    device.ledger().create("query_pool");
                    Some(pool)
                }
                Err(code) => {
                    if device.instance().validation() {
                        eprintln!(
                            "[fn3vk][warn] timestamp query pool unavailable ({code:?}); \
                             device timing will be reported as unavailable"
                        );
                    }
                    None
                }
            }
        } else {
            None
        };

        Ok(Submitter {
            device: Arc::clone(device),
            pool,
            cmd: command_buffers[0],
            fence,
            descriptor_pool,
            query_pool,
            serial: 0,
            in_flight: false,
            destroyed: false,
        })
    }

    /// Record one submission. The closure may copy, barrier and dispatch;
    /// the batch is submitted when it returns.
    pub fn record<F>(&mut self, record: F) -> Result<Submission<'_>>
    where
        F: FnOnce(&mut Recorder) -> Result<()>,
    {
        if self.in_flight {
            return Err(Error::internal(
                "Submitter::record called while its previous submission is in flight; \
                 wait for the Submission token before recording again",
            ));
        }
        // SAFETY: the previous submission completed (or this is the first use),
        // so the command buffer may be reset. The pool was created with
        // RESET_COMMAND_BUFFER.
        check(
            "vkResetCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
            },
            "reusing the per-thread command buffer after a completed wait",
        )?;
        // SAFETY: every descriptor set allocated from this pool belongs to a
        // completed submission; resetting returns all sets at once.
        check(
            "vkResetDescriptorPool",
            unsafe {
                self.device.ash().reset_descriptor_pool(
                    self.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
            },
            "reusing the transient descriptor pool after a completed wait",
        )?;

        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: the buffer was just reset and is not in flight.
        check(
            "vkBeginCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .begin_command_buffer(self.cmd, &begin_info)
            },
            "recording a dispatch batch",
        )?;

        if let Some(query_pool) = self.query_pool {
            // SAFETY: the two queries are available (the previous submission
            // was waited on) and the buffer is recording.
            unsafe {
                self.device
                    .ash()
                    .cmd_reset_query_pool(self.cmd, query_pool, 0, 2);
                self.device.ash().cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
        }

        let mut recorder = Recorder {
            device: Arc::clone(&self.device),
            cmd: self.cmd,
            descriptor_pool: self.descriptor_pool,
            dispatches: 0,
            copies: 0,
            copied_bytes: 0,
        };
        record(&mut recorder)?;

        if let Some(query_pool) = self.query_pool {
            // SAFETY: same buffer, still recording; BOTTOM_OF_PIPE means "after
            // every previously recorded command completed".
            unsafe {
                self.device.ash().cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    1,
                );
            }
        }

        // SAFETY: recording was begun on this buffer and no other thread uses
        // it.
        check(
            "vkEndCommandBuffer",
            unsafe { self.device.ash().end_command_buffer(self.cmd) },
            "finishing command recording",
        )?;

        let command_buffers = [self.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        self.device.submit(&[submit], self.fence)?;
        self.serial += 1;
        self.in_flight = true;
        let serial = self.serial;

        Ok(Submission {
            submitter: self,
            serial,
            started: Instant::now(),
            dispatches: recorder.dispatches,
            copies: recorder.copies,
            copied_bytes: recorder.copied_bytes,
            waited: false,
        })
    }

    /// Submissions completed by this submitter.
    pub fn submissions(&self) -> u64 {
        self.serial
    }

    /// `true` while a submission is in flight.
    pub fn is_in_flight(&self) -> bool {
        self.in_flight
    }

    fn finish_wait(&mut self) -> Result<Option<(u64, u64)>> {
        self.device.wait_fence(self.fence)?;
        self.device.reset_fence(self.fence)?;
        let timestamps = if let Some(query_pool) = self.query_pool {
            let mut data = [0u64; 2];
            // SAFETY: the submission completed, so both timestamps are
            // available; `data` has exactly two 64-bit slots.
            match unsafe {
                self.device.ash().get_query_pool_results(
                    query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                )
            } {
                Ok(()) => Some((data[0], data[1])),
                Err(code) => {
                    if self.device.instance().validation() {
                        eprintln!("[fn3vk][warn] reading device timestamps failed: {code:?}");
                    }
                    None
                }
            }
        } else {
            None
        };
        self.in_flight = false;
        Ok(timestamps)
    }

    /// Destroy the submitter's objects. Teardown only.
    pub fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        // A submitter can only be destroyed by its owner after the last token
        // is waited (the token borrows it), but be defensive about in-flight
        // work: the device is idle before submitters are destroyed.
        if let Some(query_pool) = self.query_pool.take() {
            // SAFETY: the device is idle (teardown) and the pool is not in use.
            unsafe { self.device.ash().destroy_query_pool(query_pool, None) };
            self.device.ledger().destroy("query_pool");
        }
        // SAFETY: the pool frees its command buffer; no submission is in
        // flight and the device is idle.
        unsafe {
            self.device
                .ash()
                .destroy_descriptor_pool(self.descriptor_pool, None)
        };
        self.device.ledger().destroy("descriptor_pool");
        // SAFETY: the fence is unused after the final wait.
        unsafe { self.device.ash().destroy_fence(self.fence, None) };
        self.device.ledger().destroy("fence");
        // SAFETY: the pool and its command buffers are idle; destroying the
        // pool frees the buffer allocated from it.
        unsafe { self.device.ash().destroy_command_pool(self.pool, None) };
        self.device.ledger().destroy("command_buffer");
        self.device.ledger().destroy("command_pool");
        self.destroyed = true;
    }
}

impl Drop for Submitter {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl std::fmt::Debug for Submitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submitter")
            .field("serial", &self.serial)
            .field("in_flight", &self.in_flight)
            .field("timestamps", &self.query_pool.is_some())
            .finish_non_exhaustive()
    }
}

/// A submitted batch that has not been waited on yet.
///
/// The token borrows the submitter, so the command buffer and fence cannot be
/// reused until [`Submission::wait`] returns (or the token is dropped, which
/// waits).
pub struct Submission<'a> {
    submitter: &'a mut Submitter,
    serial: u64,
    started: Instant,
    dispatches: u32,
    copies: u32,
    copied_bytes: u64,
    waited: bool,
}

impl<'a> Submission<'a> {
    /// Monotonic serial of this submission.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Wait for the fence and read device timestamps.
    pub fn wait(mut self) -> Result<SubmitReport> {
        let timestamps = self.submitter.finish_wait()?;
        let host_ns = self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let gpu_ns = timestamps.map(|(start, end)| {
            let ticks = end.saturating_sub(start) as f64;
            let nanoseconds = ticks * f64::from(self.submitter.device.limits().timestamp_period);
            nanoseconds.max(0.0).round().min(u64::MAX as f64) as u64
        });
        self.waited = true;
        Ok(SubmitReport {
            serial: self.serial,
            host_ns,
            gpu_ns,
            timestamps,
            submissions: 1,
            dispatches: self.dispatches,
            copies: self.copies,
            copied_bytes: self.copied_bytes,
        })
    }
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        if self.waited {
            return;
        }
        // A token dropped without an explicit wait must still complete before
        // the submitter can be reused. Errors are reported but cannot change
        // the outcome of the already-submitted work.
        if let Err(error) = self.submitter.finish_wait() {
            eprintln!(
                "[fn3vk][warn] implicit wait for submission {} failed: {error}",
                self.serial
            );
        }
    }
}

/// Records commands into one command buffer.
///
/// Every method validates its arguments before recording: checked ranges,
/// buffer usage flags, descriptor alignment, binding completeness and workgroup
/// limits. A rejected argument is an [`Error`], not a validation-layer message.
pub struct Recorder {
    device: Arc<Device>,
    cmd: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    dispatches: u32,
    copies: u32,
    copied_bytes: u64,
}

impl Recorder {
    /// Copy `len` bytes between buffers, validating both ranges and usages.
    pub fn copy_buffer(
        &mut self,
        src: &super::context::Buffer,
        src_offset: u64,
        dst: &super::context::Buffer,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        src.check_range(src_offset, len, "copy source")?;
        dst.check_range(dst_offset, len, "copy destination")?;
        if !src.usage().contains(vk::BufferUsageFlags::TRANSFER_SRC) {
            return Err(Error::internal(format!(
                "{} is not a transfer source (usage {:?}); declare TRANSFER_SRC",
                src.name(),
                src.usage()
            )));
        }
        if !dst.usage().contains(vk::BufferUsageFlags::TRANSFER_DST) {
            return Err(Error::internal(format!(
                "{} is not a transfer destination (usage {:?}); declare TRANSFER_DST",
                dst.name(),
                dst.usage()
            )));
        }
        // vkCmdCopyBuffer requires 4-byte multiples for offsets and size.
        for (what, value) in [
            ("source offset", src_offset),
            ("destination offset", dst_offset),
            ("size", len),
        ] {
            if value % 4 != 0 {
                return Err(Error::internal(format!(
                    "copy {what} {value} is not a multiple of 4, which vkCmdCopyBuffer requires"
                )));
            }
        }
        if len == 0 {
            return Err(Error::internal(
                "copy size 0 is not a valid vkCmdCopyBuffer region",
            ));
        }
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(len);
        // SAFETY: both buffers are live (they are borrowed for the call and the
        // device keeps them alive), ranges were checked above, and the command
        // buffer is recording on this thread.
        unsafe {
            self.device.ash().cmd_copy_buffer(
                self.cmd,
                src.raw(),
                dst.raw(),
                std::slice::from_ref(&region),
            );
        }
        self.copies += 1;
        self.copied_bytes += len;
        Ok(())
    }

    /// Record a buffer memory barrier.
    pub fn barrier(
        &mut self,
        buffer: &super::context::Buffer,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(self.device.queue_family())
            .dst_queue_family_index(self.device.queue_family())
            .buffer(buffer.raw())
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the command buffer is recording; the barrier refers to a live
        // buffer on the same queue family on both sides (no ownership transfer).
        unsafe {
            self.device.ash().cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );
        }
    }

    /// Allocate a transient descriptor set, write every declared binding, bind
    /// the pipeline and push constants, and dispatch.
    pub fn dispatch(&mut self, invocation: &KernelInvocation<'_>) -> Result<DispatchInfo> {
        let kernel = invocation.kernel;
        let spec = kernel.spec();
        let limits = self.device.limits();

        if invocation.bindings.len() != spec.bindings.len() {
            return Err(Error::internal(format!(
                "kernel \"{}\" declares {} binding(s) but {} were supplied",
                spec.name,
                spec.bindings.len(),
                invocation.bindings.len()
            )));
        }
        for (index, binding) in spec.bindings.iter().enumerate() {
            let supplied = invocation.bindings[index];
            if supplied.binding != binding.binding {
                return Err(Error::internal(format!(
                    "kernel \"{}\": binding argument {index} is for binding {} but the spec \
                     declares binding {} ({})",
                    spec.name, supplied.binding, binding.binding, binding.name
                )));
            }
            if supplied.range == 0 {
                return Err(Error::internal(format!(
                    "kernel \"{}\": binding {} ({}) has a zero range",
                    spec.name, binding.binding, binding.name
                )));
            }
            supplied
                .buffer
                .check_range(supplied.offset, supplied.range, "descriptor binding")?;
            let (usage, alignment, range_limit) = match binding.kind {
                super::kernels::BindingKind::StorageBuffer => (
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    limits.min_storage_buffer_offset_alignment,
                    u64::from(limits.max_storage_buffer_range),
                ),
                super::kernels::BindingKind::UniformBuffer => (
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    limits.min_uniform_buffer_offset_alignment,
                    u64::from(limits.max_uniform_buffer_range),
                ),
            };
            if !supplied.buffer.usage().contains(usage) {
                return Err(Error::internal(format!(
                    "kernel \"{}\": binding {} ({}) needs usage {:?} but buffer \"{}\" was \
                     created with {:?}",
                    spec.name,
                    binding.binding,
                    binding.name,
                    usage,
                    supplied.buffer.name(),
                    supplied.buffer.usage()
                )));
            }
            if supplied.range > range_limit {
                return Err(Error::limit(
                    "vulkan descriptor binding",
                    supplied.range,
                    range_limit,
                    format!(
                        "kernel \"{}\": binding {} ({}) on \"{}\"",
                        spec.name,
                        binding.binding,
                        binding.name,
                        self.device.adapter().name
                    ),
                ));
            }
            if alignment != 0 && !supplied.offset.is_multiple_of(alignment) {
                return Err(Error::internal(format!(
                    "kernel \"{}\": binding {} offset {} is not a multiple of {} \
                     (min buffer offset alignment)",
                    spec.name, binding.binding, supplied.offset, alignment
                )));
            }
        }
        if invocation.push_constants.len() != spec.push_constant_bytes as usize {
            return Err(Error::internal(format!(
                "kernel \"{}\" declares {} push-constant bytes but {} were supplied",
                spec.name,
                spec.push_constant_bytes,
                invocation.push_constants.len()
            )));
        }
        for axis in 0..3 {
            if invocation.workgroups[axis] == 0 {
                return Err(Error::internal(format!(
                    "kernel \"{}\": workgroup count axis {axis} is zero",
                    spec.name
                )));
            }
            if invocation.workgroups[axis] > limits.max_compute_work_group_count[axis] {
                return Err(Error::limit(
                    "vulkan workgroup count",
                    u64::from(invocation.workgroups[axis]),
                    u64::from(limits.max_compute_work_group_count[axis]),
                    format!(
                        "kernel \"{}\" axis {axis} on \"{}\"",
                        spec.name,
                        self.device.adapter().name
                    ),
                ));
            }
        }

        let layouts = [kernel.descriptor_set_layout()];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool has room for `DESCRIPTOR_SETS_PER_RECORDING` sets
        // with the declared descriptor counts; the layout is live.
        let sets = unsafe { self.device.ash().allocate_descriptor_sets(&allocate_info) }.map_err(
            |code| {
                Error::vk(
                    "vkAllocateDescriptorSets",
                    code,
                    format!(
                        "kernel \"{}\": transient set from a pool of {} sets",
                        spec.name, DESCRIPTOR_SETS_PER_RECORDING
                    ),
                )
            },
        )?;
        let set = sets[0];

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = invocation
            .bindings
            .iter()
            .map(|binding| {
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.buffer.raw())
                    .offset(binding.offset)
                    .range(binding.range)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = invocation
            .bindings
            .iter()
            .zip(buffer_infos.iter())
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding.binding)
                    .descriptor_type(
                        spec.binding(binding.binding)
                            .map(|declared| declared.kind.descriptor_type())
                            .unwrap_or(vk::DescriptorType::STORAGE_BUFFER),
                    )
                    .descriptor_count(1)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: every write targets a binding declared by the set layout, the
        // set is not in use by any submission, and `buffer_infos` outlives the
        // call.
        unsafe { self.device.ash().update_descriptor_sets(&writes, &[]) };

        // SAFETY: pipeline, layout and set are live and compatible; the set was
        // written for every declared binding above.
        unsafe {
            self.device.ash().cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.raw(),
            );
            self.device.ash().cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout(),
                0,
                &[set],
                &[],
            );
            if !invocation.push_constants.is_empty() {
                self.device.ash().cmd_push_constants(
                    self.cmd,
                    kernel.pipeline_layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    invocation.push_constants,
                );
            }
            self.device.ash().cmd_dispatch(
                self.cmd,
                invocation.workgroups[0],
                invocation.workgroups[1],
                invocation.workgroups[2],
            );
        }
        self.dispatches += 1;

        Ok(DispatchInfo {
            workgroups: invocation.workgroups,
            bindings: invocation
                .bindings
                .iter()
                .map(|binding| BindingSnapshot {
                    binding: binding.binding,
                    role: binding.role,
                    buffer_id: binding.buffer.id(),
                    offset: binding.offset,
                    range: binding.range,
                    device_local: binding.buffer.class() == super::memory::MemoryClass::DeviceLocal,
                })
                .collect(),
        })
    }

    /// `true` while the recorder is active (always, for a live recorder).
    pub fn is_recording(&self) -> bool {
        true
    }
}

/// Copy a `BindingArg` list into the snapshot form used by residency records.
pub fn snapshot_bindings(bindings: &[BindingArg<'_>]) -> Vec<BindingSnapshot> {
    bindings
        .iter()
        .map(|binding| BindingSnapshot {
            binding: binding.binding,
            role: binding.role,
            buffer_id: binding.buffer.id(),
            offset: binding.offset,
            range: binding.range,
            device_local: binding.buffer.class() == super::memory::MemoryClass::DeviceLocal,
        })
        .collect()
}
