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

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use ash::vk;

use super::context::{BindingRole, BufferTicket, KernelInvocation};
use super::device::Device;
use super::error::{check, Error, Result};
use super::quarantine::{PendingEntry, RetainedSubmitter};

/// Maximum descriptor sets one recording may allocate.
///
/// A per-op recording needs one set; a step batch needs one per op of the plan.
/// The trained GPT-2 decoder is 774 op records (the tiny model is 144), so the
/// cap carries 32% headroom over the larger pinned graph. The transient pool
/// cannot be reset while the batch is still recording, which is why the bound
/// is reserved up front rather than grown.
const DESCRIPTOR_SETS_PER_RECORDING: u32 = 1024;
/// Timestamp pairs a step batch may record (one pair per op).
///
/// The same 774-record trained plan must fit in one submission; 1024 keeps the
/// "one resident step is one submission" contract. If a device rejects the
/// query pool, `Submitter::new` degrades to no device timings rather than
/// failing, and the batch bound still holds.
const BATCH_QUERY_PAIRS: u32 = 1024;
/// Storage-buffer descriptors per set reserved in the transient pool.
const STORAGE_DESCRIPTORS_PER_SET: u32 = 16;
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
    /// Step batch this recording joined, when the submitter is batching.
    pub batch: Option<u64>,
    /// Query pair index inside that batch (`start = 2 * index`).
    pub batch_query: Option<u32>,
}

/// Result of closing a step batch.
#[derive(Debug, Clone, Default)]
pub struct BatchReport {
    /// Queue submissions this batch made (1 when it recorded anything).
    pub submissions: u32,
    /// The recording was discarded instead of submitted because a record in
    /// the same batch failed.
    pub discarded: bool,
    /// Host wall time between submit and fence signal, in nanoseconds.
    pub host_ns: u64,
    /// Device time for the whole batch, when timestamps are available.
    pub gpu_ns: Option<u64>,
    /// Raw `(start, end)` ticks for one timestamp pair per recorded op.
    pub timings: Vec<Option<(u64, u64)>>,
    /// Dispatches recorded in the batch.
    pub dispatches: u32,
    /// Buffer copies recorded in the batch.
    pub copies: u32,
    /// Bytes moved by those copies.
    pub copied_bytes: u64,
}

/// State of an open step batch on one submitter.
#[derive(Debug, Default)]
struct BatchState {
    id: u64,
    records: u32,
    dispatches: u32,
    copies: u32,
    copied_bytes: u64,
    /// A recording failed after this batch opened. The command buffer may hold
    /// a partial plan, so it must be discarded without submitting rather than
    /// closed and executed.
    failed: bool,
    /// Staging blocks whose host bytes the recorded copies still reference.
    retained: Vec<Arc<super::context::Buffer>>,
    /// Buffers the recorded commands reference, deduplicated by id. When the
    /// batch's fence wait fails, every one of them is moved to the device
    /// quarantine.
    touched: Vec<BufferTicket>,
    /// Ids already in `touched`.
    touched_ids: HashSet<u64>,
}

impl BatchState {
    /// Record one buffer reference. Returns `true` when it was new to the
    /// batch (and therefore needs one reserved reference).
    fn note(&mut self, ticket: BufferTicket) -> bool {
        if self.touched_ids.insert(ticket.id) {
            self.touched.push(ticket);
            true
        } else {
            false
        }
    }
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
    /// Buffers recorded into the in-flight submission.
    in_flight_buffers: Vec<BufferTicket>,
    /// The submitter's handles are owned by the device quarantine; `Drop` must
    /// not destroy them and the pool must not reuse it.
    quarantined: bool,
    /// One open step batch, when the backend is batching plan executions.
    batch: Option<BatchState>,
    next_batch_id: u64,
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
                .query_count(2 * BATCH_QUERY_PAIRS);
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
            in_flight_buffers: Vec::new(),
            quarantined: false,
            batch: None,
            next_batch_id: 0,
        })
    }

    /// `true` while a step batch is open on this submitter.
    pub fn is_batching(&self) -> bool {
        self.batch.is_some()
    }

    /// Id of the open batch, if any.
    pub fn batch_id(&self) -> Option<u64> {
        self.batch.as_ref().map(|batch| batch.id)
    }

    /// Hold a staging block until the open batch completes.
    ///
    /// A batched recording copies from a host-visible block that must not be
    /// reused before the submission runs; the batch owns it from here. A block
    /// the batch has not seen before also takes one device-wide in-flight
    /// reference, released only when the batch's outcome is known.
    pub fn retain_block(&mut self, block: Arc<super::context::Buffer>) {
        let ticket = block.ticket();
        let new = {
            let Some(batch) = self.batch.as_mut() else {
                // Not batching: the block was already safe to release.
                return;
            };
            let new = batch.note(ticket);
            batch.retained.push(block);
            new
        };
        if new {
            self.device.reserve_buffers(&[ticket]);
        }
    }

    /// Begin a step batch: reset and open one command buffer for every
    /// dispatch until [`Submitter::end_batch`].
    pub fn begin_batch(&mut self) -> Result<u64> {
        self.device.ensure_usable()?;
        if self.quarantined {
            return Err(Error::internal(
                "Submitter::begin_batch on a submitter owned by the device quarantine",
            ));
        }
        if self.in_flight {
            return Err(Error::internal(
                "Submitter::begin_batch while its previous submission is in flight",
            ));
        }
        if self.batch.is_some() {
            return Err(Error::internal(
                "Submitter::begin_batch while a batch is already open",
            ));
        }
        // SAFETY: the previous submission completed, and the pool was created
        // with RESET_COMMAND_BUFFER.
        if let Err(error) = check(
            "vkResetCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
            },
            "opening a step batch on a completed command buffer",
        ) {
            // The command buffer's state is unknown: never reuse this
            // submitter, but nothing was submitted, so the device itself is
            // still usable.
            self.retire("step batch begin", false, Vec::new());
            return Err(error);
        }
        // SAFETY: every descriptor set allocated from this pool belongs to a
        // completed submission; resetting returns all sets at once.
        if let Err(error) = check(
            "vkResetDescriptorPool",
            unsafe {
                self.device.ash().reset_descriptor_pool(
                    self.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
            },
            "reusing the transient descriptor pool for a step batch",
        ) {
            self.retire("step batch begin", false, Vec::new());
            return Err(error);
        }
        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: the buffer was just reset and is not in flight.
        if let Err(error) = check(
            "vkBeginCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .begin_command_buffer(self.cmd, &begin_info)
            },
            "recording a step batch",
        ) {
            self.retire("step batch begin", false, Vec::new());
            return Err(error);
        }
        if let Some(query_pool) = self.query_pool {
            // SAFETY: the buffer is recording and no submission is in flight,
            // so the whole range may be reset before any timestamp is written.
            unsafe {
                self.device.ash().cmd_reset_query_pool(
                    self.cmd,
                    query_pool,
                    0,
                    2 * BATCH_QUERY_PAIRS,
                );
            }
        }
        self.next_batch_id += 1;
        self.batch = Some(BatchState {
            id: self.next_batch_id,
            ..BatchState::default()
        });
        Ok(self.next_batch_id)
    }

    /// Close and submit the open step batch, then wait for it once.
    pub fn end_batch(&mut self) -> Result<BatchReport> {
        let started = Instant::now();
        let Some(mut batch) = self.batch.take() else {
            return Ok(BatchReport::default());
        };
        if self.quarantined {
            // The handles belong to the quarantine; this cannot submit.
            return Err(Error::internal(
                "Submitter::end_batch on a submitter owned by the device quarantine",
            ));
        }
        if batch.failed {
            // One `record` failed after this batch opened. The command buffer
            // holds a prefix of the plan; submitting it would execute a step
            // the caller was told had failed. Discard the recording instead
            // (Vulkan allows resetting a recording command buffer that is not
            // pending) and report that nothing was submitted. The retained
            // staging blocks are released with `batch`, and the references the
            // successful records reserved are proof-equivalently released.
            if let Err(error) = self.device.ensure_usable() {
                self.device.release_buffers(&batch.touched);
                self.retire("discarded step batch", false, Vec::new());
                return Err(error);
            }
            if let Err(error) = check(
                "vkResetCommandBuffer",
                unsafe {
                    self.device
                        .ash()
                        .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                },
                "discarding a failed step batch instead of submitting a partial plan",
            ) {
                self.device.release_buffers(&batch.touched);
                self.retire("discarded step batch", false, Vec::new());
                return Err(error);
            }
            self.device.release_buffers(&batch.touched);
            return Ok(BatchReport {
                discarded: true,
                ..BatchReport::default()
            });
        }
        // SAFETY: recording was begun on this buffer and no other thread uses
        // it.
        let ended = {
            #[cfg(feature = "vulkan-fault-injection")]
            if self
                .device
                .faults()
                .take(super::fault::FaultPoint::EndRecording)
            {
                Err(Error::vk(
                    "vkEndCommandBuffer",
                    vk::Result::ERROR_UNKNOWN,
                    "injected end-recording failure (FN3-27): the command buffer is not executable",
                ))
            } else {
                check(
                    "vkEndCommandBuffer",
                    unsafe { self.device.ash().end_command_buffer(self.cmd) },
                    "finishing a step batch",
                )
            }
            #[cfg(not(feature = "vulkan-fault-injection"))]
            check(
                "vkEndCommandBuffer",
                unsafe { self.device.ash().end_command_buffer(self.cmd) },
                "finishing a step batch",
            )
        };
        if let Err(error) = ended {
            // The command buffer is in an unknown state; do not reuse it.
            // Nothing was submitted, so no staged copy is in flight: release
            // the references and quarantine only the submitter's handles.
            self.device.poison("vkEndCommandBuffer", error.to_string());
            self.device.release_buffers(&batch.touched);
            self.retire("step batch recording", false, Vec::new());
            return Err(error);
        }
        let command_buffers = [self.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        if let Err(failure) = self.device.submit(&[submit], self.fence) {
            if failure.queued {
                // The submission may have reached the queue: its fence is
                // possibly pending, so every recorded buffer is retained.
                self.retire("step batch submit", true, batch.touched);
            } else {
                // Nothing was submitted: the references are proven-equivalent
                // to done and the buffers can return to their owners.
                self.device.release_buffers(&batch.touched);
                self.retire("step batch submit", false, Vec::new());
            }
            return Err(failure.error);
        }
        self.serial += 1;
        self.in_flight = true;
        self.in_flight_buffers = std::mem::take(&mut batch.touched);
        if let Err(error) = self.finish_wait() {
            let mut tickets = std::mem::take(&mut self.in_flight_buffers);
            for block in batch.retained.drain(..) {
                let ticket = block.ticket();
                if !tickets.iter().any(|seen| seen.id == ticket.id) {
                    tickets.push(ticket);
                }
            }
            self.retire("step batch wait", true, tickets);
            return Err(error);
        }
        let host_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let mut timings = Vec::with_capacity(batch.records as usize);
        if let Some(query_pool) = self.query_pool {
            let mut raw = vec![0u64; 2 * batch.records as usize];
            if !raw.is_empty() {
                // SAFETY: every recorded pair belongs to the submission that
                // just completed; `raw` has exactly two slots per record.
                match unsafe {
                    self.device.ash().get_query_pool_results(
                        query_pool,
                        0,
                        &mut raw,
                        vk::QueryResultFlags::TYPE_64,
                    )
                } {
                    Ok(()) => {
                        for pair in raw.chunks_exact(2) {
                            timings.push(Some((pair[0], pair[1])));
                        }
                    }
                    Err(code) => {
                        if self.device.instance().validation() {
                            eprintln!("[fn3vk][warn] reading batch timestamps failed: {code:?}");
                        }
                        timings.resize(batch.records as usize, None);
                    }
                }
            }
        } else {
            timings.resize(batch.records as usize, None);
        }
        // The batch's device span is the span of its per-op timestamps: the
        // first op starts and the last op ends, so no separate pair is needed.
        let gpu_ns = {
            let first = timings.iter().flatten().map(|(start, _)| *start).min();
            let last = timings.iter().flatten().map(|(_, end)| *end).max();
            match (first, last) {
                (Some(start), Some(end)) if end >= start => Some(
                    ((end - start) as f64 * f64::from(self.device.limits().timestamp_period))
                        .max(0.0)
                        .round()
                        .min(u64::MAX as f64) as u64,
                ),
                _ => None,
            }
        };
        // The retained staging blocks are safe to reuse only now: the batch's
        // successful fence wait proved that no recorded copy still reads them.
        batch.touched.clear();
        drop(batch.retained);
        Ok(BatchReport {
            submissions: 1,
            discarded: false,
            host_ns,
            gpu_ns,
            timings,
            dispatches: batch.dispatches,
            copies: batch.copies,
            copied_bytes: batch.copied_bytes,
        })
    }

    /// Record one submission. The closure may copy, barrier and dispatch;
    /// the batch is submitted when it returns.
    ///
    /// While a step batch is open the closure appends to the batch's command
    /// buffer and the returned token is *deferred*: [`Submission::wait`]
    /// reports what was recorded without waiting, and the microseconds a
    /// per-op fence wait costs disappear. A deferred recording must not read a
    /// staging buffer back — [`Submission::wait`] on one returns
    /// `submissions: 0` and the caller's host read would observe stale bytes,
    /// so the kernels reject a host output while batching.
    pub fn record<F>(&mut self, record: F) -> Result<Submission<'_>>
    where
        F: FnOnce(&mut Recorder) -> Result<()>,
    {
        self.device.ensure_usable()?;
        if self.batch.is_some() {
            return self.record_batched(record);
        }
        if self.in_flight {
            return Err(Error::internal(
                "Submitter::record called while its previous submission is in flight; \
                 wait for the Submission token before recording again",
            ));
        }
        if self.quarantined {
            return Err(Error::internal(
                "Submitter::record on a submitter owned by the device quarantine",
            ));
        }
        // SAFETY: the previous submission completed (or this is the first use),
        // so the command buffer may be reset. The pool was created with
        // RESET_COMMAND_BUFFER.
        if let Err(error) = check(
            "vkResetCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
            },
            "reusing the per-thread command buffer after a completed wait",
        ) {
            self.retire("dispatch recording", false, Vec::new());
            return Err(error);
        }
        // SAFETY: every descriptor set allocated from this pool belongs to a
        // completed submission; resetting returns all sets at once.
        if let Err(error) = check(
            "vkResetDescriptorPool",
            unsafe {
                self.device.ash().reset_descriptor_pool(
                    self.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
            },
            "reusing the transient descriptor pool after a completed wait",
        ) {
            self.retire("dispatch recording", false, Vec::new());
            return Err(error);
        }

        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: the buffer was just reset and is not in flight.
        if let Err(error) = check(
            "vkBeginCommandBuffer",
            unsafe {
                self.device
                    .ash()
                    .begin_command_buffer(self.cmd, &begin_info)
            },
            "recording a dispatch batch",
        ) {
            self.retire("dispatch recording", false, Vec::new());
            return Err(error);
        }

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
            buffers: Vec::new(),
            buffer_ids: HashSet::new(),
        };
        if let Err(error) = record(&mut recorder) {
            // Leave the command buffer reusable: the caller was told nothing
            // was submitted. If even the reset fails, the command buffer's
            // state is unknown, so the submitter is never reused.
            if let Err(reset) = check(
                "vkResetCommandBuffer",
                unsafe {
                    self.device
                        .ash()
                        .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                },
                "discarding a failed command recording",
            ) {
                self.device
                    .poison("vkResetCommandBuffer", reset.to_string());
                self.retire("dispatch recording", false, Vec::new());
            }
            return Err(error);
        }

        // The recording now references these buffers. Take one device-wide
        // reference each before the command buffer can be submitted; a later
        // submit or wait failure keeps them counted for the quarantine.
        self.device.reserve_buffers(&recorder.buffers);

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
        let ended = {
            #[cfg(feature = "vulkan-fault-injection")]
            if self
                .device
                .faults()
                .take(super::fault::FaultPoint::EndRecording)
            {
                Err(Error::vk(
                    "vkEndCommandBuffer",
                    vk::Result::ERROR_UNKNOWN,
                    "injected end-recording failure (FN3-27): the command buffer is not executable",
                ))
            } else {
                check(
                    "vkEndCommandBuffer",
                    unsafe { self.device.ash().end_command_buffer(self.cmd) },
                    "finishing command recording",
                )
            }
            #[cfg(not(feature = "vulkan-fault-injection"))]
            check(
                "vkEndCommandBuffer",
                unsafe { self.device.ash().end_command_buffer(self.cmd) },
                "finishing command recording",
            )
        };
        if let Err(error) = ended {
            // Nothing was submitted, so the references are proof-equivalently
            // done. The command buffer's state is unknown: never reuse this
            // submitter.
            self.device.release_buffers(&recorder.buffers);
            self.retire("dispatch recording", false, Vec::new());
            return Err(error);
        }

        // The recorded buffers become the submission's responsibility before
        // it is queued: if the submit or the wait fails, they are quarantined
        // rather than freed.
        self.in_flight_buffers = recorder.buffers;
        let command_buffers = [self.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        if let Err(failure) = self.device.submit(&[submit], self.fence) {
            if failure.queued {
                // The driver may have seen the submission: its fence can still
                // signal, so the buffers stay referenced and quarantined.
                self.retire("submission submit", true, Vec::new());
            } else {
                // Nothing reached the queue: drop the references as a
                // proof-equivalent discard and quarantine only the handles.
                self.device.release_buffers(&self.in_flight_buffers);
                self.in_flight_buffers.clear();
                self.retire("submission submit", false, Vec::new());
            }
            return Err(failure.error);
        }
        self.serial += 1;
        self.in_flight = true;
        let serial = self.serial;

        Ok(Submission {
            submitter: Some(self),
            serial,
            started: Instant::now(),
            dispatches: recorder.dispatches,
            copies: recorder.copies,
            copied_bytes: recorder.copied_bytes,
            waited: false,
            batch: None,
            batch_query: None,
        })
    }

    /// Append one recording to the open batch.
    fn record_batched<F>(&mut self, record: F) -> Result<Submission<'_>>
    where
        F: FnOnce(&mut Recorder) -> Result<()>,
    {
        let query_pool = self.query_pool;
        let (index, id) = {
            let batch = self.batch.as_ref().expect("checked by the caller");
            if batch.records >= BATCH_QUERY_PAIRS {
                return Err(Error::internal(format!(
                    "one step batch may record at most {BATCH_QUERY_PAIRS} dispatches; split the \
                     plan"
                )));
            }
            (batch.records, batch.id)
        };
        if let Some(query_pool) = query_pool {
            // SAFETY: the index pair is inside the pool and unused in this
            // batch, and the command buffer is recording.
            unsafe {
                self.device.ash().cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    2 * index,
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
            buffers: Vec::new(),
            buffer_ids: HashSet::new(),
        };
        if let Err(error) = record(&mut recorder) {
            // Mark the batch so `end_step` discards it instead of submitting
            // the prefix recorded so far. Nothing was submitted, so the
            // buffers this closure referenced are not in flight.
            if let Some(batch) = self.batch.as_mut() {
                batch.failed = true;
            }
            return Err(error);
        }
        if let Some(query_pool) = query_pool {
            // SAFETY: same buffer, still recording; the pair is unused.
            unsafe {
                self.device.ash().cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    2 * index + 1,
                );
            }
        }
        let mut reserved = Vec::new();
        {
            let batch = self.batch.as_mut().expect("checked above");
            for ticket in recorder.buffers.drain(..) {
                if batch.note(ticket) {
                    reserved.push(ticket);
                }
            }
            batch.records += 1;
            batch.dispatches += recorder.dispatches;
            batch.copies += recorder.copies;
            batch.copied_bytes += recorder.copied_bytes;
        }
        self.device.reserve_buffers(&reserved);
        Ok(Submission {
            submitter: None,
            serial: 0,
            started: Instant::now(),
            dispatches: recorder.dispatches,
            copies: recorder.copies,
            copied_bytes: recorder.copied_bytes,
            waited: false,
            batch: Some(id),
            batch_query: Some(index),
        })
    }

    /// Submissions completed by this submitter.
    pub fn submissions(&self) -> u64 {
        self.serial
    }

    /// `true` when this submitter's handles moved to the device quarantine.
    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    /// Hand this submitter and every in-flight buffer to the device quarantine.
    ///
    /// `submitted` says whether `vkQueueSubmit` may have reached the queue:
    /// `false` is used for a failure before submit (nothing is in flight),
    /// `true` for a submit or wait failure whose completion is unknown. The
    /// submitter is never reused after this call.
    fn retire(&mut self, label: &'static str, submitted: bool, extra: Vec<BufferTicket>) {
        if self.quarantined {
            return;
        }
        let mut tickets = std::mem::take(&mut self.in_flight_buffers);
        tickets.extend(extra);
        if let Some(mut batch) = self.batch.take() {
            tickets.append(&mut batch.touched);
            for block in batch.retained.drain(..) {
                tickets.push(block.ticket());
            }
        }
        let fence_unsignalled = submitted
            && matches!(
                // SAFETY: the fence belongs to this submitter and was
                // submitted once; querying its state is read-only.
                unsafe { self.device.ash().get_fence_status(self.fence) },
                Ok(false)
            );
        let entry = PendingEntry::new(
            format!("{label} (submitter serial {})", self.serial),
            submitted.then_some(self.fence),
            Some(RetainedSubmitter {
                command_pool: self.pool,
                command_buffer: self.cmd,
                fence: self.fence,
                descriptor_pool: self.descriptor_pool,
                query_pool: self.query_pool.take(),
            }),
            tickets,
        );
        self.device.quarantine_entry(entry, fence_unsignalled);
        self.quarantined = true;
        self.in_flight = submitted;
        self.batch = None;
    }

    fn finish_wait(&mut self) -> Result<Option<(u64, u64)>> {
        self.device.wait_fence(self.fence)?;
        // The wait proved that every recorded command completed, so each
        // reserved reference is released before the fence reset. A reset
        // failure does not un-prove the wait; the buffers are safe either way.
        self.device.release_buffers(&self.in_flight_buffers);
        self.in_flight_buffers.clear();
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
        if self.quarantined {
            // The device quarantine owns these handles and will destroy them
            // only after a proven wait or a successful teardown quiesce.
            self.destroyed = true;
            return;
        }
        if self.in_flight {
            // A token was leaked (or the submitter is being torn down while
            // its fence is unresolved). Do not destroy or recycle: quarantine
            // it instead.
            self.retire("submitter teardown", true, Vec::new());
            self.destroyed = true;
            return;
        }
        if let Some(batch) = self.batch.take() {
            // An open batch was never submitted. Its recorded commands are
            // discarded with the command buffer, so their references are
            // proof-equivalently released before the handles go.
            self.device.release_buffers(&batch.touched);
        }
        if let Some(query_pool) = self.query_pool.take() {
            // SAFETY: no submission is in flight; the pool is idle.
            unsafe { self.device.ash().destroy_query_pool(query_pool, None) };
            self.device.ledger().destroy("query_pool");
        }
        // SAFETY: the pool frees its command buffer; no submission is in
        // flight.
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
    /// `None` for a deferred (batched) recording: nothing was submitted.
    submitter: Option<&'a mut Submitter>,
    serial: u64,
    started: Instant,
    dispatches: u32,
    copies: u32,
    copied_bytes: u64,
    waited: bool,
    batch: Option<u64>,
    batch_query: Option<u32>,
}

impl<'a> Submission<'a> {
    /// Monotonic serial of this submission.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Wait for the fence and read device timestamps.
    ///
    /// A deferred (batched) recording submits nothing: the batch's single
    /// fence is waited on by `end_step`, and this call reports what was
    /// recorded without waiting.
    pub fn wait(mut self) -> Result<SubmitReport> {
        let Some(submitter) = self.submitter.as_deref_mut() else {
            self.waited = true;
            return Ok(SubmitReport {
                serial: 0,
                host_ns: 0,
                gpu_ns: None,
                timestamps: None,
                submissions: 0,
                dispatches: self.dispatches,
                copies: self.copies,
                copied_bytes: self.copied_bytes,
                batch: self.batch,
                batch_query: self.batch_query,
            });
        };
        let timestamps = match submitter.finish_wait() {
            Ok(timestamps) => timestamps,
            Err(error) => {
                // Completion is unknown: the token must not retry the wait and
                // the submitter must never be recycled. Every recorded buffer
                // and the command/descriptor pools move to the quarantine.
                submitter.retire("submission wait", true, Vec::new());
                self.waited = true;
                return Err(error);
            }
        };
        let host_ns = self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let gpu_ns = timestamps.map(|(start, end)| {
            let ticks = end.saturating_sub(start) as f64;
            let nanoseconds = ticks * f64::from(submitter.device.limits().timestamp_period);
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
            batch: None,
            batch_query: None,
        })
    }
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        if self.waited {
            return;
        }
        // A deferred recording submitted nothing, so there is nothing to wait
        // for; the batch owns the fence.
        let Some(submitter) = self.submitter.as_deref_mut() else {
            return;
        };
        // A token dropped without an explicit wait must still complete before
        // the submitter can be reused. Errors are reported but cannot change
        // the outcome of the already-submitted work: the submitter and every
        // in-flight buffer move to the quarantine instead of being freed.
        if let Err(error) = submitter.finish_wait() {
            submitter.retire("submission drop", true, Vec::new());
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
    /// Buffers this recording references, deduplicated by id.
    ///
    /// They are handed to the submission before `vkQueueSubmit`; if the wait
    /// fails, the quarantine owns them instead of the caller's `Drop`.
    buffers: Vec<BufferTicket>,
    /// Ids already in `buffers`.
    buffer_ids: HashSet<u64>,
}

impl Recorder {
    /// Note that a recorded command references `buffer`.
    fn note_buffer(&mut self, buffer: &super::context::Buffer) {
        if self.buffer_ids.insert(buffer.id()) {
            self.buffers.push(buffer.ticket());
        }
    }

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
        self.note_buffer(src);
        self.note_buffer(dst);
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
        self.note_buffer(buffer);
    }

    /// Allocate a transient descriptor set, write every declared binding, bind
    /// the pipeline and push constants, and dispatch.
    pub fn dispatch(&mut self, invocation: &KernelInvocation<'_>) -> Result<DispatchInfo> {
        #[cfg(feature = "vulkan-fault-injection")]
        if self.device.faults().take(super::fault::FaultPoint::Record) {
            // Fail inside the recording closure, exactly where a rejected
            // binding or a descriptor-pool exhaustion would: the batch has a
            // partial plan and must be discarded, not submitted.
            return Err(Error::internal(
                "injected recording failure (FN3-27): the step batch is partial",
            ));
        }
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
        for binding in invocation.bindings {
            self.note_buffer(binding.buffer);
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
}
