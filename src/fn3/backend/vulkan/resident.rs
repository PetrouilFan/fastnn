//! Device-resident tensors and persistent model data (FN3-26 / #72).
//!
//! # What problem this solves
//!
//! At the alpha every tensor is host storage, so each op uploads its operands,
//! dispatches and downloads its result. For a 32-token chain that is 1.59 GB of
//! upload traffic for a 28 MB model, and the arithmetic — 719 proven device
//! executions, `lm_head` at 89 GFLOP/s — is not the bottleneck.
//!
//! Two things are needed to close the gap, and this module provides both:
//!
//! 1. **Persistent weights.** [`PersistentTable`] holds one device copy of each
//!    immutable tensor (model weights, later folded constants) and resolves a
//!    bind by handle. The host tensor is never modified, so the CPU/fallback
//!    path keeps its bytes.
//! 2. **Resident tensors.** [`ResidentBuffer`] implements
//!    [`crate::inference::storage::DeviceResidency`], so a `Tensor` can own a device
//!    allocation instead of host memory. Kernels bind it directly; nothing is
//!    staged per op.
//!
//! # What is proven, not assumed
//!
//! Persistence is enforced, not merely observed. After
//! [`VulkanBackend::prepare_persistent`](crate::inference::backend::Backend::prepare_persistent) every registered tensor is in the
//! table, and the staging path calls
//! [`VulkanBackend::check_staged_input`](crate::inference::backend::vulkan::VulkanBackend::check_staged_input) before it copies host bytes: a
//! registered tensor that reaches a host upload is a *hard error* and
//! increments [`ResidencyStats::late_persistent_uploads`]. "Zero weight bytes
//! uploaded after prepare" is therefore a compile-time-shaped invariant with a
//! counter, not a claim about a trace.
//!
//! # Resident vs. staged
//!
//! The switch is per tensor, not per run: a host-backed tensor still stages
//! (and still produces the alpha's `host_staged = true` records), and a
//! device-resident one binds by handle. `FN3_VK_STAGED=1` makes the session
//! allocate host tensors, which brings the whole staged path back with no
//! kernel changes. That is the comparison arm for #72 and the debugging path.
//!
//! # Allocation
//!
//! Every resident tensor is one device-local buffer from the backend's
//! suballocator (device buffers are never host mapped), sized to the tensor's
//! `required_bytes()` rounded up to the 4-byte `vkCmdCopyBuffer` granularity.
//! Persistent tensors are allocated once, at prepare; a session's value slots
//! are allocated once per shape and reused by the slot pool. The counters below
//! are what the report and the tests quote.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use super::context::{Buffer, BufferDesc};
use super::device::Device;
use super::error::Error;
use super::staging::StagingPool;
use super::submission::Submitter;
use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::{Fn3Error, Fn3Result};
use crate::inference::storage::{DeviceResidency, Storage};
use crate::inference::tensor::{Tensor, TensorMeta, TensorMut, TensorView};

/// Name every resident buffer reports in diagnostics.
pub const RESIDENT_BUFFER_NAME: &str = "resident";

/// Which phase uploads are attributed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyPhase {
    /// Preparing the model: weights and other immutable tensors.
    Prepare = 0,
    /// A prefill step.
    Prefill = 1,
    /// A decode step.
    Decode = 2,
}

impl ResidencyPhase {
    /// Index used by the per-phase counters.
    pub fn index(self) -> usize {
        self as usize
    }

    /// Stable name used in reports.
    pub fn name(self) -> &'static str {
        match self {
            ResidencyPhase::Prepare => "prepare",
            ResidencyPhase::Prefill => "prefill",
            ResidencyPhase::Decode => "decode",
        }
    }

    fn from_u64(value: u64) -> ResidencyPhase {
        match value {
            1 => ResidencyPhase::Prefill,
            2 => ResidencyPhase::Decode,
            _ => ResidencyPhase::Prepare,
        }
    }
}

/// A snapshot of the residency counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidencyStats {
    /// Phase the counters are currently attributed to.
    pub phase: u64,
    /// Persistent tensors in the table.
    pub persistent_tensors: u64,
    /// Device bytes held for persistent tensors (cumulative uploads).
    pub persistent_bytes: u64,
    /// Bytes uploaded by `prepare_persistent` over the backend's lifetime.
    pub persistent_prepare_bytes: u64,
    /// Bytes uploaded by packed-weight preparation (`PackedProjection::prepare`).
    /// These are immutable weights moved once, not per-step host staging.
    pub packed_prepare_bytes: u64,
    /// Binds served by the persistent table, per phase.
    pub persistent_hits: [u64; 3],
    /// A registered persistent tensor reached a host upload. Must stay zero.
    pub late_persistent_uploads: u64,
    /// Bytes such a rejected upload would have moved.
    pub late_persistent_bytes: u64,
    /// Host uploads (non-tensor buffers included), per phase.
    pub staged_uploads: [u64; 3],
    /// Bytes moved by host uploads, per phase.
    pub staged_upload_bytes: [u64; 3],
    /// Device → host readbacks.
    pub readback_calls: u64,
    /// Bytes read back to the host.
    pub readback_bytes: u64,
    /// Device-resident tensors allocated by the backend.
    pub resident_tensors: u64,
    /// Device bytes they hold.
    pub resident_bytes: u64,
    /// Binds served by a resident tensor, per phase.
    pub resident_binds: [u64; 3],
    /// Bytes bound from resident tensors, per phase.
    pub resident_bind_bytes: [u64; 3],
    /// Submission scopes opened by `begin_step`.
    pub batches: u64,
    /// Queue submissions made by `end_step`.
    pub batch_submissions: u64,
}

/// Atomic counters behind [`ResidencyStats`].
#[derive(Debug, Default)]
pub struct ResidencyCounters {
    phase: AtomicU64,
    persistent_tensors: AtomicU64,
    persistent_bytes: AtomicU64,
    persistent_prepare_bytes: AtomicU64,
    packed_prepare_bytes: AtomicU64,
    persistent_hits: [AtomicU64; 3],
    late_persistent_uploads: AtomicU64,
    late_persistent_bytes: AtomicU64,
    staged_uploads: [AtomicU64; 3],
    staged_upload_bytes: [AtomicU64; 3],
    readback_calls: AtomicU64,
    readback_bytes: AtomicU64,
    resident_tensors: AtomicU64,
    resident_bytes: AtomicU64,
    resident_binds: [AtomicU64; 3],
    resident_bind_bytes: [AtomicU64; 3],
    batches: AtomicU64,
    batch_submissions: AtomicU64,
}

impl ResidencyCounters {
    /// An all-zero counter set attributed to [`ResidencyPhase::Prepare`].
    pub fn new() -> ResidencyCounters {
        ResidencyCounters::default()
    }

    /// Attribute subsequent events to `phase`.
    pub fn set_phase(&self, phase: ResidencyPhase) {
        self.phase.store(phase.index() as u64, Ordering::Relaxed);
    }

    /// Current phase.
    pub fn phase(&self) -> ResidencyPhase {
        ResidencyPhase::from_u64(self.phase.load(Ordering::Relaxed))
    }

    fn phase_index(&self) -> usize {
        self.phase() as usize
    }

    /// A consistent snapshot.
    pub fn snapshot(&self) -> ResidencyStats {
        let mut stats = ResidencyStats {
            phase: self.phase.load(Ordering::Relaxed),
            persistent_tensors: self.persistent_tensors.load(Ordering::Relaxed),
            persistent_bytes: self.persistent_bytes.load(Ordering::Relaxed),
            persistent_prepare_bytes: self.persistent_prepare_bytes.load(Ordering::Relaxed),
            packed_prepare_bytes: self.packed_prepare_bytes.load(Ordering::Relaxed),
            late_persistent_uploads: self.late_persistent_uploads.load(Ordering::Relaxed),
            late_persistent_bytes: self.late_persistent_bytes.load(Ordering::Relaxed),
            readback_calls: self.readback_calls.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            resident_tensors: self.resident_tensors.load(Ordering::Relaxed),
            resident_bytes: self.resident_bytes.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            batch_submissions: self.batch_submissions.load(Ordering::Relaxed),
            ..ResidencyStats::default()
        };
        for index in 0..3 {
            stats.persistent_hits[index] = self.persistent_hits[index].load(Ordering::Relaxed);
            stats.staged_uploads[index] = self.staged_uploads[index].load(Ordering::Relaxed);
            stats.staged_upload_bytes[index] =
                self.staged_upload_bytes[index].load(Ordering::Relaxed);
            stats.resident_binds[index] = self.resident_binds[index].load(Ordering::Relaxed);
            stats.resident_bind_bytes[index] =
                self.resident_bind_bytes[index].load(Ordering::Relaxed);
        }
        stats
    }

    /// Forget every counter (the phase is preserved).
    pub fn reset(&self) {
        self.persistent_tensors.store(0, Ordering::Relaxed);
        self.persistent_bytes.store(0, Ordering::Relaxed);
        self.persistent_prepare_bytes.store(0, Ordering::Relaxed);
        self.packed_prepare_bytes.store(0, Ordering::Relaxed);
        self.late_persistent_uploads.store(0, Ordering::Relaxed);
        self.late_persistent_bytes.store(0, Ordering::Relaxed);
        self.readback_calls.store(0, Ordering::Relaxed);
        self.readback_bytes.store(0, Ordering::Relaxed);
        self.resident_tensors.store(0, Ordering::Relaxed);
        self.resident_bytes.store(0, Ordering::Relaxed);
        self.batches.store(0, Ordering::Relaxed);
        self.batch_submissions.store(0, Ordering::Relaxed);
        for index in 0..3 {
            self.persistent_hits[index].store(0, Ordering::Relaxed);
            self.staged_uploads[index].store(0, Ordering::Relaxed);
            self.staged_upload_bytes[index].store(0, Ordering::Relaxed);
            self.resident_binds[index].store(0, Ordering::Relaxed);
            self.resident_bind_bytes[index].store(0, Ordering::Relaxed);
        }
    }

    fn note_persistent(&self, bytes: u64, uploaded: bool) {
        self.persistent_tensors.fetch_add(1, Ordering::Relaxed);
        self.persistent_bytes.fetch_add(bytes, Ordering::Relaxed);
        if uploaded {
            self.persistent_prepare_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
    }

    fn note_persistent_hit(&self, bytes: u64) {
        let index = self.phase_index();
        self.persistent_hits[index].fetch_add(1, Ordering::Relaxed);
        let _ = bytes;
    }

    fn note_late_persistent(&self, bytes: u64) {
        self.late_persistent_uploads.fetch_add(1, Ordering::Relaxed);
        self.late_persistent_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// One immutable packed-weight section moved to the device during prepare.
    fn note_packed_prepare(&self, bytes: u64) {
        self.packed_prepare_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn note_staged_upload(&self, bytes: u64) {
        let index = self.phase_index();
        self.staged_uploads[index].fetch_add(1, Ordering::Relaxed);
        self.staged_upload_bytes[index].fetch_add(bytes, Ordering::Relaxed);
    }

    fn note_readback(&self, bytes: u64) {
        self.readback_calls.fetch_add(1, Ordering::Relaxed);
        self.readback_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn note_resident_alloc(&self, bytes: u64) {
        self.resident_tensors.fetch_add(1, Ordering::Relaxed);
        self.resident_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn note_resident_bind(&self, bytes: u64) {
        let index = self.phase_index();
        self.resident_binds[index].fetch_add(1, Ordering::Relaxed);
        self.resident_bind_bytes[index].fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed submission scope (used by the step batch).
    pub fn note_batch(&self, submissions: u32) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.batch_submissions
            .fetch_add(u64::from(submissions), Ordering::Relaxed);
    }
}

/// A device allocation whose bytes are authoritative in device memory.
///
/// The owner is the [`Storage`] that wraps it (through an `Arc`), so dropping
/// the last tensor handle destroys the buffer. `download` is the one path that
/// brings the bytes back to the host, and it is one explicit submission.
#[derive(Debug)]
pub struct ResidentBuffer {
    buffer: Arc<Buffer>,
    device: Arc<Device>,
    device_id: DeviceId,
    label: String,
    /// Logical length: the tensor's byte count, not the padded allocation.
    logical: usize,
}

impl ResidentBuffer {
    /// Wrap a just-allocated device buffer whose logical length is `logical`.
    ///
    /// The buffer may be up to three bytes longer (the 4-byte copy
    /// granularity); those bytes are never addressable through the tensor.
    pub fn new(
        buffer: Buffer,
        device: Arc<Device>,
        device_id: DeviceId,
        label: String,
        logical: usize,
    ) -> Self {
        ResidentBuffer {
            buffer: Arc::new(buffer),
            device,
            device_id,
            label,
            logical,
        }
    }

    /// The underlying Vulkan buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// The device the allocation lives on.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }
}

impl DeviceResidency for ResidentBuffer {
    fn backend(&self) -> &'static str {
        super::VULKAN_BACKEND_NAME
    }

    fn bytes(&self) -> usize {
        self.logical
    }

    fn describe(&self) -> String {
        format!(
            "buffer {} \"{}\" {} B ({:?}) on {}",
            self.buffer.id(),
            self.label,
            self.logical,
            self.buffer.class(),
            self.device_id.diagnostic()
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn download(&self, offset: usize, dst: &mut [u8]) -> Fn3Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let size = dst.len() as u64;
        let end = (offset as u64).saturating_add(size);
        if end > self.logical as u64 {
            return Err(Fn3Error::OutOfBounds {
                what: "resident download",
                index: end,
                len: self.logical as u64,
            });
        }
        self.buffer
            .check_range(offset as u64, size, "resident download")
            .map_err(|error| error.to_fn3())?;
        #[cfg(feature = "vulkan-fault-injection")]
        if self
            .device
            .faults()
            .take(super::fault::FaultPoint::Readback)
        {
            // A host-side readback failure is recoverable: the device state is
            // intact and the caller can retry the step. It is reported at the
            // same seam as a failed `vkMapMemory`, before any submission.
            return Err(Fn3Error::Allocation {
                bytes: size,
                detail: format!(
                    "injected readback failure (FN3-27): {size} bytes at offset {offset} from \
                     \"{}\"",
                    self.label
                ),
            });
        }
        // One host-visible block, one submission, one wait. A download is an
        // explicit host-side materialisation, not the steady-state path, so a
        // fresh block per call is the honest cost model.
        let pool = StagingPool::new(Arc::clone(&self.device), size);
        let lease = pool.acquire(size).map_err(|error| error.to_fn3())?;
        let mut submitter = Submitter::new(&self.device).map_err(|error| error.to_fn3())?;
        submitter
            .record(|recorder| {
                recorder.barrier(
                    &self.buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
                recorder.copy_buffer(&self.buffer, offset as u64, lease.buffer(), 0, size)
            })
            .map_err(|error| error.to_fn3())?
            .wait()
            .map_err(|error| error.to_fn3())?;
        drop(submitter);
        lease.read_into(dst).map_err(|error| error.to_fn3())
    }
}

/// Identity of one immutable tensor in the persistent table.
///
/// `identity` is [`Storage::generation`], a process generation assigned when the
/// storage is created. It is deliberately not a host address: an address can be
/// reused by the allocator after the previous model is dropped, while a
/// generation cannot. `offset` and `len` keep two windows over one storage
/// distinct.
///
/// [`Storage::generation`]: crate::inference::storage::Storage::generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PersistentKey {
    identity: u64,
    offset: usize,
    len: usize,
}

impl PersistentKey {
    /// Key of a host view, or `None` when the view cannot be a persistent
    /// tensor (strided, resident, byte-window backed, or zero bytes).
    fn of_view(view: &TensorView<'_>) -> Option<PersistentKey> {
        // A byte-window view has no owning allocation to key on, so it can
        // never have been uploaded by `prepare_persistent`.
        let storage = view.storage()?;
        if storage.is_resident() || !view.is_contiguous() {
            return None;
        }
        let len = view.meta().byte_len().ok()?;
        if len == 0 {
            return None;
        }
        let offset = view.meta().byte_offset().ok()?;
        Some(PersistentKey {
            identity: storage.generation(),
            offset,
            len,
        })
    }

    fn of_tensor(tensor: &Tensor) -> Option<PersistentKey> {
        if tensor.storage().is_resident() || !tensor.is_contiguous() {
            return None;
        }
        let len = tensor.meta().byte_len().ok()?;
        if len == 0 {
            return None;
        }
        let offset = tensor.meta().byte_offset().ok()?;
        Some(PersistentKey {
            identity: tensor.storage().generation(),
            offset,
            len,
        })
    }
}

/// One device copy per immutable tensor, keyed by storage generation.
///
/// # Lifetime
///
/// An entry is owned by the host allocation it was uploaded from. When the
/// last [`Storage`] handle for that allocation drops, the hook registered by
/// [`VulkanBackend::prepare_persistent_tensors`] removes the entry and drops
/// the device buffer. Several sessions over one loaded model share the model's
/// weight storages, so the second session still hits the table; a later model
/// gets fresh storages (fresh generations) and can never resolve an old entry,
/// even if the host allocator reuses the address.
///
/// [`Storage`]: crate::inference::storage::Storage
/// [`VulkanBackend::prepare_persistent_tensors`]: super::VulkanBackend::prepare_persistent_tensors
#[derive(Debug, Clone, Default)]
pub struct PersistentTable {
    inner: Arc<PersistentEntries>,
}

#[derive(Debug, Default)]
struct PersistentEntries {
    entries: Mutex<HashMap<PersistentKey, Arc<Buffer>>>,
    live_bytes: AtomicU64,
    releases: AtomicU64,
}

impl PersistentEntries {
    fn entries(&self) -> std::sync::MutexGuard<'_, HashMap<PersistentKey, Arc<Buffer>>> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Remove `key` because its owner dropped, updating the live counters.
    ///
    /// `clear` removes entries without counting them as owner releases, so a
    /// missing key here is normal and must not underflow [`Self::live_bytes`].
    fn release(&self, key: &PersistentKey) {
        let removed = self.entries().remove(key);
        if removed.is_some() {
            self.live_bytes.fetch_sub(key.len as u64, Ordering::Relaxed);
            self.releases.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl PersistentTable {
    /// An empty table.
    pub fn new() -> PersistentTable {
        PersistentTable::default()
    }

    /// Number of resident copies.
    pub fn len(&self) -> usize {
        self.inner.entries().len()
    }

    /// `true` when nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device bytes currently held for persistent tensors.
    pub fn live_bytes(&self) -> u64 {
        self.inner.live_bytes.load(Ordering::Relaxed)
    }

    /// Entries released because their owning storage dropped.
    ///
    /// This counter is not reset by [`VulkanBackend::reset_residency_stats`];
    /// it is a table lifetime total, so a test can subtract a baseline.
    ///
    /// [`VulkanBackend::reset_residency_stats`]: super::VulkanBackend::reset_residency_stats
    pub fn releases(&self) -> u64 {
        self.inner.releases.load(Ordering::Relaxed)
    }

    fn get(&self, key: &PersistentKey) -> Option<Arc<Buffer>> {
        self.inner.entries().get(key).cloned()
    }

    fn contains(&self, key: &PersistentKey) -> bool {
        self.inner.entries().contains_key(key)
    }

    fn insert(&self, key: PersistentKey, buffer: Arc<Buffer>) -> bool {
        let mut entries = self.inner.entries();
        if entries.contains_key(&key) {
            return false;
        }
        entries.insert(key, buffer);
        self.inner
            .live_bytes
            .fetch_add(key.len as u64, Ordering::Relaxed);
        true
    }

    /// A hook that removes `key` when its owning storage drops.
    ///
    /// The closure holds a weak reference to the table, so a `Storage` that
    /// outlives the backend (a model may) does not keep the device alive.
    fn ownership_hook(&self, key: PersistentKey) -> impl FnOnce() + Send + 'static {
        let table = Arc::downgrade(&self.inner);
        move || {
            if let Some(table) = table.upgrade() {
                table.release(&key);
            }
        }
    }

    /// Forget every entry (the buffers drop when their last `Arc` is released).
    ///
    /// This is the explicit diagnostic reset; it does not increment
    /// [`PersistentTable::releases`], which counts owner-driven evictions only.
    pub fn clear(&self) {
        let mut entries = self.inner.entries();
        if entries.is_empty() {
            return;
        }
        entries.clear();
        self.inner.live_bytes.store(0, Ordering::Relaxed);
    }
}

/// Round a byte count up to the 4-byte granularity `vkCmdCopyBuffer` requires.
pub fn padded_bytes(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(4) * 4
    }
}

/// Require a resident operand to be a whole, contiguous, offset-zero buffer bind.
///
/// The kernels index a bound buffer from zero, so an operand that starts inside
/// its storage or strides across it cannot be bound without a shader change.
/// That case is reported by name rather than bound at the wrong offset — a wrong
/// binding would compute the wrong answer, which is far worse than an error.
pub fn require_whole_meta(meta: &TensorMeta, what: &'static str) -> Fn3Result<()> {
    if !meta.is_contiguous() {
        return Err(Fn3Error::NonContiguous {
            context: what,
            detail: format!(
                "a device-resident operand must be contiguous to be bound as one buffer; {} \
                 uses {}",
                meta.shape(),
                meta.strides()
            ),
        });
    }
    if meta.offset() != 0 {
        return Err(Fn3Error::InvalidShape {
            context: what,
            detail: format!(
                "a device-resident operand must start at element 0 of its buffer, but {} starts \
                 at element {}; materialise the window with to_contiguous first",
                meta.shape(),
                meta.offset()
            ),
        });
    }
    Ok(())
}

/// [`require_whole_meta`] for a borrowed view.
pub fn require_whole_buffer(view: &TensorView<'_>, what: &'static str) -> Fn3Result<()> {
    require_whole_meta(view.meta(), what)
}

impl super::VulkanBackend {
    /// Attribute subsequent counter events to `phase`.
    pub fn set_phase(&self, phase: ResidencyPhase) {
        self.residency_counters.set_phase(phase);
    }

    /// A snapshot of the residency counters.
    pub fn residency_stats(&self) -> ResidencyStats {
        self.residency_counters.snapshot()
    }

    /// Reset the residency counters, keeping the current phase.
    pub fn reset_residency_stats(&self) {
        self.residency_counters.reset();
    }

    /// Number of persistent tensors held on the device.
    pub fn persistent_tensor_count(&self) -> usize {
        self.persistent.len()
    }

    /// Device bytes the persistent table holds right now.
    pub fn persistent_live_bytes(&self) -> u64 {
        self.persistent.live_bytes()
    }

    /// Persistent entries released because their owning storage dropped.
    ///
    /// A table-lifetime total; [`VulkanBackend::reset_residency_stats`](crate::inference::backend::vulkan::VulkanBackend::reset_residency_stats) does
    /// not clear it.
    pub fn persistent_release_count(&self) -> u64 {
        self.persistent.releases()
    }

    /// Upload immutable tensors once and resolve them by handle afterwards.
    ///
    /// Idempotent: a tensor already in the table is a hit and moves no bytes,
    /// which is what makes several sessions over one model cheap. The host
    /// storage is never touched, so a CPU run of the same model keeps working.
    pub fn prepare_persistent_tensors(&self, tensors: &[&Tensor]) -> Fn3Result<()> {
        let _gate = self.prepare_gate_write();
        for tensor in tensors {
            let Some(key) = PersistentKey::of_tensor(tensor) else {
                // Zero-byte or strided immutable tensors are not bindable as one
                // buffer; they keep the staged path. Nothing is silently skipped
                // that a kernel could bind whole.
                continue;
            };
            if self.persistent.contains(&key) {
                self.residency_counters.note_persistent_hit(key.len as u64);
                continue;
            }
            let bytes = key.len as u64;
            let size = padded_bytes(bytes).max(4);
            let buffer =
                self.alloc_buffer(&BufferDesc::device_storage(size, "persistent.tensor"))?;
            let host = tensor.storage().host_bytes()?;
            let start = key.offset;
            let end = start + key.len;
            let window = host.get(start..end).ok_or_else(|| {
                Error::internal("persistent tensor window is outside its storage").to_fn3()
            })?;
            self.upload(&buffer, window)?;
            let buffer = Arc::new(buffer);
            if self.persistent.insert(key, Arc::clone(&buffer)) {
                // The entry is owned by this allocation: when its last handle
                // drops, remove the device copy. One hook per entry, exactly
                // once, and no host address in the key.
                tensor
                    .storage()
                    .on_last_handle_dropped(Box::new(self.persistent.ownership_hook(key)));
                self.residency_counters.note_persistent(bytes, true);
            }
        }
        Ok(())
    }

    /// The device buffer for a resident or persistent input, when there is one.
    ///
    /// `Ok(None)` means "no device copy exists; the caller stages the host
    /// bytes". A resident tensor that cannot be bound whole is an error, never a
    /// silent fallback to a host read of an empty buffer.
    pub fn resident_input(&self, view: &TensorView<'_>) -> Fn3Result<Option<Arc<Buffer>>> {
        let Some(storage) = view.storage() else {
            // A byte-window view owns no allocation, so no device copy of it can
            // exist: the caller stages its host bytes, exactly as for any other
            // non-resident input.
            return Ok(None);
        };
        if storage.is_resident() {
            require_whole_buffer(view, "vulkan resident input")?;
            let buffer = self.resident_storage_buffer(storage)?;
            let bytes = view.meta().byte_len()? as u64;
            if bytes > buffer.size() {
                return Err(Fn3Error::OutOfBounds {
                    what: "vulkan resident input (device bytes)",
                    index: bytes,
                    len: buffer.size(),
                });
            }
            self.residency_counters.note_resident_bind(bytes);
            return Ok(Some(buffer));
        }
        if let Some(key) = PersistentKey::of_view(view) {
            if let Some(buffer) = self.persistent.get(&key) {
                self.residency_counters.note_persistent_hit(key.len as u64);
                return Ok(Some(buffer));
            }
        }
        Ok(None)
    }

    /// The device buffer for a resident output, when there is one.
    pub fn resident_output(&self, output: &TensorMut<'_>) -> Fn3Result<Option<Arc<Buffer>>> {
        let Some(storage) = output.resident_storage() else {
            return Ok(None);
        };
        require_whole_meta(output.meta(), "vulkan resident output")?;
        let buffer = self.resident_storage_buffer(storage)?;
        let bytes = output.meta().byte_len()? as u64;
        if bytes > buffer.size() {
            return Err(Fn3Error::OutOfBounds {
                what: "vulkan resident output (device bytes)",
                index: bytes,
                len: buffer.size(),
            });
        }
        self.residency_counters.note_resident_bind(bytes);
        Ok(Some(buffer))
    }

    /// Reject a host upload of a tensor that `prepare_persistent` made resident.
    ///
    /// This is the enforcement half of "zero weight bytes uploaded after
    /// prepare": a kernel that forgets to ask [`VulkanBackend::resident_input`](crate::inference::backend::vulkan::VulkanBackend::resident_input)
    /// fails here instead of quietly moving the model again.
    pub fn check_staged_input(&self, view: Option<&TensorView<'_>>, bytes: u64) -> Fn3Result<()> {
        if let Some(view) = view {
            if let Some(key) = PersistentKey::of_view(view) {
                if self.persistent.contains(&key) {
                    self.residency_counters.note_late_persistent(key.len as u64);
                    return Err(Error::internal(format!(
                        "persistent tensor \"{}\" reached a host upload after prepare ({} bytes); \
                         bind it with VulkanBackend::resident_input",
                        view.storage()
                            .map_or("window", |storage| storage.residency_label()),
                        key.len
                    ))
                    .to_fn3());
                }
            }
        }
        self.residency_counters.note_staged_upload(bytes);
        Ok(())
    }

    /// Record one immutable packed-weight transfer as prepare traffic.
    pub fn note_packed_prepare(&self, bytes: u64) {
        self.residency_counters.note_packed_prepare(bytes);
    }

    /// Record a device → host readback (the explicit download path).
    pub fn note_readback(&self, bytes: u64) {
        self.residency_counters.note_readback(bytes);
    }

    /// Allocate a device-resident tensor on this backend's device.
    pub fn allocate_resident(
        &self,
        shape: &crate::inference::shape::Shape,
        dtype: DType,
    ) -> Fn3Result<Tensor> {
        let meta = TensorMeta::contiguous(shape.clone(), dtype, self.device_id())?;
        let required = meta.required_bytes()?;
        let size = padded_bytes(required as u64).max(4);
        self.capabilities.check_buffer(size)?;
        let buffer = self.alloc_buffer(&BufferDesc::device_storage(size, RESIDENT_BUFFER_NAME))?;
        let label = format!("{dtype}{shape}");
        let resident = ResidentBuffer::new(
            buffer,
            Arc::clone(self.vulkan_device()),
            self.device_id(),
            label,
            required,
        );
        let bytes = resident.bytes();
        let storage = Storage::from_resident(required, Arc::new(resident))?;
        self.residency_counters.note_resident_alloc(bytes as u64);
        Tensor::from_storage(storage, meta)
    }

    /// Upload host bytes into a device-resident tensor at `byte_offset`.
    ///
    /// This is the session-boundary input path: token ids, positions and the
    /// attention mask are computed on the host every step and must reach the
    /// device exactly once per step.
    pub fn upload_resident(
        &self,
        tensor: &Tensor,
        byte_offset: u64,
        bytes: &[u8],
    ) -> Fn3Result<()> {
        if !tensor.storage().is_resident() {
            return Err(Fn3Error::InvalidShape {
                context: "VulkanBackend::upload_resident",
                detail: "the tensor is host-backed; write its buffer directly instead".to_string(),
            });
        }
        let buffer = self.resident_storage_buffer(tensor.storage())?;
        let mut padded = bytes.to_vec();
        let size = padded_bytes(bytes.len() as u64);
        padded.resize(size as usize, 0);
        if size > buffer.size() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan resident upload",
                requested: size,
                capacity: buffer.size(),
            });
        }
        if !byte_offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(Error::internal(
                "a resident upload needs 4-byte offsets and a 4-byte-aligned length",
            )
            .to_fn3());
        }
        let pool = StagingPool::new(Arc::clone(self.vulkan_device()), size.max(4096));
        let lease = pool.acquire(size).map_err(|error| error.to_fn3())?;
        lease.write(&padded).map_err(|error| error.to_fn3())?;
        let mut submitter = self.checkout_submitter()?;
        submitter
            .record(|recorder| {
                recorder.copy_buffer(lease.buffer(), 0, &buffer, byte_offset, size)?;
                recorder.barrier(
                    &buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                );
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
            .wait()
            .map_err(|error| error.to_fn3())?;
        drop(submitter);
        self.residency_counters
            .note_staged_upload(bytes.len() as u64);
        Ok(())
    }

    /// Replace `bytes` at `byte_offset` of a resident storage from the host.
    pub fn write_resident_storage(
        &self,
        storage: &Storage,
        byte_offset: u64,
        bytes: &[u8],
    ) -> Fn3Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let buffer = self.resident_storage_buffer(storage)?;
        let mut padded = bytes.to_vec();
        let size = padded_bytes(bytes.len() as u64);
        padded.resize(size as usize, 0);
        let end = byte_offset
            .checked_add(size)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "VulkanBackend::write_resident_storage",
                axis: None,
                detail: format!("offset {byte_offset} plus {size} bytes exceeds u64::MAX"),
            })?;
        if end > buffer.size() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan resident write",
                requested: end,
                capacity: buffer.size(),
            });
        }
        if !byte_offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(Error::internal(
                "a resident write needs 4-byte offsets and a 4-byte-aligned length",
            )
            .to_fn3());
        }
        let pool = StagingPool::new(Arc::clone(self.vulkan_device()), size.max(4096));
        let lease = pool.acquire(size).map_err(|error| error.to_fn3())?;
        lease.write(&padded).map_err(|error| error.to_fn3())?;
        let mut submitter = self.checkout_submitter()?;
        submitter
            .record(|recorder| {
                recorder.copy_buffer(lease.buffer(), 0, &buffer, byte_offset, size)?;
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
            .wait()
            .map_err(|error| error.to_fn3())?;
        drop(submitter);
        self.residency_counters
            .note_staged_upload(bytes.len() as u64);
        Ok(())
    }

    /// Copy `regions` between two resident storages in one submission.
    pub fn copy_resident_storage_regions(
        &self,
        dst: &Storage,
        src: &Storage,
        regions: &[(u64, u64, u64)],
    ) -> Fn3Result<()> {
        if regions.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "vulkan-fault-injection")]
        if self
            .vulkan_device()
            .faults()
            .take(super::fault::FaultPoint::ResidentCopy)
        {
            // A copy that fails before submission leaves both buffers intact,
            // so the KV commit this copy belongs to can be rolled back and the
            // step retried. This is the "failure between the KV append and the
            // next plan" boundary of the FN3-27 matrix.
            let bytes: u64 = regions.iter().map(|(_, _, len)| *len).sum();
            return Err(Fn3Error::Allocation {
                bytes,
                detail: format!(
                    "injected device-copy failure (FN3-27): {} region(s), {bytes} bytes",
                    regions.len()
                ),
            });
        }
        let dst_buffer = self.resident_storage_buffer(dst)?;
        let src_buffer = self.resident_storage_buffer(src)?;
        let mut submitter = self.checkout_submitter()?;
        submitter
            .record(|recorder| {
                for &(dst_offset, src_offset, len) in regions {
                    recorder.copy_buffer(&src_buffer, src_offset, &dst_buffer, dst_offset, len)?;
                }
                recorder.barrier(
                    &dst_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                );
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
            .wait()
            .map_err(|error| error.to_fn3())?;
        drop(submitter);
        Ok(())
    }

    /// The device buffer behind a resident storage on this backend.
    pub fn resident_storage_buffer(&self, storage: &Storage) -> Fn3Result<Arc<Buffer>> {
        let residency = storage.residency().ok_or_else(|| Fn3Error::InvalidShape {
            context: "VulkanBackend::resident_storage_buffer",
            detail: "the storage is host-backed, so it has no device buffer".to_string(),
        })?;
        let resident = residency
            .as_any()
            .downcast_ref::<ResidentBuffer>()
            .ok_or_else(|| Fn3Error::InvalidShape {
                context: "VulkanBackend::resident_storage_buffer",
                detail: format!(
                    "the storage belongs to the {} backend, not to this Vulkan device",
                    residency.backend()
                ),
            })?;
        if resident.device_id() != self.device_id() {
            return Err(Fn3Error::DeviceMismatch {
                op: None,
                expected: self.device_id(),
                actual: resident.device_id(),
            });
        }
        Ok(Arc::clone(resident.buffer()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_rounds_to_the_copy_granularity() {
        assert_eq!(padded_bytes(0), 0);
        assert_eq!(padded_bytes(1), 4);
        assert_eq!(padded_bytes(4), 4);
        assert_eq!(padded_bytes(5), 8);
        assert_eq!(padded_bytes(201_028), 201_028);
    }

    #[test]
    fn phases_round_trip() {
        let counters = ResidencyCounters::new();
        assert_eq!(counters.phase(), ResidencyPhase::Prepare);
        counters.set_phase(ResidencyPhase::Decode);
        assert_eq!(counters.phase(), ResidencyPhase::Decode);
        assert_eq!(counters.phase().name(), "decode");
        counters.note_staged_upload(64);
        let stats = counters.snapshot();
        assert_eq!(stats.staged_uploads[2], 1);
        assert_eq!(stats.staged_upload_bytes[2], 64);
        assert_eq!(stats.staged_upload_bytes[0], 0);
        counters.reset();
        assert_eq!(counters.snapshot().staged_upload_bytes[2], 0);
    }
}
