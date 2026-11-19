//! Owned quarantine for work whose GPU completion is unknown.
//!
//! # Why this exists
//!
//! Every submission is fence-based, but a `vkWaitForFences` error or timeout
//! leaves completion *unknown*: the device may still read a staging block or
//! write a destination buffer. Dropping, destroying or recycling any resource
//! that submission references would let a later GPU write touch freed memory.
//! [`Quarantine`] is the owning home for exactly those resources until either
//!
//! 1. a later wait proves that fence signalled ([`Device::reclaim_quarantine`]),
//!    or
//! 2. device teardown quiesced first.
//!
//! A resource never leaves the quarantine on an unproven path. If teardown's
//! `vkDeviceWaitIdle` fails, the resources are deliberately leaked and the
//! reason is printed and counted — a leak is acceptable, a use-after-free is
//! not.
//!
//! # The buffer reference rule
//!
//! A buffer can be referenced by more than one submission at a time: a shared
//! persistent weight or a shared resident tensor is bound by two per-thread
//! submitters. Proof that *one* submission's fence signalled is therefore not
//! proof that a buffer is unused.
//!
//! Every buffer gets a device-wide reference count while it is inside a
//! submission's command scope:
//!
//! - incremented when the recording is committed to a submission (before
//!   `vkQueueSubmit`, so the count covers the submit window too),
//! - decremented only on a proven completion (a successful fence wait) or a
//!   proof-equivalent discard (a recording that never reached the queue),
//! - moved, not changed, when the submission retires into the quarantine.
//!
//! A buffer is destroyed only when its reference count reaches zero *and* it is
//! quarantined. `reclaim_quarantine` releases an entry whose own fence is
//! proven, but sweeps the quarantined buffers only when no live reference
//! remains; a buffer shared with a still-running submission stays owned and is
//! freed by a later reclaim (or teardown) once the last reference is released.
//! A ticket that arrives for an id another entry already quarantined joins the
//! ownership set instead of being dropped, so that later proof can still free
//! it.
//!
//! # What is owned, per kind
//!
//! | Resource | Representation | Released when |
//! |---|---|---|
//! | destination/source buffers + allocator ranges | [`BufferTicket`] | its fence is proven, or teardown |
//! | staging blocks | [`BufferTicket`] (the pool refuses to re-pool) | same as buffers |
//! | command pool, command buffer, descriptor pool (and every set it owns), query pool | [`RetainedSubmitter`] | same submission fence proven, or teardown |
//! | fence | [`RetainedSubmitter::fence`] | after the proof wait, or teardown |
//!
//! Descriptor sets have no independent lifetime: a set is allocated from the
//! transient descriptor pool and is freed by it, so quarantining the pool
//! quarantines the sets.
//!
//! # Unrecorded buffers
//!
//! A buffer that drops while the device is terminal but was never recorded in a
//! submission is *loose*: it has no fence that could prove it, so it is
//! released only at teardown, never by a reclaim pass.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use super::context::BufferTicket;
use super::device::Device;

/// Raw ownership of one submitter's Vulkan objects.
///
/// The submitter cannot be reused once this exists: its command buffer and
/// descriptor sets may still be read or written by the unproven submission.
#[derive(Debug)]
pub(crate) struct RetainedSubmitter {
    /// Command pool; freeing it frees `command_buffer`.
    pub(crate) command_pool: vk::CommandPool,
    /// The single primary command buffer allocated from `command_pool`.
    pub(crate) command_buffer: vk::CommandBuffer,
    /// Submission fence, still in whatever signal state the driver left it.
    pub(crate) fence: vk::Fence,
    /// Transient descriptor pool; freeing it frees every set it owns.
    pub(crate) descriptor_pool: vk::DescriptorPool,
    /// Device timestamps, when the adapter supports them.
    pub(crate) query_pool: Option<vk::QueryPool>,
}

/// One submission whose completion is unknown, with every resource it owns.
#[derive(Debug)]
pub(crate) struct PendingEntry {
    /// Human-readable origin (`"upload"`, `"step batch"`, ...).
    pub(crate) label: String,
    /// The submission fence. `None` means nothing was submitted before the
    /// failure, so there is no GPU work to wait for.
    pub(crate) fence: Option<vk::Fence>,
    /// Command/descriptor pools and the fence, when a submitter was in flight.
    pub(crate) submitter: Option<RetainedSubmitter>,
    /// Buffers and staging blocks the recording referenced. The buffer's
    /// reference count was reserved at commit time; this entry owns that
    /// reference until it is released.
    pub(crate) buffers: Vec<BufferTicket>,
}

impl PendingEntry {
    /// Build an entry. `fence` is `None` only when nothing reached the queue.
    pub(crate) fn new(
        label: impl Into<String>,
        fence: Option<vk::Fence>,
        submitter: Option<RetainedSubmitter>,
        buffers: Vec<BufferTicket>,
    ) -> PendingEntry {
        PendingEntry {
            label: label.into(),
            fence,
            submitter,
            buffers,
        }
    }
}

/// A snapshot of what the quarantine currently owns and has released.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuarantineStats {
    /// Entries held by the quarantine.
    pub entries: usize,
    /// Entries with a fence: a submission whose completion is unknown.
    pub pending_submissions: usize,
    /// Command/descriptor pools and fences held.
    pub retained_submitters: usize,
    /// Buffers the quarantine owns (entry-held or waiting for a zero reference).
    pub retained_buffers: usize,
    /// Buffers that dropped on a terminal device outside a recorded
    /// submission; released only at teardown.
    pub loose_buffers: usize,
    /// Distinct buffer ids with at least one live submission reference.
    pub in_flight_buffers: usize,
    /// Entries captured while their fence had not signalled (diagnostic).
    pub captured_unsignalled: u64,
    /// Entries released after a proven wait.
    pub released_entries: u64,
    /// Submitters released after a proven wait.
    pub released_submitters: u64,
    /// Buffers released after a proven wait.
    pub released_buffers: u64,
    /// Entries deliberately leaked because teardown could not quiesce.
    pub leaked_entries: u64,
    /// Buffers deliberately leaked because teardown could not quiesce.
    pub leaked_buffers: u64,
    /// Drops whose id was neither live, quarantined nor already retired.
    /// Must stay zero; a non-zero value is a bookkeeping leak.
    pub orphan_drops: u64,
    /// Recorded tickets whose id had no live buffer at retirement. Must stay
    /// zero; a non-zero value means a caller dropped an in-flight buffer.
    pub lost_tickets: u64,
}

/// One explicit reclaim pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReclaimReport {
    /// Entries whose fence was proven and whose submitter was destroyed.
    pub entries: usize,
    /// Submitters destroyed by this pass.
    pub submitters: usize,
    /// Quarantined buffers destroyed by this pass (reference count zero).
    pub buffers: usize,
    /// Entries whose fence had not signalled (still quarantined).
    pub pending: usize,
}

/// What a teardown did with the quarantine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TeardownReport {
    /// Entries present at teardown.
    pub entries: usize,
    /// Submitters present at teardown.
    pub submitters: usize,
    /// Buffers present at teardown.
    pub buffers: usize,
    /// `true` when quiescing failed and the resources were leaked.
    pub leaked: bool,
}

/// Process-wide teardown outcomes, the observable half of the leak path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TeardownTotals {
    /// Teardowns whose `vkDeviceWaitIdle` succeeded.
    pub quiesced_ok: u64,
    /// Teardown attempts whose `vkDeviceWaitIdle` failed (resources leaked).
    pub quiesced_failed: u64,
    /// Quarantined entries destroyed after a successful quiesce.
    pub released_entries: u64,
    /// Submitters destroyed after a successful quiesce.
    pub released_submitters: u64,
    /// Buffers destroyed after a successful quiesce.
    pub released_buffers: u64,
    /// Entries deliberately leaked.
    pub leaked_entries: u64,
    /// Buffers deliberately leaked.
    pub leaked_buffers: u64,
    /// Orphan drops observed over the process lifetime.
    pub orphan_drops: u64,
}

static TEARDOWN_QUIESCED_OK: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_QUIESCED_FAILED: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_RELEASED_ENTRIES: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_RELEASED_SUBMITTERS: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_RELEASED_BUFFERS: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_LEAKED_ENTRIES: AtomicU64 = AtomicU64::new(0);
static TEARDOWN_LEAKED_BUFFERS: AtomicU64 = AtomicU64::new(0);
static ORPHAN_DROPS: AtomicU64 = AtomicU64::new(0);

/// Process-wide teardown evidence, for tests and reports.
pub fn teardown_totals() -> TeardownTotals {
    TeardownTotals {
        quiesced_ok: TEARDOWN_QUIESCED_OK.load(Ordering::Relaxed),
        quiesced_failed: TEARDOWN_QUIESCED_FAILED.load(Ordering::Relaxed),
        released_entries: TEARDOWN_RELEASED_ENTRIES.load(Ordering::Relaxed),
        released_submitters: TEARDOWN_RELEASED_SUBMITTERS.load(Ordering::Relaxed),
        released_buffers: TEARDOWN_RELEASED_BUFFERS.load(Ordering::Relaxed),
        leaked_entries: TEARDOWN_LEAKED_ENTRIES.load(Ordering::Relaxed),
        leaked_buffers: TEARDOWN_LEAKED_BUFFERS.load(Ordering::Relaxed),
        orphan_drops: ORPHAN_DROPS.load(Ordering::Relaxed),
    }
}

fn record_teardown(report: &TeardownReport, boundary: bool) {
    if boundary {
        if report.leaked {
            TEARDOWN_QUIESCED_FAILED.fetch_add(1, Ordering::Relaxed);
        } else {
            TEARDOWN_QUIESCED_OK.fetch_add(1, Ordering::Relaxed);
        }
    }
    if report.leaked {
        TEARDOWN_LEAKED_ENTRIES.fetch_add(report.entries as u64, Ordering::Relaxed);
        TEARDOWN_LEAKED_BUFFERS.fetch_add(report.buffers as u64, Ordering::Relaxed);
    } else {
        TEARDOWN_RELEASED_ENTRIES.fetch_add(report.entries as u64, Ordering::Relaxed);
        TEARDOWN_RELEASED_SUBMITTERS.fetch_add(report.submitters as u64, Ordering::Relaxed);
        TEARDOWN_RELEASED_BUFFERS.fetch_add(report.buffers as u64, Ordering::Relaxed);
    }
}

/// The owning quarantine. Lives behind a mutex on [`Device`].
#[derive(Default)]
pub(crate) struct Quarantine {
    /// Buffer ids this device created and has not handed to the quarantine.
    live: HashSet<u64>,
    /// Buffer ids whose destruction the quarantine owns (entry-held or loose).
    quarantined: HashSet<u64>,
    /// Destruction data for every entry-owned quarantined buffer, keyed by id.
    /// A buffer stays here while any reference remains, even after every entry
    /// that introduced it was released.
    tickets: HashMap<u64, BufferTicket>,
    /// Live submission references per buffer id. Each committed submission and
    /// each quarantined entry holds one.
    inflight: HashMap<u64, usize>,
    /// Ids the quarantine already destroyed; a late `Buffer::drop` for one is
    /// a tombstone, not an orphan.
    gone: HashSet<u64>,
    /// Fenced (and no-submission) entries, oldest first.
    entries: Vec<PendingEntry>,
    /// Buffers that dropped on a terminal device without being recorded.
    loose: Vec<BufferTicket>,
    captured_unsignalled: u64,
    released_entries: u64,
    released_submitters: u64,
    released_buffers: u64,
    leaked_entries: u64,
    leaked_buffers: u64,
    orphan_drops: u64,
    lost_tickets: u64,
}

impl Quarantine {
    /// An empty quarantine.
    pub(crate) fn new() -> Quarantine {
        Quarantine::default()
    }

    /// Register a buffer at creation.
    pub(crate) fn register_live(&mut self, id: u64) {
        self.live.insert(id);
    }

    /// `true` when the quarantine owns this buffer's destruction.
    pub(crate) fn is_quarantined(&self, id: u64) -> bool {
        self.quarantined.contains(&id)
    }

    /// `true` after the quarantine destroyed this buffer (a tombstone).
    pub(crate) fn is_retired(&self, id: u64) -> bool {
        self.gone.contains(&id)
    }

    /// Take one reference for every buffer committed to a submission.
    ///
    /// Called before `vkQueueSubmit`, so a submit that is still in the queue
    /// keeps the count. Duplicate ids in one recording count once.
    pub(crate) fn reserve(&mut self, tickets: &[BufferTicket]) {
        let mut seen = HashSet::new();
        for ticket in tickets {
            if seen.insert(ticket.id) {
                *self.inflight.entry(ticket.id).or_insert(0) += 1;
            }
        }
    }

    /// Release references proven complete (a successful wait) or
    /// proof-equivalently discarded (nothing reached the queue).
    ///
    /// A zero count does not destroy anything here: the next reclaim pass (or
    /// teardown) sweeps a quarantined buffer whose last reference is gone.
    /// Keeping the sweep in reclaim makes "proven this pass, freed by a later
    /// reclaim" observable and keeps destruction in one place.
    pub(crate) fn release_refs(&mut self, tickets: &[BufferTicket]) {
        let mut seen = HashSet::new();
        for ticket in tickets {
            if !seen.insert(ticket.id) {
                continue;
            }
            self.drop_one_ref(ticket.id);
        }
    }

    fn drop_one_ref(&mut self, id: u64) {
        let Some(count) = self.inflight.get_mut(&id) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.inflight.remove(&id);
        }
    }

    /// Decide whether a dropping buffer must be destroyed or is already owned.
    ///
    /// Returns `true` when the caller must not destroy it: the quarantine
    /// already owns it, it was already retired, or it is unknown and counted as
    /// an orphan (never freed twice).
    pub(crate) fn defer_on_drop(&mut self, ticket: BufferTicket, terminal: bool) -> bool {
        if self.quarantined.contains(&ticket.id) {
            return true;
        }
        if !self.live.remove(&ticket.id) {
            if self.gone.remove(&ticket.id) {
                return true;
            }
            // Neither live, quarantined nor retired. Do not free a handle we no
            // longer track; count it so a bookkeeping bug is visible.
            self.orphan_drops += 1;
            ORPHAN_DROPS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if !terminal {
            return false;
        }
        self.quarantined.insert(ticket.id);
        self.loose.push(ticket);
        true
    }

    /// Take one submission with every in-flight resource.
    ///
    /// A ticket for an id another entry already quarantined joins that
    /// ownership set (its reference was reserved at commit); a ticket for an id
    /// that is neither live nor quarantined releases its reserved reference and
    /// is counted as a lost ticket instead of being silently dropped.
    pub(crate) fn push_entry(&mut self, mut entry: PendingEntry, fence_unsignalled: bool) {
        let mut retained = Vec::with_capacity(entry.buffers.len());
        let mut seen = HashSet::new();
        for ticket in entry.buffers.drain(..) {
            if !seen.insert(ticket.id) {
                continue;
            }
            if self.quarantined.contains(&ticket.id) {
                if self.loose.iter().any(|loose| loose.id == ticket.id) {
                    // The buffer dropped (and was taken loose) between the
                    // failure and this entry's registration. The fenced entry
                    // is now the authoritative owner; drop the loose copy so
                    // teardown cannot free it twice.
                    self.loose.retain(|loose| loose.id != ticket.id);
                }
                self.tickets.entry(ticket.id).or_insert(ticket);
                retained.push(ticket);
            } else if self.live.remove(&ticket.id) {
                self.quarantined.insert(ticket.id);
                self.tickets.insert(ticket.id, ticket);
                retained.push(ticket);
            } else {
                // No live resource: the caller dropped an in-flight buffer
                // before completion became unknown. Release the reserved
                // reference so it cannot keep another entry pending forever.
                self.drop_one_ref(ticket.id);
                self.lost_tickets += 1;
            }
        }
        entry.buffers = retained;
        if fence_unsignalled {
            self.captured_unsignalled += 1;
        }
        self.entries.push(entry);
    }

    /// Take every entry out for a reclaim pass.
    pub(crate) fn take_entries(&mut self) -> Vec<PendingEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Put entries whose fence did not signal back.
    pub(crate) fn restore_entries(&mut self, entries: Vec<PendingEntry>) {
        self.entries.extend(entries);
    }

    /// Release one entry whose fence is proven: its submitter can be destroyed
    /// and its buffer references are decremented. Buffers whose last reference
    /// this released are swept by [`Quarantine::sweep_zero_refs`] in the same
    /// reclaim pass.
    pub(crate) fn release_entry(&mut self, entry: &PendingEntry) {
        let mut seen = HashSet::new();
        for ticket in &entry.buffers {
            if seen.insert(ticket.id) {
                self.drop_one_ref(ticket.id);
            }
        }
        self.released_entries += 1;
        self.released_submitters += u64::from(entry.submitter.is_some());
    }

    /// Destroy every quarantined buffer whose reference count reached zero.
    ///
    /// Safe by construction: a live submission or an unproven entry still holds
    /// a reference, so a zero count means every referencing fence was proven.
    pub(crate) fn sweep_zero_refs(&mut self) -> Vec<BufferTicket> {
        let ready: Vec<u64> = self
            .tickets
            .keys()
            .copied()
            .filter(|id| self.inflight.get(id).copied().unwrap_or(0) == 0)
            .collect();
        let mut freed = Vec::with_capacity(ready.len());
        for id in ready {
            if let Some(ticket) = self.tickets.remove(&id) {
                self.quarantined.remove(&id);
                self.gone.insert(id);
                freed.push(ticket);
            }
        }
        self.released_buffers += freed.len() as u64;
        freed
    }

    /// Take everything for teardown. `idle` proves that every GPU operation
    /// finished; only then may anything be destroyed.
    pub(crate) fn take_all(
        &mut self,
        idle: bool,
    ) -> (Vec<PendingEntry>, Vec<BufferTicket>, TeardownReport) {
        let entries = std::mem::take(&mut self.entries);
        let loose = std::mem::take(&mut self.loose);
        let submitters = entries
            .iter()
            .filter(|entry| entry.submitter.is_some())
            .count();
        let mut unique = HashSet::new();
        let mut tickets = Vec::new();
        for (id, ticket) in &self.tickets {
            if unique.insert(*id) {
                tickets.push(*ticket);
            }
        }
        for entry in &entries {
            for ticket in &entry.buffers {
                if unique.insert(ticket.id) {
                    tickets.push(*self.tickets.get(&ticket.id).unwrap_or(ticket));
                }
            }
        }
        for ticket in &loose {
            if unique.insert(ticket.id) {
                tickets.push(*ticket);
            }
        }
        let report = TeardownReport {
            entries: entries.len(),
            submitters,
            buffers: tickets.len(),
            leaked: !idle,
        };
        if idle {
            for ticket in &tickets {
                self.quarantined.remove(&ticket.id);
                self.gone.insert(ticket.id);
            }
            self.tickets.clear();
            self.inflight.clear();
            self.released_entries += entries.len() as u64;
            self.released_submitters += submitters as u64;
            self.released_buffers += tickets.len() as u64;
        } else {
            self.leaked_entries += entries.len() as u64;
            self.leaked_buffers += tickets.len() as u64;
        }
        (entries, tickets, report)
    }

    /// Current counters.
    pub(crate) fn stats(&self) -> QuarantineStats {
        QuarantineStats {
            entries: self.entries.len(),
            pending_submissions: self
                .entries
                .iter()
                .filter(|entry| entry.fence.is_some())
                .count(),
            retained_submitters: self
                .entries
                .iter()
                .filter(|entry| entry.submitter.is_some())
                .count(),
            retained_buffers: self.tickets.len(),
            loose_buffers: self.loose.len(),
            in_flight_buffers: self.inflight.len(),
            captured_unsignalled: self.captured_unsignalled,
            released_entries: self.released_entries,
            released_submitters: self.released_submitters,
            released_buffers: self.released_buffers,
            leaked_entries: self.leaked_entries,
            leaked_buffers: self.leaked_buffers,
            orphan_drops: self.orphan_drops,
            lost_tickets: self.lost_tickets,
        }
    }
}

/// Destroy one retained submitter. Teardown/proof only.
pub(crate) fn destroy_submitter(device: &Device, submitter: RetainedSubmitter) {
    let ledger = device.ledger();
    if let Some(query_pool) = submitter.query_pool {
        // SAFETY: the device is quiesced (or this submitter's fence was
        // proven), and the query pool is not in use.
        unsafe { device.ash().destroy_query_pool(query_pool, None) };
        ledger.destroy("query_pool");
    }
    // SAFETY: the descriptor pool owns only sets from completed or
    // never-submitted recordings; no submission is in flight.
    unsafe {
        device
            .ash()
            .destroy_descriptor_pool(submitter.descriptor_pool, None)
    };
    ledger.destroy("descriptor_pool");
    // SAFETY: the fence is not awaited by anyone else and its last user is
    // gone; the device is quiesced or its submission is proven complete.
    unsafe { device.ash().destroy_fence(submitter.fence, None) };
    ledger.destroy("fence");
    // SAFETY: the pool frees `command_buffer`; it is idle after the proof or
    // the device-wide quiesce.
    unsafe {
        device
            .ash()
            .destroy_command_pool(submitter.command_pool, None)
    };
    let _ = submitter.command_buffer;
    ledger.destroy("command_buffer");
    ledger.destroy("command_pool");
}

/// Split proven from unproven entries with an explicit, direct wait, then
/// release the proven entries and sweep buffers whose last reference is gone.
///
/// This is the only release path that bypasses `Device::ensure_usable`: a
/// device can be terminal (poisoned after the first unknown wait) while the
/// driver can still prove that a particular fence signalled. Nothing is
/// released unless `vkWaitForFences` returns success for that entry's fence,
/// and no buffer is destroyed while another submission still references it.
pub(crate) fn reclaim(device: &Device) -> ReclaimReport {
    let entries = {
        let mut quarantine = device.lock_quarantine();
        quarantine.take_entries()
    };
    let timeout = device.fence_timeout_ns();
    let mut proven = Vec::new();
    let mut pending = Vec::new();
    for entry in entries {
        let complete = match entry.fence {
            // Nothing reached the queue, so there is no work to wait for.
            None => true,
            Some(fence) => {
                // SAFETY: the fence was created by this device and submitted
                // exactly once; waiting is read-only with respect to the
                // resource state and cannot free anything.
                matches!(
                    unsafe { device.ash().wait_for_fences(&[fence], true, timeout) },
                    Ok(())
                )
            }
        };
        if complete {
            proven.push(entry);
        } else {
            pending.push(entry);
        }
    }
    let pending_count = pending.len();
    let released_entries = proven.len();
    let released_submitters = proven
        .iter()
        .filter(|entry| entry.submitter.is_some())
        .count();
    let swept = {
        let mut quarantine = device.lock_quarantine();
        quarantine.restore_entries(pending);
        for entry in &proven {
            quarantine.release_entry(entry);
        }
        quarantine.sweep_zero_refs()
    };
    for entry in proven {
        if let Some(submitter) = entry.submitter {
            destroy_submitter(device, submitter);
        }
    }
    for ticket in &swept {
        // The id is out of the quarantine and in the tombstone set: a
        // concurrent `Buffer::drop` cannot destroy it, and this is its only
        // destroy site.
        super::context::destroy_buffer(device, *ticket);
    }
    ReclaimReport {
        entries: released_entries,
        submitters: released_submitters,
        buffers: swept.len(),
        pending: pending_count,
    }
}

/// Quiesce-then-destroy the quarantine at a teardown boundary.
///
/// `idle` is the result of `vkDeviceWaitIdle` (or the equivalent). On `false`,
/// every resource is deliberately leaked and the reason is recorded; nothing
/// is destroyed. `boundary` distinguishes the first teardown record from a
/// later drain of resources that deferred after it (the latter releases
/// resources but does not count as a second successful quiesce; a later
/// *failure* is always counted because it is new evidence).
pub(crate) fn teardown(device: &Device, idle: bool, boundary: bool) -> TeardownReport {
    let (entries, tickets, report) = {
        let mut quarantine = device.lock_quarantine();
        quarantine.take_all(idle)
    };
    if idle {
        for entry in entries {
            if let Some(submitter) = entry.submitter {
                destroy_submitter(device, submitter);
            }
        }
        for ticket in tickets {
            super::context::destroy_buffer(device, ticket);
        }
    } else {
        if report.entries != 0 || report.submitters != 0 || report.buffers != 0 {
            eprintln!(
                "[fn3vk][warn] teardown could not quiesce the device; deliberately leaking {} \
                 quarantined entr{} ({} submitter(s), {} buffer(s)) instead of freeing resources \
                 the GPU may still reference",
                report.entries,
                if report.entries == 1 { "y" } else { "ies" },
                report.submitters,
                report.buffers
            );
        } else {
            eprintln!(
                "[fn3vk][warn] teardown could not quiesce the device; nothing was quarantined, \
                 but the device, its chunks and the loader instance are leaked"
            );
        }
        for entry in &entries {
            eprintln!(
                "[fn3vk][warn]   leaked: {} ({:?}, {} buffer(s))",
                entry.label,
                entry.fence.map(|_| "submitted"),
                entry.buffers.len()
            );
        }
    }
    record_teardown(&report, boundary);
    report
}

/// Record a failed quiesce attempt without draining the quarantine (a retry
/// after a boundary that already promised to leak).
pub(crate) fn record_failed_quiesce() {
    TEARDOWN_QUIESCED_FAILED.fetch_add(1, Ordering::Relaxed);
}

/// The mutex is only ever held for bookkeeping, never across a Vulkan call.
impl std::fmt::Debug for Quarantine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quarantine")
            .field("entries", &self.entries.len())
            .field("tickets", &self.tickets.len())
            .field("loose", &self.loose.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket(id: u64) -> BufferTicket {
        BufferTicket {
            id,
            raw: vk::Buffer::null(),
            allocation: super::super::memory::MemoryAllocation {
                chunk: 1,
                offset: 0,
                size: 4,
                memory_type_index: 0,
            },
            logical_size: 4,
            name: "test",
            kind: "device_buffer",
        }
    }

    #[test]
    fn a_terminal_device_takes_the_loose_path_and_a_healthy_one_does_not() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(1);
        quarantine.register_live(2);
        assert!(
            !quarantine.defer_on_drop(ticket(1), false),
            "a healthy device destroys its own buffer"
        );
        assert!(
            quarantine.defer_on_drop(ticket(2), true),
            "a terminal device defers destruction"
        );
        let stats = quarantine.stats();
        assert_eq!(stats.loose_buffers, 1);
        assert!(quarantine.is_quarantined(2));
        assert!(!quarantine.is_quarantined(1));
        // A second drop of the same id is never a second free.
        assert!(quarantine.defer_on_drop(ticket(2), true));
        assert_eq!(quarantine.stats().loose_buffers, 1);
    }

    #[test]
    fn a_drop_with_no_live_record_counts_as_an_orphan_not_a_silent_defer() {
        let mut quarantine = Quarantine::new();
        assert!(quarantine.defer_on_drop(ticket(99), true));
        let stats = quarantine.stats();
        assert_eq!(stats.orphan_drops, 1);
        assert_eq!(stats.loose_buffers, 0);
        assert!(!quarantine.is_quarantined(99));
    }

    #[test]
    fn a_retired_id_is_a_tombstone_not_an_orphan() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(4);
        quarantine.reserve(&[ticket(4)]);
        quarantine.push_entry(
            PendingEntry::new("test", None, None, vec![ticket(4)]),
            false,
        );
        let entries = quarantine.take_entries();
        quarantine.release_entry(&entries[0]);
        let freed = quarantine.sweep_zero_refs();
        assert_eq!(freed.len(), 1);
        assert!(quarantine.defer_on_drop(ticket(4), true));
        assert_eq!(quarantine.stats().orphan_drops, 0);
    }

    #[test]
    fn an_entry_retains_each_live_buffer_exactly_once() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(7);
        quarantine.reserve(&[ticket(7), ticket(7)]);
        let entry = PendingEntry::new("test", None, None, vec![ticket(7), ticket(7), ticket(8)]);
        quarantine.push_entry(entry, false);
        let stats = quarantine.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.retained_buffers, 1, "deduplicated by id");
        assert_eq!(stats.in_flight_buffers, 1);
        assert_eq!(stats.lost_tickets, 1, "id 8 had no live buffer");
        // The buffer's own Drop must not destroy it.
        assert!(quarantine.defer_on_drop(ticket(7), false));
        assert_eq!(quarantine.stats().loose_buffers, 0);
    }

    #[test]
    fn a_buffer_shared_by_two_entries_is_freed_only_by_the_last_release() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(9);
        quarantine.reserve(&[ticket(9)]);
        quarantine.push_entry(
            PendingEntry::new("first", None, None, vec![ticket(9)]),
            false,
        );
        quarantine.reserve(&[ticket(9)]);
        quarantine.push_entry(
            PendingEntry::new("second", None, None, vec![ticket(9)]),
            false,
        );
        let entries = quarantine.take_entries();
        quarantine.release_entry(&entries[0]);
        assert!(
            quarantine.sweep_zero_refs().is_empty(),
            "the second, unproven entry still references the buffer"
        );
        assert!(quarantine.is_quarantined(9));
        quarantine.release_entry(&entries[1]);
        let freed = quarantine.sweep_zero_refs();
        assert_eq!(freed.len(), 1, "the last proven reference frees it");
        assert!(!quarantine.is_quarantined(9));
        assert_eq!(quarantine.stats().released_buffers, 1);
    }

    #[test]
    fn a_live_submission_keeps_a_quarantined_buffer_owned() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(10);
        quarantine.reserve(&[ticket(10)]);
        // The second submission commits while the buffer is still live.
        quarantine.reserve(&[ticket(10)]);
        quarantine.push_entry(
            PendingEntry::new("failed", None, None, vec![ticket(10)]),
            false,
        );
        let entries = quarantine.take_entries();
        quarantine.release_entry(&entries[0]);
        assert!(quarantine.sweep_zero_refs().is_empty());
        assert!(quarantine.is_quarantined(10));
        // The live submission completes successfully.
        quarantine.release_refs(&[ticket(10)]);
        let freed = quarantine.sweep_zero_refs();
        assert_eq!(freed.len(), 1);
        assert_eq!(quarantine.stats().released_buffers, 1);
    }

    #[test]
    fn a_loose_buffer_promoted_by_a_late_entry_is_still_freed_once() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(11);
        quarantine.reserve(&[ticket(11)]);
        // The buffer drops on the terminal device before the entry registers.
        assert!(quarantine.defer_on_drop(ticket(11), true));
        assert_eq!(quarantine.stats().loose_buffers, 1);
        // The entry then claims it as an in-flight resource.
        quarantine.push_entry(
            PendingEntry::new("late", None, None, vec![ticket(11)]),
            false,
        );
        assert_eq!(
            quarantine.stats().loose_buffers,
            0,
            "the fenced entry replaces the loose copy"
        );
        let entries = quarantine.take_entries();
        quarantine.release_entry(&entries[0]);
        let freed = quarantine.sweep_zero_refs();
        assert_eq!(freed.len(), 1);
        let (_entries, tickets, report) = quarantine.take_all(true);
        assert!(
            tickets.is_empty() && report.buffers == 0,
            "teardown must not see a second copy of the same buffer"
        );
    }

    #[test]
    fn a_failed_quiesce_reports_a_leak_and_keeps_the_ids_owned() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(3);
        quarantine.reserve(&[ticket(3)]);
        quarantine.push_entry(
            PendingEntry::new("test", None, None, vec![ticket(3)]),
            false,
        );
        let (_entries, _tickets, report) = quarantine.take_all(false);
        assert!(report.leaked);
        assert_eq!(report.entries, 1);
        assert_eq!(report.buffers, 1);
        assert!(
            quarantine.is_quarantined(3),
            "a leaked buffer stays owned; nothing may free it later"
        );
        assert_eq!(quarantine.stats().leaked_entries, 1);
        assert_eq!(quarantine.stats().leaked_buffers, 1);
    }

    #[test]
    fn a_successful_quiesce_releases_and_counts_once() {
        let mut quarantine = Quarantine::new();
        quarantine.register_live(4);
        quarantine.reserve(&[ticket(4)]);
        quarantine.push_entry(
            PendingEntry::new("test", None, None, vec![ticket(4)]),
            false,
        );
        let (_entries, tickets, report) = quarantine.take_all(true);
        assert_eq!(tickets.len(), 1);
        assert!(!report.leaked);
        assert!(!quarantine.is_quarantined(4));
        assert!(quarantine.is_retired(4));
        assert_eq!(quarantine.stats().released_entries, 1);
        assert_eq!(quarantine.stats().released_buffers, 1);
        // A late drop of the same id is a tombstone, not a second free.
        assert!(quarantine.defer_on_drop(ticket(4), false));
        assert_eq!(quarantine.stats().loose_buffers, 0);
        assert_eq!(quarantine.stats().orphan_drops, 0);
    }
}
