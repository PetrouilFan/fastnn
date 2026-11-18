//! Execution evidence: what ran, where, and how the bytes moved.
//!
//! # Why this exists
//!
//! The 3.0 release gate says a Vulkan run must be provably a Vulkan run. A
//! function returning the right numbers is not evidence: a silent CPU fallback
//! would do that too. [`ResidencyLedger`] records, for every op the backend
//! executes:
//!
//! - the op id and the kernel that serviced it,
//! - the number of queue submissions and the dispatch geometry,
//! - every buffer binding with its role, id, range and memory class,
//! - bytes uploaded and bytes read back,
//! - whether any *input* was read back to the host (it must never be, for an
//!   input that the kernel consumes),
//! - device timestamps when the queue family supports them.
//!
//! #62 can then assert, from the public API, that every instruction in a model
//! has a record with `backend == "vulkan"`, `submissions >= 1`, a non-zero
//! dispatch and at least one device-local binding — and that the record count
//! matches the instruction count. That is the "no operation silently ran on
//! CPU" demonstration.
//!
//! # Current limitation (reported to #48/#52)
//!
//! `Tensor` storage is host-backed `AlignedBuffer` in the alpha contract, so
//! the backend uploads inputs and downloads the output for every op
//! (`host_staged = true`). The arithmetic is still executed by the GPU kernel and
//! counted here. Device-resident tensor storage needs a `Storage` variant owned
//! by #52/#48; the platform layer already exposes device buffers and a
//! dispatch-without-round-trip path for it (`VulkanBackend::dispatch`).

use std::sync::Mutex;

use super::submission::BindingSnapshot;
use crate::fn3::device::DeviceId;
use crate::fn3::ops::OpId;

/// Backend name used in every record.
pub const VULKAN_BACKEND_NAME: &str = "vulkan";

/// One executed operation.
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Monotonic record number within the backend.
    pub sequence: u64,
    /// Op that ran.
    pub op: OpId,
    /// Kernel that serviced it.
    pub kernel: &'static str,
    /// Backend name (`"vulkan"`).
    pub backend: &'static str,
    /// Device the work ran on.
    pub device: DeviceId,
    /// Queue submissions used by this op (at least 1).
    pub submissions: u32,
    /// Workgroup counts.
    pub workgroups: [u32; 3],
    /// Every descriptor binding, in binding order.
    pub bindings: Vec<BindingSnapshot>,
    /// Bytes copied host → device for this op.
    pub upload_bytes: u64,
    /// Bytes copied device → host for this op.
    pub readback_bytes: u64,
    /// `true` if any kernel input was read back to the host. Must be false.
    pub inputs_read_back: bool,
    /// `true` when inputs and outputs travelled through host staging (the
    /// alpha tensor-storage model).
    pub host_staged: bool,
    /// Sum of device-local binding ranges.
    pub device_local_bytes: u64,
    /// Host wall time for the submission, in nanoseconds.
    pub host_ns: u64,
    /// Device time, when the queue family supports timestamps.
    pub gpu_ns: Option<u64>,
    /// Raw `(start, end)` device timestamp ticks, for audit.
    pub gpu_ticks: Option<(u64, u64)>,
}

impl ExecutionRecord {
    /// Start a record with the identity fields filled in and everything else
    /// zeroed/false. Callers set the submission, binding and byte fields.
    pub fn new(sequence: u64, op: OpId, kernel: &'static str, device: DeviceId) -> ExecutionRecord {
        ExecutionRecord {
            sequence,
            op,
            kernel,
            backend: VULKAN_BACKEND_NAME,
            device,
            submissions: 0,
            workgroups: [0, 0, 0],
            bindings: Vec::new(),
            upload_bytes: 0,
            readback_bytes: 0,
            inputs_read_back: false,
            host_staged: true,
            device_local_bytes: 0,
            host_ns: 0,
            gpu_ns: None,
            gpu_ticks: None,
        }
    }

    /// `true` when the record proves device work: a submission, a dispatch and
    /// at least one device-local binding.
    pub fn proves_device_execution(&self) -> bool {
        self.submissions >= 1
            && self.workgroups.iter().all(|count| *count > 0)
            && self.bindings.iter().any(|binding| binding.device_local)
            && !self.inputs_read_back
    }
}

/// Aggregate view of the ledger.
#[derive(Debug, Clone, Default)]
pub struct ResidencySummary {
    /// Executions recorded.
    pub executions: u64,
    /// Queue submissions over all executions.
    pub submissions: u64,
    /// Bytes uploaded.
    pub upload_bytes: u64,
    /// Bytes read back.
    pub readback_bytes: u64,
    /// Records where an input was read back (must be 0).
    pub input_readbacks: u64,
    /// Records that prove device execution.
    pub proven_device_executions: u64,
    /// Total device time, when every record reported it.
    pub gpu_ns: Option<u64>,
    /// Ops seen, with their execution counts, stable-sorted by op id.
    pub ops: Vec<(OpId, u64)>,
}

/// Append-only (until [`ResidencyLedger::clear`]) evidence ledger.
#[derive(Debug, Default)]
pub struct ResidencyLedger {
    inner: Mutex<LedgerInner>,
}

#[derive(Debug, Default)]
struct LedgerInner {
    records: Vec<ExecutionRecord>,
    next_sequence: u64,
}

impl ResidencyLedger {
    /// An empty ledger.
    pub fn new() -> ResidencyLedger {
        ResidencyLedger::default()
    }

    /// Reserve the next sequence number and append a completed record.
    pub fn record(&self, mut record: ExecutionRecord) -> u64 {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let sequence = inner.next_sequence;
        inner.next_sequence += 1;
        record.sequence = sequence;
        inner.records.push(record);
        sequence
    }

    /// All records, oldest first.
    pub fn records(&self) -> Vec<ExecutionRecord> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .records
            .clone()
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .records
            .len()
    }

    /// `true` when nothing has run.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every record. Sequence numbers keep increasing.
    pub fn clear(&self) {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .records
            .clear();
    }

    /// Aggregate counters.
    pub fn summary(&self) -> ResidencySummary {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut summary = ResidencySummary::default();
        let mut gpu_total = 0u64;
        let mut gpu_complete = true;
        let mut ops: Vec<(OpId, u64)> = Vec::new();
        for record in &inner.records {
            summary.executions += 1;
            summary.submissions += u64::from(record.submissions);
            summary.upload_bytes += record.upload_bytes;
            summary.readback_bytes += record.readback_bytes;
            if record.inputs_read_back {
                summary.input_readbacks += 1;
            }
            if record.proves_device_execution() {
                summary.proven_device_executions += 1;
            }
            match record.gpu_ns {
                Some(nanoseconds) => gpu_total += nanoseconds,
                None => gpu_complete = false,
            }
            match ops.iter_mut().find(|(op, _)| *op == record.op) {
                Some((_, count)) => *count += 1,
                None => ops.push((record.op, 1)),
            }
        }
        ops.sort_by_key(|(op, _)| op.code());
        summary.ops = ops;
        summary.gpu_ns = if gpu_complete && !inner.records.is_empty() {
            Some(gpu_total)
        } else {
            None
        };
        summary
    }

    /// Human-readable evidence block, one line per execution.
    pub fn report(&self) -> String {
        let records = self.records();
        if records.is_empty() {
            return "  (no executions)\n".to_string();
        }
        let mut text = String::new();
        for record in &records {
            text.push_str(&format!(
                "  #{:03} op={} kernel={} device={} submissions={} workgroups={:?} \
                 upload={} readback={} inputs_read_back={} device_local={} host_ns={} gpu_ns={}\n",
                record.sequence,
                record.op.name(),
                record.kernel,
                record.device.diagnostic(),
                record.submissions,
                record.workgroups,
                record.upload_bytes,
                record.readback_bytes,
                record.inputs_read_back,
                record.device_local_bytes,
                record.host_ns,
                record
                    .gpu_ns
                    .map(|nanoseconds| nanoseconds.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::super::context::BindingRole;
    use super::super::submission::BindingSnapshot;
    use super::*;
    use crate::fn3::device::PhysicalDeviceId;

    fn record() -> ExecutionRecord {
        let mut record = ExecutionRecord::new(
            0,
            OpId::Add,
            "smoke.add.f32",
            DeviceId::Vulkan(PhysicalDeviceId::new(0)),
        );
        record.submissions = 1;
        record.workgroups = [2, 1, 1];
        record.bindings.push(BindingSnapshot {
            binding: 0,
            role: BindingRole::Input,
            buffer_id: 1,
            offset: 0,
            range: 64,
            device_local: true,
        });
        record
    }

    #[test]
    fn a_proven_record_needs_submission_dispatch_and_device_memory() {
        let mut sample = record();
        assert!(sample.proves_device_execution());
        sample.workgroups = [0, 0, 0];
        assert!(!sample.proves_device_execution());
        sample.workgroups = [1, 1, 1];
        sample.inputs_read_back = true;
        assert!(!sample.proves_device_execution());
    }

    #[test]
    fn summary_counts_ops_and_readbacks() {
        let ledger = ResidencyLedger::new();
        ledger.record(record());
        ledger.record(record());
        let summary = ledger.summary();
        assert_eq!(summary.executions, 2);
        assert_eq!(summary.submissions, 2);
        assert_eq!(summary.ops, vec![(OpId::Add, 2)]);
        assert_eq!(summary.input_readbacks, 0);
        assert_eq!(summary.proven_device_executions, 2);
        assert!(ledger.report().contains("op=add"));
    }
}
