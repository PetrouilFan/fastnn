//! Per-segment wall-clock accounting, compiled only with `segment-timing`.
//!
//! # What this is
//!
//! A measurement tool, not a product feature. A normal build compiles none of
//! this module: the feature is off by default, [`Session`](super::Session) has
//! no timing field, and [`Session::execute`](super::Session::execute) carries no
//! clock read. With the feature on, each plan segment's dispatch is surrounded
//! by one `Instant` pair and the elapsed time is added to an entry for its
//! [`OpId`].
//!
//! # Why there is no shared clock map
//!
//! A session is `Send` but deliberately `!Sync` and runs one forward at a time
//! ([ADR-0001 §5](../../../docs/v3/adr-0001-one-runtime.md)). The timing
//! table is a plain field of that session, updated by the thread that called
//! `execute` after the segment's work — including the work a segment hands to
//! its pool — has finished. It is therefore never read or written concurrently,
//! needs no lock or atomic, and cannot be observed by another thread while a
//! forward is running. A global map keyed by op name would be shared mutable
//! state between sessions and threads for no benefit.
//!
//! # What the numbers mean
//!
//! A segment's time is the arena frame, the operand binding and the dispatch
//! (scalar kernel or oneDNN blocks, including waiting for pool workers). For a
//! scalar segment that parallelises over rows, this is wall time, not CPU time.
//! The sum over entries is at most a forward's wall time because input
//! validation and output read-back are outside the segment loop.
//!
//! Entries are kept in first-execution order (which is plan order), so a report
//! is deterministic. A caller that wants a ranked table sorts its own copy.

use std::time::Duration;

use crate::inference::ops::OpId;

/// One plan segment's accumulated time, keyed by its position in the plan.
///
/// The label table answers "how much does `Conv+Swish` cost in total"; this
/// answers "which convolution". Both are recorded from the same call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentOccurrence {
    /// Position in `Plan::segments`.
    pub index: usize,
    /// The op the segment is anchored on.
    pub op: OpId,
    /// What actually ran, including a fused chain.
    pub label: String,
    /// How many times the segment ran since the last reset.
    pub calls: u64,
    /// Summed wall time of those calls.
    pub total: Duration,
}

/// One segment label's accumulated wall time in one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTiming {
    /// The op the segment is anchored on.
    pub op: OpId,
    /// What actually ran: the op name, or the fused chain (`Conv+Swish`).
    ///
    /// A fused segment does the work of several graph ops, so charging its
    /// whole time to the anchor would report the epilogue as free. The label
    /// keeps the chain visible in the table.
    pub label: String,
    /// How many segments with this op ran since the last reset.
    pub calls: u64,
    /// Summed wall time of those segments.
    pub total: Duration,
    /// The slowest single segment seen, useful to spot an outlier.
    pub longest: Duration,
}

impl SegmentTiming {
    /// Mean wall time per call; zero when the op did not run.
    pub fn mean(&self) -> Duration {
        self.total
            .checked_div(self.calls.max(1) as u32)
            .unwrap_or(Duration::ZERO)
    }
}

/// The timing table of one session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegmentTimings {
    entries: Vec<SegmentTiming>,
    occurrences: Vec<SegmentOccurrence>,
    total: Duration,
}

impl SegmentTimings {
    /// Clear every entry, keeping the allocation for the next forward.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.occurrences.clear();
        self.total = Duration::ZERO;
    }

    /// Add one segment's elapsed time. Called by the session's executor.
    pub(crate) fn record(&mut self, index: usize, op: OpId, label: &str, elapsed: Duration) {
        self.total = self.total.saturating_add(elapsed);
        match self.entries.iter_mut().find(|entry| entry.label == label) {
            Some(entry) => {
                entry.calls += 1;
                entry.total = entry.total.saturating_add(elapsed);
                entry.longest = entry.longest.max(elapsed);
            }
            None => self.entries.push(SegmentTiming {
                op,
                label: label.to_string(),
                calls: 1,
                total: elapsed,
                longest: elapsed,
            }),
        }
        match self
            .occurrences
            .iter_mut()
            .find(|entry| entry.index == index)
        {
            Some(entry) => {
                entry.calls += 1;
                entry.total = entry.total.saturating_add(elapsed);
            }
            None => self.occurrences.push(SegmentOccurrence {
                index,
                op,
                label: label.to_string(),
                calls: 1,
                total: elapsed,
            }),
        }
    }

    /// One entry per op that ran, in first-execution order.
    pub fn entries(&self) -> &[SegmentTiming] {
        &self.entries
    }

    /// One entry per plan segment that ran, in plan order.
    pub fn occurrences(&self) -> &[SegmentOccurrence] {
        &self.occurrences
    }

    /// Summed segment time; at most a forward's wall time.
    pub fn total(&self) -> Duration {
        self.total
    }

    /// `true` when nothing has been recorded since construction or reset.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_accumulate_per_op_in_first_seen_order() {
        let mut timings = SegmentTimings::default();
        timings.record(0, OpId::Conv, "Conv", Duration::from_millis(3));
        timings.record(1, OpId::Sigmoid, "Sigmoid", Duration::from_millis(1));
        timings.record(2, OpId::Conv, "Conv", Duration::from_millis(5));
        let entries = timings.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, OpId::Conv);
        assert_eq!(entries[0].calls, 2);
        assert_eq!(entries[0].total, Duration::from_millis(8));
        assert_eq!(entries[0].longest, Duration::from_millis(5));
        assert_eq!(entries[0].mean(), Duration::from_millis(4));
        assert_eq!(entries[1].op, OpId::Sigmoid);
        assert_eq!(timings.occurrences().len(), 3);
        assert_eq!(timings.occurrences()[2].index, 2);
        assert_eq!(timings.occurrences()[2].total, Duration::from_millis(5));
        assert_eq!(timings.total(), Duration::from_millis(9));
        timings.reset();
        assert!(timings.is_empty());
        assert!(timings.occurrences().is_empty());
        assert_eq!(timings.total(), Duration::ZERO);
    }

    #[test]
    fn a_single_call_has_a_mean_equal_to_its_total() {
        let mut timings = SegmentTimings::default();
        timings.record(0, OpId::Transpose, "Transpose", Duration::from_micros(7));
        assert_eq!(timings.entries()[0].mean(), Duration::from_micros(7));
    }
}
