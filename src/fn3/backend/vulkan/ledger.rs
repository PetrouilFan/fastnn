//! Handle bookkeeping.
//!
//! The validation layer catches most lifetime bugs, but it reports them only at
//! `vkDestroyDevice`. This ledger is the cheap, always-on second witness: every
//! handle this module creates is counted, every destroy is counted, and the
//! test suite asserts that transient kinds balance while the backend is still
//! alive. It is deliberately tiny (a `Mutex<BTreeMap<&'static str, Counts>>`)
//! and dependency-free.

use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    created: u64,
    destroyed: u64,
    bytes_created: u64,
    bytes_destroyed: u64,
}

/// Per-kind create/destroy and byte counters.
#[derive(Debug, Default)]
pub struct HandleLedger {
    inner: Mutex<BTreeMap<&'static str, Counts>>,
}

impl HandleLedger {
    /// Record a created handle.
    pub fn create(&self, kind: &'static str) {
        self.with(kind, |counts| counts.created += 1);
    }

    /// Record a destroyed handle.
    pub fn destroy(&self, kind: &'static str) {
        self.with(kind, |counts| counts.destroyed += 1);
    }

    /// Record allocated bytes for a kind.
    pub fn allocate_bytes(&self, kind: &'static str, bytes: u64) {
        self.with(kind, |counts| counts.bytes_created += bytes);
    }

    /// Record freed bytes for a kind.
    pub fn free_bytes(&self, kind: &'static str, bytes: u64) {
        self.with(kind, |counts| counts.bytes_destroyed += bytes);
    }

    /// Kinds with a non-zero created-minus-destroyed count or byte imbalance.
    pub fn outstanding(&self) -> Vec<String> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut out = Vec::new();
        for (kind, counts) in inner.iter() {
            if counts.created != counts.destroyed {
                out.push(format!(
                    "{kind}: created={} destroyed={}",
                    counts.created, counts.destroyed
                ));
            }
            if counts.bytes_created != counts.bytes_destroyed {
                out.push(format!(
                    "{kind} bytes: allocated={} freed={}",
                    counts.bytes_created, counts.bytes_destroyed
                ));
            }
        }
        out
    }

    /// `true` when every created handle has been destroyed.
    pub fn is_balanced(&self) -> bool {
        self.outstanding().is_empty()
    }

    /// Outstanding count for one kind.
    pub fn outstanding_count(&self, kind: &str) -> u64 {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner
            .iter()
            .filter(|(name, _)| **name == kind)
            .map(|(_, counts)| counts.created.saturating_sub(counts.destroyed))
            .sum()
    }

    /// One line per kind, for evidence logs.
    pub fn report(&self) -> String {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if inner.is_empty() {
            return "  (no handles)\n".to_string();
        }
        let mut out = String::new();
        for (kind, counts) in inner.iter() {
            out.push_str(&format!(
                "  {kind}: created={} destroyed={} outstanding={}",
                counts.created,
                counts.destroyed,
                counts.created.saturating_sub(counts.destroyed)
            ));
            if counts.bytes_created != 0 || counts.bytes_destroyed != 0 {
                out.push_str(&format!(
                    " bytes_allocated={} bytes_freed={}",
                    counts.bytes_created, counts.bytes_destroyed
                ));
            }
            out.push('\n');
        }
        out
    }

    fn with(&self, kind: &'static str, update: impl FnOnce(&mut Counts)) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        update(inner.entry(kind).or_default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_requires_every_create_to_be_destroyed() {
        let ledger = HandleLedger::default();
        ledger.create("buffer");
        ledger.create("buffer");
        assert_eq!(ledger.outstanding_count("buffer"), 2);
        assert!(!ledger.is_balanced());
        ledger.destroy("buffer");
        ledger.destroy("buffer");
        assert!(ledger.is_balanced());
        assert!(ledger.report().contains("buffer: created=2 destroyed=2"));
    }
}
