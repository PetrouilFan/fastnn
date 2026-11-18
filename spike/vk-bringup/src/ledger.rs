//! Allocation/destruction bookkeeping.
//!
//! The spike uses this to prove the teardown is complete without relying only
//! on the validation layer: every created handle and every allocated byte is
//! recorded and every destroy is recorded, and the run fails if the two sides
//! differ. It is deliberately tiny and dependency-free.

use std::collections::BTreeMap;

#[derive(Default)]
pub struct Ledger {
    created: BTreeMap<&'static str, u64>,
    destroyed: BTreeMap<&'static str, u64>,
    bytes_created: BTreeMap<&'static str, u64>,
    bytes_destroyed: BTreeMap<&'static str, u64>,
}

impl Ledger {
    pub fn create(&mut self, kind: &'static str) {
        *self.created.entry(kind).or_default() += 1;
    }

    pub fn destroy(&mut self, kind: &'static str) {
        *self.destroyed.entry(kind).or_default() += 1;
    }

    pub fn allocate_bytes(&mut self, kind: &'static str, bytes: u64) {
        *self.bytes_created.entry(kind).or_default() += bytes;
    }

    pub fn free_bytes(&mut self, kind: &'static str, bytes: u64) {
        *self.bytes_destroyed.entry(kind).or_default() += bytes;
    }

    /// Kinds with a non-zero created-minus-destroyed count or byte imbalance.
    pub fn outstanding(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (kind, created) in &self.created {
            let destroyed = self.destroyed.get(kind).copied().unwrap_or(0);
            if *created != destroyed {
                out.push(format!("{kind}: created={created} destroyed={destroyed}"));
            }
        }
        for (kind, created) in &self.bytes_created {
            let destroyed = self.bytes_destroyed.get(kind).copied().unwrap_or(0);
            if *created != destroyed {
                out.push(format!(
                    "{kind} bytes: allocated={created} freed={destroyed}"
                ));
            }
        }
        out
    }

    pub fn is_balanced(&self) -> bool {
        self.outstanding().is_empty()
    }

    /// One-line-per-kind human-readable summary for the evidence log.
    pub fn report(&self) -> String {
        let mut kinds: Vec<&'static str> = self
            .created
            .keys()
            .chain(self.bytes_created.keys())
            .copied()
            .collect();
        kinds.sort_unstable();
        kinds.dedup();

        let mut out = String::new();
        for kind in kinds {
            let created = self.created.get(kind).copied().unwrap_or(0);
            let destroyed = self.destroyed.get(kind).copied().unwrap_or(0);
            let bytes_created = self.bytes_created.get(kind).copied().unwrap_or(0);
            let bytes_destroyed = self.bytes_destroyed.get(kind).copied().unwrap_or(0);
            out.push_str(&format!(
                "  {kind}: created={created} destroyed={destroyed}"
            ));
            if bytes_created != 0 || bytes_destroyed != 0 {
                out.push_str(&format!(
                    " bytes_allocated={bytes_created} bytes_freed={bytes_destroyed}"
                ));
            }
            out.push('\n');
        }
        if out.is_empty() {
            out.push_str("  (no handles)\n");
        }
        out
    }
}
