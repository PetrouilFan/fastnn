//! FN3 3.0 runtime adapter seam.
//!
//! The 3.0 core (`fastnn::fn3`) is being written concurrently (FN3-02) and does
//! not exist in this revision. This module defines the exact runtime surface the
//! harness expects, and fails loudly instead of emitting placeholder numbers.
//!
//! When `fastnn::fn3` lands, implement [`Fn3Runtime`] for a wrapper around it
//! and enable the real path in [`run`]. Do not touch the schema, suites, or
//! comparison tooling: they are runtime-agnostic.

#![allow(dead_code)]

use super::{BenchConfig, Suite, SuiteOutcome};
use std::collections::BTreeMap;

/// Base revision of this worktree; recorded in the failure message so a result
/// can never be mistaken for a fn3 measurement.
pub const BASE_REVISION: &str = "a2c0759fff8bdcb7c943e53442f3382b86e59b66";

/// The entry points the 3.0 runtime must expose to the harness.
///
/// One prepared model is loaded once, prepared once, then used for prefill and
/// steady decode. `prefill` and `decode` must synchronize the device before
/// returning; `last_device_time_ms` reports backend device time when available.
pub trait Fn3Runtime {
    type PreparedModel;

    fn load(&self) -> Result<Self::PreparedModel, String>;
    fn prepare(&self, model: Self::PreparedModel) -> Result<Self::PreparedModel, String>;
    fn prefill(
        &mut self,
        model: &mut Self::PreparedModel,
        prompt_tokens: &[u32],
    ) -> Result<(), String>;
    fn decode(&mut self, model: &mut Self::PreparedModel, token: u32) -> Result<(), String>;
    /// Device time for the most recent operation, when the backend reports it.
    fn last_device_time_ms(&self) -> Option<f64>;
    /// Peak device memory in bytes, when queryable.
    fn peak_device_memory_bytes(&self) -> Option<u64>;
    /// Counters for the most recent operation (allocation count, arena copies,
    /// ...). Returning `None` is acceptable and is reported as explicitly
    /// unavailable, never as zero.
    fn last_operation_counters(&self) -> Option<BTreeMap<String, u64>> {
        None
    }
}

/// Whether this build contains a linked fn3 runtime.
pub fn is_available() -> bool {
    false
}

pub fn run(_suite: Suite, _config: &BenchConfig) -> Result<SuiteOutcome, String> {
    Err(format!(
        "fn3 adapter is not built in this revision (base {BASE_REVISION}).\n\
         The 3.0 runtime `fastnn::fn3` does not exist in this worktree, so no fn3 \
         measurement can be produced. This failure is intentional: the harness \
         refuses to emit baseline numbers under the fn3 adapter name.\n\
         Once FN3-02 lands, implement `adapters::fn3::Fn3Runtime` for the new \
         runtime and wire it into `adapters::fn3::run`. The harness schema, \
         suites, and comparison tooling do not need to change."
    ))
}
