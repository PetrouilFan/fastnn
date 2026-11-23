//! The prepared-model artifact and the weight payload it carries.
//!
//! A prepared model is one file. [`format`] owns that file: the `FNN3ART2`
//! header, the serialized typed graph, one plan per target, the weight payload
//! and the precision record. Every section is hashed and 64-byte aligned;
//! [`format::Artifact`] validates the bytes before anything executes them, and
//! the payload is mapped rather than copied on load.
//!
//! [`weights`] is the import-side store. A `WeightSet` identifies each
//! initializer (or folded `Constant`) by [`weights::WeightId`] and never
//! mutates its bytes after import; the artifact writer and the session both
//! borrow from it.
//!
//! Session state — the arena, the worker pool, the prepared engine blocks — is
//! not here. It belongs to `runtime`, because it is created per session and
//! discarded with it.

pub mod format;
pub mod weights;

pub use weights::{WeightId, WeightSet, WeightSource, WeightTensor};
