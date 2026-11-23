//! The versioned prepared-model artifact, format version 2.
//!
//! # What this module is for
//!
//! An inference deployment needs one reproducible file that carries everything
//! the runtime executes: the typed graph, one execution plan per target
//! profile, the weights in a memory-mappable payload, quantization metadata and
//! the hashes that prove the bytes were not corrupted. Everything else — a
//! session's arena, its worker pool, KV state, a device choice — is per-session
//! state and is never written into a model file.
//!
//! ```text
//! ONNX bytes ──► import::ImportedGraph ──► PreparedArtifact::from_imported
//!                                                   │
//!                    ┌──────────────────────────────┴──────────────┐
//!                    ▼                                             ▼
//!            to_bytes / write_file                     Artifact::open (mmap)
//!            (canonical, deterministic)                (hashes checked first)
//! ```
//!
//! `prepare` is the only step that needs ONNX. `load` maps the artifact, decodes
//! the stored plan and runs it; there is no second planner on that path.
//!
//! # Sections
//!
//! | Tag | Name | Content |
//! |---:|---|---|
//! | 0 | `HEADER` | fixed 128-byte frame, not a payload section |
//! | 1 | `GRAPH` | typed IR values, nodes, full `OpParams`, `SourceLoc`, interface |
//! | 2 | `PLANS` | one plan per target: engines, layout, reorders, arena, fusion |
//! | 3 | `WEIGHTS` | one contiguous, 64-byte-aligned payload plus its table |
//! | 4 | `QUANT` | per-tensor scale/zero-point/axis metadata (empty for F32) |
//! | 5 | `HASHES` | SHA-256 of `HEADER`, `GRAPH`, `PLANS`, `WEIGHTS`, `QUANT` |
//!
//! Every section starts on a 64-byte boundary. The `WEIGHTS` payload starts on a
//! 64-byte boundary inside its section, so every weight's first byte is 64-byte
//! aligned in the mapping.
//!
//! # Format policy
//!
//! There is one format. Version 1 (`FNN3PREP`) was decoder-shaped and stored
//! only three `OpParams` fields; it is not read. A version-1 file fails with
//! [`FormatErrorKind::ObsoleteFormat`], whose message says to re-prepare the
//! source ONNX. There is no compatibility shim and no partial read: a partially
//! understood model is a wrong model.
//!
//! # Untrusted input
//!
//! The reader never trusts a count, length or offset. Every count is checked
//! against the documented budgets and the bytes that could encode it before a
//! `Vec` is allocated; every section range is checked against the file length
//! and its 64-byte alignment; every weight range is checked against the payload.
//! A structural failure is [`FormatErrorKind::MalformedAt`] with the section
//! name and absolute byte offset. A hash mismatch is
//! [`FormatErrorKind::HashMismatch`] naming the section.
//!
//! # Determinism
//!
//! Tables keep declaration order, there are no maps in the byte stream, and
//! alignment padding is zeros and is required to be zeros on read. `to_bytes`
//! twice is byte-identical, and `from_bytes` → `to_bytes` is byte-identical.
//!
//! # What a file may not claim
//!
//! After decoding, the artifact cross-checks the graph, the weight table and
//! every plan: one slot per graph value with the same type and byte count, weight
//! ids in range with matching dtype and payload length, interface names in the
//! declared order, operand and result ids in range, fusion entries that point at
//! fused segments, and a reorder byte sum equal to its list. Disagreement is
//! [`FormatErrorKind::Inconsistent`].

pub mod error;
mod wire;

mod artifact;
mod graph;
mod plan;
mod weights;

pub use artifact::{
    inspect_file, parse_directory, resolve_targets, Artifact, ArtifactHeader, ArtifactInspection,
    PlanSummary, PreparedArtifact, QuantMetadata, QuantTensor, SectionHash, SectionInfo,
    SectionRange, DIRECTORY_BYTES, DTYPE_ENCODING, FORMAT_VERSION, HEADER_BYTES, LEGACY_MAGIC,
    MAGIC, MAX_INSPECT_SECTION_BYTES, MAX_NODES, MAX_PARAM_AXES, MAX_PAYLOAD_BYTES, MAX_RANK,
    MAX_VALUES, MAX_WEIGHTS, SECTION_COUNT, TARGETS, TARGET_ALL, TARGET_CPU, TARGET_CPU_REFERENCE,
};
pub use error::{FormatError, FormatErrorKind};
pub use graph::ArtifactMetadata;
pub use plan::{ArtifactPlan, Precision, MAX_TARGETS};
pub use weights::ArtifactWeight;
