//! The versioned, shareable prepared-model artifact (FN3-09 / #55).
//!
//! # What this module is for
//!
//! An inference deployment needs one reproducible file that says "these are the
//! weights, this is the graph, this is the architecture, these are the dynamic
//! bounds this build accepts". Everything else — KV cache contents, step
//! counters, the device choice, the specialised plan — is either per-session or
//! per-device and must never be shared through a model file.
//!
//! ```text
//! ONNX bytes ──► import::ModelImport ──► PreparedArtifact::from_import
//!                                              │
//!            ┌─────────────────────────────────┴──────────────┐
//!            ▼                                                ▼
//!     to_bytes / write_file                        from_bytes / read_file
//!     (canonical, deterministic)                   (strict, checked, bounded)
//! ```
//!
//! The importer (#54) is the only producer; the planner (#56), the session
//! (#58), the CLI (#63) and the end-to-end gate (#62) are consumers. #62 loads
//! an artifact and never touches ONNX.
//!
//! # The format, in full
//!
//! ## Container
//!
//! ```text
//! header   64 bytes
//!   0..8    b"FNN3PREP"            magic
//!   8..10   u16 format version     = 1
//!  10..12   u16 header length      = 64
//!  12..16   u32 dtype encoding     = 1   (one DType::code byte per tensor)
//!  16..20   u32 flags              = 0   (reserved, must be zero)
//!  20..24   u32 reserved           = 0
//!  24..32   u64 file length        = header length + payload length
//!  32..40   u64 payload length
//!  40..44   u32 payload CRC-32
//!  44..64   20 reserved bytes      = 0
//! payload  (all of it is covered by the CRC)
//!   u32 section count              = 6
//!   u32 reserved                   = 0
//!   6 × [u32 tag, u32 reserved = 0, u64 section length]
//!   section bytes in tag order, each zero-padded to a multiple of 8
//! ```
//!
//! All integers are little-endian. All strings are `u32` byte length followed
//! by UTF-8 bytes (no terminator, no NUL). All floats are stored as their
//! `f32` bit pattern, never as text. Booleans are `u8` 0 or 1. Enumerations are
//! `u8` codes with a documented meaning; an unknown code is an error, never a
//! default.
//!
//! ## Sections
//!
//! | Tag | Name | Content |
//! |---:|---|---|
//! | 1 | `META` | ONNX IR version, opset domain+version, producer, producer version, graph name |
//! | 2 | `GEOM` | layers, hidden, MLP width, query/KV heads, head width, vocabulary, position embedding, LayerNorm epsilon, token-embedding weight, logits value, per-layer KV interface |
//! | 3 | `ENV` | the seven dynamic symbols, each as `u64 min, u64 max`, in `Symbol::ALL` order |
//! | 4 | `VALS` | the value table (name, dtype, shape, origin), then graph inputs, then graph outputs |
//! | 5 | `NODE` | the op list: ONNX provenance, `OpId` code, operand handles, `OpParams` |
//! | 6 | `WGHT` | the weight table: name, dtype, extents, source, exact payload bytes |
//!
//! Section 4 and 5 are the graph description the planner consumes; section 6 is
//! the weight table it references. Weight payloads begin on an 8-byte boundary
//! relative to the section start (and therefore to the file), with zero padding.
//!
//! ## Checksum
//!
//! CRC-32, IEEE 802.3 (reflected polynomial `0xEDB88320`, init `0xFFFFFFFF`,
//! final XOR `0xFFFFFFFF`, check value `0xCBF43926` for `"123456789"`), over the
//! payload only. The header is validated field by field instead, so a corrupt
//! header names the bad field while a corrupt payload is a checksum mismatch.
//! This detects accidental corruption; it is not a signature.
//!
//! ## Versioning policy
//!
//! Magic and version are the first ten bytes, so an incompatible file is
//! rejected before any structure is interpreted:
//!
//! * same version → read;
//! * any other version → [`FormatErrorKind::UnsupportedVersion`] with both
//!   numbers ("this file is prepared-model format version 2, but this build
//!   reads version 1 only");
//! * a different dtype encoding → [`FormatErrorKind::UnsupportedDtypeEncoding`].
//!
//! A change that adds a section, reorders a field, changes a meaning or assigns
//! a dtype code **must** bump the version. There is no in-place compatibility
//! and no best-effort partial read: a partially understood model is a wrong
//! model.
//!
//! ## 2.x policy
//!
//! The 2.x `.fnnc` artifact is raw `bincode` with no magic and no version, and
//! it records a compiler plan for the 2.x IR. It is on the 2.x do-not-carry
//! list. **This build never reads `.fnnc` and never re-exports one.** A `.fnnc`
//! file fails at the magic check with
//! [`FormatErrorKind::BadMagic`], which is tested explicitly
//! (`legacy_fnnc_bytes_are_rejected_at_the_magic`). Regenerating a 3.0 artifact
//! always starts from the ONNX (or another 3.0 artifact), never from a 2.x file.
//!
//! ## What a file may not claim
//!
//! The reader enforces, before allocating: file and payload lengths against the
//! bytes present, every section length against the payload, every count against
//! both a policy cap ([`MAX_VALUES`], [`MAX_NODES`], [`MAX_WEIGHTS`],
//! [`MAX_RANK`], [`MAX_PARAM_AXES`]) and the bytes that could encode it, and
//! every weight payload length against its section. After decoding it enforces
//! that the geometry, the value table, the weight table and the envelope agree
//! ([`PreparedArtifact::validate`]): a vocabulary the token-embedding weight
//! does not have, a layer count without KV interfaces, a fixed head geometry
//! outside the declared bounds, or a value handle out of range is an
//! [`FormatErrorKind::Inconsistent`] error.
//!
//! # Determinism
//!
//! There is exactly one encoding of a model: tables keep declaration order, there
//! are no maps in the byte stream, every optional field has a canonical absent
//! form, alignment padding is zeros and is *required* to be zeros on read. So
//! `to_bytes` twice is byte-identical, and `from_bytes` → `to_bytes` is
//! byte-identical.
//!
//! # Session state is not in the file
//!
//! The artifact has no KV capacity, no live extent, no batch size, no step
//! counter, no phase and no device. Those are session or planner state. The
//! geometry section describes layer/head/vocabulary shape (a property of the
//! model); how many tokens to allocate is decided by [`crate::fn3::model::Session`]
//! through [`crate::fn3::import::DecoderGeometry::kv_cache_config`].

pub mod error;
mod wire;

mod artifact;
mod graph;

pub use artifact::{
    ArtifactMetadata, PreparedArtifact, PreparedHeader, FORMAT_VERSION, HEADER_BYTES, MAGIC,
    MAX_NODES, MAX_PARAM_AXES, MAX_PAYLOAD_BYTES, MAX_RANK, MAX_VALUES, MAX_VOCAB_SIZE,
    MAX_WEIGHTS,
};
pub use error::{FormatError, FormatErrorKind};
pub use graph::PreparedGraph;
