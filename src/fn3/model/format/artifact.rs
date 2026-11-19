//! The artifact: a validated, self-describing, checksummed model file.
//!
//! # Layout (version 1)
//!
//! ```text
//! header   (64 bytes, fixed)
//!   0..8    magic                    b"FNN3PREP"
//!   8..10   u16 format version       1
//!  10..12   u16 header length        64
//!  12..16   u32 dtype encoding       1   (fn3 DType::code values, one byte each)
//!  16..20   u32 flags                0   (reserved; must be zero)
//!  20..24   u32 reserved             0
//!  24..32   u64 file length          header length + payload length
//!  32..40   u64 payload length
//!  40..44   u32 payload CRC-32
//!  44..64   reserved (20 bytes)      0
//! payload  (everything after the header, covered by the CRC)
//!   u32 section count                6
//!   u32 reserved                     0
//!   6 × [u32 tag, u32 reserved, u64 length]
//!   sections in tag order, each zero-padded to a multiple of 8 bytes:
//!     1 META  provenance: ONNX IR version, opset, producer, graph name
//!     2 GEOM  architecture geometry and its value/weight handles
//!     3 ENV   the declared dynamic bounds (7 × min/max)
//!     4 VALS  value table + graph inputs + graph outputs
//!     5 NODE  op list with operand handles and attributes
//!     6 WGHT  weight table with the exact payload bytes
//! ```
//!
//! # Why each part is where it is
//!
//! * **Magic first, version second.** A file from another format (including a
//!   2.x `.fnnc`, which is raw `bincode` with neither) is rejected at byte 0,
//!   and a file from another *version* of this format is rejected at byte 8,
//!   before a single field is interpreted. That is what makes "this file is
//!   version 9, this build reads version 1" possible instead of a parse error
//!   from the middle of the data.
//! * **A fixed-size header.** Every header field is at a constant offset, so
//!   the whole header is validated without allocating or interpreting a
//!   variable-length structure.
//! * **Lengths before contents.** `file length`, `payload length` and every
//!   section length are checked against the bytes that are actually present
//!   before any `Vec` is created. A hostile count cannot become an allocation.
//! * **A checksum over the payload only.** The header fields are individually
//!   validated (magic, version, sizes, reserved), and the payload gets one
//!   CRC-32, so a single flipped byte anywhere in the file is reported — by a
//!   field check in the header or by the CRC in the payload.
//! * **Sections, not a single blob.** Each table has one owner function, its own
//!   length, and an exact-consumption check, so a section that is too short or
//!   too long is a local error rather than a shifted read. The reserved `tag`
//!   space and the `dtype encoding` field are the extension points for the
//!   quantized metadata milestone (#68).
//! * **Eight-byte alignment of weight payloads**, relative to the section start
//!   (and therefore to the file: the header is 64 bytes and the directory is
//!   104, both multiples of 8). Padding is written as zeros and *required* to be
//!   zeros on read, so the encoding is canonical: one model, one byte string.
//!
//! # What is deliberately absent
//!
//! No KV cache capacity, no batch size, no live extent, no step counter, no
//! backend or device choice: those are session state (FN3-12 / #58) or a device
//! decision (FN3-10 / #56), not properties of a shareable model. The geometry
//! section carries the layer/head/vocabulary shape; a session decides how many
//! tokens to allocate. Two sessions over one artifact share no byte of mutable
//! state, and the artifact never changes after it is written.

use std::path::Path;

use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::import::{
    DecoderGeometry, Dim, KvInterface, ModelImport, ModelProvenance, OpsetImport,
    PositionEmbedding, ValueId, ValueInfo, ValueOrigin, ValueShape,
};
use crate::inference::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};
use crate::inference::model::{ModelWeights, MAX_KV_LAYERS};
use crate::inference::ops::OpId;
use crate::inference::shape::{DimBound, Shape, ShapeEnvelope, Symbol};

use super::error::{FormatError, FormatErrorKind};
use super::graph::PreparedGraph;
use super::wire::{crc32, Reader, Writer};

// ---------------------------------------------------------------------------
// Framing constants
// ---------------------------------------------------------------------------

/// The eight magic bytes of a versioned prepared-model artifact.
///
/// `FNN3` is the tree, `PREP` is the artifact. Deliberately not a NUL-padded
/// string: all eight bytes carry information, so a truncated copy rarely looks
/// like a valid magic.
pub const MAGIC: [u8; 8] = *b"FNN3PREP";

/// The only format version this build writes and reads.
pub const FORMAT_VERSION: u16 = 1;

/// Size of the fixed header, in bytes.
pub const HEADER_BYTES: usize = 64;

/// Dtype encoding id: one byte per tensor, carrying [`DType::code`].
///
/// A future encoding (for example a quantized one with a wider code) must use a
/// new id, so an old build reports "this file uses dtype encoding 2" instead of
/// misreading a code.
const DTYPE_ENCODING: u32 = 1;

/// Number of sections. Fixed for version 1.
const SECTION_COUNT: usize = 6;

/// Payload directory size: a count, a reserved word, and one 16-byte entry per
/// section. A multiple of eight, so the first section starts aligned.
const DIRECTORY_BYTES: usize = 8 + SECTION_COUNT * 16;

/// Section tags, in the only order version 1 accepts.
mod tag {
    pub(super) const META: u32 = 1;
    pub(super) const GEOMETRY: u32 = 2;
    pub(super) const ENVELOPE: u32 = 3;
    pub(super) const VALUES: u32 = 4;
    pub(super) const NODES: u32 = 5;
    pub(super) const WEIGHTS: u32 = 6;

    /// Every tag, in write order.
    pub(super) const ALL: [u32; super::SECTION_COUNT] =
        [META, GEOMETRY, ENVELOPE, VALUES, NODES, WEIGHTS];

    /// A human name for a tag, used in malformed-section diagnostics.
    pub(super) fn name(tag: u32) -> &'static str {
        match tag {
            META => "META",
            GEOMETRY => "GEOM",
            ENVELOPE => "ENV",
            VALUES => "VALS",
            NODES => "NODE",
            WEIGHTS => "WGHT",
            _ => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Largest artifact payload this build accepts, in bytes.
///
/// The artifact is read from one buffer, so this is the memory a load may cost.
/// Eight GiB is far above the 498 MB trained GPT-2 and far below "the process
/// dies before reporting an error".
pub const MAX_PAYLOAD_BYTES: u64 = 8 * (1 << 30);

/// Largest value table this build accepts.
pub const MAX_VALUES: u64 = 1 << 20;

/// Largest op list this build accepts.
pub const MAX_NODES: u64 = 1 << 20;

/// Largest weight table this build accepts.
pub const MAX_WEIGHTS: u64 = 1 << 20;

/// Largest tensor rank this build accepts anywhere in the artifact.
pub const MAX_RANK: u64 = 64;

/// Largest `OpParams.axes` vector this build accepts.
pub const MAX_PARAM_AXES: u64 = 1024;

/// Largest vocabulary size this build accepts.
///
/// A policy cap, not a hardware limit: real vocabularies are around 50k and the
/// largest production ones are under 300k.
pub const MAX_VOCAB_SIZE: u32 = 1 << 24;

// Smallest possible encoded size of one entry, used to prove that a declared
// count can fit in the bytes that remain before anything is allocated. Each is a
// lower bound on the encoding the reader below actually consumes, so a count
// that fails the proof cannot fit.
//
// A value entry is a name (4-byte length + at least one byte, because
// `PreparedGraph::check` rejects an empty name), a dtype, a shape kind and an
// origin. A weight or node origin carries a 4-byte handle; a graph-input origin
// carries no handle but every such value needs a 4-byte slot in the graph-input
// table that follows, so 11 is the floor for all three.
const MIN_VALUE_ENTRY: u64 = 4 + 1 + 1 + 1 + 4;
// A node always has at least one output, so the output count's 4 bytes are
// followed by at least one 4-byte value handle.
const MIN_NODE_ENTRY: u64 = 4 + 4 + 4 + 2 + 4 + 4 + 4 + 1 + 4 + 4;
const MIN_WEIGHT_ENTRY: u64 = 4 + 1 + 4 + 1 + 8;
const MIN_DIRECTORY_ENTRY: u64 = 16;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// The validated fixed header of an artifact.
///
/// [`PreparedHeader::parse`] reads only the framing: magic, version, encoding,
/// lengths and reserved fields. It is the cheap entry point a tool or a CLI uses
/// to answer "what is this file?" before committing to a full load — and the
/// first thing [`PreparedArtifact::from_bytes`] calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedHeader {
    /// Format version (always [`FORMAT_VERSION`] when `parse` succeeds).
    pub version: u16,
    /// Dtype encoding id (always `1` when `parse` succeeds).
    pub dtype_encoding: u32,
    /// Total file length declared by the header.
    pub file_bytes: u64,
    /// Payload length declared by the header.
    pub payload_bytes: u64,
    /// CRC-32 of the payload as stored in the header.
    pub payload_crc32: u32,
}

impl PreparedHeader {
    /// Validate the framing of `bytes`.
    ///
    /// Checks, in this order: minimum size, magic, format version, header
    /// length, dtype encoding, reserved fields, file length, payload length and
    /// the payload cap. The checksum is *not* recomputed here; it is a whole
    /// payload pass and belongs to [`PreparedArtifact::from_bytes`].
    pub fn parse(bytes: &[u8]) -> Result<PreparedHeader, FormatError> {
        if bytes.len() < HEADER_BYTES {
            return Err(FormatError::truncated(
                "prepared-model header",
                HEADER_BYTES as u64,
                bytes.len() as u64,
            ));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if magic != MAGIC {
            return Err(FormatError::new(FormatErrorKind::BadMagic { found: magic }));
        }
        let mut reader = Reader::new(&bytes[..HEADER_BYTES]);
        reader.take(8, "magic")?;

        let version = reader.u16("format version")?;
        if version != FORMAT_VERSION {
            return Err(FormatError::new(FormatErrorKind::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            }));
        }
        let header_length = u64::from(reader.u16("header length")?);
        if header_length != HEADER_BYTES as u64 {
            return Err(FormatError::new(FormatErrorKind::LengthMismatch {
                what: "header length",
                declared: header_length,
                actual: HEADER_BYTES as u64,
            }));
        }
        let dtype_encoding = reader.u32("dtype encoding")?;
        if dtype_encoding != DTYPE_ENCODING {
            return Err(FormatError::new(
                FormatErrorKind::UnsupportedDtypeEncoding {
                    found: dtype_encoding,
                    supported: DTYPE_ENCODING,
                },
            ));
        }
        reader.zero_u32("flags")?;
        reader.zero_u32("reserved header word")?;
        let file_bytes = reader.u64("file length")?;
        let payload_bytes = reader.u64("payload length")?;
        let payload_crc32 = reader.u32("payload checksum")?;
        let reserved = reader.take(20, "reserved header bytes")?;
        if let Some(position) = reserved.iter().position(|&byte| byte != 0) {
            return Err(FormatError::new(FormatErrorKind::ReservedField {
                field: "reserved header bytes",
                value: u64::from(reserved[position]),
            }));
        }

        let actual_bytes = bytes.len() as u64;
        if file_bytes != actual_bytes {
            return Err(FormatError::new(FormatErrorKind::LengthMismatch {
                what: "file length",
                declared: file_bytes,
                actual: actual_bytes,
            }));
        }
        // `actual_bytes >= HEADER_BYTES` was established above.
        let expected_payload = actual_bytes - HEADER_BYTES as u64;
        if payload_bytes != expected_payload {
            return Err(FormatError::new(FormatErrorKind::LengthMismatch {
                what: "payload length",
                declared: payload_bytes,
                actual: expected_payload,
            }));
        }
        if payload_bytes > MAX_PAYLOAD_BYTES {
            return Err(FormatError::limit(
                "payload length",
                payload_bytes,
                MAX_PAYLOAD_BYTES,
            ));
        }
        Ok(PreparedHeader {
            version,
            dtype_encoding,
            file_bytes,
            payload_bytes,
            payload_crc32,
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Provenance carried by the artifact.
///
/// Not required to *run* the model, but the first thing a person needs when two
/// artifacts behave differently: which ONNX IR and opset produced this graph,
/// from which exporter, and what the graph was called. It is also the section a
/// reproducible-build check compares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMetadata {
    /// ONNX IR version of the source model.
    pub ir_version: i64,
    /// Opset of the source model's default domain.
    pub opset: OpsetImport,
    /// Exporter name (`"pytorch"`).
    pub producer: String,
    /// Exporter version (`"2.8.0"`).
    pub producer_version: String,
    /// Source graph name.
    pub graph_name: String,
}

impl ArtifactMetadata {
    /// Read the provenance off an import.
    pub fn from_import(import: &ModelImport) -> ArtifactMetadata {
        ArtifactMetadata {
            ir_version: import.ir_version(),
            opset: import.opset().clone(),
            producer: import.producer().to_string(),
            producer_version: import.producer_version().to_string(),
            graph_name: import.graph_name().to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// A validated prepared model in its shareable, versioned form.
///
/// Build one from an import, write it, read it back. It owns weights, the graph
/// (values, ops and the graph interface), the architecture geometry and the
/// declared dynamic bounds — and nothing else. It holds no session state and is
/// independent of any device.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use fastnn::inference::import::import_onnx_file;
/// use fastnn::inference::model::format::PreparedArtifact;
/// use fastnn::inference::shape::ShapeEnvelope;
///
/// let imported = import_onnx_file("model.onnx")?;
/// let artifact = PreparedArtifact::from_import(&imported, ShapeEnvelope::uniform(1024)?)?;
/// artifact.write_file("model.fn3")?;
///
/// let loaded = PreparedArtifact::read_file("model.fn3")?;
/// assert_eq!(loaded.geometry(), artifact.geometry());
/// assert_eq!(loaded.to_bytes()?, artifact.to_bytes()?);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedArtifact {
    metadata: ArtifactMetadata,
    graph: PreparedGraph,
    weights: WeightSet,
    geometry: DecoderGeometry,
    envelope: ShapeEnvelope,
}

impl PreparedArtifact {
    /// Build from an import and the declared bounds.
    ///
    /// Copies the weight payload once (the import owns it); use
    /// [`PreparedArtifact::from_parts`] when the caller already owns the parts
    /// and wants to avoid that copy for a large model.
    ///
    /// The envelope is *declared*, not detected: how many tokens a deployment
    /// may ask for is a deployment decision, so it is an argument. It is
    /// validated here, so an envelope this build cannot honour never reaches a
    /// file.
    pub fn from_import(
        import: &ModelImport,
        envelope: ShapeEnvelope,
    ) -> Result<PreparedArtifact, FormatError> {
        let artifact = PreparedArtifact {
            metadata: ArtifactMetadata::from_import(import),
            graph: PreparedGraph::from_graph(import.graph()),
            weights: import.weights().clone(),
            geometry: import.geometry().clone(),
            envelope,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Build from already-owned parts.
    ///
    /// The zero-copy seam for a caller that consumed a [`ModelImport`] with
    /// [`ModelImport::into_parts`].
    pub fn from_parts(
        metadata: ArtifactMetadata,
        graph: PreparedGraph,
        weights: WeightSet,
        geometry: DecoderGeometry,
        envelope: ShapeEnvelope,
    ) -> Result<PreparedArtifact, FormatError> {
        let artifact = PreparedArtifact {
            metadata,
            graph,
            weights,
            geometry,
            envelope,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Decode an artifact from bytes.
    ///
    /// Fails on a bad magic, an unreadable version, a checksum mismatch, a
    /// truncated or over-long section, an unknown dtype or op code, and on any
    /// internal inconsistency between the sections. Never panics and never
    /// allocates from an unvalidated count.
    pub fn from_bytes(bytes: &[u8]) -> Result<PreparedArtifact, FormatError> {
        let header = PreparedHeader::parse(bytes)?;
        let payload = &bytes[HEADER_BYTES..];
        let computed = crc32(payload);
        if computed != header.payload_crc32 {
            return Err(FormatError::new(FormatErrorKind::ChecksumMismatch {
                stored: header.payload_crc32,
                computed,
            }));
        }
        let sections = read_directory(payload)?;
        let artifact = PreparedArtifact {
            metadata: decode_metadata(sections[0])?,
            geometry: decode_geometry(sections[1])?,
            envelope: decode_envelope(sections[2])?,
            graph: decode_graph(sections[3], sections[4])?,
            weights: decode_weights(sections[5])?,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Read an artifact from a file.
    pub fn read_file(path: impl AsRef<Path>) -> Result<PreparedArtifact, FormatError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?;
        PreparedArtifact::from_bytes(&bytes)
    }

    /// Encode to the canonical byte string.
    ///
    /// Deterministic: the same artifact always produces the same bytes, and
    /// `from_bytes(to_bytes())` round-trips byte-identically because the
    /// encoding has no nullable, unordered or padded-by-choice fields.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        self.validate()?;
        let payload = self.encode_payload()?;
        let payload_bytes = payload.len() as u64;
        if payload_bytes > MAX_PAYLOAD_BYTES {
            return Err(FormatError::limit(
                "payload length",
                payload_bytes,
                MAX_PAYLOAD_BYTES,
            ));
        }
        let file_bytes = HEADER_BYTES as u64 + payload_bytes;

        let mut file = Vec::with_capacity(file_bytes as usize);
        file.extend_from_slice(&MAGIC);
        file.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        file.extend_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
        file.extend_from_slice(&DTYPE_ENCODING.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&file_bytes.to_le_bytes());
        file.extend_from_slice(&payload_bytes.to_le_bytes());
        file.extend_from_slice(&crc32(&payload).to_le_bytes());
        file.resize(HEADER_BYTES, 0);
        debug_assert_eq!(file.len(), HEADER_BYTES);
        file.extend_from_slice(&payload);
        Ok(file)
    }

    /// Write the artifact to a file, replacing any existing file.
    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<(), FormatError> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        std::fs::write(path, &bytes)
            .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))
    }

    /// Provenance.
    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }

    /// The graph: values, ops and the graph interface.
    pub fn graph(&self) -> &PreparedGraph {
        &self.graph
    }

    /// The weight table, in declaration order.
    pub fn weights(&self) -> &WeightSet {
        &self.weights
    }

    /// The architecture geometry.
    pub fn geometry(&self) -> &DecoderGeometry {
        &self.geometry
    }

    /// The declared dynamic bounds.
    pub fn envelope(&self) -> ShapeEnvelope {
        self.envelope
    }

    /// Bytes of weight payload the artifact carries.
    pub fn weight_bytes(&self) -> u64 {
        self.weights.total_bytes()
    }

    /// Copy every weight into host tensors on the CPU.
    ///
    /// The loaded artifact stays independent: the tensors own aligned copies.
    pub fn to_model_weights(&self) -> Result<ModelWeights, Fn3Error> {
        self.weights.to_model_weights()
    }

    /// Consume into the five parts.
    pub fn into_parts(
        self,
    ) -> (
        ArtifactMetadata,
        PreparedGraph,
        WeightSet,
        DecoderGeometry,
        ShapeEnvelope,
    ) {
        (
            self.metadata,
            self.graph,
            self.weights,
            self.geometry,
            self.envelope,
        )
    }

    /// Turn the artifact into the planner and session input, with no ONNX file.
    ///
    /// This is the whole point of the artifact: the graph, the weights, the
    /// geometry and the envelope in the file are what a plan is built from, so
    /// a prepared deployment never needs the source it was prepared from.
    /// Nothing is re-detected and nothing is copied twice: the artifact is
    /// consumed and its weight bytes move into the import.
    ///
    /// The result satisfies the round-trip identity a verifier can check:
    /// `ArtifactMetadata::from_import(&artifact.to_model_import()) == *artifact.metadata()`,
    /// and [`Session::with_runtime`](crate::inference::model::Session::with_runtime)
    /// produces the same plans as an import of the source ONNX.
    pub fn into_model_import(self) -> ModelImport {
        let metadata = self.metadata;
        ModelImport::from_parts(
            self.graph.into_graph_ir(),
            self.weights,
            self.geometry,
            ModelProvenance {
                ir_version: metadata.ir_version,
                opset: metadata.opset,
                producer: metadata.producer,
                producer_version: metadata.producer_version,
                graph_name: metadata.graph_name,
            },
        )
    }

    /// [`PreparedArtifact::into_model_import`] without consuming the artifact.
    ///
    /// Copies the weight bytes once; prefer the consuming form when the
    /// artifact is not needed afterwards.
    pub fn to_model_import(&self) -> ModelImport {
        self.clone().into_model_import()
    }

    /// Check every invariant the reader and writer rely on.
    ///
    /// Called by [`PreparedArtifact::from_bytes`] (so a decoded file is always
    /// consistent) and by [`PreparedArtifact::to_bytes`] (so a hand-built
    /// artifact can never be written into an unreadable file).
    pub fn validate(&self) -> Result<(), FormatError> {
        self.envelope
            .validate()
            .map_err(|error| FormatError::contract("envelope", error))?;
        self.graph.check()?;
        self.graph.check_weights(self.weights.len())?;
        self.check_geometry()
    }

    /// Cross-check the geometry against the value table, the weight table and
    /// the envelope.
    ///
    /// This is the "bounded" rule: a file cannot claim a vocabulary, a layer
    /// count or a head geometry that the rest of the file does not support, and
    /// cannot declare fixed geometry that its own dynamic bounds exclude.
    fn check_geometry(&self) -> Result<(), FormatError> {
        let geometry = &self.geometry;
        let layer_cap = MAX_KV_LAYERS.min(u64::from(u32::MAX));
        if u64::from(geometry.layers) > layer_cap {
            return Err(FormatError::limit(
                "layer count",
                u64::from(geometry.layers),
                layer_cap,
            ));
        }
        if geometry.vocab_size > MAX_VOCAB_SIZE {
            return Err(FormatError::limit(
                "vocabulary size",
                u64::from(geometry.vocab_size),
                u64::from(MAX_VOCAB_SIZE),
            ));
        }
        if geometry.vocab_size == 0 {
            return Err(FormatError::inconsistent(
                "geometry",
                "the vocabulary size is 0; a model with no tokens cannot produce logits",
            ));
        }
        for (name, value) in [
            ("hidden size", geometry.hidden_size),
            ("MLP hidden size", geometry.mlp_hidden_size),
            ("query head count", geometry.query_heads),
            ("KV head count", geometry.kv_heads),
            ("head width", geometry.head_dim),
        ] {
            if value == 0 {
                return Err(FormatError::inconsistent(
                    "geometry",
                    format!("{name} must be positive, found 0"),
                ));
            }
        }
        let projected = u64::from(geometry.query_heads) * u64::from(geometry.head_dim);
        if projected != u64::from(geometry.hidden_size) {
            return Err(FormatError::inconsistent(
                "geometry",
                format!(
                    "query heads ({}) x head width ({}) = {projected}, but the model width is {}",
                    geometry.query_heads, geometry.head_dim, geometry.hidden_size
                ),
            ));
        }
        if geometry.kv.len() != geometry.layers as usize {
            return Err(FormatError::inconsistent(
                "geometry",
                format!(
                    "the model declares {} layers but {} KV interfaces",
                    geometry.layers,
                    geometry.kv.len()
                ),
            ));
        }

        let token_embedding = self.weight(geometry.token_embedding, "token embedding")?;
        let expected = [
            u64::from(geometry.vocab_size),
            u64::from(geometry.hidden_size),
        ];
        if token_embedding.shape().dims() != expected {
            return Err(FormatError::inconsistent(
                "geometry",
                format!(
                    "the token embedding weight `{}` is {}, but the declared vocabulary and \
                     width need [vocab_size, hidden_size] = {expected:?}",
                    token_embedding.name(),
                    token_embedding.shape()
                ),
            ));
        }

        match geometry.positions {
            PositionEmbedding::Learned {
                weight,
                max_positions,
            } => {
                if max_positions == 0 {
                    return Err(FormatError::inconsistent(
                        "geometry",
                        "the position table declares zero positions",
                    ));
                }
                let table = self.weight(weight, "position embedding")?;
                let expected = [u64::from(max_positions), u64::from(geometry.hidden_size)];
                if table.shape().dims() != expected {
                    return Err(FormatError::inconsistent(
                        "geometry",
                        format!(
                            "the position table `{}` is {}, but {max_positions} positions of \
                             width {} need {expected:?}",
                            table.name(),
                            table.shape(),
                            geometry.hidden_size
                        ),
                    ));
                }
            }
        }

        if let Some(epsilon) = geometry.layer_norm_epsilon {
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(FormatError::inconsistent(
                    "geometry",
                    format!("the LayerNorm epsilon is {epsilon}, which is not a positive float"),
                ));
            }
        }

        let logits = self.value(geometry.logits, "logits")?;
        if logits.dtype != DType::F32 {
            return Err(FormatError::inconsistent(
                "geometry",
                format!(
                    "the logits value `{}` is {}, but logits must be f32",
                    logits.name, logits.dtype
                ),
            ));
        }
        for (layer, interface) in geometry.kv.iter().enumerate() {
            for (field, id) in [
                ("past key", interface.past_key),
                ("past value", interface.past_value),
                ("present key", interface.present_key),
                ("present value", interface.present_value),
            ] {
                let value = self.value(id, "KV interface")?;
                if value.dtype != DType::F32 {
                    return Err(FormatError::inconsistent(
                        "geometry",
                        format!(
                            "layer {layer} {field} value `{}` is {}, but a KV cache is f32",
                            value.name, value.dtype
                        ),
                    ));
                }
            }
        }

        // Fixed geometry must be inside the declared dynamic bounds, or no
        // session could ever run this model.
        for (symbol, value) in [
            (Symbol::NHeads, u64::from(geometry.query_heads)),
            (Symbol::NKvHeads, u64::from(geometry.kv_heads)),
            (Symbol::HeadDim, u64::from(geometry.head_dim)),
        ] {
            self.envelope
                .get(symbol)
                .check(symbol, value)
                .map_err(|error| {
                    FormatError::inconsistent(
                        "geometry vs envelope",
                        format!(
                            "the model's fixed {symbol} is {value}, which its own declared \
                             envelope excludes: {error}"
                        ),
                    )
                })?;
        }
        Ok(())
    }

    fn value(&self, id: ValueId, what: &str) -> Result<&ValueInfo, FormatError> {
        self.graph.values().get(id.index()).ok_or_else(|| {
            FormatError::inconsistent(
                "geometry",
                format!(
                    "the {what} handle names value {}, but the graph has {} values",
                    id.index(),
                    self.graph.value_count()
                ),
            )
        })
    }

    fn weight(&self, id: WeightId, what: &str) -> Result<&WeightTensor, FormatError> {
        self.weights.get(id).ok_or_else(|| {
            FormatError::inconsistent(
                "geometry",
                format!(
                    "the {what} handle names weight {}, but the weight table has {} entries",
                    id.index(),
                    self.weights.len()
                ),
            )
        })
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    fn encode_payload(&self) -> Result<Vec<u8>, FormatError> {
        let sections: [Vec<u8>; SECTION_COUNT] = [
            encode_metadata(&self.metadata)?,
            encode_geometry(&self.geometry)?,
            encode_envelope(&self.envelope),
            encode_values(&self.graph)?,
            encode_nodes(&self.graph)?,
            encode_weights(&self.weights)?,
        ];

        let total: usize = sections.iter().map(Vec::len).sum();
        let mut payload = Writer::with_capacity(DIRECTORY_BYTES + total);
        payload.u32(SECTION_COUNT as u32);
        payload.u32(0);
        for (position, bytes) in sections.iter().enumerate() {
            payload.u32(tag::ALL[position]);
            payload.u32(0);
            payload.u64(bytes.len() as u64);
        }
        for bytes in &sections {
            payload.bytes(bytes);
            payload.pad_to_8();
        }
        debug_assert_eq!(
            payload.len() % 8,
            0,
            "every section is padded to an eight-byte boundary"
        );
        Ok(payload.into_bytes())
    }
}

// ---------------------------------------------------------------------------
// Payload directory
// ---------------------------------------------------------------------------

/// Read the section directory and return each section's bytes.
fn read_directory(payload: &[u8]) -> Result<Vec<&[u8]>, FormatError> {
    let mut reader = Reader::new(payload);
    let count = reader.count(
        "section directory",
        SECTION_COUNT as u64,
        MIN_DIRECTORY_ENTRY,
    )?;
    if count != SECTION_COUNT {
        return Err(FormatError::malformed(
            "payload directory",
            format!(
                "version {FORMAT_VERSION} has {SECTION_COUNT} sections, the file declares {count}"
            ),
        ));
    }
    reader.zero_u32("payload directory reserved word")?;

    let mut lengths = Vec::with_capacity(SECTION_COUNT);
    for expected in tag::ALL {
        let found = reader.u32("section tag")?;
        reader.zero_u32("section tag reserved word")?;
        let length = reader.u64("section length")?;
        if found != expected {
            return Err(FormatError::malformed(
                "payload directory",
                format!(
                    "section {} is tagged {found} ({}), expected {expected} ({})",
                    lengths.len(),
                    tag::name(found),
                    tag::name(expected)
                ),
            ));
        }
        if length > reader.remaining() {
            return Err(FormatError::truncated(
                format!("{} section", tag::name(expected)),
                length,
                reader.remaining(),
            ));
        }
        lengths.push((expected, length));
    }

    let mut sections = Vec::with_capacity(SECTION_COUNT);
    for (tag, length) in lengths {
        debug_assert_eq!(
            reader.pos() % 8,
            0,
            "a section must start on an eight-byte boundary"
        );
        let bytes = reader.take(length, format!("{} section", tag::name(tag)))?;
        reader.pad_to_8(format!("{} section padding", tag::name(tag)))?;
        sections.push(bytes);
    }
    reader.finish("payload")?;
    Ok(sections)
}

// ---------------------------------------------------------------------------
// META
// ---------------------------------------------------------------------------

fn encode_metadata(metadata: &ArtifactMetadata) -> Result<Vec<u8>, FormatError> {
    let mut section = Writer::with_capacity(96);
    section.i64(metadata.ir_version);
    section.string(&metadata.opset.domain)?;
    section.i64(metadata.opset.version);
    section.string(&metadata.producer)?;
    section.string(&metadata.producer_version)?;
    section.string(&metadata.graph_name)?;
    Ok(section.into_bytes())
}

fn decode_metadata(section: &[u8]) -> Result<ArtifactMetadata, FormatError> {
    let mut reader = Reader::new(section);
    let ir_version = reader.i64("metadata ir_version")?;
    let domain = reader.string("opset domain")?;
    let version = reader.i64("opset version")?;
    let producer = reader.string("producer name")?;
    let producer_version = reader.string("producer version")?;
    let graph_name = reader.string("graph name")?;
    reader.finish("META section")?;
    Ok(ArtifactMetadata {
        ir_version,
        opset: OpsetImport { domain, version },
        producer,
        producer_version,
        graph_name,
    })
}

// ---------------------------------------------------------------------------
// GEOM
// ---------------------------------------------------------------------------

fn encode_geometry(geometry: &DecoderGeometry) -> Result<Vec<u8>, FormatError> {
    let mut section = Writer::with_capacity(72);
    section.u32(geometry.layers);
    section.u32(geometry.hidden_size);
    section.u32(geometry.mlp_hidden_size);
    section.u32(geometry.query_heads);
    section.u32(geometry.kv_heads);
    section.u32(geometry.head_dim);
    section.u32(geometry.vocab_size);
    match geometry.positions {
        PositionEmbedding::Learned {
            weight,
            max_positions,
        } => {
            section.u8(0);
            section.u32(weight.index() as u32);
            section.u32(max_positions);
        }
    }
    match geometry.layer_norm_epsilon {
        Some(epsilon) => {
            section.u8(1);
            section.f32(epsilon);
        }
        None => {
            section.u8(0);
            section.f32(0.0);
        }
    }
    section.u32(geometry.token_embedding.index() as u32);
    section.u32(geometry.logits.index() as u32);
    let kv_count = u32::try_from(geometry.kv.len()).map_err(|_| {
        FormatError::limit(
            "KV interface count",
            geometry.kv.len() as u64,
            u32::MAX as u64,
        )
    })?;
    section.u32(kv_count);
    for interface in &geometry.kv {
        section.u32(interface.past_key.index() as u32);
        section.u32(interface.past_value.index() as u32);
        section.u32(interface.present_key.index() as u32);
        section.u32(interface.present_value.index() as u32);
    }
    Ok(section.into_bytes())
}

fn decode_geometry(section: &[u8]) -> Result<DecoderGeometry, FormatError> {
    let mut reader = Reader::new(section);
    let layers = reader.u32("geometry layer count")?;
    let hidden_size = reader.u32("geometry hidden size")?;
    let mlp_hidden_size = reader.u32("geometry MLP size")?;
    let query_heads = reader.u32("geometry query head count")?;
    let kv_heads = reader.u32("geometry KV head count")?;
    let head_dim = reader.u32("geometry head width")?;
    let vocab_size = reader.u32("geometry vocabulary size")?;

    let positions_kind = reader.u8("geometry position kind")?;
    if positions_kind != 0 {
        return Err(FormatError::malformed(
            "geometry",
            format!(
                "position embedding kind {positions_kind} is not defined; this build knows \
                 kind 0 (learned position table)"
            ),
        ));
    }
    let position_weight = WeightId::new(reader.u32("geometry position weight")?);
    let max_positions = reader.u32("geometry position count")?;

    let epsilon_present = reader.u8("geometry epsilon flag")?;
    let epsilon_bits = reader.u32("geometry epsilon")?;
    let layer_norm_epsilon = match epsilon_present {
        0 if epsilon_bits == 0 => None,
        0 => {
            return Err(FormatError::malformed(
                "geometry",
                format!("the epsilon flag is 0 but its value bits are {epsilon_bits:#010x}"),
            ))
        }
        1 => Some(f32::from_bits(epsilon_bits)),
        other => {
            return Err(FormatError::malformed(
                "geometry",
                format!("the epsilon flag is {other}, expected 0 or 1"),
            ))
        }
    };

    let token_embedding = WeightId::new(reader.u32("geometry token embedding")?);
    let logits = ValueId::new(reader.u32("geometry logits")?);
    let kv_count = reader.count("KV interface table", MAX_KV_LAYERS, 16)?;
    let mut kv = Vec::with_capacity(kv_count);
    for layer in 0..kv_count {
        kv.push(KvInterface {
            past_key: ValueId::new(reader.u32(format!("layer {layer} past key"))?),
            past_value: ValueId::new(reader.u32(format!("layer {layer} past value"))?),
            present_key: ValueId::new(reader.u32(format!("layer {layer} present key"))?),
            present_value: ValueId::new(reader.u32(format!("layer {layer} present value"))?),
        });
    }
    reader.finish("GEOM section")?;
    Ok(DecoderGeometry {
        layers,
        hidden_size,
        mlp_hidden_size,
        query_heads,
        kv_heads,
        head_dim,
        vocab_size,
        positions: PositionEmbedding::Learned {
            weight: position_weight,
            max_positions,
        },
        layer_norm_epsilon,
        token_embedding,
        logits,
        kv,
    })
}

// ---------------------------------------------------------------------------
// ENV
// ---------------------------------------------------------------------------

fn encode_envelope(envelope: &ShapeEnvelope) -> Vec<u8> {
    let mut section = Writer::with_capacity(Symbol::COUNT * 16);
    for symbol in Symbol::ALL {
        let bound = envelope.get(symbol);
        section.u64(bound.min());
        section.u64(bound.max());
    }
    section.into_bytes()
}

fn decode_envelope(section: &[u8]) -> Result<ShapeEnvelope, FormatError> {
    let mut reader = Reader::new(section);
    let mut bounds = [DimBound::new(0, 0).expect("0,0 is a valid bound"); Symbol::COUNT];
    for (position, symbol) in Symbol::ALL.into_iter().enumerate() {
        let min = reader.u64(format!("envelope {} minimum", symbol.name()))?;
        let max = reader.u64(format!("envelope {} maximum", symbol.name()))?;
        bounds[position] =
            DimBound::new(min, max).map_err(|error| FormatError::contract("envelope", error))?;
    }
    reader.finish("ENV section")?;
    Ok(ShapeEnvelope::new(bounds))
}

// ---------------------------------------------------------------------------
// VALS + NODE
// ---------------------------------------------------------------------------

fn encode_values(graph: &PreparedGraph) -> Result<Vec<u8>, FormatError> {
    let count = u32::try_from(graph.value_count()).map_err(|_| {
        FormatError::limit("value count", graph.value_count() as u64, u32::MAX as u64)
    })?;
    let mut section = Writer::with_capacity(64 + graph.value_count() * 32);
    section.u32(count);
    for value in graph.values() {
        section.string(&value.name)?;
        section.u8(value.dtype.code());
        match &value.shape {
            ValueShape::Unknown => section.u8(0),
            ValueShape::Dims(dims) => {
                section.u8(1);
                let rank = u32::try_from(dims.len()).map_err(|_| {
                    FormatError::limit("value rank", dims.len() as u64, u32::MAX as u64)
                })?;
                section.u32(rank);
                for dim in dims {
                    match dim {
                        Dim::Known(extent) => {
                            section.u8(0);
                            section.u64(*extent);
                        }
                        Dim::Symbol(symbol) => {
                            section.u8(1);
                            section.u8(symbol_code(*symbol));
                        }
                        Dim::Unknown(reason) => {
                            section.u8(2);
                            section.string(reason)?;
                        }
                    }
                }
            }
        }
        match value.origin {
            ValueOrigin::GraphInput => section.u8(0),
            ValueOrigin::Weight(weight) => {
                section.u8(1);
                section.u32(weight.index() as u32);
            }
            ValueOrigin::Node(node) => {
                section.u8(2);
                section.u32(node);
            }
        }
    }
    let input_count = u32::try_from(graph.inputs().len()).map_err(|_| {
        FormatError::limit(
            "graph input count",
            graph.inputs().len() as u64,
            u32::MAX as u64,
        )
    })?;
    section.u32(input_count);
    for id in graph.inputs() {
        section.u32(id.index() as u32);
    }
    let output_count = u32::try_from(graph.outputs().len()).map_err(|_| {
        FormatError::limit(
            "graph output count",
            graph.outputs().len() as u64,
            u32::MAX as u64,
        )
    })?;
    section.u32(output_count);
    for id in graph.outputs() {
        section.u32(id.index() as u32);
    }
    Ok(section.into_bytes())
}

fn encode_nodes(graph: &PreparedGraph) -> Result<Vec<u8>, FormatError> {
    let count = u32::try_from(graph.nodes().len()).map_err(|_| {
        FormatError::limit("node count", graph.nodes().len() as u64, u32::MAX as u64)
    })?;
    let mut section = Writer::with_capacity(64 + graph.nodes().len() * 64);
    section.u32(count);
    for node in graph.nodes() {
        section.u32(node.onnx_index);
        section.string(&node.name)?;
        section.string(&node.onnx_op)?;
        section.u16(node.op.code());
        let input_count = u32::try_from(node.inputs.len()).map_err(|_| {
            FormatError::limit(
                "node input count",
                node.inputs.len() as u64,
                u32::MAX as u64,
            )
        })?;
        section.u32(input_count);
        for id in &node.inputs {
            section.u32(id.index() as u32);
        }
        let output_count = u32::try_from(node.outputs.len()).map_err(|_| {
            FormatError::limit(
                "node output count",
                node.outputs.len() as u64,
                u32::MAX as u64,
            )
        })?;
        section.u32(output_count);
        for id in &node.outputs {
            section.u32(id.index() as u32);
        }
        let axes_count = u32::try_from(node.params.axes.len()).map_err(|_| {
            FormatError::limit(
                "op axes count",
                node.params.axes.len() as u64,
                u32::MAX as u64,
            )
        })?;
        section.u32(axes_count);
        for axis in &node.params.axes {
            section.i64(*axis);
        }
        section.u8(u8::from(node.params.keep_dims));
        section.f32(node.params.epsilon);
    }
    Ok(section.into_bytes())
}

fn decode_graph(values_section: &[u8], nodes_section: &[u8]) -> Result<PreparedGraph, FormatError> {
    let mut reader = Reader::new(values_section);
    let count = reader.count("value table", MAX_VALUES, MIN_VALUE_ENTRY)?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let name = reader.string(format!("value {index} name"))?;
        let dtype = dtype(reader.u8(format!("value {index} dtype"))?, "value table")?;
        let shape = decode_shape(&mut reader, index)?;
        let origin_kind = reader.u8(format!("value {index} origin"))?;
        let origin = match origin_kind {
            0 => ValueOrigin::GraphInput,
            1 => ValueOrigin::Weight(WeightId::new(
                reader.u32(format!("value {index} weight handle"))?,
            )),
            2 => ValueOrigin::Node(reader.u32(format!("value {index} producer node"))?),
            other => {
                return Err(FormatError::malformed(
                    "value table",
                    format!(
                        "value {index} (`{name}`) has origin kind {other}, expected 0 (graph \
                         input), 1 (weight) or 2 (node)"
                    ),
                ))
            }
        };
        values.push(ValueInfo {
            name,
            dtype,
            shape,
            origin,
        });
    }
    let input_count = reader.count("graph input table", MAX_VALUES, 4)?;
    let mut inputs = Vec::with_capacity(input_count);
    for position in 0..input_count {
        inputs.push(ValueId::new(reader.u32(format!("graph input {position}"))?));
    }
    let output_count = reader.count("graph output table", MAX_VALUES, 4)?;
    let mut outputs = Vec::with_capacity(output_count);
    for position in 0..output_count {
        outputs.push(ValueId::new(
            reader.u32(format!("graph output {position}"))?,
        ));
    }
    reader.finish("VALS section")?;

    let mut reader = Reader::new(nodes_section);
    let count = reader.count("node table", MAX_NODES, MIN_NODE_ENTRY)?;
    let mut nodes = Vec::with_capacity(count);
    for index in 0..count {
        let onnx_index = reader.u32(format!("node {index} ONNX index"))?;
        let name = reader.string(format!("node {index} name"))?;
        let onnx_op = reader.string(format!("node {index} ONNX op"))?;
        let code = reader.u16(format!("node {index} op code"))?;
        let op = OpId::from_u16(code)
            .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
        let input_count = reader.count(format!("node {index} inputs"), MAX_PARAM_AXES, 4)?;
        let mut inputs = Vec::with_capacity(input_count);
        for position in 0..input_count {
            inputs.push(ValueId::new(
                reader.u32(format!("node {index} input {position}"))?,
            ));
        }
        let output_count = reader.count(format!("node {index} outputs"), MAX_PARAM_AXES, 4)?;
        let mut node_outputs = Vec::with_capacity(output_count);
        for position in 0..output_count {
            node_outputs.push(ValueId::new(
                reader.u32(format!("node {index} output {position}"))?,
            ));
        }
        let axes_count = reader.count(format!("node {index} axes"), MAX_PARAM_AXES, 8)?;
        let mut axes = Vec::with_capacity(axes_count);
        for position in 0..axes_count {
            axes.push(reader.i64(format!("node {index} axis {position}"))?);
        }
        let keep_dims = match reader.u8(format!("node {index} keep_dims"))? {
            0 => false,
            1 => true,
            other => {
                return Err(FormatError::malformed(
                    "node table",
                    format!("node {index} (`{name}`) has keep_dims {other}, expected 0 or 1"),
                ))
            }
        };
        let epsilon = reader.f32(format!("node {index} epsilon"))?;
        nodes.push(crate::inference::import::GraphNode {
            onnx_index,
            name,
            onnx_op,
            op,
            inputs,
            outputs: node_outputs,
            params: crate::inference::backend::OpParams {
                axes,
                keep_dims,
                epsilon,
            },
        });
    }
    reader.finish("NODE section")?;

    PreparedGraph::new(values, nodes, inputs, outputs)
}

fn decode_shape(reader: &mut Reader<'_>, index: usize) -> Result<ValueShape, FormatError> {
    let kind = reader.u8(format!("value {index} shape kind"))?;
    match kind {
        0 => Ok(ValueShape::Unknown),
        1 => {
            let rank = reader.count(format!("value {index} rank"), MAX_RANK, 2)?;
            let mut dims = Vec::with_capacity(rank);
            for axis in 0..rank {
                let dim_kind = reader.u8(format!("value {index} axis {axis} kind"))?;
                dims.push(match dim_kind {
                    0 => Dim::Known(reader.u64(format!("value {index} axis {axis} extent"))?),
                    1 => {
                        let code = reader.u8(format!("value {index} axis {axis} symbol"))?;
                        Dim::Symbol(symbol_from_code(code).map_err(|detail| {
                            FormatError::malformed(
                                "value table",
                                format!("value {index} axis {axis}: {detail}"),
                            )
                        })?)
                    }
                    2 => Dim::Unknown(
                        reader.string(format!("value {index} axis {axis} unknown reason"))?,
                    ),
                    other => {
                        return Err(FormatError::malformed(
                            "value table",
                            format!(
                                "value {index} axis {axis} has dim kind {other}, expected 0 \
                                 (known), 1 (symbol) or 2 (unknown)"
                            ),
                        ))
                    }
                });
            }
            Ok(ValueShape::Dims(dims))
        }
        other => Err(FormatError::malformed(
            "value table",
            format!("value {index} has shape kind {other}, expected 0 (unknown) or 1 (ranked)"),
        )),
    }
}

// ---------------------------------------------------------------------------
// WGHT
// ---------------------------------------------------------------------------

fn encode_weights(weights: &WeightSet) -> Result<Vec<u8>, FormatError> {
    let count = u32::try_from(weights.len())
        .map_err(|_| FormatError::limit("weight count", weights.len() as u64, u32::MAX as u64))?;
    let payload_bytes: usize = weights.iter().map(WeightTensor::byte_len).sum();
    let mut section = Writer::with_capacity(32 + weights.len() * 64 + payload_bytes);
    section.u32(count);
    for weight in weights.iter() {
        section.string(weight.name())?;
        section.u8(weight.dtype().code());
        let rank = u32::try_from(weight.shape().rank()).map_err(|_| {
            FormatError::limit("weight rank", weight.shape().rank() as u64, u32::MAX as u64)
        })?;
        section.u32(rank);
        for extent in weight.shape().dims() {
            section.u64(*extent);
        }
        section.u8(match weight.source() {
            WeightSource::Initializer => 0,
            WeightSource::AttributeConstant => 1,
        });
        section.u64(weight.byte_len() as u64);
        section.pad_to_8();
        section.bytes(weight.bytes());
    }
    Ok(section.into_bytes())
}

fn decode_weights(section_bytes: &[u8]) -> Result<WeightSet, FormatError> {
    let mut reader = Reader::new(section_bytes);
    let count = reader.count("weight table", MAX_WEIGHTS, MIN_WEIGHT_ENTRY)?;
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let name = reader.string(format!("weight {index} name"))?;
        let dtype = dtype(reader.u8(format!("weight {index} dtype"))?, "weight table")?;
        let rank = reader.count(format!("weight {index} rank"), MAX_RANK, 8)?;
        let mut dims = Vec::with_capacity(rank);
        for axis in 0..rank {
            dims.push(reader.u64(format!("weight {index} axis {axis}"))?);
        }
        let source = match reader.u8(format!("weight {index} source"))? {
            0 => WeightSource::Initializer,
            1 => WeightSource::AttributeConstant,
            other => {
                return Err(FormatError::malformed(
                    "weight table",
                    format!(
                        "weight {index} (`{name}`) has source {other}, expected 0 (initializer) \
                         or 1 (attribute constant)"
                    ),
                ))
            }
        };
        let payload_len = reader.u64(format!("weight {index} payload length"))?;
        reader.pad_to_8(format!("weight {index} padding"))?;
        let payload = reader.take(payload_len, format!("weight {index} payload"))?;
        entries.push(
            WeightTensor::new(
                name,
                dtype,
                Shape::from_dims(dims),
                payload.to_vec(),
                source,
            )
            .map_err(|error| FormatError::contract("weight table", error))?,
        );
    }
    reader.finish("WGHT section")?;
    WeightSet::new(entries).map_err(|error| FormatError::contract("weight table", error))
}

// ---------------------------------------------------------------------------
// Shared field codecs
// ---------------------------------------------------------------------------

/// Decode a dtype code, naming the reader that rejected it.
fn dtype(code: u8, what: &'static str) -> Result<DType, FormatError> {
    DType::from_code(code, what).map_err(FormatError::from)
}

/// Stable on-disk code for a dynamic-dimension symbol.
///
/// The match is exhaustive on purpose: adding a [`Symbol`] variant must fail to
/// compile here so a code is assigned deliberately rather than by declaration
/// order.
fn symbol_code(symbol: Symbol) -> u8 {
    match symbol {
        Symbol::Batch => 0,
        Symbol::Seq => 1,
        Symbol::Past => 2,
        Symbol::Total => 3,
        Symbol::NHeads => 4,
        Symbol::NKvHeads => 5,
        Symbol::HeadDim => 6,
    }
}

fn symbol_from_code(code: u8) -> Result<Symbol, String> {
    match code {
        0 => Ok(Symbol::Batch),
        1 => Ok(Symbol::Seq),
        2 => Ok(Symbol::Past),
        3 => Ok(Symbol::Total),
        4 => Ok(Symbol::NHeads),
        5 => Ok(Symbol::NKvHeads),
        6 => Ok(Symbol::HeadDim),
        other => Err(format!(
            "shape symbol code {other} is not defined; this build knows 0..=6"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inference::import::{Dim, ValueShape};
    use crate::inference::model::KvCacheLayout;

    fn tiny_geometry() -> DecoderGeometry {
        DecoderGeometry {
            layers: 1,
            hidden_size: 4,
            mlp_hidden_size: 8,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            vocab_size: 6,
            positions: PositionEmbedding::Learned {
                weight: WeightId::new(1),
                max_positions: 3,
            },
            layer_norm_epsilon: Some(1e-5),
            token_embedding: WeightId::new(0),
            logits: ValueId::new(4),
            kv: vec![KvInterface {
                past_key: ValueId::new(1),
                past_value: ValueId::new(2),
                present_key: ValueId::new(3),
                present_value: ValueId::new(4),
            }],
        }
    }

    fn tiny_graph() -> PreparedGraph {
        use crate::inference::backend::OpParams;
        use crate::inference::import::GraphNode;
        PreparedGraph::new(
            vec![
                ValueInfo {
                    name: "past.0.key".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::Known(1), Dim::Known(2)]),
                    origin: ValueOrigin::GraphInput,
                },
                ValueInfo {
                    name: "past.0.value".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::Symbol(Symbol::Past)]),
                    origin: ValueOrigin::GraphInput,
                },
                ValueInfo {
                    name: "present.0.key".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Unknown,
                    origin: ValueOrigin::GraphInput,
                },
                ValueInfo {
                    name: "present.0.value".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::unknown("past + seq")]),
                    origin: ValueOrigin::GraphInput,
                },
                ValueInfo {
                    name: "logits".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::Known(6)]),
                    origin: ValueOrigin::Node(0),
                },
                ValueInfo {
                    name: "wpe".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::Known(3), Dim::Known(4)]),
                    origin: ValueOrigin::Weight(WeightId::new(1)),
                },
                ValueInfo {
                    name: "wte".to_string(),
                    dtype: DType::F32,
                    shape: ValueShape::Dims(vec![Dim::Known(6), Dim::Known(4)]),
                    origin: ValueOrigin::Weight(WeightId::new(0)),
                },
            ],
            vec![GraphNode {
                onnx_index: 3,
                name: "matmul".to_string(),
                onnx_op: "MatMul".to_string(),
                op: OpId::MatMul,
                inputs: vec![ValueId::new(0), ValueId::new(6)],
                outputs: vec![ValueId::new(4)],
                params: OpParams {
                    axes: vec![-1, 0, 3],
                    keep_dims: true,
                    epsilon: 1e-5,
                },
            }],
            vec![
                ValueId::new(0),
                ValueId::new(1),
                ValueId::new(2),
                ValueId::new(3),
            ],
            vec![ValueId::new(4)],
        )
        .unwrap()
    }

    fn tiny_weights() -> WeightSet {
        WeightSet::new(vec![
            WeightTensor::new(
                "wte",
                DType::F32,
                Shape::from_dims([6, 4]),
                vec![0x11; 6 * 4 * 4],
                WeightSource::Initializer,
            )
            .unwrap(),
            WeightTensor::new(
                "wpe",
                DType::F32,
                Shape::from_dims([3, 4]),
                vec![0x22; 3 * 4 * 4],
                WeightSource::AttributeConstant,
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn tiny_artifact() -> PreparedArtifact {
        PreparedArtifact::from_parts(
            ArtifactMetadata {
                ir_version: 7,
                opset: OpsetImport {
                    domain: String::new(),
                    version: 14,
                },
                producer: "pytorch".to_string(),
                producer_version: "2.8.0".to_string(),
                graph_name: "main_graph".to_string(),
            },
            tiny_graph(),
            tiny_weights(),
            tiny_geometry(),
            ShapeEnvelope::uniform(8).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn a_minimal_artifact_round_trips_byte_identically() {
        let artifact = tiny_artifact();
        let bytes = artifact.to_bytes().unwrap();
        let loaded = PreparedArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(loaded, artifact);
        assert_eq!(loaded.to_bytes().unwrap(), bytes);
        assert_eq!(artifact.geometry().kv_cache_config(4, 1).dtype, DType::F32);
        assert_eq!(
            artifact.geometry().kv_cache_config(4, 1).layout,
            KvCacheLayout::BatchHeadsCapacityHeadDim
        );
    }

    #[test]
    fn the_header_layout_matches_the_documented_offsets() {
        let artifact = tiny_artifact();
        let bytes = artifact.to_bytes().unwrap();
        assert_eq!(&bytes[0..8], b"FNN3PREP");
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
        assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 64);
        assert_eq!(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            1
        );
        assert_eq!(bytes[16..24], [0u8; 8]);
        assert_eq!(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            bytes.len() as u64
        );
        assert_eq!(
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            (bytes.len() - 64) as u64
        );
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            crc32(&bytes[64..])
        );
        assert_eq!(bytes[44..64], [0u8; 20]);
        let header = PreparedHeader::parse(&bytes).unwrap();
        assert_eq!(header.payload_bytes, (bytes.len() - 64) as u64);
    }

    #[test]
    fn weight_payloads_are_eight_byte_aligned_and_zero_padded() {
        let bytes = tiny_artifact().to_bytes().unwrap();
        // Re-walk the payload directory like a tool would, then check the
        // alignment rule: every weight payload starts at a file offset that is a
        // multiple of 8.
        let payload = &bytes[HEADER_BYTES..];
        let mut reader = Reader::new(payload);
        let count = reader.count("directory", 8, 16).unwrap();
        assert_eq!(count, SECTION_COUNT);
        reader.u32("reserved").unwrap();
        let mut lengths = Vec::new();
        for _ in 0..count {
            reader.u32("tag").unwrap();
            reader.u32("reserved").unwrap();
            lengths.push(reader.u64("length").unwrap());
        }
        let mut offsets = Vec::new();
        for (position, length) in lengths.iter().enumerate() {
            let offset = reader.pos() as u64;
            offsets.push(offset);
            reader.take(*length, format!("section {position}")).unwrap();
            reader.pad_to_8("padding").unwrap();
        }
        assert_eq!(offsets[5] % 8, 0);
        // The first weight payload follows its header fields, so just assert the
        // rule the reader enforces: the section itself decodes and every payload
        // reader position is aligned.
        let loaded = PreparedArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.weights().len(), 2);
        for weight in loaded.weights().iter() {
            assert_eq!(
                weight.byte_len() as u64,
                weight.shape().numel().unwrap() * 4
            );
        }
    }

    #[test]
    fn a_wrong_dtype_code_names_the_code_and_the_reader() {
        let artifact = tiny_artifact();
        let bytes = artifact.to_bytes().unwrap();
        let mut mutated = bytes.clone();
        // Find the WGHT section and corrupt the first weight's dtype byte: the
        // first weight's dtype sits after the section's `u32 count`, a `u32`
        // name length and the 3-byte name "wte".
        let wght = find_section(&mutated, tag::WEIGHTS);
        let dtype_position = wght + 4 + 4 + 3;
        mutated[dtype_position] = 9;
        // Fix the checksum so the dtype code, not the CRC, is what fails.
        let payload = &mutated[HEADER_BYTES..];
        let crc = crc32(payload);
        mutated[40..44].copy_from_slice(&crc.to_le_bytes());
        let error = PreparedArtifact::from_bytes(&mutated).unwrap_err();
        assert!(
            matches!(error.kind(), FormatErrorKind::UnknownDType(_)),
            "{error}"
        );
        assert!(
            error.to_string().contains("unknown dtype code 9"),
            "{error}"
        );
        assert!(error.to_string().contains("weight table"), "{error}");
    }

    #[test]
    fn an_unknown_op_code_is_reported_by_name() {
        let artifact = tiny_artifact();
        let mut bytes = artifact.to_bytes().unwrap();
        let node = find_section(&bytes, tag::NODES);
        // NODE section: u32 count, u32 onnx_index, u32 name length, name bytes,
        // u32 op-name length, op-name bytes, then the u16 op code.
        let name_len = u32::from_le_bytes(bytes[node + 8..node + 12].try_into().unwrap()) as usize;
        let op_len_position = node + 12 + name_len;
        let op_len = u32::from_le_bytes(
            bytes[op_len_position..op_len_position + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let code_position = op_len_position + 4 + op_len;
        bytes[code_position] = 0x01;
        bytes[code_position + 1] = 0x02; // 0x0201, a reserved range
        let payload = &bytes[HEADER_BYTES..];
        let crc = crc32(payload);
        bytes[40..44].copy_from_slice(&crc.to_le_bytes());
        let error = PreparedArtifact::from_bytes(&bytes).unwrap_err();
        assert!(
            matches!(
                error.kind(),
                FormatErrorKind::UnknownOpCode { code: 0x0201 }
            ),
            "{error}"
        );
        assert!(error.to_string().contains("reserved range"), "{error}");
    }

    /// File offset of a section's first byte, by tag.
    fn find_section(bytes: &[u8], tag: u32) -> usize {
        let mut position = HEADER_BYTES + 8;
        let mut offset = HEADER_BYTES + DIRECTORY_BYTES;
        for _ in 0..SECTION_COUNT {
            let found = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
            let length = u64::from_le_bytes(bytes[position + 8..position + 16].try_into().unwrap());
            if found == tag {
                return offset;
            }
            position += 16;
            offset += length as usize;
            offset += (8 - (length as usize % 8)) % 8;
        }
        panic!("section {tag} not found");
    }

    #[test]
    fn the_envelope_is_written_in_symbol_order() {
        let envelope = ShapeEnvelope::new([
            DimBound::new(1, 2).unwrap(),
            DimBound::new(3, 4).unwrap(),
            DimBound::new(5, 6).unwrap(),
            DimBound::new(7, 8).unwrap(),
            DimBound::new(9, 10).unwrap(),
            DimBound::new(11, 12).unwrap(),
            DimBound::new(13, 14).unwrap(),
        ]);
        let bytes = encode_envelope(&envelope);
        assert_eq!(bytes.len(), 112);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(bytes[96..104].try_into().unwrap()), 13);
        assert_eq!(u64::from_le_bytes(bytes[104..112].try_into().unwrap()), 14);
        assert_eq!(decode_envelope(&bytes).unwrap(), envelope);
    }

    #[test]
    fn geometry_rejects_a_vocabulary_that_the_weights_do_not_support() {
        let mut geometry = tiny_geometry();
        geometry.vocab_size = 7;
        let error = PreparedArtifact::from_parts(
            ArtifactMetadata {
                ir_version: 7,
                opset: OpsetImport {
                    domain: String::new(),
                    version: 14,
                },
                producer: "test".to_string(),
                producer_version: "0".to_string(),
                graph_name: "g".to_string(),
            },
            tiny_graph(),
            tiny_weights(),
            geometry,
            ShapeEnvelope::uniform(8).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("token embedding"), "{error}");
    }

    #[test]
    fn geometry_rejects_bounds_that_exclude_its_own_fixed_heads() {
        let envelope = ShapeEnvelope::uniform(8)
            .unwrap()
            .with(Symbol::NHeads, DimBound::new(8, 9).unwrap());
        let error = PreparedArtifact::from_parts(
            ArtifactMetadata {
                ir_version: 7,
                opset: OpsetImport {
                    domain: String::new(),
                    version: 14,
                },
                producer: "test".to_string(),
                producer_version: "0".to_string(),
                graph_name: "g".to_string(),
            },
            tiny_graph(),
            tiny_weights(),
            tiny_geometry(),
            envelope,
        )
        .unwrap_err();
        assert!(error.to_string().contains("fixed n_heads is 2"), "{error}");
    }

    #[test]
    fn symbols_are_coded_stably() {
        for (expected, symbol) in Symbol::ALL.into_iter().enumerate() {
            assert_eq!(symbol_code(symbol), expected as u8);
            assert_eq!(symbol_from_code(expected as u8), Ok(symbol));
        }
        assert!(symbol_from_code(7).is_err());
        assert!(symbol_from_code(u8::MAX).is_err());
    }
}
