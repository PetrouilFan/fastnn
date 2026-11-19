//! A self-contained, bounds-checked reader for the ONNX protobuf schema.
//!
//! # Why this exists
//!
//! The 2.x tree has no protobuf decoder: `src/onnx/` consumes a JSON blob that
//! the Python `onnx` package produced, and 3.0 must not require Python at
//! runtime. So this module decodes the wire format directly. It has no
//! dependency and no `unsafe`.
//!
//! # What it decodes
//!
//! Only the messages the importer needs, with only the fields it reads:
//!
//! | Message | Fields used |
//! |---|---|
//! | `ModelProto` | `ir_version`, producer, `opset_import`, `graph`, `functions`, `training_info` |
//! | `GraphProto` | `node`, `name`, `initializer`, `input`, `output`, `value_info`, sparse/quantization counts |
//! | `NodeProto` | `input`, `output`, `name`, `op_type`, `attribute`, `domain` |
//! | `AttributeProto` | `name`, `type`, scalar/list payloads, tensor payloads, subgraph markers |
//! | `TensorProto` | `dims`, `data_type`, `raw_data`, typed payloads, name, external/segment markers |
//! | `ValueInfoProto` | `name`, `type` (`TypeProto.Tensor` only) |
//! | `TypeProto` | `elem_type`, `shape` (`TensorShapeProto.Dimension`) |
//! | `OperatorSetIdProto` | `domain`, `version` |
//!
//! Unknown fields are skipped by wire type, which is what protobuf requires: a
//! newer exporter may add fields without breaking this reader. A *known* field
//! with the wrong wire type is an error, because that means the bytes are not
//! what the schema says they are.
//!
//! # No panics, no recursion blow-up
//!
//! Every read is bounds-checked and returns [`ProtoError`]. The reader descends
//! at most [`MAX_DEPTH`] levels; a nested `GraphProto` inside an attribute is
//! *marked*, never parsed, so a hostile file cannot cause unbounded recursion.
//! Length prefixes are validated against the bytes that remain before any slice
//! is taken, and no `Vec` is sized from a declared count.
//!
//! # Borrowed strings
//!
//! Names, domains and `dim_param` values borrow from the input bytes and are
//! validated as UTF-8 once, at decode time. Tensor payloads borrow too, so
//! decoding a 28 MB model copies no weight bytes.

use super::budget::{
    MAX_ATTRIBUTES, MAX_NODES, MAX_OPERANDS, MAX_OPSET_IMPORTS, MAX_PARAM_AXES, MAX_RANK,
    MAX_STRING_BYTES, MAX_TYPED_ELEMENTS, MAX_VALUES, MAX_WEIGHTS,
};
use super::error::ProtoError;

/// Maximum nesting depth the decoder descends.
///
/// `ModelProto → GraphProto → NodeProto → AttributeProto → TensorProto` is five
/// levels; the `ValueInfoProto → TypeProto → TensorShapeProto → Dimension` chain
/// is four. Eight leaves room for one schema addition without permitting
/// unbounded recursion.
pub(crate) const MAX_DEPTH: u8 = 8;

// ---------------------------------------------------------------------------
// Wire reader
// ---------------------------------------------------------------------------

/// A cursor over one protobuf message.
#[derive(Clone)]
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    base: usize,
    depth: u8,
}

impl<'a> Reader<'a> {
    /// Start reading `bytes`, which begin at absolute offset `base` in the file.
    fn new(bytes: &'a [u8], base: usize) -> Reader<'a> {
        Reader {
            bytes,
            pos: 0,
            base,
            depth: 0,
        }
    }

    /// Bytes left in this message.
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// `true` when the message is consumed.
    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Absolute offset of the next byte.
    fn offset(&self) -> usize {
        self.base + self.pos
    }

    /// Read one base-128 varint.
    fn varint(&mut self) -> Result<u64, ProtoError> {
        let start = self.offset();
        let mut value: u64 = 0;
        let mut shift = 0u32;
        for index in 0..10usize {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return Err(ProtoError::Truncated {
                    offset: start,
                    needed: 1,
                    available: 0,
                });
            };
            self.pos += 1;
            if index == 9 && byte > 1 {
                return Err(ProtoError::VarintTooLong { offset: start });
            }
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err(ProtoError::VarintTooLong { offset: start })
    }

    /// Read a field tag: `(field_number, wire_type)`.
    fn tag(&mut self) -> Result<(u32, u8), ProtoError> {
        let offset = self.offset();
        let key = self.varint()?;
        let field = key >> 3;
        let wire = (key & 7) as u8;
        if field == 0 {
            return Err(ProtoError::InvalidFieldNumber { offset });
        }
        let field = u32::try_from(field).map_err(|_| ProtoError::InvalidFieldNumber { offset })?;
        Ok((field, wire))
    }

    /// Take `len` bytes that must all be present.
    fn take(&mut self, len: usize) -> Result<&'a [u8], ProtoError> {
        let offset = self.offset();
        let end = self.pos.checked_add(len).ok_or(ProtoError::Truncated {
            offset,
            needed: len,
            available: self.remaining(),
        })?;
        let slice = self.bytes.get(self.pos..end).ok_or(ProtoError::Truncated {
            offset,
            needed: len,
            available: self.remaining(),
        })?;
        self.pos = end;
        Ok(slice)
    }

    /// Read a length-delimited payload.
    fn bytes(&mut self) -> Result<&'a [u8], ProtoError> {
        let len = self.varint()?;
        let len = usize::try_from(len).map_err(|_| ProtoError::Truncated {
            offset: self.offset(),
            needed: usize::MAX,
            available: self.remaining(),
        })?;
        self.take(len)
    }

    /// Read one little-endian `float`.
    fn float(&mut self) -> Result<f32, ProtoError> {
        let bytes = self.take(4)?;
        let mut word = [0u8; 4];
        word.copy_from_slice(bytes);
        Ok(f32::from_bits(u32::from_le_bytes(word)))
    }

    /// Read one little-endian `double`.
    fn double(&mut self) -> Result<f64, ProtoError> {
        let bytes = self.take(8)?;
        let mut word = [0u8; 8];
        word.copy_from_slice(bytes);
        Ok(f64::from_bits(u64::from_le_bytes(word)))
    }

    /// Read a nested message.
    fn sub(&mut self) -> Result<Reader<'a>, ProtoError> {
        let offset = self.offset();
        let payload = self.bytes()?;
        if self.depth + 1 > MAX_DEPTH {
            return Err(ProtoError::DepthExceeded { offset });
        }
        Ok(Reader {
            bytes: payload,
            pos: 0,
            base: self.base + self.pos - payload.len(),
            depth: self.depth + 1,
        })
    }

    /// Skip a field this reader does not model.
    fn skip(&mut self, field: u32, wire: u8) -> Result<(), ProtoError> {
        let offset = self.offset();
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.take(4)?;
            }
            other => {
                return Err(ProtoError::UnsupportedWireType {
                    field,
                    wire: other,
                    offset,
                })
            }
        }
        Ok(())
    }

    /// Require `wire` and report what the schema expects otherwise.
    fn expect(
        &self,
        field: u32,
        wire: u8,
        expected: u8,
        name: &'static str,
    ) -> Result<(), ProtoError> {
        if wire == expected {
            Ok(())
        } else {
            Err(ProtoError::WrongWireType {
                field,
                expected: name,
                found: wire,
                offset: self.offset(),
            })
        }
    }

    /// Read a `string` field and validate UTF-8.
    ///
    /// The declared length is checked against [`MAX_STRING_BYTES`] *before* the
    /// slice is taken, so a five-byte field claiming four gigabytes fails on one
    /// comparison rather than after the name is copied.
    fn text(&mut self, what: &'static str) -> Result<&'a str, ProtoError> {
        let offset = self.offset();
        let len = self.varint()?;
        let len = usize::try_from(len).map_err(|_| ProtoError::Truncated {
            offset,
            needed: usize::MAX,
            available: self.remaining(),
        })?;
        if len > MAX_STRING_BYTES {
            return Err(ProtoError::LimitExceeded {
                what,
                limit: MAX_STRING_BYTES,
                offset,
            });
        }
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map_err(|_| ProtoError::InvalidUtf8 { what, offset })
    }

    /// Refuse a repeated field that has reached its budget.
    fn room_for(&self, what: &'static str, len: usize, limit: usize) -> Result<(), ProtoError> {
        if len >= limit {
            Err(ProtoError::LimitExceeded {
                what,
                limit,
                offset: self.offset(),
            })
        } else {
            Ok(())
        }
    }

    /// Read a repeated `int64` field, packed or unpacked.
    fn repeated_i64(
        &mut self,
        field: u32,
        wire: u8,
        what: &'static str,
        limit: usize,
        out: &mut Vec<i64>,
    ) -> Result<(), ProtoError> {
        match wire {
            0 => {
                self.room_for(what, out.len(), limit)?;
                out.push(self.varint()? as i64)
            }
            2 => {
                let payload = self.bytes()?;
                let mut inner = Reader {
                    bytes: payload,
                    pos: 0,
                    base: self.base + self.pos - payload.len(),
                    depth: self.depth,
                };
                while !inner.is_empty() {
                    inner.room_for(what, out.len(), limit)?;
                    out.push(inner.varint()? as i64);
                }
            }
            other => {
                return Err(ProtoError::WrongWireType {
                    field,
                    expected: "varint or packed varints",
                    found: other,
                    offset: self.offset(),
                })
            }
        }
        Ok(())
    }

    /// Read a repeated `uint64` field, packed or unpacked.
    fn repeated_u64(
        &mut self,
        field: u32,
        wire: u8,
        what: &'static str,
        limit: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), ProtoError> {
        match wire {
            0 => {
                self.room_for(what, out.len(), limit)?;
                out.push(self.varint()?)
            }
            2 => {
                let payload = self.bytes()?;
                let mut inner = Reader {
                    bytes: payload,
                    pos: 0,
                    base: self.base + self.pos - payload.len(),
                    depth: self.depth,
                };
                while !inner.is_empty() {
                    inner.room_for(what, out.len(), limit)?;
                    out.push(inner.varint()?);
                }
            }
            other => {
                return Err(ProtoError::WrongWireType {
                    field,
                    expected: "varint or packed varints",
                    found: other,
                    offset: self.offset(),
                })
            }
        }
        Ok(())
    }

    /// Read a repeated `int32` field, packed or unpacked.
    fn repeated_i32(
        &mut self,
        field: u32,
        wire: u8,
        what: &'static str,
        limit: usize,
        out: &mut Vec<i32>,
    ) -> Result<(), ProtoError> {
        match wire {
            0 => {
                self.room_for(what, out.len(), limit)?;
                out.push(self.varint()? as i32)
            }
            2 => {
                let payload = self.bytes()?;
                let mut inner = Reader {
                    bytes: payload,
                    pos: 0,
                    base: self.base + self.pos - payload.len(),
                    depth: self.depth,
                };
                while !inner.is_empty() {
                    inner.room_for(what, out.len(), limit)?;
                    out.push(inner.varint()? as i32);
                }
            }
            other => {
                return Err(ProtoError::WrongWireType {
                    field,
                    expected: "varint or packed varints",
                    found: other,
                    offset: self.offset(),
                })
            }
        }
        Ok(())
    }

    /// Read a repeated `float` field, packed or unpacked.
    fn repeated_f32(
        &mut self,
        field: u32,
        wire: u8,
        what: &'static str,
        limit: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), ProtoError> {
        match wire {
            5 => {
                self.room_for(what, out.len(), limit)?;
                out.push(self.float()?)
            }
            2 => {
                let payload = self.bytes()?;
                if payload.len() % 4 != 0 {
                    return Err(ProtoError::Truncated {
                        offset: self.offset(),
                        needed: 4,
                        available: payload.len() % 4,
                    });
                }
                for chunk in payload.chunks_exact(4) {
                    self.room_for(what, out.len(), limit)?;
                    let mut word = [0u8; 4];
                    word.copy_from_slice(chunk);
                    out.push(f32::from_bits(u32::from_le_bytes(word)));
                }
            }
            other => {
                return Err(ProtoError::WrongWireType {
                    field,
                    expected: "fixed32 or packed fixed32",
                    found: other,
                    offset: self.offset(),
                })
            }
        }
        Ok(())
    }

    /// Read a repeated `double` field, packed or unpacked.
    fn repeated_f64(
        &mut self,
        field: u32,
        wire: u8,
        what: &'static str,
        limit: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), ProtoError> {
        match wire {
            1 => {
                self.room_for(what, out.len(), limit)?;
                out.push(self.double()?)
            }
            2 => {
                let payload = self.bytes()?;
                if payload.len() % 8 != 0 {
                    return Err(ProtoError::Truncated {
                        offset: self.offset(),
                        needed: 8,
                        available: payload.len() % 8,
                    });
                }
                for chunk in payload.chunks_exact(8) {
                    self.room_for(what, out.len(), limit)?;
                    let mut word = [0u8; 8];
                    word.copy_from_slice(chunk);
                    out.push(f64::from_bits(u64::from_le_bytes(word)));
                }
            }
            other => {
                return Err(ProtoError::WrongWireType {
                    field,
                    expected: "fixed64 or packed fixed64",
                    found: other,
                    offset: self.offset(),
                })
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoded model
// ---------------------------------------------------------------------------

/// `OperatorSetIdProto`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpsetId<'a> {
    /// Empty for the default ONNX domain.
    pub(crate) domain: &'a str,
    /// Opset version.
    pub(crate) version: i64,
}

/// `ModelProto`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Model<'a> {
    pub(crate) ir_version: i64,
    pub(crate) producer_name: &'a str,
    pub(crate) producer_version: &'a str,
    pub(crate) opset_imports: Vec<OpsetId<'a>>,
    pub(crate) graph: Option<Graph<'a>>,
    pub(crate) function_count: usize,
    pub(crate) has_training_info: bool,
}

/// `GraphProto`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Graph<'a> {
    pub(crate) name: &'a str,
    pub(crate) nodes: Vec<Node<'a>>,
    pub(crate) initializers: Vec<Tensor<'a>>,
    pub(crate) inputs: Vec<ValueInfo<'a>>,
    pub(crate) outputs: Vec<ValueInfo<'a>>,
    pub(crate) value_info: Vec<ValueInfo<'a>>,
    pub(crate) sparse_initializer_count: usize,
    pub(crate) quantization_annotation_count: usize,
}

/// `NodeProto`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node<'a> {
    pub(crate) inputs: Vec<&'a str>,
    pub(crate) outputs: Vec<&'a str>,
    pub(crate) name: &'a str,
    pub(crate) op_type: &'a str,
    pub(crate) domain: &'a str,
    pub(crate) attributes: Vec<Attribute<'a>>,
}

/// A decoded `AttributeProto` payload.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AttributeValue<'a> {
    /// Type `UNDEFINED`, or no payload field at all.
    Undefined,
    /// Type `FLOAT`.
    Float(f32),
    /// Type `INT`.
    Int(i64),
    /// Type `STRING`; raw bytes, because ONNX does not require UTF-8 here.
    String(&'a [u8]),
    /// Type `TENSOR`.
    Tensor(Tensor<'a>),
    /// Type `FLOATS`.
    Floats(Vec<f32>),
    /// Type `INTS`.
    Ints(Vec<i64>),
    /// Type `STRINGS`.
    Strings(Vec<&'a [u8]>),
    /// Type `TENSORS`.
    Tensors(Vec<Tensor<'a>>),
    /// Type `GRAPH` or `GRAPHS`: a subgraph, deliberately not parsed.
    Subgraph,
    /// Type `SPARSE_TENSOR` or `SPARSE_TENSORS`.
    SparseTensor,
    /// Type `TYPE_PROTO` or `TYPE_PROTOS`.
    TypeProto,
}

impl AttributeValue<'_> {
    /// ONNX name of the payload kind, used in diagnostics.
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            AttributeValue::Undefined => "UNDEFINED",
            AttributeValue::Float(_) => "FLOAT",
            AttributeValue::Int(_) => "INT",
            AttributeValue::String(_) => "STRING",
            AttributeValue::Tensor(_) => "TENSOR",
            AttributeValue::Floats(_) => "FLOATS",
            AttributeValue::Ints(_) => "INTS",
            AttributeValue::Strings(_) => "STRINGS",
            AttributeValue::Tensors(_) => "TENSORS",
            AttributeValue::Subgraph => "GRAPH",
            AttributeValue::SparseTensor => "SPARSE_TENSOR",
            AttributeValue::TypeProto => "TYPE_PROTO",
        }
    }

    /// ONNX name of an `AttributeProto.AttributeType` code.
    pub(crate) fn type_name(code: i32) -> &'static str {
        match code {
            0 => "UNDEFINED",
            1 => "FLOAT",
            2 => "INT",
            3 => "STRING",
            4 => "TENSOR",
            5 => "GRAPH",
            6 => "FLOATS",
            7 => "INTS",
            8 => "STRINGS",
            9 => "TENSORS",
            10 => "GRAPHS",
            11 => "SPARSE_TENSOR",
            12 => "SPARSE_TENSORS",
            13 => "TYPE_PROTO",
            14 => "TYPE_PROTOS",
            _ => "UNKNOWN",
        }
    }

    /// `true` when this payload agrees with a declared `AttributeType` code.
    pub(crate) fn matches_type(&self, code: i32) -> bool {
        match self {
            AttributeValue::Undefined => code == 0,
            AttributeValue::Float(_) => code == 1,
            AttributeValue::Int(_) => code == 2,
            AttributeValue::String(_) => code == 3,
            AttributeValue::Tensor(_) => code == 4,
            AttributeValue::Floats(_) => code == 6,
            AttributeValue::Ints(_) => code == 7,
            AttributeValue::Strings(_) => code == 8,
            AttributeValue::Tensors(_) => code == 9,
            AttributeValue::Subgraph => code == 5 || code == 10,
            AttributeValue::SparseTensor => code == 11 || code == 12,
            AttributeValue::TypeProto => code == 13 || code == 14,
        }
    }
}

/// `AttributeProto`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Attribute<'a> {
    pub(crate) name: &'a str,
    /// The `type` field, when the exporter set one.
    pub(crate) declared_type: Option<i32>,
    pub(crate) value: AttributeValue<'a>,
}

/// A tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Dim {
    /// `dim_value`.
    Value(i64),
    /// `dim_param`; a symbolic name such as `"batch_size"`.
    Param(String),
    /// Neither field was set, so the extent is unknown.
    Unknown,
}

/// `TypeProto.Tensor`, the only type this build imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorType {
    pub(crate) elem_type: i32,
    /// `None` when the shape field is absent, so the rank is unknown.
    pub(crate) dims: Option<Vec<Dim>>,
}

/// `ValueInfoProto`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValueInfo<'a> {
    pub(crate) name: &'a str,
    pub(crate) ty: Option<TensorType>,
    /// Name of the non-tensor `TypeProto` variant that was present, if any.
    pub(crate) unsupported_type: Option<&'static str>,
}

/// `TensorProto`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dims: Vec<i64>,
    pub(crate) data_type: i32,
    pub(crate) raw_data: Option<&'a [u8]>,
    pub(crate) float_data: Vec<f32>,
    pub(crate) int32_data: Vec<i32>,
    pub(crate) int64_data: Vec<i64>,
    pub(crate) uint64_data: Vec<u64>,
    pub(crate) double_data: Vec<f64>,
    pub(crate) string_data_count: usize,
    pub(crate) has_external_data: bool,
    pub(crate) data_location: i32,
    pub(crate) has_segment: bool,
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode a complete `ModelProto`.
pub(crate) fn decode_model(bytes: &[u8]) -> Result<Model<'_>, ProtoError> {
    let mut reader = Reader::new(bytes, 0);
    let mut model = Model {
        ir_version: 0,
        producer_name: "",
        producer_version: "",
        opset_imports: Vec::new(),
        graph: None,
        function_count: 0,
        has_training_info: false,
    };
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 0, "varint")?;
                model.ir_version = reader.varint()? as i64;
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                model.producer_name = reader.text("ModelProto.producer_name")?;
            }
            3 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                model.producer_version = reader.text("ModelProto.producer_version")?;
            }
            7 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut graph = reader.sub()?;
                model.graph = Some(decode_graph(&mut graph)?);
            }
            8 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut opset = reader.sub()?;
                reader.room_for(
                    "opset imports",
                    model.opset_imports.len(),
                    MAX_OPSET_IMPORTS,
                )?;
                model.opset_imports.push(decode_opset(&mut opset)?);
            }
            20 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                model.has_training_info = true;
                reader.skip(field, wire)?;
            }
            25 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                model.function_count += 1;
                reader.skip(field, wire)?;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    if model.ir_version == 0 {
        return Err(ProtoError::MissingField {
            message: "ModelProto",
            field: "ir_version",
        });
    }
    Ok(model)
}

/// Decode `OperatorSetIdProto`.
fn decode_opset<'a>(reader: &mut Reader<'a>) -> Result<OpsetId<'a>, ProtoError> {
    let mut opset = OpsetId {
        domain: "",
        version: 0,
    };
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                opset.domain = reader.text("OperatorSetIdProto.domain")?;
            }
            2 => {
                reader.expect(field, wire, 0, "varint")?;
                opset.version = reader.varint()? as i64;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(opset)
}

/// Decode `GraphProto`.
fn decode_graph<'a>(reader: &mut Reader<'a>) -> Result<Graph<'a>, ProtoError> {
    let mut graph = Graph {
        name: "",
        nodes: Vec::new(),
        initializers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        value_info: Vec::new(),
        sparse_initializer_count: 0,
        quantization_annotation_count: 0,
    };
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut node = reader.sub()?;
                reader.room_for("graph nodes", graph.nodes.len(), MAX_NODES)?;
                graph.nodes.push(decode_node(&mut node)?);
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                graph.name = reader.text("GraphProto.name")?;
            }
            5 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut tensor = reader.sub()?;
                reader.room_for("graph initializers", graph.initializers.len(), MAX_WEIGHTS)?;
                graph.initializers.push(decode_tensor(&mut tensor)?);
            }
            11 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut value = reader.sub()?;
                reader.room_for("graph inputs", graph.inputs.len(), MAX_VALUES)?;
                graph.inputs.push(decode_value_info(&mut value)?);
            }
            12 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut value = reader.sub()?;
                reader.room_for("graph outputs", graph.outputs.len(), MAX_VALUES)?;
                graph.outputs.push(decode_value_info(&mut value)?);
            }
            13 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut value = reader.sub()?;
                reader.room_for("graph value_info", graph.value_info.len(), MAX_VALUES)?;
                graph.value_info.push(decode_value_info(&mut value)?);
            }
            14 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                graph.quantization_annotation_count += 1;
                reader.skip(field, wire)?;
            }
            15 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                graph.sparse_initializer_count += 1;
                reader.skip(field, wire)?;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(graph)
}

/// Decode `NodeProto`.
fn decode_node<'a>(reader: &mut Reader<'a>) -> Result<Node<'a>, ProtoError> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut name = "";
    let mut op_type = "";
    let mut domain = "";
    let mut attributes = Vec::new();
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.room_for("node inputs", inputs.len(), MAX_OPERANDS)?;
                inputs.push(reader.text("NodeProto.input")?);
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.room_for("node outputs", outputs.len(), MAX_OPERANDS)?;
                outputs.push(reader.text("NodeProto.output")?);
            }
            3 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                name = reader.text("NodeProto.name")?;
            }
            4 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                op_type = reader.text("NodeProto.op_type")?;
            }
            5 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut attribute = reader.sub()?;
                reader.room_for("node attributes", attributes.len(), MAX_ATTRIBUTES)?;
                attributes.push(decode_attribute(&mut attribute)?);
            }
            7 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                domain = reader.text("NodeProto.domain")?;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(Node {
        inputs,
        outputs,
        name,
        op_type,
        domain,
        attributes,
    })
}

/// Decode `AttributeProto`.
fn decode_attribute<'a>(reader: &mut Reader<'a>) -> Result<Attribute<'a>, ProtoError> {
    let mut name = "";
    let mut declared_type: Option<i32> = None;
    let mut value: Option<AttributeValue<'a>> = None;
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                name = reader.text("AttributeProto.name")?;
            }
            2 => {
                reader.expect(field, wire, 5, "fixed32")?;
                let float = reader.float()?;
                set_payload(&mut value, AttributeValue::Float(float), "AttributeProto")?;
            }
            3 => {
                reader.expect(field, wire, 0, "varint")?;
                let int = reader.varint()? as i64;
                set_payload(&mut value, AttributeValue::Int(int), "AttributeProto")?;
            }
            4 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let bytes = reader.bytes()?;
                set_payload(&mut value, AttributeValue::String(bytes), "AttributeProto")?;
            }
            5 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut tensor = reader.sub()?;
                let tensor = decode_tensor(&mut tensor)?;
                set_payload(&mut value, AttributeValue::Tensor(tensor), "AttributeProto")?;
            }
            6 | 11 => {
                // GRAPH / GRAPHS: advance past the payload without recursing.
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                set_payload(&mut value, AttributeValue::Subgraph, "AttributeProto")?;
            }
            7 => {
                let mut floats = take_list(value, "AttributeProto", "FLOATS")?;
                reader.repeated_f32(
                    field,
                    wire,
                    "attribute floats",
                    MAX_PARAM_AXES,
                    &mut floats,
                )?;
                value = Some(AttributeValue::Floats(floats));
            }
            8 => {
                let mut ints = take_list(value, "AttributeProto", "INTS")?;
                reader.repeated_i64(field, wire, "attribute ints", MAX_PARAM_AXES, &mut ints)?;
                value = Some(AttributeValue::Ints(ints));
            }
            9 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let bytes = reader.bytes()?;
                let mut strings = take_list(value, "AttributeProto", "STRINGS")?;
                reader.room_for("attribute strings", strings.len(), MAX_PARAM_AXES)?;
                strings.push(bytes);
                value = Some(AttributeValue::Strings(strings));
            }
            10 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut tensor = reader.sub()?;
                let tensor = decode_tensor(&mut tensor)?;
                let mut tensors = take_list(value, "AttributeProto", "TENSORS")?;
                reader.room_for("attribute tensors", tensors.len(), MAX_PARAM_AXES)?;
                tensors.push(tensor);
                value = Some(AttributeValue::Tensors(tensors));
            }
            14 | 15 | 22 | 23 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                let replacement = if field == 22 || field == 23 {
                    AttributeValue::SparseTensor
                } else {
                    AttributeValue::TypeProto
                };
                set_payload(&mut value, replacement, "AttributeProto")?;
            }
            20 => {
                reader.expect(field, wire, 0, "varint")?;
                declared_type = Some(reader.varint()? as i32);
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(Attribute {
        name,
        declared_type,
        value: value.unwrap_or(AttributeValue::Undefined),
    })
}

/// Take an existing repeated-list payload of the expected kind, if any.
fn take_list<'v, T>(
    slot: Option<AttributeValue<'v>>,
    context: &'static str,
    expected: &'static str,
) -> Result<Vec<T>, ProtoError>
where
    T: FromAttributeList<'v>,
{
    match slot {
        None => Ok(Vec::new()),
        Some(value) => T::take(value).ok_or_else(|| ProtoError::ConflictingPayload {
            message: context,
            detail: format!("a non-{expected} payload is already present"),
        }),
    }
}

/// Conversion from a decoded list payload back into its element vector.
trait FromAttributeList<'v>: Sized {
    fn take(value: AttributeValue<'v>) -> Option<Vec<Self>>;
}

impl<'v> FromAttributeList<'v> for f32 {
    fn take(value: AttributeValue<'v>) -> Option<Vec<f32>> {
        match value {
            AttributeValue::Floats(values) => Some(values),
            _ => None,
        }
    }
}

impl<'v> FromAttributeList<'v> for i64 {
    fn take(value: AttributeValue<'v>) -> Option<Vec<i64>> {
        match value {
            AttributeValue::Ints(values) => Some(values),
            _ => None,
        }
    }
}

impl<'v> FromAttributeList<'v> for &'v [u8] {
    fn take(value: AttributeValue<'v>) -> Option<Vec<&'v [u8]>> {
        match value {
            AttributeValue::Strings(values) => Some(values),
            _ => None,
        }
    }
}

impl<'v> FromAttributeList<'v> for Tensor<'v> {
    fn take(value: AttributeValue<'v>) -> Option<Vec<Tensor<'v>>> {
        match value {
            AttributeValue::Tensors(values) => Some(values),
            _ => None,
        }
    }
}

/// Store a singular payload, rejecting a second, different one.
fn set_payload<'a>(
    slot: &mut Option<AttributeValue<'a>>,
    replacement: AttributeValue<'a>,
    context: &'static str,
) -> Result<(), ProtoError> {
    match slot {
        None => {
            *slot = Some(replacement);
            Ok(())
        }
        Some(existing) => {
            if std::mem::discriminant(existing) == std::mem::discriminant(&replacement) {
                // Repeated occurrences of a singular field: protobuf keeps the
                // last value.
                *slot = Some(replacement);
                Ok(())
            } else {
                Err(ProtoError::ConflictingPayload {
                    message: context,
                    detail: format!(
                        "{} and {} are both present",
                        existing.kind_name(),
                        replacement.kind_name()
                    ),
                })
            }
        }
    }
}

/// Decode `TensorProto`.
fn decode_tensor<'a>(reader: &mut Reader<'a>) -> Result<Tensor<'a>, ProtoError> {
    let mut tensor = Tensor {
        name: "",
        dims: Vec::new(),
        data_type: 0,
        raw_data: None,
        float_data: Vec::new(),
        int32_data: Vec::new(),
        int64_data: Vec::new(),
        uint64_data: Vec::new(),
        double_data: Vec::new(),
        string_data_count: 0,
        has_external_data: false,
        data_location: 0,
        has_segment: false,
    };
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => reader.repeated_i64(field, wire, "tensor rank", MAX_RANK, &mut tensor.dims)?,
            2 => {
                reader.expect(field, wire, 0, "varint")?;
                tensor.data_type = reader.varint()? as i32;
            }
            3 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                tensor.has_segment = true;
            }
            4 => reader.repeated_f32(
                field,
                wire,
                "typed tensor payload",
                MAX_TYPED_ELEMENTS,
                &mut tensor.float_data,
            )?,
            5 => reader.repeated_i32(
                field,
                wire,
                "typed tensor payload",
                MAX_TYPED_ELEMENTS,
                &mut tensor.int32_data,
            )?,
            6 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                tensor.string_data_count += 1;
            }
            7 => reader.repeated_i64(
                field,
                wire,
                "typed tensor payload",
                MAX_TYPED_ELEMENTS,
                &mut tensor.int64_data,
            )?,
            8 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                tensor.name = reader.text("TensorProto.name")?;
            }
            9 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                tensor.raw_data = Some(reader.bytes()?);
            }
            10 => reader.repeated_f64(
                field,
                wire,
                "typed tensor payload",
                MAX_TYPED_ELEMENTS,
                &mut tensor.double_data,
            )?,
            11 => reader.repeated_u64(
                field,
                wire,
                "typed tensor payload",
                MAX_TYPED_ELEMENTS,
                &mut tensor.uint64_data,
            )?,
            13 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                tensor.has_external_data = true;
            }
            14 => {
                reader.expect(field, wire, 0, "varint")?;
                tensor.data_location = reader.varint()? as i32;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(tensor)
}

/// Decode `ValueInfoProto`.
fn decode_value_info<'a>(reader: &mut Reader<'a>) -> Result<ValueInfo<'a>, ProtoError> {
    let mut name = "";
    let mut ty = None;
    let mut unsupported_type = None;
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                name = reader.text("ValueInfoProto.name")?;
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut value_type = reader.sub()?;
                let (decoded, unsupported) = decode_type(&mut value_type)?;
                ty = decoded;
                unsupported_type = unsupported;
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(ValueInfo {
        name,
        ty,
        unsupported_type,
    })
}

/// Decode `TypeProto`; only `tensor_type` is supported.
fn decode_type<'a>(
    reader: &mut Reader<'a>,
) -> Result<(Option<TensorType>, Option<&'static str>), ProtoError> {
    let mut tensor_type = None;
    let mut unsupported = None;
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut inner = reader.sub()?;
                tensor_type = Some(decode_tensor_type(&mut inner)?);
            }
            2 | 3 | 4 | 5 | 8 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                reader.bytes()?;
                unsupported = Some(match field {
                    2 => "sequence_type",
                    3 => "map_type",
                    4 => "optional_type",
                    5 => "sparse_tensor_type",
                    _ => "opaque_type",
                });
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok((tensor_type, unsupported))
}

/// Decode `TypeProto.Tensor`.
fn decode_tensor_type(reader: &mut Reader<'_>) -> Result<TensorType, ProtoError> {
    let mut elem_type = 0;
    let mut dims = None;
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 0, "varint")?;
                elem_type = reader.varint()? as i32;
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut shape = reader.sub()?;
                dims = Some(decode_shape(&mut shape)?);
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(TensorType { elem_type, dims })
}

/// Decode `TensorShapeProto`.
fn decode_shape(reader: &mut Reader<'_>) -> Result<Vec<Dim>, ProtoError> {
    let mut dims = Vec::new();
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                let mut dimension = reader.sub()?;
                reader.room_for("shape dimensions", dims.len(), MAX_RANK)?;
                dims.push(decode_dimension(&mut dimension)?);
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(dims)
}

/// Decode `TensorShapeProto.Dimension`.
///
/// A dimension that sets both `dim_value` and `dim_param` is contradictory.
/// The symbolic name wins, because it is the stronger statement about a dynamic
/// axis and the importer treats an unexplained symbol as unknown rather than
/// guessing an extent.
fn decode_dimension(reader: &mut Reader<'_>) -> Result<Dim, ProtoError> {
    let mut value = None;
    let mut param = None;
    while !reader.is_empty() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                reader.expect(field, wire, 0, "varint")?;
                value = Some(reader.varint()? as i64);
            }
            2 => {
                reader.expect(field, wire, 2, "length-delimited")?;
                param = Some(
                    reader
                        .text("TensorShapeProto.Dimension.dim_param")?
                        .to_string(),
                );
            }
            _ => reader.skip(field, wire)?,
        }
    }
    Ok(match (param, value) {
        (Some(param), _) => Dim::Param(param),
        (None, Some(value)) => Dim::Value(value),
        (None, None) => Dim::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append a base-128 varint.
    fn push_varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7F) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    /// Encode one field tag.
    fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
        push_varint((u64::from(field) << 3) | u64::from(wire), out);
    }

    /// Encode one varint field.
    fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
        tag(field, 0, out);
        push_varint(value, out);
    }

    /// Encode one length-delimited field.
    fn field_bytes(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        tag(field, 2, out);
        push_varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    #[test]
    fn decodes_a_minimal_model_and_borrows_raw_data() {
        let mut tensor = Vec::new();
        field_varint(1, 2, &mut tensor); // dims = [2]
        field_varint(2, 1, &mut tensor); // FLOAT
        field_bytes(8, b"w", &mut tensor);
        field_bytes(9, &[0, 0, 128, 63, 0, 0, 0, 64], &mut tensor);

        let mut graph = Vec::new();
        field_bytes(5, &tensor, &mut graph);

        let mut model = Vec::new();
        field_varint(1, 7, &mut model);
        field_bytes(7, &graph, &mut model);

        let decoded = decode_model(&model).unwrap();
        assert_eq!(decoded.ir_version, 7);
        let graph = decoded.graph.unwrap();
        assert_eq!(graph.initializers.len(), 1);
        let tensor = &graph.initializers[0];
        assert_eq!(tensor.name, "w");
        assert_eq!(tensor.dims, vec![2]);
        assert_eq!(tensor.raw_data.unwrap().len(), 8);
        // The payload is a window into the caller's bytes, not a copy.
        let start = model
            .windows(4)
            .position(|window| window == [0, 0, 128, 63])
            .expect("payload present in the source buffer");
        let pointer = tensor.raw_data.unwrap().as_ptr() as usize;
        assert_eq!(pointer, model.as_ptr() as usize + start);
    }

    #[test]
    fn truncation_is_reported_with_an_offset_and_never_panics() {
        let mut graph = Vec::new();
        field_bytes(2, b"main_graph", &mut graph);
        let mut model = Vec::new();
        field_varint(1, 7, &mut model);
        field_bytes(7, &graph, &mut model);
        // Cutting into the graph payload must fail; the two-byte prefix is a
        // complete model with no graph and legitimately decodes.
        for length in 0..model.len() {
            let result = decode_model(&model[..length]);
            if length >= 3 {
                assert!(result.is_err(), "prefix of {length} bytes decoded");
            } else {
                // Two bytes are `ir_version = 7` with no graph, which is a
                // legal (if useless) ModelProto.
                let _ = result;
            }
        }
    }

    #[test]
    fn a_hostile_length_prefix_is_rejected() {
        let mut model = Vec::new();
        field_varint(1, 7, &mut model);
        // Field 7 (graph) with a length far beyond the buffer.
        tag(7, 2, &mut model);
        push_varint(0xFFFF_FFFF, &mut model);
        let error = decode_model(&model).unwrap_err();
        assert!(
            matches!(error, ProtoError::Truncated { .. }),
            "unexpected {error:?}"
        );
    }

    #[test]
    fn a_varint_longer_than_ten_bytes_is_rejected() {
        let mut model = Vec::new();
        tag(1, 0, &mut model);
        model.extend_from_slice(&[0xFF; 11]);
        let error = decode_model(&model).unwrap_err();
        assert!(
            matches!(error, ProtoError::VarintTooLong { .. }),
            "unexpected {error:?}"
        );
    }

    #[test]
    fn protobuf_groups_are_rejected_not_skipped() {
        let mut model = Vec::new();
        field_varint(1, 7, &mut model);
        // An unknown field that uses the (deprecated) group wire type.
        tag(30, 3, &mut model);
        let error = decode_model(&model).unwrap_err();
        assert!(
            matches!(
                error,
                ProtoError::UnsupportedWireType {
                    field: 30,
                    wire: 3,
                    ..
                }
            ),
            "unexpected {error:?}"
        );
    }

    #[test]
    fn unknown_fields_are_skipped_by_wire_type() {
        let mut model = Vec::new();
        field_varint(1, 7, &mut model);
        field_bytes(99, b"future field", &mut model);
        field_varint(98, 42, &mut model);
        let decoded = decode_model(&model).unwrap();
        assert_eq!(decoded.ir_version, 7);
        assert!(decoded.graph.is_none());
    }

    #[test]
    fn a_missing_ir_version_is_named() {
        let error = decode_model(&[]).unwrap_err();
        assert_eq!(
            error,
            ProtoError::MissingField {
                message: "ModelProto",
                field: "ir_version"
            }
        );
    }

    #[test]
    fn a_known_field_with_the_wrong_wire_type_is_rejected() {
        let mut model = Vec::new();
        // `ir_version` is a varint; a length-delimited payload for it is not a
        // model this reader can trust.
        field_bytes(1, b"seven", &mut model);
        let error = decode_model(&model).unwrap_err();
        assert!(
            matches!(
                error,
                ProtoError::WrongWireType {
                    field: 1,
                    expected: "varint",
                    ..
                }
            ),
            "unexpected {error:?}"
        );
    }

    #[test]
    fn two_payloads_in_one_attribute_are_rejected() {
        let mut attribute = Vec::new();
        field_bytes(1, b"axes", &mut attribute);
        tag(2, 5, &mut attribute); // f = 1.0
        attribute.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        field_varint(3, 7, &mut attribute); // i = 7
        let error = decode_attribute(&mut Reader::new(&attribute, 0)).unwrap_err();
        assert!(
            matches!(error, ProtoError::ConflictingPayload { .. }),
            "unexpected {error:?}"
        );
    }

    #[test]
    fn a_dimension_with_both_fields_prefers_the_symbol() {
        let mut dimension = Vec::new();
        field_varint(1, 5, &mut dimension);
        field_bytes(2, b"seq", &mut dimension);
        let decoded = decode_dimension(&mut Reader::new(&dimension, 0)).unwrap();
        assert_eq!(decoded, Dim::Param("seq".to_string()));
    }

    #[test]
    fn a_nested_message_beyond_the_depth_limit_is_rejected() {
        // A nested message at the depth limit must be refused rather than
        // recursed into; subgraphs are marked during attribute decoding, so this
        // guards the schema-shaped nesting the reader does follow.
        let mut payload = Vec::new();
        field_bytes(1, b"x", &mut payload);
        let mut reader = Reader::new(&payload, 0);
        reader.depth = MAX_DEPTH;
        let (field, wire) = reader.tag().unwrap();
        reader.expect(field, wire, 2, "length-delimited").unwrap();
        match reader.sub() {
            Err(ProtoError::DepthExceeded { .. }) => {}
            Err(other) => panic!("unexpected {other:?}"),
            Ok(_) => panic!("a message beyond the depth limit was accepted"),
        }
    }
}
