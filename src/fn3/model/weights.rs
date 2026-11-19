//! Typed, device-independent weight tensors produced by model import.
//!
//! # Why a byte-level weight table
//!
//! The runtime type [`crate::inference::model::ModelWeights`] holds `Tensor`s that
//! live on a device. Import must not tie weights to a device before a planner
//! chooses one, and the versioned prepared-model format (FN3-09 / #55) has to
//! serialize them. Both needs are satisfied by one intermediate form: a name, a
//! [`DType`], a [`Shape`] and the model's own bytes, in the model's own order.
//!
//! ```text
//! Onnx initializers ──► WeightSet (name, dtype, shape, bytes)
//!                            ├──► to_model_weights() ──► ModelWeights (host tensors)
//!                            └──► #55 writes the four fields per entry
//! ```
//!
//! # Determinism
//!
//! Entries keep the order they were constructed in. The importer builds the set
//! in ONNX initializer order followed by lifted attribute constants in node
//! order, so the same bytes always produce the same layout. Names are unique;
//! [`WeightSet::new`] rejects a duplicate rather than letting a later entry
//! shadow an earlier one.
//!
//! # Fidelity
//!
//! Bytes are stored exactly as the file carried them. Nothing here converts a
//! dtype, reorders elements or normalizes endianness: the ONNX wire format is
//! little-endian and so is every target this alpha supports, and a byte-order
//! conversion would be a numerical change disguised as I/O.

use std::collections::BTreeMap;

use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::model::ModelWeights;
use crate::inference::shape::Shape;
use crate::inference::storage::Storage;
use crate::inference::tensor::{Tensor, TensorMeta};

/// Position of one weight inside its [`WeightSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WeightId(u32);

impl WeightId {
    /// Construct from a position. Used by the importer, which owns the order.
    pub fn new(index: u32) -> WeightId {
        WeightId(index)
    }

    /// Zero-based position in the set.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for WeightId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "weight {}", self.0)
    }
}

/// Where a weight came from, so a caller can tell the model's tensors from
/// constants the importer synthesized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightSource {
    /// A `GraphProto.initializer`.
    Initializer,
    /// A constant lifted out of a node attribute, for example
    /// `ConstantOfShape.value`, which [`crate::inference::backend::OpParams`] cannot
    /// express. It becomes a real weight so the planner can reference it like
    /// any other constant input.
    AttributeConstant,
}

impl WeightSource {
    /// Stable name used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            WeightSource::Initializer => "initializer",
            WeightSource::AttributeConstant => "attribute constant",
        }
    }
}

/// One typed weight tensor: name, dtype, shape and the exact payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTensor {
    name: String,
    dtype: DType,
    shape: Shape,
    bytes: Vec<u8>,
    source: WeightSource,
}

impl WeightTensor {
    /// Construct, checking that the payload length matches `shape` and `dtype`.
    ///
    /// The length check is the last line of defence for a corrupt or hostile
    /// file: a caller that computed `expected` from a different shape cannot
    /// produce a tensor whose bytes disagree with its metadata.
    pub fn new(
        name: impl Into<String>,
        dtype: DType,
        shape: Shape,
        bytes: Vec<u8>,
        source: WeightSource,
    ) -> Result<WeightTensor, Fn3Error> {
        let expected = expected_bytes(&shape, dtype)?;
        if expected != bytes.len() as u64 {
            return Err(Fn3Error::InvalidFormat {
                format: "fn3 weights",
                detail: format!(
                    "weight `{}` declares {dtype} {shape}, which needs {expected} payload bytes, \
                     but {} bytes were supplied",
                    name.into(),
                    bytes.len()
                ),
            });
        }
        Ok(WeightTensor {
            name: name.into(),
            dtype,
            shape,
            bytes,
            source,
        })
    }

    /// Name exactly as the model spelled it (or the importer's synthesized
    /// name, for a lifted attribute constant).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Declared shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Payload bytes, exactly as the model carried them.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Where the weight came from.
    pub fn source(&self) -> WeightSource {
        self.source
    }

    /// Element count implied by the shape, checked.
    pub fn numel(&self) -> Result<u64, Fn3Error> {
        self.shape.numel()
    }

    /// Payload length in bytes.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Copy the payload into a host tensor on the CPU.
    ///
    /// The tensor borrows nothing: it owns an aligned copy, so it outlives the
    /// model bytes it was built from.
    ///
    /// # Rank-one extent-1 shapes
    ///
    /// `Shape::default_strides` gives an interior or trailing extent-1 axis the
    /// stride of the next axis, which `Strides::check_no_overlap` rejects, so
    /// `TensorMeta::contiguous` currently fails for shapes such as `[2, 1]` or
    /// `[1, 3, 1]`. No weight in the pinned graph has one (all are rank 0, rank 1
    /// with a non-unit extent, or rank 2 with both extents above one), and the
    /// fix belongs to the shape contract owner (reported by FN3-07 / #53), not
    /// here. A weight with such a shape returns the core error unchanged.
    pub fn to_tensor(&self) -> Result<Tensor, Fn3Error> {
        let meta = TensorMeta::contiguous(self.shape.clone(), self.dtype, DeviceId::Cpu)?;
        let storage = Storage::from_bytes(&self.bytes)?;
        Tensor::from_storage(storage, meta)
    }
}

/// Bytes that `shape` and `dtype` require, checked for overflow.
pub fn expected_bytes(shape: &Shape, dtype: DType) -> Result<u64, Fn3Error> {
    shape
        .numel()?
        .checked_mul(dtype.size_bytes() as u64)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "WeightTensor::expected_bytes",
            axis: None,
            detail: format!("{shape} elements of {dtype}"),
        })
}

/// Every weight of one model, in declaration order, with a name index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightSet {
    entries: Vec<WeightTensor>,
    index: BTreeMap<String, u32>,
}

impl WeightSet {
    /// Build from entries, rejecting a duplicate name.
    pub fn new(entries: Vec<WeightTensor>) -> Result<WeightSet, Fn3Error> {
        let mut index = BTreeMap::new();
        for (position, entry) in entries.iter().enumerate() {
            let position = u32::try_from(position).map_err(|_| Fn3Error::CapacityExceeded {
                what: "weight count",
                requested: entries.len() as u64,
                capacity: u32::MAX as u64,
            })?;
            if let Some(existing) = index.insert(entry.name.clone(), position) {
                return Err(Fn3Error::InvalidFormat {
                    format: "fn3 weights",
                    detail: format!(
                        "duplicate weight name `{}` at positions {existing} and {position}",
                        entry.name
                    ),
                });
            }
        }
        Ok(WeightSet { entries, index })
    }

    /// Number of weights.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the set is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A weight by position.
    pub fn get(&self, id: WeightId) -> Option<&WeightTensor> {
        self.entries.get(id.index())
    }

    /// A weight by exact name.
    pub fn find(&self, name: &str) -> Option<&WeightTensor> {
        self.index
            .get(name)
            .and_then(|position| self.entries.get(*position as usize))
    }

    /// Position of a weight by exact name.
    pub fn id_of(&self, name: &str) -> Option<WeightId> {
        self.index.get(name).copied().map(WeightId)
    }

    /// Every weight, in declaration order.
    pub fn iter(&self) -> std::slice::Iter<'_, WeightTensor> {
        self.entries.iter()
    }

    /// Every weight with its position.
    pub fn iter_ids(&self) -> impl Iterator<Item = (WeightId, &WeightTensor)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(position, weight)| (WeightId(position as u32), weight))
    }

    /// Only the weights that came from `GraphProto.initializer`.
    pub fn initializers(&self) -> impl Iterator<Item = &WeightTensor> {
        self.entries
            .iter()
            .filter(|weight| weight.source == WeightSource::Initializer)
    }

    /// Total payload size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|weight| weight.byte_len() as u64)
            .sum()
    }

    /// Copy every weight into host tensors and assemble [`ModelWeights`].
    ///
    /// The resulting tensors are on [`DeviceId::Cpu`]; a planner that runs on
    /// another device uploads them and builds its own `ModelWeights`, which is
    /// why this conversion is a separate, explicit call.
    pub fn to_model_weights(&self) -> Result<ModelWeights, Fn3Error> {
        let mut named = Vec::with_capacity(self.entries.len());
        for weight in &self.entries {
            named.push((weight.name.clone(), weight.to_tensor()?));
        }
        ModelWeights::with_names(named)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[u64]) -> Shape {
        Shape::from_dims(dims.iter().copied())
    }

    #[test]
    fn byte_length_must_match_shape_and_dtype() {
        let error = WeightTensor::new(
            "w",
            DType::F32,
            shape(&[2, 3]),
            vec![0; 20],
            WeightSource::Initializer,
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("24 payload bytes"), "{text}");
        assert!(text.contains("20 bytes were supplied"), "{text}");
    }

    #[test]
    fn a_scalar_weight_is_one_element() {
        let weight = WeightTensor::new(
            "eps",
            DType::F32,
            Shape::scalar(),
            1e-5f32.to_le_bytes().to_vec(),
            WeightSource::Initializer,
        )
        .unwrap();
        assert_eq!(weight.numel().unwrap(), 1);
        let tensor = weight.to_tensor().unwrap();
        assert_eq!(tensor.shape().rank(), 0);
        assert_eq!(tensor.as_f32_slice().unwrap(), &[1e-5]);
    }

    #[test]
    fn a_scalar_i64_weight_round_trips_its_exact_bytes() {
        let weight = WeightTensor::new(
            "k",
            DType::I64,
            Shape::scalar(),
            42i64.to_le_bytes().to_vec(),
            WeightSource::Initializer,
        )
        .unwrap();
        let tensor = weight.to_tensor().unwrap();
        assert_eq!(tensor.as_i64_slice().unwrap(), &[42]);
    }

    #[test]
    fn lookup_by_name_and_id_are_consistent() {
        let entries = vec![
            WeightTensor::new(
                "a",
                DType::F32,
                shape(&[1]),
                vec![0; 4],
                WeightSource::Initializer,
            )
            .unwrap(),
            WeightTensor::new(
                "b",
                DType::I64,
                shape(&[1]),
                vec![0; 8],
                WeightSource::AttributeConstant,
            )
            .unwrap(),
        ];
        let set = WeightSet::new(entries).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.find("b").unwrap().dtype(), DType::I64);
        let id = set.id_of("b").unwrap();
        assert_eq!(id.index(), 1);
        assert_eq!(set.get(id).unwrap().name(), "b");
        assert_eq!(set.initializers().count(), 1);
        assert_eq!(set.total_bytes(), 12);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let entry = WeightTensor::new(
            "w",
            DType::F32,
            shape(&[1]),
            vec![0; 4],
            WeightSource::Initializer,
        )
        .unwrap();
        let error = WeightSet::new(vec![entry.clone(), entry]).unwrap_err();
        assert!(
            error.to_string().contains("duplicate weight name"),
            "{error}"
        );
    }

    #[test]
    fn model_weights_share_one_device_and_keep_names() {
        let set = WeightSet::new(vec![
            WeightTensor::new(
                "a",
                DType::F32,
                shape(&[2]),
                vec![0; 8],
                WeightSource::Initializer,
            )
            .unwrap(),
            WeightTensor::new(
                "b",
                DType::F32,
                shape(&[1]),
                vec![0; 4],
                WeightSource::Initializer,
            )
            .unwrap(),
        ])
        .unwrap();
        let weights = set.to_model_weights().unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights.device(), Some(DeviceId::Cpu));
        assert!(weights.named("b").is_some());
        assert!(weights.named("missing").is_none());
    }
}
