//! Native GGUF file reader.
//!
//! Parses the GGUF header, metadata key-value pairs, and tensor info array,
//! then provides direct access to each tensor's raw bytes.  No Python
//! dependency, no dequantization — just parse and hand off raw blocks to
//! `GgmlQuantizedTensor`.
//!
//! TODO: Add mmap support via `memmap2` when the dependency is available.
//!
//! GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use crate::quants::{GgmlQuantizedTensor, QuantizedDType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// In-memory representation of a GGUF file.
pub struct GgufFile {
    pub version: u32,
    /// Raw metadata key/value pairs (key → raw value bytes).
    pub metadata: HashMap<String, Vec<u8>>,
    /// Tensor descriptors.
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data starts (aligned).
    pub data_start: usize,
    /// Full file contents.
    data: Vec<u8>,
}

/// Descriptor for one tensor inside a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub dtype: QuantizedDType,
    /// Offset from `data_start` to the first byte of this tensor's data.
    pub offset: u64,
    /// Size in bytes (computed from dtype + dimensions).
    pub size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic: expected 'GGUF', got {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedDtype(u32),
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Cursor over a byte slice, tracking consumed position.
struct ByteCursor {
    buf: Vec<u8>,
    pos: usize,
}

impl ByteCursor {
    fn new(buf: Vec<u8>) -> Self {
        ByteCursor { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        if self.remaining() < 1 {
            return Err(GgufError::Parse("need 1 byte".into()));
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32, GgufError> {
        if self.remaining() < 4 {
            return Err(GgufError::Parse("need 4 bytes for u32".into()));
        }
        let v = u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64_le(&mut self) -> Result<u64, GgufError> {
        if self.remaining() < 8 {
            return Err(GgufError::Parse("need 8 bytes for u64".into()));
        }
        let v = u64::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
            self.buf[self.pos + 4],
            self.buf[self.pos + 5],
            self.buf[self.pos + 6],
            self.buf[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64_le()? as usize;
        if self.remaining() < len {
            return Err(GgufError::Parse("need bytes for string".into()));
        }
        let s = String::from_utf8(self.buf[self.pos..self.pos + len].to_vec())
            .map_err(|e| GgufError::Parse(format!("invalid utf8: {}", e)))?;
        self.pos += len;
        Ok(s)
    }

    fn skip_value(&mut self, val_type: u32) -> Result<(), GgufError> {
        match val_type {
            0 | 1 | 7 => { self.read_u8()?; }
            2 | 3 => { self.pos += 2; }
            4 | 5 | 6 => { self.pos += 4; }
            10 | 11 | 12 => { self.pos += 8; }
            8 => { let _ = self.read_string()?; }
            9 => {
                let elem_type = self.read_u32_le()?;
                let len = self.read_u64_le()? as usize;
                for _ in 0..len {
                    self.skip_value(elem_type)?;
                }
            }
            _ => return Err(GgufError::Parse(format!("unknown value type {}", val_type))),
        }
        Ok(())
    }
}

impl GgufFile {
    /// Load a GGUF file into RAM (full copy, fast random access).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Self::from_bytes(buf)
    }

    /// Parse from an owned `Vec<u8>`.
    pub fn from_bytes(buf: Vec<u8>) -> Result<Self, GgufError> {
        let (version, metadata, tensors, data_start) = parse(&buf)?;
        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_start,
            data: buf,
        })
    }

    /// Access a tensor by name.
    ///
    /// Copies the raw tensor bytes from the file into a new `GgmlQuantizedTensor`.
    pub fn get_tensor(&self, name: &str) -> Option<GgmlQuantizedTensor> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let start = self.data_start + info.offset as usize;
        let end = start + info.size as usize;
        let bytes = self.data[start..end].to_vec();

        // Derive shape from dims (trim trailing 1s, keep at least rank 2)
        let mut shape = [1usize; 2];
        let dims: Vec<usize> = info
            .dims[..info.n_dims as usize]
            .iter()
            .map(|d| *d as usize)
            .collect();

        // GGML convention: dims[0] = ne[0] = in_features (columns),
        // dims[1] = ne[1] = out_features (rows).
        // Our convention: shape[0] = out_features, shape[1] = in_features.
        if dims.len() >= 2 {
            shape[0] = dims[1];
            shape[1] = dims[0];
        } else if dims.len() == 1 {
            shape[0] = 1;
            shape[1] = dims[0];
        }

        Some(GgmlQuantizedTensor::from_gguf_bytes(
            info.dtype,
            shape,
            bytes,
        ))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn summary(&self) -> String {
        let total_mb: f64 = self
            .tensors
            .iter()
            .map(|t| t.size as f64)
            .sum::<f64>()
            / (1024.0 * 1024.0);
        format!(
            "GGUF v{} | {} metadata keys | {} tensors | {:.1} MB total",
            self.version,
            self.metadata.len(),
            self.tensors.len(),
            total_mb
        )
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn parse(buf: &[u8]) -> Result<(
    u32,
    HashMap<String, Vec<u8>>,
    Vec<GgufTensorInfo>,
    usize,
), GgufError> {
    let mut cursor = ByteCursor::new(buf.to_vec());

    // Magic
    let magic = cursor.read_u32_le()?;
    let magic_bytes = magic.to_le_bytes();
    if &magic_bytes != b"GGUF" {
        return Err(GgufError::InvalidMagic([
            magic_bytes[0], magic_bytes[1], magic_bytes[2], magic_bytes[3],
        ]));
    }

    let version = cursor.read_u32_le()?;
    if version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u64_le()? as usize;
    let metadata_kv_count = cursor.read_u64_le()? as usize;

    // Metadata
    let mut metadata = HashMap::with_capacity(metadata_kv_count);
    for _ in 0..metadata_kv_count {
        let key = cursor.read_string()?;
        let val_type = cursor.read_u32_le()?;
        let start_pos = cursor.pos;
        cursor.skip_value(val_type)?;
        let val_bytes = cursor.buf[start_pos..cursor.pos].to_vec();
        metadata.insert(key, val_bytes);
    }

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        let n_dims = cursor.read_u32_le()?;
        let mut dims = [0u64; 4];
        for i in 0..n_dims as usize {
            dims[i] = cursor.read_u64_le()?;
        }
        let dtype_raw = cursor.read_u32_le()?;
        let dtype = decode_dtype(dtype_raw)?;
        let offset = cursor.read_u64_le()?;

        let numel: usize = dims[..n_dims as usize]
            .iter()
            .map(|d| *d as usize)
            .product();
        let size = if let Some(bs) = dtype.block_size() {
            let bb = dtype.block_bytes().expect("block dtype must have block_bytes");
            (numel.div_ceil(bs) * bb) as u64
        } else {
            let bpw = dtype.bits_per_weight() as f64;
            (numel as f64 * bpw / 8.0).ceil() as u64
        };

        tensors.push(GgufTensorInfo {
            name,
            n_dims,
            dims,
            dtype,
            offset,
            size,
        });
    }

    // Align tensor data to 32-byte boundary
    let alignment: usize = 32;
    let padding = (alignment - (cursor.pos % alignment)) % alignment;
    let data_start = cursor.pos + padding;

    Ok((version, metadata, tensors, data_start))
}

fn decode_dtype(raw: u32) -> Result<QuantizedDType, GgufError> {
    use QuantizedDType::*;
    let dt = match raw {
        0 => F32,
        1 => F16,
        2 => Q4_0,
        3 => Q4_1,
        6 => Q5_0,
        7 => Q5_1,
        8 => Q8_0,
        9 => Q8_1,
        10 => Q2_K,
        11 => Q3_K,
        12 => Q4_K,
        13 => Q5_K,
        14 => Q6_K,
        15 => Q8_K,
        16 => IQ2_XXS,
        17 => IQ2_XS,
        18 => IQ3_XXS,
        19 => IQ1_S,
        20 => IQ4_NL,
        21 => IQ3_S,
        22 => IQ2_S,
        23 => IQ4_XS,
        24 => I8,
        25 => I16,
        26 => I32,
        27 => I64,
        28 => F64,
        29 => Iq1M,
        30 => Bf16,
        34 => Tq1_0,
        35 => Tq2_0,
        39 => Mxfp4,
        40 => Nvfp4,
        41 => Q1_0,
        _ => return Err(GgufError::UnsupportedDtype(raw)),
    };
    Ok(dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_gguf_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version = 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Tensor 0: name="test", n_dims=2, dims=[4,8], dtype=Q4_0 (0), offset=0
        let name = b"test";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // Q4_0
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset=0
        // Pad to 32 bytes
        let pad = 32 - (buf.len() % 32);
        buf.resize(buf.len() + pad, 0);
        // Tensor data: 4×8=32 elements, Q4_0: 1 block = 18 bytes
        buf.resize(buf.len() + 18, 0);
        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = minimal_gguf_bytes();
        let gguf = GgufFile::from_bytes(data).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "test");
        assert_eq!(gguf.tensors[0].n_dims, 2);
        assert_eq!(gguf.tensors[0].dims[0], 4);
        assert_eq!(gguf.tensors[0].dims[1], 8);
        assert_eq!(gguf.tensors[0].dtype, QuantizedDType::Q4_0);
        assert_eq!(gguf.tensors[0].size, 18); // 1 block × 18 bytes
    }

    #[test]
    fn test_get_tensor() {
        let data = minimal_gguf_bytes();
        let gguf = GgufFile::from_bytes(data).unwrap();
        let tensor = gguf.get_tensor("test").unwrap();
        assert_eq!(tensor.out_features(), 4);
        assert_eq!(tensor.in_features(), 8);
        assert_eq!(tensor.memory_bytes(), 18);
    }
}
