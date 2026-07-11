use crate::error::{FastnnError, FastnnResult};
use crate::nn::Module;
use crate::storage::{CpuStorage, DType, Storage};
use crate::tensor::{Tensor, TensorImpl};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

// Magic bytes to identify fastnn checkpoint format
const MAGIC_BYTES: [u8; 4] = [0x46, 0x4E, 0x4E, 0x00]; // "FNN\0"
                                                       // Magic bytes to identify fastnn optimizer state format
#[allow(dead_code)] // Reserved for future optimizer state serialization
const OPTIMIZER_MAGIC: [u8; 4] = [0x46, 0x4E, 0x4F, 0x00]; // "FNO\0"
                                                           // Format version for forward compatibility
const FORMAT_VERSION: u32 = 3;

/// Write a length-prefixed byte slice (length as little-endian u64, then data).
fn write_length_prefixed(writer: &mut impl Write, bytes: &[u8]) -> FastnnResult<()> {
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(FastnnError::Io)?;
    writer.write_all(bytes).map_err(FastnnError::Io)?;
    Ok(())
}

/// Read a length-prefixed byte slice (expects length as little-endian u64, then data).
fn read_length_prefixed(reader: &mut impl Read) -> FastnnResult<Vec<u8>> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes).map_err(FastnnError::Io)?;
    let len = u64::from_le_bytes(len_bytes) as usize;
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data).map_err(FastnnError::Io)?;
    Ok(data)
}

/// Write a slice of i64 values with length prefix.
fn write_slice_i64(writer: &mut impl Write, slice: &[i64]) -> FastnnResult<()> {
    write_length_prefixed(writer, bytemuck::cast_slice(slice))?;
    Ok(())
}

/// Read a slice of i64 values with length prefix.
/// Uses safe byte-by-byte conversion to avoid alignment UB.
fn read_slice_i64(reader: &mut impl Read) -> FastnnResult<Vec<i64>> {
    let bytes = read_length_prefixed(reader)?;
    let result: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    Ok(result)
}

/// Read a tensor using the v1 format (used by Python's write_tensor/read_tensor).
/// Returns (name, numpy_data) where numpy_data is a Vec<f32> with shape information.
fn read_tensor_v1(reader: &mut impl Read) -> FastnnResult<(String, Vec<u32>, Vec<f32>)> {
    // Read name length and name
    let mut name_len_bytes = [0u8; 8];
    reader
        .read_exact(&mut name_len_bytes)
        .map_err(FastnnError::Io)?;
    let name_len = u64::from_le_bytes(name_len_bytes) as usize;
    let mut name_bytes = vec![0u8; name_len];
    reader
        .read_exact(&mut name_bytes)
        .map_err(FastnnError::Io)?;
    let name = String::from_utf8(name_bytes).map_err(FastnnError::Utf8)?;

    // Read shape length and shape
    let mut shape_len_bytes = [0u8; 8];
    reader
        .read_exact(&mut shape_len_bytes)
        .map_err(FastnnError::Io)?;
    let shape_len = u64::from_le_bytes(shape_len_bytes) as usize;
    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes).map_err(FastnnError::Io)?;
        shape.push(i64::from_le_bytes(dim_bytes) as u32);
    }

    // Read data length and data
    let mut data_len_bytes = [0u8; 8];
    reader
        .read_exact(&mut data_len_bytes)
        .map_err(FastnnError::Io)?;
    let data_len = u64::from_le_bytes(data_len_bytes) as usize;
    let byte_count = data_len.checked_mul(4).ok_or_else(|| {
        FastnnError::Serialization(format!(
            "Integer overflow: data_len {} * 4 exceeds usize",
            data_len
        ))
    })?;
    let mut data_bytes = vec![0u8; byte_count];
    reader
        .read_exact(&mut data_bytes)
        .map_err(FastnnError::Io)?;

    // Convert bytes to f32 values
    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((name, shape, data))
}

pub fn save_state_dict(state_dict: Vec<(String, Tensor)>, path: &str) -> FastnnResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(&MAGIC_BYTES)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

    let num_params = state_dict.len() as u64;
    writer.write_all(&num_params.to_le_bytes())?;

    for (name, tensor) in &state_dict {
        write_length_prefixed(&mut writer, name.as_bytes())?;
        write_slice_i64(&mut writer, &tensor.shape())?;
        writer.write_all(&[tensor.dtype() as u8])?;

        if let Some(bytes) = tensor.as_byte_slice() {
            write_length_prefixed(&mut writer, bytes)?;
        } else {
            let data = tensor.to_numpy()?;
            // Convert f32 vec to bytes using safe reinterpretation.
            let bytes = bytemuck::cast_slice(&data);
            write_length_prefixed(&mut writer, bytes)?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub fn save_model(model: &dyn Module, path: &str) -> FastnnResult<()> {
    save_state_dict(model.named_parameters(), path)
}

#[allow(dead_code)]
pub fn load_model(path: &str) -> FastnnResult<HashMap<String, Tensor>> {
    let mut result = HashMap::new();

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read and validate magic bytes
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    if magic_bytes != MAGIC_BYTES {
        return Err(FastnnError::Serialization(format!(
            "Invalid file format: expected magic bytes {:?}, got {:?}",
            MAGIC_BYTES, magic_bytes
        )));
    }

    // Read format version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version > FORMAT_VERSION {
        return Err(FastnnError::Serialization(format!(
            "Unsupported format version: {}. Maximum supported version is {}",
            version, FORMAT_VERSION
        )));
    }

    // Read number of parameters
    let mut num_params_bytes = [0u8; 8];
    reader.read_exact(&mut num_params_bytes)?;
    let num_params = u64::from_le_bytes(num_params_bytes);

    if version == 1 {
        // Version 1 format: uses read_tensor (old format with name + data)
        for _ in 0..num_params {
            let (name, shape, data) = read_tensor_v1(&mut reader)?;
            let tensor = Tensor::from_vec(data, shape.into_iter().map(|d| d as i64).collect());
            result.insert(name, tensor);
        }
    } else if version == 2 {
        // Version 2 format: length-prefixed, no dtype byte (assumes F32)
        for _ in 0..num_params {
            let name_bytes = read_length_prefixed(&mut reader)?;
            let name = String::from_utf8(name_bytes).map_err(FastnnError::Utf8)?;

            let shape = read_slice_i64(&mut reader)?;

            let data_bytes = read_length_prefixed(&mut reader)?;

            let expected_numel: usize = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
                .ok_or_else(|| {
                    FastnnError::Serialization(format!(
                        "Integer overflow computing numel from shape {:?} for parameter '{}'",
                        shape, name
                    ))
                })?;
            let data_len = data_bytes.len() / 4; // f32 is 4 bytes
            if expected_numel != data_len {
                return Err(FastnnError::Serialization(format!(
                    "Shape mismatch for parameter '{}': shape {:?} expects {} elements, but data has {} elements",
                    name, shape, expected_numel, data_len
                )));
            }

            // SAFE: Use safe byte-to-float conversion instead of unsafe `from_raw_parts`
            // to avoid undefined behaviour from misaligned pointers.
            let data: Vec<f32> = data_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_vec(data, shape);
            result.insert(name, tensor);
        }
    } else {
        // Version 3+ format: length-prefixed with dtype byte
        for _ in 0..num_params {
            let name_bytes = read_length_prefixed(&mut reader)?;
            let name = String::from_utf8(name_bytes).map_err(FastnnError::Utf8)?;

            let shape = read_slice_i64(&mut reader)?;

            let mut dtype_byte = [0u8; 1];
            reader.read_exact(&mut dtype_byte)?;
            let dtype = match dtype_byte[0] {
                0 => DType::F32,
                1 => DType::F64,
                2 => DType::I32,
                3 => DType::I64,
                4 => DType::Bool,
                5 => DType::F16,
                6 => DType::BF16,
                7 => DType::I4,
                8 => DType::I8Scaled,
                9 => DType::F8,
                10 => DType::F8R,
                11 => DType::F4,
                other => {
                    return Err(FastnnError::Serialization(format!(
                        "Unknown dtype tag {} for parameter '{}'",
                        other, name
                    )));
                }
            };

            let data_bytes = read_length_prefixed(&mut reader)?;
            let nbytes = data_bytes.len();

            let expected_numel: usize = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
                .ok_or_else(|| {
                    FastnnError::Serialization(format!(
                        "Integer overflow computing numel from shape {:?} for parameter '{}'",
                        shape, name
                    ))
                })?;
            let elem_size = dtype.size();
            let data_len = nbytes / elem_size;
            if expected_numel != data_len {
                return Err(FastnnError::Serialization(format!(
                    "Shape mismatch for parameter '{}': shape {:?} expects {} elements of {:?}, but data has {} elements",
                    name, shape, expected_numel, dtype, data_len
                )));
            }

            let sizes: SmallVec<[i64; 8]> = shape.into();
            let storage = Arc::new(Storage::Cpu(CpuStorage::from_vec(data_bytes, nbytes)));
            let tensor = Tensor::new(TensorImpl::new(storage, sizes, dtype));
            result.insert(name, tensor);
        }
    }

    Ok(result)
}
