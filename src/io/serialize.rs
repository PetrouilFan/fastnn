use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::mem::size_of;

// Magic bytes to identify fastnn checkpoint format
const MAGIC_BYTES: [u8; 4] = [0x46, 0x4E, 0x4E, 0x00]; // "FNN\0"
                                                       // Format version for forward compatibility
const FORMAT_VERSION: u32 = 1;

#[allow(dead_code)]
pub fn save_model(model: &dyn Module, path: &str) -> Result<(), String> {
    let params = model.named_parameters();

    // Open file for writing
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);

    // Write magic bytes for format identification
    writer
        .write_all(&MAGIC_BYTES)
        .map_err(|e| format!("Failed to write magic bytes: {}", e))?;

    // Write format version
    writer
        .write_all(&FORMAT_VERSION.to_le_bytes())
        .map_err(|e| format!("Failed to write format version: {}", e))?;

    // Write number of parameters
    let num_params = params.len() as u64;
    writer
        .write_all(&num_params.to_le_bytes())
        .map_err(|e| format!("Failed to write param count: {}", e))?;

    // Write each parameter
    for (name, tensor) in &params {
        // Write name length and name
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u64;
        writer
            .write_all(&name_len.to_le_bytes())
            .map_err(|e| format!("Failed to write name length: {}", e))?;
        writer
            .write_all(name_bytes)
            .map_err(|e| format!("Failed to write name: {}", e))?;

        // Write shape
        let shape = tensor.shape();
        let shape_len = shape.len() as u64;
        writer
            .write_all(&shape_len.to_le_bytes())
            .map_err(|e| format!("Failed to write shape length: {}", e))?;
        for &dim in &shape {
            writer
                .write_all(&(dim as u64).to_le_bytes())
                .map_err(|e| format!("Failed to write shape dim: {}", e))?;
        }

        // Write data
        let data = tensor.to_numpy();
        let data_len = data.len() as u64;
        writer
            .write_all(&data_len.to_le_bytes())
            .map_err(|e| format!("Failed to write data length: {}", e))?;

        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<f32>())
        };
        writer
            .write_all(bytes)
            .map_err(|e| format!("Failed to write data: {}", e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Failed to flush: {}", e))?;
    Ok(())
}

#[allow(dead_code)]
pub fn load_model(
    path: &str,
    _model_class: Option<&str>,
) -> Result<HashMap<String, Tensor>, String> {
    let mut result = HashMap::new();

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read and validate magic bytes
    let mut magic_bytes = [0u8; 4];
    reader
        .read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic bytes: {}", e))?;
    if magic_bytes != MAGIC_BYTES {
        return Err(format!(
            "Invalid file format: expected magic bytes {:?}, got {:?}",
            MAGIC_BYTES, magic_bytes
        ));
    }

    // Read format version
    let mut version_bytes = [0u8; 4];
    reader
        .read_exact(&mut version_bytes)
        .map_err(|e| format!("Failed to read format version: {}", e))?;
    let version = u32::from_le_bytes(version_bytes);
    if version > FORMAT_VERSION {
        return Err(format!(
            "Unsupported format version: {}. Maximum supported version is {}",
            version, FORMAT_VERSION
        ));
    }

    // Read number of parameters
    let mut num_params_bytes = [0u8; 8];
    reader
        .read_exact(&mut num_params_bytes)
        .map_err(|e| format!("Failed to read param count: {}", e))?;
    let num_params = u64::from_le_bytes(num_params_bytes);

    for _ in 0..num_params {
        // Read name
        let mut name_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut name_len_bytes)
            .map_err(|e| format!("Failed to read name length: {}", e))?;
        let name_len = u64::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader
            .read_exact(&mut name_bytes)
            .map_err(|e| format!("Failed to read name: {}", e))?;
        let name =
            String::from_utf8(name_bytes).map_err(|e| format!("Invalid UTF-8 in name: {}", e))?;

        // Read shape
        let mut shape_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut shape_len_bytes)
            .map_err(|e| format!("Failed to read shape length: {}", e))?;
        let shape_len = u64::from_le_bytes(shape_len_bytes) as usize;

        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            reader
                .read_exact(&mut dim_bytes)
                .map_err(|e| format!("Failed to read shape dim: {}", e))?;
            shape.push(u64::from_le_bytes(dim_bytes) as i64);
        }

        // Read data
        let mut data_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut data_len_bytes)
            .map_err(|e| format!("Failed to read data length: {}", e))?;
        let data_len = u64::from_le_bytes(data_len_bytes) as usize;

        // Validate shape matches data length
        let expected_numel: usize = shape.iter().map(|&d| d as usize).product();
        if expected_numel != data_len {
            return Err(format!(
                "Shape mismatch for parameter '{}': shape {:?} expects {} elements, but data has {} elements",
                name, shape, expected_numel, data_len
            ));
        }

        let mut data = vec![0.0f32; data_len];
        let data_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data_len * size_of::<f32>(),
            )
        };
        reader
            .read_exact(data_bytes)
            .map_err(|e| format!("Failed to read data: {}", e))?;

        let tensor = Tensor::from_vec(data, shape);
        result.insert(name, tensor);
    }

    Ok(result)
}
