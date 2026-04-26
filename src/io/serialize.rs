use crate::error::{FastnnError, FastnnResult};
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
pub fn save_model(model: &dyn Module, path: &str) -> FastnnResult<()> {
    let params = model.named_parameters();

    // Open file for writing
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write magic bytes for format identification
    writer.write_all(&MAGIC_BYTES)?;

    // Write format version
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

    // Write number of parameters
    let num_params = params.len() as u64;
    writer.write_all(&num_params.to_le_bytes())?;

    // Write each parameter
    for (name, tensor) in &params {
        // Write name length and name
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u64;
        writer.write_all(&name_len.to_le_bytes())?;
        writer.write_all(name_bytes)?;

        // Write shape
        let shape = tensor.shape();
        let shape_len = shape.len() as u64;
        writer.write_all(&shape_len.to_le_bytes())?;
        for &dim in &shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        // Write data
        let data = tensor.to_numpy();
        let data_len = data.len() as u64;
        writer.write_all(&data_len.to_le_bytes())?;

        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<f32>())
        };
        writer.write_all(bytes)?;
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
) -> FastnnResult<HashMap<String, Tensor>> {
    let mut result = HashMap::new();

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read and validate magic bytes
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    if magic_bytes != MAGIC_BYTES {
        return Err(FastnnError::io(format!(
            "Invalid file format: expected magic bytes {:?}, got {:?}",
            MAGIC_BYTES, magic_bytes
        )));
    }

    // Read format version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version > FORMAT_VERSION {
        return Err(FastnnError::io(format!(
            "Unsupported format version: {}. Maximum supported version is {}",
            version, FORMAT_VERSION
        )));
    }

    // Read number of parameters
    let mut num_params_bytes = [0u8; 8];
    reader.read_exact(&mut num_params_bytes)?;
    let num_params = u64::from_le_bytes(num_params_bytes);

    for _ in 0..num_params {
        // Read name
        let mut name_len_bytes = [0u8; 8];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u64::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name =
            String::from_utf8(name_bytes).map_err(|e| FastnnError::io(format!("Invalid UTF-8 in name: {}", e)))?;

        // Read shape
        let mut shape_len_bytes = [0u8; 8];
        reader.read_exact(&mut shape_len_bytes)?;
        let shape_len = u64::from_le_bytes(shape_len_bytes) as usize;

        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as i64);
        }

        // Read data
        let mut data_len_bytes = [0u8; 8];
        reader.read_exact(&mut data_len_bytes)?;
        let data_len = u64::from_le_bytes(data_len_bytes) as usize;

        // Validate shape matches data length
        let expected_numel: usize = shape.iter().map(|&d| d as usize).product();
        if expected_numel != data_len {
            return Err(FastnnError::shape(format!(
                "Shape mismatch for parameter '{}': shape {:?} expects {} elements, but data has {} elements",
                name, shape, expected_numel, data_len
            )));
        }

        // Validate tensor size to prevent integer overflow
        // Check if total elements exceed reasonable limits
        const MAX_TENSOR_ELEMENTS: usize = 1usize << 40; // 1 trillion elements max
        if expected_numel > MAX_TENSOR_ELEMENTS {
            return Err(FastnnError::overflow(format!(
                "Tensor '{}' has {} elements, which exceeds maximum allowed size of {}",
                name, expected_numel, MAX_TENSOR_ELEMENTS
            )));
        }

        // Validate byte size doesn't overflow
        let byte_size = data_len.checked_mul(size_of::<f32>())
            .ok_or_else(|| FastnnError::overflow(
                format!("Tensor '{}' byte size overflow", name)
            ))?;

        let mut data_bytes = vec![0u8; byte_size];
        reader.read_exact(&mut data_bytes)?;

        let data =
            unsafe { std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, data_len) }
                .to_vec();

        let tensor = Tensor::from_vec(data, shape);
        result.insert(name, tensor);
    }

    Ok(result)
}
