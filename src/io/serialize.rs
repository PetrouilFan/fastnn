use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::mem::size_of;

#[allow(dead_code)]
pub fn save_model(model: &dyn Module, path: &str) -> Result<(), String> {
    let params = model.named_parameters();

    // Open file for writing
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);

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
    println!("Saved model to {} with {} parameters", path, params.len());
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

        let mut data_bytes = vec![0u8; data_len * size_of::<f32>()];
        reader
            .read_exact(&mut data_bytes)
            .map_err(|e| format!("Failed to read data: {}", e))?;

        let data =
            unsafe { std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, data_len) }
                .to_vec();

        let tensor = Tensor::from_vec(data, shape);
        result.insert(name, tensor);
    }

    println!(
        "Loaded model from {} with {} parameters",
        path,
        result.len()
    );
    Ok(result)
}
