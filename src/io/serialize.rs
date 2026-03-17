use crate::nn::Module;
use crate::tensor::Tensor;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[allow(dead_code)]
pub fn save_model(model: &dyn Module, path: &str) {
    let params = model.named_parameters();

    // Build tensors map for safetensors
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

    for (name, tensor) in &params {
        // Get the raw data as bytes
        let data = tensor.to_numpy();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };

        // Create TensorView with the data and shape
        let shape: Vec<usize> = tensor.shape().iter().map(|&s| s as usize).collect();
        let view = TensorView::new(safetensors::DType::F32, &shape, bytes)
            .expect("Failed to create TensorView");

        tensors.insert(name.clone(), view);

        metadata.insert(
            name.clone(),
            serde_json::json!({
                "shape": tensor.shape(),
                "dtype": tensor.dtype().as_str(),
            }),
        );
    }

    // Serialize to file
    if let Ok(file) = File::create(path) {
        let mut writer = BufWriter::new(file);
        if let Err(e) = safetensors::serialize_to_file(&tensors, &Some(&metadata), &mut writer) {
            eprintln!("Failed to save model: {}", e);
        } else {
            println!("Saved model to {} with {} parameters", path, params.len());
        }
    } else {
        eprintln!("Failed to create file: {}", path);
    }
}

#[allow(dead_code)]
pub fn load_model(path: &str, _model_class: Option<&str>) -> HashMap<String, Tensor> {
    let mut result = HashMap::new();

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        match safetensors::load(reader) {
            Ok(safe_tensors) => {
                for (name, tensor) in safe_tensors {
                    let shape: Vec<i64> = tensor.shape().iter().map(|&s| s as i64).collect();
                    let data = tensor.data();
                    let float_slice = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const f32,
                            data.len() / std::mem::size_of::<f32>(),
                        )
                    };
                    let vec_data = float_slice.to_vec();
                    result.insert(name, Tensor::from_vec(vec_data, shape));
                }
                println!(
                    "Loaded model from {} with {} parameters",
                    path,
                    result.len()
                );
            }
            Err(e) => eprintln!("Failed to load model: {}", e),
        }
    } else {
        eprintln!("Failed to open file: {}", path);
    }

    result
}
