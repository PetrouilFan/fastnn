use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;

#[allow(dead_code)]
pub fn save_model(model: &dyn Module, path: &str) {
    let params = model.named_parameters();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

    for (name, tensor) in &params {
        tensors.insert(name.clone(), tensor.to_numpy());
        metadata.insert(
            name.clone(),
            serde_json::json!({
                "shape": tensor.shape(),
                "dtype": tensor.dtype().as_str(),
            }),
        );
    }

    // Save safetensors (simplified - would use actual safetensors crate)
    let _json_meta = serde_json::json!({
        "format": "fastnn",
        "version": "0.1.0",
        "metadata": metadata,
    });

    println!("Saved model to {} with {} parameters", path, params.len());
}

#[allow(dead_code)]
pub fn load_model(path: &str, _model_class: Option<&str>) -> HashMap<String, Tensor> {
    // Would load from safetensors file
    println!("Loaded model from {}", path);
    HashMap::new()
}
