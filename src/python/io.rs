#[pyfunction]
fn save_model(model: &PyTransformerEncoder, path: String) -> PyResult<()> {
    core_io::serialize::save_model(&model.inner, &path)
        .map_err(|e| IoError::new_err(format!("Failed to save model: {}", e)))
}

#[pyfunction]
fn load_model(path: String) -> PyResult<HashMap<String, PyTensor>> {
    let state_dict = core_io::serialize::load_model(&path, None)
        .map_err(|e| IoError::new_err(format!("Failed to load model: {}", e)))?;
    Ok(state_dict
        .into_iter()
        .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
        .collect())
}

