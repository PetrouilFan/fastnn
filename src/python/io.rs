#[pyfunction]
fn save_model(model: &PyTransformerEncoder, path: String) -> PyResult<()> {
    crate::io::serialize::save_model(&model.inner, &path)
        .map_err(|e| IoError::new_err(e.to_string()))
}

#[pyfunction]
fn load_model(path: String) -> PyResult<HashMap<String, PyTensor>> {
    let state_dict = crate::io::serialize::load_model(&path)
        .map_err(|e| IoError::new_err(e.to_string()))?;
    Ok(state_dict
        .into_iter()
        .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
        .collect())
}

