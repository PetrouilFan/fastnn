#[pyfunction]
fn save_model(model: &Bound<'_, PyAny>, path: String) -> PyResult<()> {
    let np = model
        .call_method0("named_parameters")
        .map_err(|e| {
            IoError::new_err(format!(
                "save_model: model does not expose named_parameters(): {}",
                e
            ))
        })?;
    let params: Vec<(String, PyTensor)> = np.extract().map_err(|e| {
        IoError::new_err(format!(
            "save_model: failed to extract named_parameters: {}",
            e
        ))
    })?;
    let state_dict: Vec<(String, Tensor)> =
        params.into_iter().map(|(name, pt)| (name, pt.inner)).collect();
    crate::io::serialize::save_state_dict(state_dict, &path)
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

