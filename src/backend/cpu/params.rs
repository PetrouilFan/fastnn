use crate::backend::BackendError;
use crate::ir::node::{DimExpr, ShapeEnv};

/// Resolve kernel dimension params at dispatch time using the runtime shape
/// environment. Returns `params` unchanged if no symbolic dims are present.
///
/// Resolve symbolic parameters to concrete values using the runtime ShapeEnv.
///
/// Returns `Err(BackendError::Dispatch)` when `param_dims` is malformed
/// (backend lowering bug) or a symbolic dimension cannot be resolved.
pub(super) fn resolve_params(
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    expected: usize,
) -> Result<Vec<usize>, BackendError> {
    if let Some(dims) = param_dims {
        if dims.len() < expected {
            return Err(BackendError::Dispatch(format!(
                "resolve_params: param_dims has {} elements but expected {} (params={:?})",
                dims.len(),
                expected,
                params
            )));
        }
        dims[..expected]
            .iter()
            .map(|d| {
                d.evaluate_with_env(shape_env)
                    .map(|v| v as usize)
                    .map_err(|e| BackendError::Dispatch(format!("resolve_params: {e}")))
            })
            .collect()
    } else {
        Ok(params.to_vec())
    }
}
