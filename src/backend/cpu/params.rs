use crate::backend::BackendError;
use crate::ir::{DimExpr, ShapeEnv};
use std::borrow::Cow;

/// Resolve kernel dimension params at dispatch time using the runtime shape
/// environment. Returns `params` unchanged (as a borrowed `Cow`) if no
/// symbolic dims are present, avoiding a heap allocation on the common path.
///
/// Returns `Err(BackendError::Dispatch)` when `param_dims` is malformed
/// (backend lowering bug) or a symbolic dimension cannot be resolved.
pub(super) fn resolve_params<'a>(
    params: &'a [usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    expected: usize,
) -> Result<Cow<'a, [usize]>, BackendError> {
    if let Some(dims) = param_dims {
        if dims.len() < expected {
            return Err(BackendError::Dispatch(format!(
                "resolve_params: param_dims has {} elements but expected {} (params={:?})",
                dims.len(),
                expected,
                params
            )));
        }
        let resolved: Vec<usize> = dims[..expected]
            .iter()
            .map(|d| {
                d.evaluate_with_env(shape_env)
                    .map(|v| v as usize)
                    .map_err(|e| BackendError::Dispatch(format!("resolve_params: {e}")))
            })
            .collect::<Result<_, _>>()?;
        Ok(Cow::Owned(resolved))
    } else {
        Ok(Cow::Borrowed(params))
    }
}
