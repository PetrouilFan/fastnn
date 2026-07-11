use crate::autograd::{make_node_info, AutogradMeta};
use crate::ir::{ComputeGraph, DimExpr, IRNode, NodeId};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// Parse a shape attribute string like `"[1,64,56,56]"` into a `Vec<DimExpr>`.
pub fn parse_shape_attr(s: &str) -> Vec<DimExpr> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|part| {
            let p = part.trim();
            if let Ok(v) = p.parse::<u64>() {
                DimExpr::Known(v)
            } else if p.is_empty() {
                DimExpr::Known(1)
            } else {
                DimExpr::Symbol(p.to_string())
            }
        })
        .collect()
}

/// Attach gradient tracking to an output tensor.
/// Call this inside `Module::forward` after computing `output` from `inputs`.
/// `op_name` should be the backward pass name (e.g. `"MaxPool2dBackward"`).
pub fn attach_grad(output: &mut Tensor, op_name: &'static str, inputs: &[&Tensor]) {
    if inputs.iter().any(|t| t.requires_grad()) {
        let owned_inputs: Vec<Tensor> = inputs.iter().map(|t| (*t).clone()).collect();
        let mut meta = AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(make_node_info(op_name, owned_inputs));
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(parking_lot::Mutex::new(meta)));
    }
}

/// Iterate over all nodes in topological order, calling `f` for each.
/// Returns any error propagated from `f`, or skips deleted nodes.
pub fn traverse_graph<F>(graph: &ComputeGraph, mut f: F) -> Result<(), String>
where
    F: FnMut(NodeId, &IRNode) -> Result<(), String>,
{
    let order = graph.topological_sort();
    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };
        f(node_id, node)?;
    }
    Ok(())
}

/// Mutable version — use when you need `graph.get_node_mut`.
pub fn traverse_graph_mut<F>(graph: &mut ComputeGraph, mut f: F) -> Result<(), String>
where
    F: FnMut(NodeId, &mut IRNode) -> Result<(), String>,
{
    let order = graph.topological_sort();
    for &node_id in &order {
        let has_node = graph.get_node(node_id).is_some();
        if !has_node {
            continue;
        }
        let node = graph.get_node_mut(node_id).unwrap();
        f(node_id, node)?;
    }
    Ok(())
}

/// Parse common convolution attributes from a string-keyed attribute map.
/// Returns `(stride, padding, dilation)`.
pub fn parse_conv_attrs(attrs: &HashMap<String, String>) -> (i64, i64, i64) {
    let stride = attrs
        .get("stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let padding = attrs
        .get("padding")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    let dilation = attrs
        .get("dilation")
        .and_then(|d| d.parse().ok())
        .unwrap_or(1);
    (stride, padding, dilation)
}

/// Parse optional axis attribute — negative values count from end.
pub fn parse_axis(attrs: &HashMap<String, String>, ndim: usize) -> Option<usize> {
    let axis_str = attrs.get("axis")?;
    let axis: i64 = axis_str.parse().ok()?;
    if axis < 0 {
        Some((ndim as i64 + axis) as usize)
    } else {
        Some(axis as usize)
    }
}

/// Compute the spatial output dimension for a convolution-like operation.
pub fn spatial_output_dim(
    input_dim: &DimExpr,
    kernel_dim: &DimExpr,
    stride: i64,
    padding: i64,
    dilation: i64,
) -> Result<DimExpr, String> {
    match (input_dim, kernel_dim) {
        (DimExpr::Known(h), DimExpr::Known(k)) => {
            let result = (*h as i64 + 2 * padding - dilation * (*k as i64 - 1) - 1) / stride + 1;
            if result <= 0 {
                return Err(format!(
                    "output spatial dimension is non-positive ({})",
                    result
                ));
            }
            Ok(DimExpr::Known(result as u64))
        }
        _ => {
            let h_val = input_dim.evaluate();
            let k_val = kernel_dim.evaluate();
            match (h_val, k_val) {
                (Some(h), Some(k)) => {
                    let result =
                        (h as i64 + 2 * padding - dilation * (k as i64 - 1) - 1) / stride + 1;
                    if result <= 0 {
                        return Err(format!(
                            "output spatial dimension is non-positive ({})",
                            result
                        ));
                    }
                    Ok(DimExpr::Known(result as u64))
                }
                _ => Ok(DimExpr::Symbol("?".to_string())),
            }
        }
    }
}
