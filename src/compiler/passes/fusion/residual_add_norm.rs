use super::FusionPass;
use crate::ir::node::{ComputeGraph, NodeId, Opcode};
use std::collections::HashMap;

pub struct FusedResidualAddNorm;

impl FusionPass for FusedResidualAddNorm {
    fn name() -> &'static str {
        "FusedResidualAddNorm"
    }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
        apply_fused_residual_add_norm(graph)
    }
}

fn apply_fused_residual_add_norm(graph: &mut ComputeGraph) -> Result<bool, String> {
    let norm_ids: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.opcode, Opcode::LayerNorm | Opcode::RMSNorm))
        .map(|n| n.id)
        .collect();

    let mut changed = false;
    let mut to_remove: Vec<NodeId> = Vec::new();

    for norm_id in norm_ids {
        if to_remove.contains(&norm_id) {
            continue;
        }

        let norm_node = match graph.get_node(norm_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        // LayerNorm has 3 inputs [data, weight, bias], RMSNorm has 2 [data, weight]
        let num_norm_inputs = norm_node.inputs.len();
        if num_norm_inputs < 2 {
            continue;
        }
        let add_id = norm_node.inputs[0];

        if to_remove.contains(&add_id) {
            continue;
        }

        let add_node = match graph.get_node(add_id) {
            Some(n) if n.opcode == Opcode::Add => n.clone(),
            _ => continue,
        };

        if add_node.inputs.len() != 2 {
            continue;
        }

        let residual_id = add_node.inputs[0];
        let main_output_id = add_node.inputs[1];

        // Check the Add node only feeds the norm (otherwise leave Add in place)
        let add_consumers: Vec<NodeId> = graph.consumers(add_id);
        let can_remove_add = add_consumers.len() == 1 && add_consumers[0] == norm_id;

        let eps = norm_node
            .attrs
            .get("eps")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-5);

        let norm_type = match norm_node.opcode {
            Opcode::LayerNorm => "layer_norm",
            Opcode::RMSNorm => "rms_norm",
            _ => unreachable!(),
        };

        let mut attrs = HashMap::new();
        attrs.insert("eps".to_string(), eps.to_string());
        attrs.insert("norm_type".to_string(), norm_type.to_string());

        // Preserve weight (and bias for LayerNorm) inputs from the original norm node
        let weight_bias_inputs: Vec<NodeId> = norm_node.inputs[1..].to_vec();

        if let Some(norm_mut) = graph.get_node_mut(norm_id) {
            norm_mut.opcode = Opcode::FusedResidualAddNorm;
            let mut fused_inputs = vec![residual_id, main_output_id];
            fused_inputs.extend(weight_bias_inputs);
            norm_mut.inputs = fused_inputs;
            norm_mut.attrs = attrs;
        }

        if can_remove_add {
            to_remove.push(add_id);
        }

        changed = true;
    }

    for id in &to_remove {
        graph.remove_node(*id);
    }

    Ok(changed)
}
