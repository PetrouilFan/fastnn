//! Compiler pass to insert F8x4R gradient quantization around optimizer inputs.
//!
//! After the backward graph is built (but before optimizer injection),
//! this pass inserts `QuantizeGradient` → `DequantizeGradient` pairs on
//! each gradient edge feeding an optimizer update node.

use std::collections::HashMap;

use crate::ir::node::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};

/// Insert gradient quantization for all optimizer update nodes.
///
/// For each optimizer node, the gradient input (index 1) is wrapped with:
///   grad_f32 → QuantizeGradient → grad_f8x4r → DequantizeGradient → grad_f32'
pub fn quantize_gradients(graph: &mut ComputeGraph) {
    let node_ids = graph.topological_sort();
    let mut rewrites: Vec<(NodeId, NodeId, f32)> = Vec::new();

    for &node_id in &node_ids {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };
        if !is_optimizer_op(&node.opcode) {
            continue;
        }
        // Gradient input is at index 1 (after param at index 0)
        let grad_id = match node.inputs.get(1) {
            Some(&id) => id,
            None => continue,
        };
        // Use default scale = 1.0 (calibration can adjust later)
        rewrites.push((node_id, grad_id, 1.0));
    }

    for (opt_id, grad_id, scale) in rewrites {
        let grad_type = graph
            .get_node(grad_id)
            .map(|n| n.output_type.clone())
            .unwrap_or(TensorType::new(vec![], IrDType::F32));

        let f8x4r_type = TensorType::new(
            grad_type.shape.clone(),
            IrDType::F8R { scales: vec![scale] },
        );

        let mut attrs = HashMap::new();
        attrs.insert("scale".to_string(), scale.to_string());
        let q_id = graph.add_node_with_attrs(
            Opcode::QuantizeGradient,
            vec![grad_id],
            f8x4r_type,
            attrs,
        );

        let dq_type = TensorType::new(grad_type.shape.clone(), IrDType::F32);
        let dq_id = graph.add_node(
            Opcode::DequantizeGradient,
            vec![q_id],
            dq_type,
        );

        if let Some(opt) = graph.get_node_mut(opt_id) {
            opt.inputs[1] = dq_id;
        }
    }
}

fn is_optimizer_op(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::SgdUpdate
            | Opcode::AdamUpdate
            | Opcode::AdamWUpdate
            | Opcode::MuonUpdate
            | Opcode::LionUpdate
            | Opcode::RmspropUpdate
    )
}
