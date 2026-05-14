#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, Opcode, IRNode, NodeId};
use std::collections::HashMap;

pub fn fuse_operators(graph: &mut ComputeGraph) -> Result<(), String> {
    let mut changed = true;
    while changed {
        changed = false;
        changed |= fuse_matmul_add_relu(graph)?;
        changed |= fuse_op_relu(graph)?;
    }
    Ok(())
}

/// Fuse any op (that has a single consumer) with its subsequent Relu.
///
/// Pattern: `Op -> Relu` becomes a single `Op(fused_op=OpRelu)` node.
/// The original op node is mutated in place; the Relu node is removed.
/// This saves one intermediate allocation in memory planning.
fn fuse_op_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    let mut fused = false;
    let mut to_remove: Vec<NodeId> = Vec::new();

    // Collect node IDs first to avoid borrow issues
    let node_ids: Vec<NodeId> = graph.nodes.iter().map(|n| n.id).collect();

    for &op_id in &node_ids {
        if to_remove.contains(&op_id) {
            continue;
        }

        // Get the op node — skip if already fused, or is a terminal op
        let op_node = match graph.get_node(op_id) {
            Some(n) => n.clone(),
            None => continue,
        };
        match op_node.opcode {
            Opcode::Relu | Opcode::Input | Opcode::Constant(_) => continue,
            _ => {}
        }
        // Skip if already fused (don't re-fuse)
        if op_node.attrs.get("fused_op").is_some() {
            continue;
        }

        // The op must have exactly one consumer (the Relu)
        let op_consumers: Vec<NodeId> = graph.consumers(op_id);
        if op_consumers.len() != 1 {
            continue;
        }

        let relu_id = op_consumers[0];
        if to_remove.contains(&relu_id) {
            continue;
        }

        // The consumer must be a Relu
        let relu = match graph.get_node(relu_id) {
            Some(n) if n.opcode == Opcode::Relu => n.clone(),
            _ => continue,
        };

        // Fuse: mutate the op node to absorb the Relu
        if let Some(op_mut) = graph.get_node_mut(op_id) {
            // Set the fused op attribute
            op_mut.attrs.insert("fused_op".to_string(), "OpRelu".to_string());
            // The output type becomes the Relu's output type (typically same shape as op output)
            op_mut.output_type = relu.output_type.clone();
        }

        to_remove.push(relu_id);

        // Rewire: consumers of Relu now consume the fused op node
        let relu_consumers: Vec<NodeId> = graph.consumers(relu_id);
        for consumer_id in relu_consumers {
            if let Some(consumer) = graph.get_node_mut(consumer_id) {
                for input in consumer.inputs.iter_mut() {
                    if *input == relu_id {
                        *input = op_id;
                    }
                }
            }
        }

        // If Relu was a graph output, replace with the op node
        if let Some(output) = graph.outputs.iter_mut().find(|o| **o == relu_id) {
            *output = op_id;
        }

        fused = true;
    }

    // Remove old Relu nodes
    for id in &to_remove {
        graph.remove_node(*id);
    }

    Ok(fused)
}

fn fuse_matmul_add_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    let mut fused = false;
    let mut to_remove: Vec<NodeId> = Vec::new();
    let mut new_nodes: Vec<IRNode> = Vec::new();

    let node_ids: Vec<NodeId> = graph.nodes.iter().map(|n| n.id).collect();

    for &matmul_id in &node_ids {
        if to_remove.contains(&matmul_id) {
            continue;
        }

        let matmul = match graph.get_node(matmul_id) {
            Some(n) if n.opcode == Opcode::MatMul => n.clone(),
            _ => continue,
        };

        // MatMul must have exactly one consumer (the Add/BiasAdd)
        let matmul_consumers: Vec<NodeId> = graph.consumers(matmul_id);
        if matmul_consumers.len() != 1 {
            continue;
        }

        let add_id = matmul_consumers[0];
        if to_remove.contains(&add_id) {
            continue;
        }

        let add = match graph.get_node(add_id) {
            Some(n) if n.opcode == Opcode::Add || n.opcode == Opcode::BiasAdd => n.clone(),
            _ => continue,
        };

        // Add/BiasAdd must have exactly one consumer (the Relu)
        let add_consumers: Vec<NodeId> = graph.consumers(add_id);
        if add_consumers.len() != 1 {
            continue;
        }

        let relu_id = add_consumers[0];
        if to_remove.contains(&relu_id) {
            continue;
        }

        let relu = match graph.get_node(relu_id) {
            Some(n) if n.opcode == Opcode::Relu => n.clone(),
            _ => continue,
        };

        // Extract bias input from Add (the input that isn't the MatMul output)
        let bias_input: Vec<NodeId> = add.inputs.iter()
            .filter(|&&i| i != matmul_id)
            .copied()
            .collect();

        // Create fused node
        let fused_id = graph.next_id;
        graph.next_id += 1;

        let mut fused_inputs = matmul.inputs.clone();
        fused_inputs.extend(bias_input);

        let mut attrs = HashMap::new();
        attrs.insert("fused_op".to_string(), "MatMulAddRelu".to_string());

        let fused_node = IRNode {
            id: fused_id,
            opcode: Opcode::MatMul,
            inputs: fused_inputs,
            output_type: relu.output_type.clone(),
            secondary_output_type: None,
            attrs,
            name: format!("fused_{}", matmul.name),
        };

        new_nodes.push(fused_node);

        to_remove.push(matmul_id);
        to_remove.push(add_id);
        to_remove.push(relu_id);

        // Rewire: consumers of Relu now consume the fused node
        let relu_consumers: Vec<NodeId> = graph.consumers(relu_id);
        for consumer_id in relu_consumers {
            if let Some(consumer) = graph.get_node_mut(consumer_id) {
                for input in consumer.inputs.iter_mut() {
                    if *input == relu_id {
                        *input = fused_id;
                    }
                }
            }
        }

        // If Relu was a graph output, replace with fused node
        if let Some(output) = graph.outputs.iter_mut().find(|o| **o == relu_id) {
            *output = fused_id;
        }

        fused = true;
    }

    // Remove old nodes
    for id in &to_remove {
        graph.remove_node(*id);
    }

    // Add fused nodes
    for node in new_nodes {
        graph.nodes.push(node);
    }

    Ok(fused)
}
