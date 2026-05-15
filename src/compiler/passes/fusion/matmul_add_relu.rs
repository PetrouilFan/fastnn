use super::FusionPass;
use crate::ir::node::{ComputeGraph, IRNode, NodeId, Opcode};
use std::collections::HashMap;

pub struct MatMulAddRelu;

impl FusionPass for MatMulAddRelu {
    fn name() -> &'static str {
        "MatMulAddRelu"
    }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
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

            let bias_input: Vec<NodeId> = add
                .inputs
                .iter()
                .filter(|&&i| i != matmul_id)
                .copied()
                .collect();

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

            if let Some(output) = graph.outputs.iter_mut().find(|o| **o == relu_id) {
                *output = fused_id;
            }

            fused = true;
        }

        for id in &to_remove {
            graph.remove_node(*id);
        }

        for node in new_nodes {
            graph.nodes.push(node);
        }

        Ok(fused)
    }
}
