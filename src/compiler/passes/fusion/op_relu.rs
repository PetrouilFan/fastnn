use super::FusionPass;
use crate::ir::node::{ComputeGraph, NodeId, Opcode};

pub struct OpRelu;

impl FusionPass for OpRelu {
    fn name() -> &'static str {
        "OpRelu"
    }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
        let mut fused = false;
        let mut to_remove: Vec<NodeId> = Vec::new();
        let node_ids: Vec<NodeId> = graph.nodes.iter().map(|n| n.id).collect();

        for &op_id in &node_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            let op_node = match graph.get_node(op_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            match op_node.opcode {
                Opcode::Relu | Opcode::Input | Opcode::Constant(_) => continue,
                _ => {}
            }
            if op_node.attrs.contains_key("fused_op") {
                continue;
            }

            let op_consumers: Vec<NodeId> = graph.consumers(op_id);
            if op_consumers.len() != 1 {
                continue;
            }

            let relu_id = op_consumers[0];
            if to_remove.contains(&relu_id) {
                continue;
            }

            let relu = match graph.get_node(relu_id) {
                Some(n) if n.opcode == Opcode::Relu => n.clone(),
                _ => continue,
            };

            if let Some(op_mut) = graph.get_node_mut(op_id) {
                op_mut
                    .attrs
                    .insert("fused_op".to_string(), "OpRelu".to_string());
                op_mut.output_type = relu.output_type.clone();
            }

            to_remove.push(relu_id);

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

            if let Some(output) = graph.outputs.iter_mut().find(|o| **o == relu_id) {
                *output = op_id;
            }

            fused = true;
        }

        for id in &to_remove {
            graph.remove_node(*id);
        }

        Ok(fused)
    }
}
