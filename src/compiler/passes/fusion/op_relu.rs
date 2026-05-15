use super::FusionPass;
use crate::ir::node::{ComputeGraph, NodeId, Opcode};

fn activation_fused_name(opcode: Opcode) -> Option<&'static str> {
    match opcode {
        Opcode::Relu => Some("OpRelu"),
        Opcode::Gelu => Some("OpGelu"),
        Opcode::Silu => Some("OpSilu"),
        _ => None,
    }
}

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
                Opcode::Relu
                | Opcode::Gelu
                | Opcode::Silu
                | Opcode::Input
                | Opcode::Constant(_) => continue,
                _ => {}
            }
            if op_node.attrs.contains_key("fused_op") {
                continue;
            }

            let op_consumers: Vec<NodeId> = graph.consumers(op_id);
            if op_consumers.len() != 1 {
                continue;
            }

            let act_id = op_consumers[0];
            if to_remove.contains(&act_id) {
                continue;
            }

            let act = match graph.get_node(act_id) {
                Some(n) => n.clone(),
                _ => continue,
            };
            let fused_name = match activation_fused_name(act.opcode) {
                Some(name) => name,
                None => continue,
            };

            if let Some(op_mut) = graph.get_node_mut(op_id) {
                op_mut
                    .attrs
                    .insert("fused_op".to_string(), fused_name.to_string());
                op_mut.output_type = act.output_type.clone();
            }

            to_remove.push(act_id);

            let act_consumers: Vec<NodeId> = graph.consumers(act_id);
            for consumer_id in act_consumers {
                if let Some(consumer) = graph.get_node_mut(consumer_id) {
                    for input in consumer.inputs.iter_mut() {
                        if *input == act_id {
                            *input = op_id;
                        }
                    }
                }
            }

            if let Some(output) = graph.outputs.iter_mut().find(|o| **o == act_id) {
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
