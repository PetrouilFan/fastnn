use crate::ir::node::{ComputeGraph, NodeId, Opcode, TensorValue};

pub fn arithmetic_simplify(graph: &mut ComputeGraph) -> usize {
    let order = graph.topological_sort();
    let mut simplified = 0;

    let mut rewrites: Vec<(NodeId, RewriteAction)> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        let action = match node.opcode {
            Opcode::Mul => {
                let is_one = |id: NodeId| -> bool {
                    graph.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 1.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 1,
                        _ => false,
                    })
                };
                let is_zero = |id: NodeId| -> bool {
                    graph.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 0.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 0,
                        _ => false,
                    })
                };

                if node.inputs.len() >= 2 {
                    if is_one(node.inputs[1]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[0]))
                    } else if is_one(node.inputs[0]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[1]))
                    } else if is_zero(node.inputs[1]) {
                        Some(RewriteAction::ReplaceWithZero(node.inputs[1]))
                    } else if is_zero(node.inputs[0]) {
                        Some(RewriteAction::ReplaceWithZero(node.inputs[0]))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            Opcode::Add => {
                let is_zero = |id: NodeId| -> bool {
                    graph.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 0.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 0,
                        _ => false,
                    })
                };

                if node.inputs.len() >= 2 {
                    if is_zero(node.inputs[1]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[0]))
                    } else if is_zero(node.inputs[0]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[1]))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            Opcode::Sub => {
                let is_zero = |id: NodeId| -> bool {
                    graph.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 0.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 0,
                        _ => false,
                    })
                };

                if node.inputs.len() >= 2 && is_zero(node.inputs[1]) {
                    Some(RewriteAction::ReplaceWith(node.inputs[0]))
                } else {
                    None
                }
            }

            Opcode::Div => {
                let is_one = |id: NodeId| -> bool {
                    graph.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 1.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 1,
                        _ => false,
                    })
                };

                if node.inputs.len() >= 2 && is_one(node.inputs[1]) {
                    Some(RewriteAction::ReplaceWith(node.inputs[0]))
                } else {
                    None
                }
            }

            Opcode::Neg => {
                if let Some(&inner_id) = node.inputs.first() {
                    let is_neg = graph.get_node(inner_id).is_some_and(|n| n.opcode == Opcode::Neg);
                    if is_neg {
                        if let Some(&inner_input) = graph.get_node(inner_id).and_then(|n| n.inputs.first()) {
                            Some(RewriteAction::ReplaceWith(inner_input))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            Opcode::Abs => {
                if let Some(&inner_id) = node.inputs.first() {
                    let is_abs = graph.get_node(inner_id).is_some_and(|n| n.opcode == Opcode::Abs);
                    if is_abs {
                        Some(RewriteAction::ReplaceWith(inner_id))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(action) = action {
            rewrites.push((node_id, action));
        }
    }

    for (node_id, action) in &rewrites {
        match action {
            RewriteAction::ReplaceWith(replacement) => {
                rewire_consumers(graph, *node_id, *replacement);
                graph.remove_node(*node_id);
                simplified += 1;
            }
            RewriteAction::ReplaceWithZero(zero_const_id) => {
                rewire_consumers(graph, *node_id, *zero_const_id);
                graph.remove_node(*node_id);
                simplified += 1;
            }
        }
    }

    simplified
}

enum RewriteAction {
    ReplaceWith(NodeId),
    ReplaceWithZero(NodeId),
}

fn rewire_consumers(graph: &mut ComputeGraph, from: NodeId, to: NodeId) {
    let consumers: Vec<NodeId> = graph.consumers(from);
    for consumer_id in consumers {
        if let Some(consumer) = graph.get_node_mut(consumer_id) {
            for input in consumer.inputs.iter_mut() {
                if *input == from {
                    *input = to;
                }
            }
        }
    }
}
