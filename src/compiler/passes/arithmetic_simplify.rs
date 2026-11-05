use crate::ir::{ComputeGraph, NodeId, Opcode, TensorValue};

pub fn arithmetic_simplify(graph: &mut ComputeGraph) -> usize {
    let mut simplified = 0;

    let mut rewrites: Vec<(NodeId, RewriteAction)> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    let _ = crate::utils::traverse_graph(graph_ref, |_node_id, node| {
        let action = match node.opcode {
            Opcode::Mul => {
                let is_one = |id: NodeId| -> bool {
                    graph_ref.get_node(id).is_some_and(|n| match &n.opcode {
                        Opcode::Constant(TensorValue::Float(v)) => *v == 1.0,
                        Opcode::Constant(TensorValue::Int(v)) => *v == 1,
                        _ => false,
                    })
                };

                if node.inputs.len() >= 2 {
                    if is_one(node.inputs[1]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[0]))
                    } else if is_one(node.inputs[0]) {
                        Some(RewriteAction::ReplaceWith(node.inputs[1]))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            Opcode::Add => {
                let is_zero = |id: NodeId| -> bool {
                    graph_ref.get_node(id).is_some_and(|n| match &n.opcode {
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
                    graph_ref.get_node(id).is_some_and(|n| match &n.opcode {
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
                    graph_ref.get_node(id).is_some_and(|n| match &n.opcode {
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
                    let is_neg = graph_ref
                        .get_node(inner_id)
                        .is_some_and(|n| n.opcode == Opcode::Neg);
                    if is_neg {
                        if let Some(&inner_input) =
                            graph_ref.get_node(inner_id).and_then(|n| n.inputs.first())
                        {
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
                    let is_abs = graph_ref
                        .get_node(inner_id)
                        .is_some_and(|n| n.opcode == Opcode::Abs);
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
            rewrites.push((_node_id, action));
        }
        Ok(())
    });

    for (node_id, action) in &rewrites {
        let RewriteAction::ReplaceWith(replacement) = action;
        let same_type = graph
            .get_node(*node_id)
            .zip(graph.get_node(*replacement))
            .is_some_and(|(node, input)| {
                node.output_type == input.output_type
                    && node
                        .output_type
                        .shape
                        .iter()
                        .all(|d| matches!(d, crate::ir::DimExpr::Known(_)))
            });
        if !same_type {
            continue;
        }
        if let Some(node) = graph.get_node_mut(*node_id) {
            // Preserve output identity and chained consumers. DCE resolves these
            // identity reshapes after all candidates have been collected.
            let shape = node
                .output_type
                .shape
                .iter()
                .map(|d| d.evaluate().unwrap().to_string())
                .collect::<Vec<_>>()
                .join(",");
            node.opcode = Opcode::Reshape;
            node.inputs = vec![*replacement];
            node.attrs.clear();
            node.attrs.insert("shape".into(), shape);
            simplified += 1;
        }
    }
    if simplified > 0 {
        graph.mark_mutated();
    }
    simplified
}

enum RewriteAction {
    ReplaceWith(NodeId),
}
