use crate::ir::node::ComputeGraph;
use std::collections::HashSet;

/// Remove nodes that are not reachable from `graph.inputs`, `graph.outputs`,
/// or `graph.required_nodes`.
///
/// If `graph.outputs` is empty the pass is a no-op — some callers build
/// graphs without explicitly setting outputs, and we conservatively assume
/// every node is live.
///
/// Returns the number of removed nodes.
pub fn eliminate_dead_code(graph: &mut ComputeGraph) -> usize {
    if graph.outputs.is_empty() {
        return 0;
    }

    let mut reachable: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = Vec::new();

    for &id in &graph.inputs {
        stack.push(id);
    }
    for &id in &graph.outputs {
        stack.push(id);
    }
    for &id in &graph.required_nodes {
        stack.push(id);
    }

    while let Some(id) = stack.pop() {
        if reachable.insert(id) {
            if let Some(node) = graph.get_node(id) {
                for &input_id in &node.inputs {
                    stack.push(input_id);
                }
            }
        }
    }

    let before = graph.nodes.len();
    graph.nodes.retain(|node| reachable.contains(&node.id));
    let removed = before - graph.nodes.len();

    if removed > 0 {
        graph.inputs.retain(|id| reachable.contains(id));
        graph.outputs.retain(|id| reachable.contains(id));
        graph.required_nodes.retain(|id| reachable.contains(id));
        for node in &mut graph.nodes {
            node.inputs.retain(|id| reachable.contains(id));
        }
        graph.rebuild_node_index();
        graph.mark_mutated();
    }

    removed
}
