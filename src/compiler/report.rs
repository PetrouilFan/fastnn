use crate::ir::GraphKind;
use serde::{Deserialize, Serialize};

/// Structured summary of a successful compiler pipeline run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileReport {
    pub graph_kind: GraphKind,
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub passes: Vec<PassReport>,
}

impl CompileReport {
    pub(crate) fn new(graph_kind: GraphKind, initial_nodes: usize) -> Self {
        Self {
            graph_kind,
            initial_nodes,
            final_nodes: initial_nodes,
            passes: Vec::new(),
        }
    }

    pub(crate) fn record(&mut self, name: &'static str, before_nodes: usize, after_nodes: usize) {
        self.passes.push(PassReport {
            name: name.to_string(),
            before_nodes,
            after_nodes,
        });
        self.final_nodes = after_nodes;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassReport {
    pub name: String,
    pub before_nodes: usize,
    pub after_nodes: usize,
}

impl PassReport {
    pub fn node_delta(&self) -> isize {
        self.after_nodes as isize - self.before_nodes as isize
    }
}
