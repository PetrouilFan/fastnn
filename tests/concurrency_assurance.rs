use fastnn::ir::{ComputeGraph, DimExpr, GraphResourceLimits, IrDType, Opcode, TensorType};
use std::sync::{Arc, Barrier};
use std::thread;

fn vector_type() -> TensorType {
    TensorType::new(vec![DimExpr::Known(16)], IrDType::F32)
}

#[test]
fn immutable_graph_indexes_are_stable_under_concurrent_reads() {
    let mut graph = ComputeGraph::new();
    let lhs = graph.add_node(Opcode::Input, vec![], vector_type());
    let rhs = graph.add_node(Opcode::Input, vec![], vector_type());
    let add = graph.add_node(Opcode::Add, vec![lhs, rhs], vector_type());
    let relu = graph.add_node(Opcode::Relu, vec![add], vector_type());
    graph.set_inputs(vec![lhs, rhs]);
    graph.set_outputs(vec![relu]);
    graph
        .validate_with_limits(&GraphResourceLimits::default())
        .unwrap();

    let graph = Arc::new(graph);
    let barrier = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let graph = Arc::clone(&graph);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..1_000 {
                assert_eq!(graph.try_topological_sort().unwrap(), [lhs, rhs, add, relu]);
                assert_eq!(graph.consumers(lhs), [add]);
                assert_eq!(graph.consumers(rhs), [add]);
                assert_eq!(graph.consumers(add), [relu]);
                assert!(graph.consumers(relu).is_empty());
                assert_eq!(graph.get_node(relu).unwrap().opcode, Opcode::Relu);
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
}
