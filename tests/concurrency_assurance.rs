use fastnn::ir::{ComputeGraph, DimExpr, GraphResourceLimits, IrDType, Opcode, TensorType};
use std::sync::{Arc, Barrier};
use std::thread;

#[cfg(feature = "prepared-plan")]
use fastnn::backend::cpu::CpuBackend;
#[cfg(feature = "prepared-plan")]
use fastnn::backend::executor::GraphExecutor;
#[cfg(feature = "prepared-plan")]
use fastnn::backend::prepared::prepare_executable_plan;
#[cfg(feature = "prepared-plan")]
use fastnn::ir::TensorValue;

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

#[cfg(feature = "prepared-plan")]
#[test]
fn prepared_execution_is_repeatable_across_concurrent_executors() {
    let mut graph = ComputeGraph::new();
    let input = graph.add_node(Opcode::Input, vec![], vector_type());
    let constant_values = vec![2.0_f32; 16];
    let constant = graph.add_constant(TensorValue::Data {
        bytes: bytemuck::cast_slice(&constant_values).to_vec(),
        tensor_type: vector_type(),
    });
    let output = graph.add_node(Opcode::Add, vec![input, constant], vector_type());
    graph.set_inputs(vec![input]);
    graph.set_outputs(vec![output]);

    let compiler = GraphExecutor::new(CpuBackend);
    let (plan, memory_plan, graph) = compiler
        .compile_with_plan_and_quantize(graph, None, None)
        .unwrap();
    let graph = Arc::new(graph);
    let prepared = Arc::new(prepare_executable_plan(&plan).unwrap());
    let input_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
    let input_bytes = bytemuck::cast_slice(&input_values);
    let expected = {
        let mut baseline_plan = plan.clone();
        let mut baseline_executor = GraphExecutor::new(CpuBackend);
        baseline_executor
            .execute_prepared_fallback(
                &graph,
                &mut baseline_plan,
                &memory_plan,
                &[input_bytes],
                &prepared,
            )
            .unwrap()
            .remove(0)
    };
    let expected = Arc::new(expected);
    let plan = Arc::new(plan);
    let memory_plan = Arc::new(memory_plan);
    let barrier = Arc::new(Barrier::new(8));

    let mut workers = Vec::new();
    for _ in 0..8 {
        let graph = Arc::clone(&graph);
        let mut plan = (*plan).clone();
        let memory_plan = Arc::clone(&memory_plan);
        let prepared = Arc::clone(&prepared);
        let expected = Arc::clone(&expected);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            let input_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
            let input_bytes = bytemuck::cast_slice(&input_values);
            barrier.wait();
            for _ in 0..100 {
                let mut executor = GraphExecutor::new(CpuBackend);
                let outputs = executor
                    .execute_prepared_fallback(
                        &graph,
                        &mut plan,
                        &memory_plan,
                        &[input_bytes],
                        &prepared,
                    )
                    .unwrap();
                assert_eq!(outputs[0], *expected);
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
}
