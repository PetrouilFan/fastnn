use fastnn::compiler::pipeline::CompilerPipeline;
use fastnn::ir::{ComputeGraph, DimExpr, GraphKind, IrDType, Opcode, TensorType};
use fastnn::types::CompileTarget;

#[test]
fn compiler_emits_structured_pass_report() {
    let mut graph = ComputeGraph::with_kind(GraphKind::Inference);
    let input = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
    );
    graph.inputs.push(input);
    graph.outputs.push(input);
    graph.required_nodes.insert(input);

    let compiled = CompilerPipeline::new(CompileTarget::Native, None)
        .run(graph)
        .expect("native inference graph should compile");

    assert_eq!(compiled.report.graph_kind, GraphKind::Inference);
    assert_eq!(compiled.report.initial_nodes, 1);
    assert_eq!(compiled.report.final_nodes, compiled.graph.nodes.len());
    assert_eq!(
        compiled.report.passes.first().unwrap().name,
        "shape inference"
    );
    assert_eq!(
        compiled.report.passes.last().unwrap().name,
        "memory planning"
    );
    assert!(compiled
        .report
        .passes
        .iter()
        .any(|pass| pass.name == "representation validation"));
}
