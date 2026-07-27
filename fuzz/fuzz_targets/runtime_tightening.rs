#![no_main]

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::tighten_slices;
use fastnn::backend::{Backend, ExecutablePlan};
use fastnn::compiler::passes::memory_planning::plan_memory;
use fastnn::compiler::plan::MemoryPlan;
use fastnn::ir::{ComputeGraph, DimExpr, IrDType, Opcode, ShapeEnv, TensorType};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

fn artifacts() -> &'static (ComputeGraph, ExecutablePlan, MemoryPlan) {
    static ARTIFACTS: OnceLock<(ComputeGraph, ExecutablePlan, MemoryPlan)> = OnceLock::new();
    ARTIFACTS.get_or_init(|| {
        let mut graph = ComputeGraph::new();
        let tensor_type = TensorType::new(
            vec![DimExpr::Symbol("batch".to_owned()), DimExpr::Known(4)],
            IrDType::F32,
        );
        let input = graph.add_node(Opcode::Input, vec![], tensor_type.clone());
        let output = graph.add_node(Opcode::Relu, vec![input], tensor_type);
        graph.set_inputs(vec![input]);
        graph.set_outputs(vec![output]);
        let memory_plan = plan_memory(&graph).unwrap();
        let executable = CpuBackend.compile(&graph, &memory_plan).unwrap();
        (graph, executable, memory_plan)
    })
}

fuzz_target!(|data: &[u8]| {
    let Some(bytes) = data.get(..8) else {
        return;
    };
    let batch = u64::from_le_bytes(bytes.try_into().unwrap());
    let mut shape_env = ShapeEnv::new();
    shape_env.try_bind("batch", batch).unwrap();

    let (graph, executable, memory_plan) = artifacts();
    let Ok(tightened_memory) = memory_plan.try_tighten(graph, &shape_env) else {
        return;
    };
    let mut tightened_executable = executable.clone();
    if tighten_slices(
        &mut tightened_executable,
        memory_plan,
        &tightened_memory,
        graph,
    )
    .is_ok()
    {
        let _ = tightened_memory.validate();
        let _ = tightened_executable.validate();
    }
});
