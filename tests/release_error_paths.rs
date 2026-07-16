use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::prepared::PersistentPreparedWeights;
use fastnn::backend::{Backend, BufferSlice, ExecutablePlan, Instruction};
use fastnn::compiler::passes::memory_planning::plan_memory;
use fastnn::compiler::{AllocSlot, MemoryPlan};
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::{IrDType, Opcode, ShapeEnv};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

const HELPER_ENV: &str = "FASTNN_RELEASE_ERROR_HELPER";

fn run_malformed_dispatch() {
    let backend = CpuBackend;
    let plan = ExecutablePlan {
        instructions: vec![Instruction::CallKernel {
            kernel_name: "adam_update_f32".into(),
            input_slices: vec![],
            output_slice: BufferSlice::new(0, 4),
            secondary_output_slice: None,
            params: vec![],
            param_dims: None,
            node_id: None,
            weight_meta: None,
        }],
        arena_size: 4,
        levels: vec![0],
    };
    let arena = backend
        .try_allocate_arena(plan.arena_size)
        .expect("small helper arena should allocate");
    assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());

    let mut slots = HashMap::new();
    slots.insert(
        0,
        AllocSlot {
            offset: usize::MAX,
            size: 2,
            node_id: 0,
            output_index: 0,
        },
    );
    let memory = MemoryPlan {
        total_size: usize::MAX,
        slots,
        inputs: vec![0],
        secondary_slots: HashMap::new(),
        outputs: vec![0],
        tightened_params: HashMap::new(),
    };
    assert!(memory.validate().is_err());

    let persistent_plan = ExecutablePlan {
        instructions: vec![Instruction::CallKernel {
            kernel_name: "matmul".into(),
            input_slices: vec![BufferSlice::new(0, 4), BufferSlice::new(4, 4)],
            output_slice: BufferSlice::new(8, 4),
            secondary_output_slice: None,
            params: vec![1, 2, 1],
            param_dims: None,
            node_id: Some(0),
            weight_meta: None,
        }],
        arena_size: 12,
        levels: vec![0],
    };
    let arena = backend
        .try_allocate_arena(persistent_plan.arena_size)
        .expect("small persistent helper arena should allocate");
    let mut persistent = PersistentPreparedWeights::new();
    assert!(persistent.insert((4, 4), Arc::new(vec![1.0])));
    assert!(backend
        .dispatch_with_persistent_view(
            &persistent_plan,
            &arena,
            &ShapeEnv::new(),
            Some(&persistent),
        )
        .is_err());

    let builder = GraphBuilder::new();
    let input = builder.input(&[2, 2], IrDType::F32);
    let output = builder.softmax(&input, 1);
    let mut graph = builder.to_graph();
    graph.set_outputs(vec![output.node_id()]);
    graph
        .nodes
        .iter_mut()
        .find(|node| matches!(node.opcode, Opcode::Softmax))
        .expect("softmax node should exist")
        .attrs
        .insert("axis".into(), "-3".into());
    let memory = plan_memory(&graph).expect("small graph should plan");
    assert!(backend.compile(&graph, &memory).is_err());

    let builder = GraphBuilder::new();
    let left = builder.input(&[1, 2], IrDType::F32);
    let right = builder.input(&[1, 2], IrDType::F32);
    let output = builder.concat(&[&left, &right], 1);
    let mut graph = builder.to_graph();
    graph.set_outputs(vec![output.node_id()]);
    graph
        .nodes
        .iter_mut()
        .find(|node| matches!(node.opcode, Opcode::Concat))
        .expect("concat node should exist")
        .attrs
        .insert("axis".into(), "9".into());
    let memory = plan_memory(&graph).expect("small graph should plan");
    assert!(backend.compile(&graph, &memory).is_err());
}

#[test]
fn malformed_executable_plan_exits_normally() {
    if std::env::var_os(HELPER_ENV).is_some() {
        run_malformed_dispatch();
        return;
    }

    let executable = std::env::current_exe().expect("test executable path should be available");
    let output = Command::new(executable)
        .arg("--exact")
        .arg("malformed_executable_plan_exits_normally")
        .arg("--nocapture")
        .env(HELPER_ENV, "1")
        .output()
        .expect("release-error helper should start");

    assert!(
        output.status.success(),
        "malformed public input terminated the subprocess abnormally: status={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
