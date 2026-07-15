use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::{Backend, BufferSlice, ExecutablePlan, Instruction};
use fastnn::compiler::{AllocSlot, MemoryPlan};
use fastnn::ir::ShapeEnv;
use std::collections::HashMap;
use std::process::Command;

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
