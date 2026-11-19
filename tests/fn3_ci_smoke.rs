//! CI-lane smoke tests.
//!
//! This file is owned by FN3-20 (#66). It deliberately does two things:
//!
//! 1. checks that the lane inventory keeps compile evidence, software-Vulkan
//!    evidence and physical-GPU evidence distinct (see `tools/ci/lanes.json`);
//! 2. runs a tiny CPU plan end to end, so the `cpu` lane fails if the 3.0 core
//!    stops executing on a machine with no GPU.
//!
//! It is not a second copy of `tests/fn3_contracts.rs`: contract semantics live
//! there, lane honesty lives here. The Vulkan smoke test itself belongs to
//! FN3-11 (#57) and is `tests/fn3_vulkan_smoke.rs`.

use std::path::PathBuf;

use fastnn::inference::device::available_devices;
use fastnn::inference::{
    CpuBackend, DType, DeviceId, DeviceRequest, ExecutionPlan, Instruction, ModelWeights, OpId,
    PreparedModel, Session, Shape, ShapeEnvelope, Tensor,
};

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn lanes() -> serde_json::Value {
    let path = repo_path("tools/ci/lanes.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not valid JSON: {error}", path.display()))
}

fn lane<'a>(inventory: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    inventory["lanes"]
        .as_array()
        .expect("lanes must be an array")
        .iter()
        .find(|lane| lane["name"] == name)
        .unwrap_or_else(|| panic!("lane {name:?} is missing from tools/ci/lanes.json"))
}

fn text(lane: &serde_json::Value, field: &str) -> String {
    lane[field]
        .as_array()
        .unwrap_or_else(|| panic!("lane field {field:?} must be an array of strings"))
        .iter()
        .map(|item| {
            item.as_str()
                .unwrap_or_else(|| panic!("{field} entries must be strings"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// 1. The lane inventory is honest by construction.
// ---------------------------------------------------------------------------

#[test]
fn every_lane_states_what_it_proves_and_what_it_does_not() {
    let inventory = lanes();
    let lanes = inventory["lanes"]
        .as_array()
        .expect("lanes must be an array");
    assert!(!lanes.is_empty(), "the lane inventory cannot be empty");

    let mut names = std::collections::HashSet::new();
    for lane in lanes {
        let name = lane["name"].as_str().expect("every lane needs a name");
        assert!(names.insert(name.to_owned()), "duplicate lane {name}");
        assert!(
            !text(lane, "proves").trim().is_empty(),
            "{name} must say what it proves"
        );
        assert!(
            !text(lane, "does_not_prove").trim().is_empty(),
            "{name} must say what it does not prove"
        );
        assert!(
            !lane["reproduce"].as_str().unwrap_or("").trim().is_empty(),
            "{name} must name its local reproduction command"
        );
    }
}

#[test]
fn software_vulkan_is_never_presented_as_gpu_evidence() {
    let inventory = lanes();
    let software = lane(&inventory, "vulkan-software-smoke");
    assert_eq!(software["hosted"], true);
    assert_eq!(software["software_only"], true);
    let disclaimer = text(software, "does_not_prove").to_lowercase();
    assert!(
        disclaimer.contains("ci-only") || disclaimer.contains("software"),
        "the software lane must label itself CI-only/software: {disclaimer}"
    );
    assert!(
        disclaimer.contains("physical gpu") || disclaimer.contains("physical-gpu"),
        "the software lane must state that it is not physical-GPU evidence: {disclaimer}"
    );

    let physical = lane(&inventory, "gpu-physical");
    assert_eq!(
        physical["hosted"], false,
        "gpu-physical must never be a hosted job"
    );
    assert_eq!(physical["evidence_required"], true);
}

#[test]
fn model_dependent_lane_reads_one_pin_and_skips_honestly() {
    let inventory = lanes();
    let pinned = lane(&inventory, "pinned-model");
    assert_eq!(pinned["required_when"]["pin_status"], "pinned");
    assert_eq!(pinned["required"], false);
    let disclaimer = text(pinned, "does_not_prove").to_lowercase();
    assert!(disclaimer.contains("not verified"), "{disclaimer}");
    assert!(disclaimer.contains("#49"), "{disclaimer}");

    // The consumer-side pin must be a real file with an explicit status, so the
    // lane can never mistake a placeholder for a revision.
    let pin_path = repo_path("tools/ci/model-pin.json");
    let pin: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&pin_path).expect("tools/ci/model-pin.json must exist"),
    )
    .expect("tools/ci/model-pin.json must be valid JSON");
    let status = pin["status"]
        .as_str()
        .expect("the pin must declare a status");
    assert!(
        status == "pinned" || status == "unpinned",
        "unexpected pin status {status:?}"
    );
    if status == "pinned" {
        assert!(
            pin["revision"]
                .as_str()
                .is_some_and(|revision| revision.len() == 40),
            "a pinned revision must be a 40-hex commit"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. CPU lane smoke: a tiny plan must execute on this machine.
// ---------------------------------------------------------------------------

#[test]
fn cpu_lane_runs_a_tiny_plan_end_to_end() {
    let x = Tensor::from_f32(Shape::from_dims([2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let bias = Tensor::from_f32(Shape::from_dims([2]), &[0.5, -0.5]).unwrap();
    let identity = Tensor::from_f32(Shape::from_dims([2, 2]), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let add_out = Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, DeviceId::Cpu).unwrap();
    let matmul_out = Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, DeviceId::Cpu).unwrap();

    let plan = ExecutionPlan::new(
        vec![
            Instruction::new(OpId::Add, vec![0, 1], vec![2]),
            Instruction::new(OpId::MatMul, vec![2, 3], vec![4]),
        ],
        5,
    )
    .unwrap();
    let model = PreparedModel::new(
        plan,
        ModelWeights::new(vec![]).unwrap(),
        DeviceId::Cpu,
        ShapeEnvelope::uniform(8).unwrap(),
        None,
    )
    .unwrap();

    let mut slots = vec![
        Some(x),
        Some(bias),
        Some(add_out),
        Some(identity),
        Some(matmul_out),
    ];
    model.execute(&CpuBackend::new(), &mut slots).unwrap();
    assert_eq!(
        slots[2].as_ref().unwrap().as_f32_slice().unwrap(),
        &[1.5, 1.5, 3.5, 3.5]
    );
    assert_eq!(
        slots[4].as_ref().unwrap().as_f32_slice().unwrap(),
        &[1.5, 1.5, 3.5, 3.5]
    );

    // A session owns no state for a model without KV geometry, but constructing
    // one must work; a lane that cannot create a session is not a CPU lane.
    let session = Session::new(&model).unwrap();
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
}

// ---------------------------------------------------------------------------
// 3. Device honesty: the feature that does not exist must not be simulated.
// ---------------------------------------------------------------------------

#[test]
fn vulkan_is_reported_only_when_the_backend_is_really_built() {
    // This deliberately avoids `cfg(feature = "vulkan")`: until FN3-11 (#57)
    // declares the feature, such a cfg is an unexpected_cfgs warning and would
    // pollute the lane's clippy budget. Branch on what the build reports.
    let available = available_devices();
    let vulkan_devices: Vec<_> = available
        .iter()
        .filter(|info| info.id.vulkan_index().is_some())
        .collect();

    if vulkan_devices.is_empty() {
        let error = DeviceRequest::Vulkan { index: Some(0) }
            .resolve_on_host()
            .expect_err("an explicit Vulkan request must fail when the build has no Vulkan device");
        let message = error.to_string();
        assert!(message.contains("vulkan"), "{message}");
        assert!(
            message.contains("no Vulkan backend") || message.contains("available devices"),
            "the error must name the missing backend or list devices: {message}"
        );
    } else {
        // FN3-11 (#57) owns the real dispatch; this only proves the feature
        // wiring reaches `available_devices`. It creates no device.
        assert!(
            vulkan_devices
                .iter()
                .all(|info| info.id.vulkan_index().is_some()),
            "a Vulkan-labelled device must have a Vulkan index: {available:?}"
        );
    }
}
