//! FN3-26 (#72) regression: an identity `Reshape` feeding a `Gather`.
//!
//! The trained GPT-2's first embedding lookup is
//! `Gather(transformer.wte.weight, Reshape(input_ids, [1, 13]))`. Until the
//! planner folded that reshape, the plan put the reshape output in a
//! device-resident slot, and `Gather` validates its indices on the host:
//! `view.as_i64_slice()` on a resident tensor returns
//! `Fn3Error::NotImplemented` before the shader is recorded. The tiny model was
//! unaffected because its `Gather` reads `input_ids` directly.
//!
//! This test builds the same shape of graph synthetically, with no ONNX file
//! and no model weights, so the fold is proven independently of the pinned
//! trained graph:
//!
//! ```text
//! input_ids [1, 13] i64 ──► Reshape([1, 13]) ──► Gather(wte, axis 0) ──► [1, 13, 3] f32
//! ```
//!
//! The planner test proves the structural fold in any feature configuration.
//! The Vulkan test executes the plan with the resident session's allocation and
//! batching policy (`allocate_resident` for every instruction output,
//! `begin_step(..., batch = true)`), asserts the exact output, and asserts that
//! the index operand is the host `input_ids` slot with no input readback.
//!
//! ```bash
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//!   cargo test --features vulkan --test fn3_vulkan_identity_reshape \
//!   -- --nocapture --test-threads=1
//! ```

use fastnn::fn3::import::{
    DecoderGeometry, Dim, GraphIr, GraphNode, ModelImport, ModelProvenance, OpsetImport,
    PositionEmbedding, ValueInfo, ValueOrigin, ValueShape,
};
use fastnn::fn3::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};
use fastnn::fn3::model::{FoldKind, Plan, Planner, SlotRole};
use fastnn::fn3::{CpuBackend, DType, LiveDims, OpId, OpParams, Shape};

const VOCAB: u64 = 4;
const HIDDEN: u64 = 3;
const SEQ: u64 = 13;
/// Deliberately includes every row and repeats, so a permutation cannot pass.
#[cfg(feature = "vulkan")]
const TOKENS: [i64; SEQ as usize] = [0, 3, 1, 0, 2, 3, 0, 1, 3, 2, 0, 1, 3];

fn shape(dims: &[u64]) -> Shape {
    Shape::from_dims(dims.iter().copied())
}

fn value(name: &str, dtype: DType, dims: &[u64], origin: ValueOrigin) -> ValueInfo {
    ValueInfo {
        name: name.to_string(),
        dtype,
        shape: ValueShape::Dims(dims.iter().map(|&extent| Dim::Known(extent)).collect()),
        origin,
    }
}

fn params(axes: &[i64]) -> OpParams {
    OpParams::new().with_axes(axes.iter().copied())
}

/// A graph with exactly one thing to prove: `Reshape(input_ids)` is an
/// identity in front of a host-validated `Gather`.
fn synthetic_import() -> ModelImport {
    let values = vec![
        value("input_ids", DType::I64, &[1, SEQ], ValueOrigin::GraphInput),
        value(
            "table",
            DType::F32,
            &[VOCAB, HIDDEN],
            ValueOrigin::Weight(WeightId::new(0)),
        ),
        value(
            "reshape_target",
            DType::I64,
            &[2],
            ValueOrigin::Weight(WeightId::new(1)),
        ),
        value(
            "/Reshape_output_0",
            DType::I64,
            &[1, SEQ],
            ValueOrigin::Node(0),
        ),
        value(
            "logits",
            DType::F32,
            &[1, SEQ, HIDDEN],
            ValueOrigin::Node(1),
        ),
    ];
    let nodes = vec![
        GraphNode {
            onnx_index: 0,
            name: "Reshape".to_string(),
            onnx_op: "Reshape".to_string(),
            op: OpId::Reshape,
            inputs: vec![
                fastnn::fn3::import::ValueId::new(0),
                fastnn::fn3::import::ValueId::new(2),
            ],
            outputs: vec![fastnn::fn3::import::ValueId::new(3)],
            params: params(&[]),
        },
        GraphNode {
            onnx_index: 1,
            name: "Gather".to_string(),
            onnx_op: "Gather".to_string(),
            op: OpId::Gather,
            inputs: vec![
                fastnn::fn3::import::ValueId::new(1),
                fastnn::fn3::import::ValueId::new(3),
            ],
            outputs: vec![fastnn::fn3::import::ValueId::new(4)],
            params: params(&[0]),
        },
    ];
    let graph = GraphIr::from_parts(
        values,
        nodes,
        vec![fastnn::fn3::import::ValueId::new(0)],
        vec![fastnn::fn3::import::ValueId::new(4)],
    );

    let table: Vec<f32> = (0..VOCAB)
        .flat_map(|row| {
            let row = row as f32;
            [row, row + 0.5, row + 1.0]
        })
        .collect();
    let table_bytes = table.iter().flat_map(|value| value.to_le_bytes()).collect();
    let target_bytes: Vec<u8> = [1i64, SEQ as i64]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let weights = WeightSet::new(vec![
        WeightTensor::new(
            "table",
            DType::F32,
            shape(&[VOCAB, HIDDEN]),
            table_bytes,
            WeightSource::Initializer,
        )
        .expect("table weight"),
        WeightTensor::new(
            "reshape_target",
            DType::I64,
            shape(&[2]),
            target_bytes,
            WeightSource::Initializer,
        )
        .expect("target weight"),
    ])
    .expect("unique weight names");

    let geometry = DecoderGeometry {
        layers: 1,
        hidden_size: HIDDEN as u32,
        mlp_hidden_size: HIDDEN as u32,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        vocab_size: VOCAB as u32,
        positions: PositionEmbedding::Learned {
            weight: WeightId::new(0),
            max_positions: 32,
        },
        layer_norm_epsilon: None,
        token_embedding: WeightId::new(0),
        logits: fastnn::fn3::import::ValueId::new(4),
        kv: Vec::new(),
    };
    ModelImport::from_parts(
        graph,
        weights,
        geometry,
        ModelProvenance {
            ir_version: 8,
            opset: OpsetImport {
                domain: String::new(),
                version: 14,
            },
            producer: "fn3 synthetic identity-reshape test".to_string(),
            producer_version: "1".to_string(),
            graph_name: "synthetic_identity_reshape".to_string(),
        },
    )
}

fn live_dims() -> LiveDims {
    LiveDims::new(1, SEQ, 0, 1, 1, 1).expect("synthetic dims")
}

#[cfg(feature = "vulkan")]
fn expected_output() -> Vec<f32> {
    TOKENS
        .iter()
        .flat_map(|&token| {
            let token = token as f32;
            [token, token + 0.5, token + 1.0]
        })
        .collect()
}

/// The same identity, but as a graph output: the fold must not remove the
/// value's own slot binding.
fn reshape_output_import() -> ModelImport {
    let values = vec![
        value("x", DType::F32, &[1, SEQ], ValueOrigin::GraphInput),
        value(
            "reshape_target",
            DType::I64,
            &[2],
            ValueOrigin::Weight(WeightId::new(0)),
        ),
        value("y", DType::F32, &[1, SEQ], ValueOrigin::Node(0)),
    ];
    let nodes = vec![GraphNode {
        onnx_index: 0,
        name: "Reshape".to_string(),
        onnx_op: "Reshape".to_string(),
        op: OpId::Reshape,
        inputs: vec![
            fastnn::fn3::import::ValueId::new(0),
            fastnn::fn3::import::ValueId::new(1),
        ],
        outputs: vec![fastnn::fn3::import::ValueId::new(2)],
        params: params(&[]),
    }];
    let graph = GraphIr::from_parts(
        values,
        nodes,
        vec![fastnn::fn3::import::ValueId::new(0)],
        vec![fastnn::fn3::import::ValueId::new(2)],
    );
    let target_bytes: Vec<u8> = [1i64, SEQ as i64]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let weights = WeightSet::new(vec![WeightTensor::new(
        "reshape_target",
        DType::I64,
        shape(&[2]),
        target_bytes,
        WeightSource::Initializer,
    )
    .expect("target weight")])
    .expect("weight set");
    let geometry = DecoderGeometry {
        layers: 1,
        hidden_size: SEQ as u32,
        mlp_hidden_size: SEQ as u32,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        vocab_size: 2,
        positions: PositionEmbedding::Learned {
            weight: WeightId::new(0),
            max_positions: 32,
        },
        layer_norm_epsilon: None,
        token_embedding: WeightId::new(0),
        logits: fastnn::fn3::import::ValueId::new(2),
        kv: Vec::new(),
    };
    ModelImport::from_parts(
        graph,
        weights,
        geometry,
        ModelProvenance {
            ir_version: 8,
            opset: OpsetImport {
                domain: String::new(),
                version: 14,
            },
            producer: "fn3 synthetic identity-reshape output test".to_string(),
            producer_version: "1".to_string(),
            graph_name: "synthetic_reshape_output".to_string(),
        },
    )
}

fn gather_instruction(plan: &Plan) -> &fastnn::fn3::Instruction {
    let gather = plan
        .instructions()
        .iter()
        .find(|instruction| instruction.op == OpId::Gather)
        .expect("the plan has a Gather instruction");
    assert_eq!(
        plan.slot(gather.inputs[0]).expect("table slot").name(),
        "table",
        "the Gather must read the synthetic table"
    );
    gather
}

#[test]
fn identity_reshape_is_folded_and_gather_reads_the_graph_input_slot() {
    let import = synthetic_import();
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
    let plan = planner.plan(&envelope, live_dims()).expect("plan");

    // The reshape emits no instruction and has no slot of its own ...
    assert_eq!(
        plan.instructions().len(),
        1,
        "the identity Reshape must not survive as an instruction"
    );
    assert!(
        plan.instructions()
            .iter()
            .all(|instruction| instruction.op != OpId::Reshape),
        "a Reshape instruction survived the identity fold"
    );
    let folded = plan
        .folded_node(0)
        .expect("ONNX node 0 (Reshape) must be recorded as folded");
    assert_eq!(folded.op, OpId::Reshape);
    assert_eq!(
        folded.kind,
        FoldKind::Identity,
        "the fold is an identity alias, not constant evaluation"
    );
    assert!(
        plan.slot_of("/Reshape_output_0").is_none(),
        "the folded value must not own a slot"
    );

    // ... so Gather's index operand is the host `input_ids` slot itself.
    let gather = gather_instruction(&plan);
    let index = plan.slot(gather.inputs[1]).expect("index slot");
    assert_eq!(
        index.role(),
        &SlotRole::GraphInput,
        "Gather indices must be host-visible; got role {:?}",
        index.role()
    );
    assert_eq!(index.name(), "input_ids");
    assert_eq!(index.shape().dims(), &[1, SEQ]);
    assert_eq!(
        index.name(),
        plan.input("input_ids").expect("graph input").name()
    );
}

#[test]
fn graph_output_reshape_keeps_its_own_slot() {
    let import = reshape_output_import();
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
    let plan = planner.plan(&envelope, live_dims()).expect("plan");
    assert_eq!(
        plan.instructions().len(),
        1,
        "the Reshape must execute because its value is a graph output"
    );
    assert_eq!(plan.instructions()[0].op, OpId::Reshape);
    assert!(
        plan.folded_node(0).is_none(),
        "a graph-output Reshape must not be folded"
    );
    let output = plan.output("y").expect("graph output");
    assert_eq!(
        plan.slot(output.slot()).expect("output slot").role(),
        &SlotRole::InstructionOutput { instruction: 0 },
        "a graph output must keep its own instruction-output binding"
    );
}

#[cfg(feature = "vulkan")]
mod resident {
    use super::*;

    use std::sync::Arc;

    use fastnn::fn3::backend::vulkan::instance::validation_counts;
    use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
    use fastnn::fn3::device::DeviceRequest;
    use fastnn::fn3::model::ModelWeights;
    use fastnn::fn3::{Backend, DeviceId, Tensor, TensorMeta};

    fn require_physical() -> bool {
        std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
    }

    fn pinned_uuid() -> Option<String> {
        std::env::var("FN3_VK_DEVICE_UUID")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
    }

    fn open_physical() -> Option<Arc<VulkanBackend>> {
        let index = adapters()
            .iter()
            .find(|adapter| {
                adapter.has_compute_queue()
                    && !adapter.is_software()
                    && pinned_uuid()
                        .as_deref()
                        .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                        .unwrap_or(true)
            })
            .map(|adapter| adapter.index)?;
        let backend = VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: Some(index) },
            VulkanConfig::validated(),
        )
        .expect("open the physical Vulkan device");
        assert!(!backend.is_software(), "the resident arm needs a real GPU");
        Some(backend)
    }

    fn retag(weights: &ModelWeights, device: DeviceId) -> ModelWeights {
        let mut named = Vec::with_capacity(weights.len());
        for index in 0..weights.len() {
            let tensor = weights.get(index).expect("weight");
            let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
                .expect("weight metadata");
            named.push((
                weights.name(index).expect("weight name").to_string(),
                Tensor::from_storage(tensor.storage().clone(), meta).expect("retag"),
            ));
        }
        ModelWeights::with_names(named).expect("weights share one device")
    }

    /// Allocate every slot as a resident session does: graph inputs stay for
    /// the caller to bind, weights share the model's storage, and every
    /// instruction output is a device buffer.
    fn materialise_resident(
        plan: &Plan,
        weights: &ModelWeights,
        backend: &VulkanBackend,
    ) -> Vec<Option<Tensor>> {
        let device = backend.device_id();
        let mut slots = Vec::with_capacity(plan.slots().len());
        for slot in plan.slots() {
            let tensor = match slot.role() {
                SlotRole::GraphInput => None,
                SlotRole::Weight { id } => {
                    let tensor = weights.get(id.index()).expect("weight").clone();
                    assert_eq!(tensor.dtype(), slot.dtype());
                    assert_eq!(tensor.shape(), slot.shape());
                    Some(tensor)
                }
                SlotRole::Constant { index } => {
                    let tensor = plan.constants()[*index as usize]
                        .to_tensor()
                        .expect("constant tensor");
                    let meta = TensorMeta::contiguous(slot.shape().clone(), slot.dtype(), device)
                        .expect("constant metadata");
                    Some(
                        Tensor::from_storage(tensor.storage().clone(), meta)
                            .expect("re-tagged constant"),
                    )
                }
                SlotRole::InstructionOutput { .. } => {
                    if slot.shape().numel().expect("slot elements") > 0 {
                        Some(
                            backend
                                .allocate_resident(slot.shape(), slot.dtype())
                                .expect("resident output"),
                        )
                    } else {
                        None
                    }
                }
            };
            slots.push(tensor);
        }
        slots
    }

    #[test]
    fn resident_gather_reads_host_input_ids_after_the_fold() {
        let Some(backend) = open_physical() else {
            assert!(
                !require_physical(),
                "FN3_VK_REQUIRE_PHYSICAL=1 but no physical compute device with the pinned UUID {:?} \
                 is present",
                pinned_uuid()
            );
            println!("SKIP: no physical Vulkan compute device");
            return;
        };
        println!(
            "[fn3-72][synthetic] device=\"{}\" uuid={} validation={}",
            backend.adapter().name,
            backend.adapter().uuid_string(),
            backend.validation_enabled()
        );
        assert!(
            backend.validation_enabled(),
            "this arm requires the validation layer"
        );
        let before = validation_counts();

        let import = synthetic_import();
        let planner = Planner::new(&import, backend.as_ref());
        let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
        let plan = planner.plan(&envelope, live_dims()).expect("plan");
        let gather_position = plan
            .instructions()
            .iter()
            .position(|instruction| instruction.op == OpId::Gather)
            .expect("the plan has a Gather instruction");
        let gather = &plan.instructions()[gather_position];
        let index_slot = gather.inputs[1];
        let index = plan.slot(index_slot).expect("index slot");

        let device = backend.device_id();
        let weights = retag(&import.to_model_weights().expect("weights"), device);
        backend
            .as_ref()
            .prepare_persistent(&persistent_weights(&weights))
            .expect("persistent weights");
        let mut slots = materialise_resident(&plan, &weights, &backend);

        let tokens = Tensor::from_i64(shape(&[1, SEQ]), &TOKENS).expect("token tensor");
        let tokens = Tensor::from_storage(
            tokens.storage().clone(),
            TensorMeta::contiguous(shape(&[1, SEQ]), DType::I64, device).expect("token metadata"),
        )
        .expect("retag tokens");
        slots[plan
            .input("input_ids")
            .expect("input_ids is a graph input")
            .slot() as usize] = Some(tokens);

        // The resident session's execution shape: every output is a device
        // buffer and the whole plan records into one submission.
        backend
            .as_ref()
            .begin_step("synthetic-prefill", true)
            .expect("begin step");
        let executed = plan.execute(&*backend, &mut slots);
        let closed = backend.as_ref().end_step();
        match executed {
            Ok(()) => {}
            Err(error) => {
                panic!(
                    "resident plan execution failed: {error}; the Gather index slot is {:?} \
                     (device_resident={})",
                    index.role(),
                    slots[index_slot as usize]
                        .as_ref()
                        .map(Tensor::is_device_resident)
                        .unwrap_or(false)
                );
            }
        }
        closed.expect("end step");

        // The execution succeeded; now prove *why*: the reshape was folded and
        // the Gather index operand is the host graph input, not a device slot.
        assert_eq!(
            plan.instructions().len(),
            1,
            "after the identity fold the synthetic plan contains only the Gather"
        );
        assert_eq!(
            index.role(),
            &SlotRole::GraphInput,
            "the resident plan must read host indices, not a device slot"
        );
        assert_eq!(index.name(), "input_ids");
        assert_eq!(
            plan.folded_node(0).expect("Reshape fold").kind,
            FoldKind::Identity
        );

        let summary = backend.residency().summary();
        assert_eq!(
            summary.input_readbacks, 0,
            "a batched resident step read a device tensor back on the host"
        );
        assert_eq!(summary.executions, 1);
        assert_eq!(summary.proven_device_executions, 1);

        let downloaded = slots[gather.outputs[0] as usize]
            .as_ref()
            .expect("Gather wrote its output")
            .to_host()
            .expect("download the explicit result");
        let values = downloaded.as_f32_slice().expect("f32 logits");
        assert_eq!(
            values,
            expected_output().as_slice(),
            "the gathered rows are not the exact table rows in token order"
        );
        assert_eq!(
            downloaded.shape().dims(),
            &[1, SEQ, HIDDEN],
            "Gather output shape"
        );

        let after = validation_counts();
        assert_eq!(
            after, before,
            "validation messages changed during the synthetic resident step"
        );
        println!(
            "[fn3-72][synthetic] exact output {:?}, input_readbacks={}, validation errors {} \
             warnings {}",
            &values[..HIDDEN as usize],
            summary.input_readbacks,
            after.0,
            after.1
        );
    }

    fn persistent_weights(weights: &ModelWeights) -> Vec<&Tensor> {
        (0..weights.len())
            .filter_map(|index| weights.get(index))
            .collect()
    }
}
