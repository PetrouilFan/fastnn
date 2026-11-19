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
//! The public planner retains the copy for external slot bindings. The
//! resident session folds it because it owns the contiguous token binding.
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
use fastnn::fn3::model::{Plan, Planner, SlotRole};
use fastnn::fn3::{
    CpuBackend, DType, DeviceId, Fn3Error, LiveDims, OpId, OpParams, Shape, Strides, Tensor,
    TensorMeta,
};

const VOCAB: u64 = 4;
const HIDDEN: u64 = 3;
const SEQ: u64 = 13;
/// Deliberately includes every row and repeats, so a permutation cannot pass.
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

/// A reshape of `[1, SEQ]` indices feeding a host-validated `Gather`.
fn synthetic_import(reshape_dims: [u64; 2]) -> ModelImport {
    synthetic_import_with_declared_shape(reshape_dims, reshape_dims)
}

fn synthetic_import_with_declared_shape(
    reshape_dims: [u64; 2],
    declared_dims: [u64; 2],
) -> ModelImport {
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
            &declared_dims,
            ValueOrigin::Node(0),
        ),
        value(
            "logits",
            DType::F32,
            &[reshape_dims[0], reshape_dims[1], HIDDEN],
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
    let target_bytes: Vec<u8> = reshape_dims
        .iter()
        .flat_map(|&value| (value as i64).to_le_bytes())
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
fn public_reshape_copies_strided_inputs_for_both_execution_entry_points() {
    let import = synthetic_import([1, SEQ]);
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
    let plan = planner.plan(&envelope, live_dims()).expect("plan");

    assert!(plan.folded_node(0).is_none());
    let reshape = &plan.instructions()[0];
    assert_eq!(reshape.op, OpId::Reshape);
    let input = plan.input("input_ids").expect("input").slot() as usize;
    let reshaped = reshape.outputs[0] as usize;
    let gather = gather_instruction(&plan);
    assert_eq!(gather.inputs[1] as usize, reshaped);
    assert_ne!(input, reshaped);

    let padded: Vec<_> = TOKENS.iter().flat_map(|&token| [token, i64::MAX]).collect();
    let storage = Tensor::from_i64(shape(&[2 * SEQ]), &padded).expect("padded tokens");
    let meta = TensorMeta::strided(
        shape(&[1, SEQ]),
        Strides::from_elements(vec![2 * SEQ as i64, 2]),
        0,
        DType::I64,
        DeviceId::Cpu,
    )
    .expect("strided token metadata");
    let tokens = Tensor::from_storage(storage.storage().clone(), meta).expect("strided tokens");
    assert!(!tokens.as_view().expect("token view").is_contiguous());
    let weights = import.to_model_weights().expect("weights");
    for use_execution in [false, true] {
        let mut slots = plan.materialise(&weights, DeviceId::Cpu).expect("slots");
        slots[input] = Some(tokens.clone());
        if use_execution {
            plan.execution()
                .execute(&backend, &mut slots)
                .expect("execution().execute");
        } else {
            plan.execute(&backend, &mut slots).expect("Plan::execute");
        }
        assert_eq!(
            slots[reshaped].as_ref().unwrap().as_i64_slice().unwrap(),
            &TOKENS
        );
        assert_eq!(
            slots[gather.outputs[0] as usize]
                .as_ref()
                .unwrap()
                .as_f32_slice()
                .unwrap(),
            expected_output()
        );
    }
}

#[test]
fn identity_reshape_rejects_a_conflicting_declared_shape() {
    let import = synthetic_import_with_declared_shape([1, SEQ], [SEQ, 1]);
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
    let result = planner.plan(&envelope, live_dims());
    assert!(
        matches!(&result, Err(Fn3Error::InvalidShape { context: "specialize::shape", detail })
            if detail.contains("/Reshape_output_0") && detail.contains("the two must agree")),
        "expected a declared reshape shape error, got {result:?}"
    );
}

#[test]
fn shape_changing_reshape_with_equal_element_count_keeps_its_own_slot() {
    let import = synthetic_import([SEQ, 1]);
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
    let plan = planner.plan(&envelope, live_dims()).expect("plan");

    assert_eq!(plan.instructions().len(), 2);
    assert!(plan.folded_node(0).is_none());
    let reshape = &plan.instructions()[0];
    assert_eq!(reshape.op, OpId::Reshape);
    let input_slot = plan.input("input_ids").expect("graph input").slot();
    assert_eq!(reshape.inputs[0], input_slot);
    let reshaped_slot = reshape.outputs[0];
    assert_ne!(reshaped_slot, input_slot);
    let reshaped = plan.slot(reshaped_slot).expect("reshape output slot");
    assert_eq!(reshaped.name(), "/Reshape_output_0");
    assert_eq!(reshaped.shape().dims(), &[SEQ, 1]);
    assert_eq!(
        reshaped.role(),
        &SlotRole::InstructionOutput { instruction: 0 }
    );
    assert_eq!(gather_instruction(&plan).inputs[1], reshaped_slot);
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
    use fastnn::fn3::import::{KvInterface, ValueId};
    use fastnn::fn3::model::{ModelWeights, PreparedModel, Session};

    // Supply the decoder interface needed by a real prefill. The cache has
    // one lane and both key and value use the same synthetic present tensor.
    fn session_import(declared_dims: [u64; 2]) -> ModelImport {
        let import = synthetic_import_with_declared_shape([1, SEQ], declared_dims);
        let mut geometry = import.geometry().clone();
        let (graph, weights) = import.into_parts();
        let mut values = graph.values().to_vec();
        let mut nodes = graph.nodes().to_vec();
        let mut inputs = graph.inputs().to_vec();
        let mut outputs = graph.outputs().to_vec();
        for (name, dtype, dims) in [
            ("position_ids", DType::I64, vec![1, SEQ]),
            ("attention_mask", DType::I64, vec![1, SEQ]),
            ("past_key", DType::F32, vec![1, 1, 0, 1]),
            ("past_value", DType::F32, vec![1, 1, 0, 1]),
        ] {
            inputs.push(ValueId::new(values.len() as u32));
            values.push(value(name, dtype, &dims, ValueOrigin::GraphInput));
        }
        let past_key = inputs[3];
        let past_value = inputs[4];
        outputs.push(past_value);
        let chunk = ValueId::new(values.len() as u32);
        let cache_dims = [1, 1, SEQ, 1];
        values.push(value(
            "cache_chunk",
            DType::F32,
            &cache_dims,
            ValueOrigin::Weight(WeightId::new(2)),
        ));
        let present = ValueId::new(values.len() as u32);
        values.push(value(
            "present",
            DType::F32,
            &cache_dims,
            ValueOrigin::Node(2),
        ));
        nodes.push(GraphNode {
            onnx_index: 2,
            name: "Concat".into(),
            onnx_op: "Concat".into(),
            op: OpId::Concat,
            inputs: vec![past_key, chunk],
            outputs: vec![present],
            params: params(&[2]),
        });
        outputs.push(present);
        for (name, input) in [("positions", inputs[1]), ("mask", inputs[2])] {
            let output = ValueId::new(values.len() as u32);
            let node = nodes.len() as u32;
            values.push(value(name, DType::F32, &[1, SEQ], ValueOrigin::Node(node)));
            nodes.push(GraphNode {
                onnx_index: node,
                name: name.into(),
                onnx_op: "Cast".into(),
                op: OpId::Cast,
                inputs: vec![input],
                outputs: vec![output],
                params: OpParams::new(),
            });
            outputs.push(output);
        }
        geometry.kv.push(KvInterface {
            past_key,
            past_value,
            present_key: present,
            present_value: present,
        });
        let mut weights: Vec<_> = weights.iter().cloned().collect();
        weights.push(
            WeightTensor::new(
                "cache_chunk",
                DType::F32,
                shape(&cache_dims),
                vec![0; SEQ as usize * 4],
                WeightSource::Initializer,
            )
            .expect("cache chunk"),
        );
        ModelImport::from_parts(
            GraphIr::from_parts(values, nodes, inputs, outputs),
            WeightSet::new(weights).expect("weights"),
            geometry,
            ModelProvenance {
                ir_version: 8,
                opset: OpsetImport {
                    domain: String::new(),
                    version: 14,
                },
                producer: "synthetic session test".into(),
                producer_version: "1".into(),
                graph_name: "synthetic_reshape_session".into(),
            },
        )
    }

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

        let import = session_import([1, SEQ]);
        let cpu = CpuBackend::new();
        let planner = Planner::new(&import, &cpu);
        let envelope = planner.envelope(1, SEQ, 0, SEQ).expect("envelope");
        let plan = planner.plan(&envelope, live_dims()).expect("plan");
        let device = backend.device_id();
        let weights = retag(&import.to_model_weights().expect("weights"), device);
        let model = PreparedModel::new(
            plan.execution().clone(),
            weights,
            device,
            envelope,
            Some(import.geometry().kv_cache_config(SEQ, 1)),
        )
        .expect("prepared model");
        let mut session =
            Session::with_runtime(&model, &import, backend.as_ref()).expect("session");
        let tokens = TOKENS.map(|token| token as u32);
        let logits = session
            .prefill(&tokens)
            .expect("resident Gather needs host indices");
        let expected = expected_output();
        let values = logits.as_f32_slice().expect("f32 logits");
        assert_eq!(values, &expected[expected.len() - HIDDEN as usize..]);
        assert_eq!(logits.shape().dims(), &[1, 1, HIDDEN]);
        assert_eq!(
            session.slot_allocations(),
            4,
            "Gather, Concat and two casts, but no Reshape"
        );
        let summary = backend.residency().summary();
        assert_eq!(
            summary.input_readbacks, 0,
            "a batched resident step read a device tensor back on the host"
        );
        let mut rejected =
            Session::with_runtime(&model, &import, backend.as_ref()).expect("session");
        let mut invalid = tokens;
        invalid[SEQ as usize - 1] = VOCAB as u32;
        let result = rejected.prefill(&invalid);
        assert!(
            matches!(
                result,
                Err(Fn3Error::OutOfBounds {
                    what: "gather index",
                    index: VOCAB,
                    len: VOCAB
                })
            ),
            "expected Gather's typed bounds error, got {result:?}"
        );

        let malformed_import = session_import([SEQ, 1]);
        let mut malformed =
            Session::with_runtime(&model, &malformed_import, backend.as_ref()).expect("session");
        let result = malformed.prefill(&tokens);
        assert!(
            matches!(&result, Err(Fn3Error::InvalidShape { context: "specialize::shape", detail })
                if detail.contains("/Reshape_output_0") && detail.contains("the two must agree")),
            "expected a declared shape error from the session planner, got {result:?}"
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
}
