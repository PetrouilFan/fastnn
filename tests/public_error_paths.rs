use fastnn::autograd;
use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::Backend;
use fastnn::compiler::passes::dead_code_elimination::eliminate_dead_code;
use fastnn::compiler::passes::memory_planning::plan_memory;
use fastnn::compiler::passes::quantization::quantize_weights;
use fastnn::compiler::passes::shape_inference::infer_shapes;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::{ComputeGraph, DimExpr, IrDType, Opcode, TensorType, TensorValue};
use fastnn::storage::{DType, Device};
use fastnn::tensor::{dtype_to_ir, ir_to_dtype, Tensor};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

#[test]
fn item_on_non_scalar_tensor_returns_shape_error() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let error = tensor
        .item()
        .expect_err("item on a multi-element tensor must fail");
    assert!(error.to_string().contains("requires one element"));
}

#[test]
fn conflicting_shape_binding_returns_error() {
    let mut env = fastnn::ir::ShapeEnv::new();
    env.try_bind("N", 2).unwrap();
    let error = env
        .try_bind("N", 3)
        .expect_err("conflicting shape binding must fail");
    assert!(error.contains("inconsistently"));
}

#[test]
fn unsupported_numpy_dtype_returns_error() {
    let tensor = Tensor::zeros(vec![1], DType::Bool, Device::Cpu);
    let error = tensor
        .to_numpy()
        .expect_err("unsupported NumPy dtype must fail");
    assert!(error.to_string().contains("does not support"));
}

#[test]
fn invalid_graph_quantization_width_returns_error() {
    let graph = GraphBuilder::new();
    let input = graph.input(&[4], IrDType::F32);

    let signed_error = graph
        .quantize(&input, 3)
        .expect_err("invalid signed quantization width must fail");
    assert!(signed_error.to_string().contains("must be 4 or 8"));

    let unsigned_error = graph
        .quantize_unsigned(&input, 16)
        .expect_err("invalid unsigned quantization width must fail");
    assert!(unsigned_error.to_string().contains("must be 4 or 8"));
}

#[test]
fn runtime_activation_dtype_cannot_escape_to_eager_tensor() {
    let error = ir_to_dtype(IrDType::I8)
        .expect_err("runtime activation dtype must not become an eager tensor dtype");
    assert!(error.to_string().contains("not a Tensor-level dtype"));
}

#[test]
fn unsupported_eager_dtype_cannot_enter_executable_ir() {
    let error =
        dtype_to_ir(DType::F64).expect_err("unsupported eager dtype must not enter executable IR");
    assert!(error.to_string().contains("not supported in executable IR"));
}

#[test]
fn malformed_reshape_returns_structured_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);

    let mismatch = tensor
        .try_reshape(vec![3])
        .expect_err("mismatched reshape must fail");
    assert!(mismatch.to_string().contains("Shape error"));

    let multiple_inferred = tensor
        .try_view(vec![-1, -1])
        .expect_err("multiple inferred dimensions must fail");
    assert!(multiple_inferred
        .to_string()
        .contains("infer one dimension"));

    let overflow = tensor
        .try_reshape(vec![i64::MAX, 2])
        .expect_err("overflowing shape product must fail");
    assert!(overflow.to_string().contains("overflow"));
}

#[test]
fn invalid_dimension_reordering_returns_structured_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let transpose = tensor
        .try_transpose(0, 2)
        .expect_err("out-of-range transpose must fail");
    assert!(transpose.to_string().contains("out of range"));

    let duplicate = tensor
        .try_permute(vec![0, 0])
        .expect_err("duplicate permutation dimensions must fail");
    assert!(duplicate.to_string().contains("not a permutation"));

    let wrong_rank = tensor
        .try_permute(vec![0])
        .expect_err("wrong-rank permutation must fail");
    assert!(wrong_rank.to_string().contains("requires 2 dimensions"));
}

#[test]
fn invalid_expand_and_unsqueeze_return_structured_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);

    let unsqueeze = tensor
        .try_unsqueeze(3)
        .expect_err("out-of-range unsqueeze must fail");
    assert!(unsqueeze.to_string().contains("out of range"));

    let rank = tensor
        .try_expand(vec![2])
        .expect_err("lower-rank expansion must fail");
    assert!(rank.to_string().contains("target has 1 dimensions"));

    let incompatible = tensor
        .try_expand(vec![3, 3])
        .expect_err("non-singleton expansion must fail");
    assert!(incompatible.to_string().contains("only size-1 dimensions"));

    let negative = tensor
        .try_expand(vec![-1, 2])
        .expect_err("negative expansion dimension must fail");
    assert!(negative.to_string().contains("non-negative"));
}

#[test]
fn invalid_cat_stack_and_repeat_return_structured_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![1, 3]);

    let empty = Tensor::try_cat(&[], 0).expect_err("empty cat must fail");
    assert!(empty.to_string().contains("at least one"));

    let mismatch =
        Tensor::try_cat(&[a.clone(), b.clone()], 0).expect_err("cat shape mismatch must fail");
    assert!(mismatch.to_string().contains("differ at dimension"));

    let stack = Tensor::try_stack(&[a.clone(), b], 0).expect_err("stack shape mismatch must fail");
    assert!(stack.to_string().contains("same shape"));

    let repeat_rank = a
        .try_repeat(&[2])
        .expect_err("too few repeat dimensions must fail");
    assert!(repeat_rank.to_string().contains("at least 2"));

    let repeat_negative = a
        .try_repeat(&[1, -1])
        .expect_err("negative repeat must fail");
    assert!(repeat_negative.to_string().contains("non-negative"));
}

#[test]
fn invalid_slice_returns_structured_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let dimension = tensor
        .try_slice(2, 0, 1, 1)
        .expect_err("out-of-range slice dimension must fail");
    assert!(dimension.to_string().contains("out of range"));

    let zero_step = tensor
        .try_slice(0, 0, 1, 0)
        .expect_err("zero slice step must fail");
    assert!(zero_step.to_string().contains("must be positive"));

    let reversed = tensor
        .try_slice(0, 2, 1, 1)
        .expect_err("reversed slice range must fail");
    assert!(reversed.to_string().contains("reversed"));

    let clamped = tensor
        .try_slice(0, i64::MIN, i64::MAX, 1)
        .expect("extreme bounds should clamp safely");
    assert_eq!(clamped.shape(), vec![2, 2]);
}

#[test]
fn invalid_argmax_and_gather_return_structured_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let indices = Tensor::from_vec(vec![0.0], vec![1]);

    let argmax = tensor
        .try_argmax(Some(2))
        .expect_err("out-of-range argmax dimension must fail");
    assert!(argmax.to_string().contains("out of range"));

    let gather = tensor
        .try_gather(-3, &indices)
        .expect_err("out-of-range gather axis must fail");
    assert!(gather.to_string().contains("out of range"));

    let invalid_indices = Tensor::zeros(vec![1], DType::I64, Device::Cpu);
    let dtype = tensor
        .try_gather(0, &invalid_indices)
        .expect_err("unsupported gather index dtype must fail");
    assert!(dtype.to_string().contains("requires F32"));
}

#[test]
fn backward_rejects_mismatched_gradient_shape() {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad_(true);
    let loss = input.sum(0, false);
    let wrong_gradient = Tensor::from_vec(vec![1.0], vec![1]);

    let error = autograd::backward(&loss, Some(wrong_gradient))
        .expect_err("mismatched backward gradient must fail");
    assert!(error.to_string().contains("gradient shape"));
}

#[test]
fn symbolic_input_size_conflict_returns_dispatch_error_instead_of_panicking() {
    let g = GraphBuilder::new();
    let x = g.input_with_dims(
        &[DimExpr::Symbol("N".into()), DimExpr::Known(4)],
        IrDType::F32,
    );
    let _y = g.input_with_dims(&[DimExpr::Symbol("N".into())], IrDType::F32);

    let x_bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y_bytes = f32_bytes(&[9.0, 10.0, 11.0]);

    let err = g
        .compile_and_execute(&[&x], CpuBackend, &[&x_bytes, &y_bytes])
        .expect_err("conflicting symbolic dimensions should return an error");

    let message = err.to_string();
    assert!(
        message.contains("shape env:"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("symbol 'N'") || message.contains("symbol \"N\""),
        "unexpected error: {message}"
    );
}

#[test]
fn malformed_quantized_weight_metadata_returns_error_instead_of_panicking() {
    let mut graph = ComputeGraph::new();

    let act_scale = 1.0f32;
    let act_zp = 0.0f32;
    let act_i8: Vec<i8> = vec![1, 2, 3, 4];
    let mut payload = Vec::new();
    payload.extend_from_slice(&act_scale.to_le_bytes());
    payload.extend_from_slice(&act_zp.to_le_bytes());
    for &v in &act_i8 {
        payload.push(v as u8);
    }
    let act_tt = TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::I8);
    let act_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: payload,
            tensor_type: act_tt.clone(),
        }),
        vec![],
        act_tt,
    );

    let w_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];
    let w_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32);
    let w_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: bytemuck::cast_slice(&w_data).to_vec(),
            tensor_type: w_tt.clone(),
        }),
        vec![],
        w_tt,
    );

    let mm_id = graph.add_node(
        Opcode::MatMul,
        vec![act_id, w_id],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
    );
    graph.set_inputs(vec![]);
    graph.set_outputs(vec![mm_id]);

    infer_shapes(&mut graph).unwrap();
    quantize_weights(&mut graph, 8, true, None).unwrap();
    eliminate_dead_code(&mut graph);

    let matmul_node = graph.get_node(mm_id).unwrap().clone();
    let weight_id = matmul_node.inputs[1];
    let weight_node = graph.get_node_mut(weight_id).unwrap();
    weight_node.output_type.dtype = IrDType::I8Scaled {
        scales: vec![],
        dequant_offsets: vec![],
    };
    if let Opcode::Constant(TensorValue::Data { tensor_type, .. }) = &mut weight_node.opcode {
        tensor_type.dtype = IrDType::I8Scaled {
            scales: vec![],
            dequant_offsets: vec![],
        };
    }

    let mem = plan_memory(&graph).expect("memory planning should succeed");
    let mut plan = CpuBackend
        .compile(&graph, &mem)
        .expect("compile should succeed");
    let mut executor = GraphExecutor::new(CpuBackend);
    let err = executor
        .execute(&graph, &mut plan, &mem, &[])
        .expect_err("malformed quantized metadata should return an error");

    let message = err.to_string();
    assert!(
        message.contains("quant") || message.contains("scale") || message.contains("zero"),
        "unexpected error: {message}"
    );
}
