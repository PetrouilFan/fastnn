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
use fastnn::tensor::{ir_to_dtype, Tensor};

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
