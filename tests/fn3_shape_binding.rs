//! Input-shape binding: a dynamic export becomes plannable.
//!
//! A stock export declares its input as `["batch", 3, "height", "width"]`, and
//! a `Reshape` inside the graph multiplies those symbols. The typed IR may keep
//! a symbol (it is a legal extent), but the planner cannot: every slot in a plan
//! has one concrete shape. `import_onnx_graph_with_shapes` replaces a graph
//! input's declared extents with the shape the model will run at, so everything
//! downstream follows from the model's own rules.
//!
//! The fixture is a one-node `Relu` whose input is declared `["batch", 4]`.
//! Regenerate it with the `onnx` package:
//!
//! ```python
//! import onnx
//! from onnx import TensorProto, helper
//! x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
//! y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
//! graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "shape-binding", [x], [y])
//! model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
//! model.ir_version = 8
//! onnx.save(model, "tests/fixtures/shape-binding.onnx")
//! ```
//!
//! It is 107 bytes and checked in, so the test needs no Python and no model
//! checkout.

use fastnn::inference::cli::{run, EXIT_MODEL, EXIT_OK};
use fastnn::inference::import::{
    declared_inputs, import_onnx_graph, import_onnx_graph_with_shapes, ImportProblemKind,
};
use fastnn::inference::ir::DimExpr;
use fastnn::inference::runtime::Plan;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/shape-binding.onnx"
);

/// `Resize(x, roi, scales)` with a zero `roi` and a symbolic batch axis.
///
/// ONNX defines roi's values only for `tf_crop_and_resize`, so a zero
/// placeholder describes the same resize as an omitted one. Generated with the
/// `onnx` package: roi `zeros(8)`, scales `[1, 1, 2, 2]`, `asymmetric`/`floor`.
const RESIZE_ROI: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/resize-roi.onnx"
);

fn bytes() -> Vec<u8> {
    std::fs::read(FIXTURE).expect("the checked-in fixture is readable")
}

fn bind(name: &str, extents: &[u64]) -> (String, Vec<u64>) {
    (name.to_string(), extents.to_vec())
}

#[test]
fn the_declared_input_is_reported_before_any_binding() {
    let inputs = declared_inputs(&bytes()).expect("the fixture decodes");
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].name, "x");
    assert_eq!(inputs[0].shape, "[batch, 4]");
}

#[test]
fn a_symbolic_extent_imports_but_cannot_be_planned() {
    let imported = import_onnx_graph(&bytes()).expect("a symbol is a legal typed extent");
    let dims = imported
        .graph()
        .value(imported.inputs()[0])
        .expect("the input value exists")
        .ty
        .dims()
        .to_vec();
    assert!(
        matches!(dims[0], DimExpr::Symbol(_)),
        "the symbol must survive the unbound import: {dims:?}"
    );
    let error = Plan::from_imported(&imported).expect_err("a plan needs one concrete shape");
    assert!(
        error.to_string().contains("batch") || error.to_string().contains("static"),
        "the refusal must name the problem: {error}"
    );
}

#[test]
fn binding_the_input_makes_the_graph_plannable() {
    let imported = import_onnx_graph_with_shapes(&bytes(), &[bind("x", &[2, 4])])
        .expect("a bound input imports");
    let value = imported
        .graph()
        .value(imported.inputs()[0])
        .expect("the input value exists");
    assert_eq!(
        value.ty.dims(),
        &[DimExpr::constant(2), DimExpr::constant(4)]
    );
    let plan = Plan::from_imported(&imported).expect("a bound graph plans");
    let slot = plan.input_slot("x").expect("the input is in the plan");
    let slot = plan.slot(slot).expect("the slot exists");
    assert_eq!(
        slot.ty.dims(),
        &[DimExpr::constant(2), DimExpr::constant(4)]
    );
}

#[test]
fn a_binding_for_a_name_the_model_does_not_declare_is_a_typed_problem() {
    let error = import_onnx_graph_with_shapes(&bytes(), &[bind("images", &[1, 3, 8, 8])])
        .expect_err("the model has no `images` input");
    assert!(
        error.contains(|kind| matches!(kind, ImportProblemKind::ShapeBinding { .. })),
        "{error}"
    );
    assert!(error.to_string().contains("no graph input"), "{error}");
}

#[test]
fn a_binding_with_the_wrong_rank_is_a_typed_problem() {
    let error = import_onnx_graph_with_shapes(&bytes(), &[bind("x", &[1, 2, 4])])
        .expect_err("the model declares rank 2");
    assert!(
        error.contains(|kind| matches!(kind, ImportProblemKind::ShapeBinding { .. })),
        "{error}"
    );
    assert!(error.to_string().contains("rank 2"), "{error}");
}

#[test]
fn binding_one_input_twice_is_rejected() {
    let error = import_onnx_graph_with_shapes(&bytes(), &[bind("x", &[2, 4]), bind("x", &[3, 4])])
        .expect_err("one input has one shape");
    assert!(error.to_string().contains("more than once"), "{error}");
}

#[test]
fn the_cli_prepares_and_runs_the_bound_model() {
    let directory =
        std::env::temp_dir().join(format!("fastnn-shape-binding-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("scratch");
    let artifact = directory.join("bound.fnn3");
    let artifact = artifact.to_str().expect("utf-8 path");

    // Without a binding the doctor says what to bind.
    let mut out = Vec::new();
    let mut err = Vec::new();
    let argv: Vec<String> = vec!["doctor".to_string(), FIXTURE.to_string()];
    let code = run(&argv, &mut out, &mut err);
    let stdout = String::from_utf8_lossy(&out).into_owned();
    assert_eq!(code, EXIT_MODEL, "stdout:\n{stdout}");
    assert!(stdout.contains("inputs: x: [batch, 4]"), "{stdout}");

    // With the binding, prepare and run.
    let mut out = Vec::new();
    let mut err = Vec::new();
    let argv: Vec<String> = vec![
        "prepare".to_string(),
        "--onnx".to_string(),
        FIXTURE.to_string(),
        "--out".to_string(),
        artifact.to_string(),
        "--shape".to_string(),
        "x=2,4".to_string(),
    ];
    let code = run(&argv, &mut out, &mut err);
    let stdout = String::from_utf8_lossy(&out).into_owned();
    let stderr = String::from_utf8_lossy(&err).into_owned();
    assert_eq!(code, EXIT_OK, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("bound:    x=2,4"), "{stdout}");

    let mut out = Vec::new();
    let mut err = Vec::new();
    let argv: Vec<String> = vec![
        "run".to_string(),
        artifact.to_string(),
        "--input".to_string(),
        "x=ones".to_string(),
    ];
    let code = run(&argv, &mut out, &mut err);
    let stdout = String::from_utf8_lossy(&out).into_owned();
    assert_eq!(code, EXIT_OK, "stdout:\n{stdout}");
    assert!(stdout.contains("output y: shape [2x4]"), "{stdout}");

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn a_zero_roi_placeholder_does_not_block_a_resize() {
    let bytes = std::fs::read(RESIZE_ROI).expect("the checked-in fixture is readable");
    let imported = import_onnx_graph_with_shapes(&bytes, &[bind("x", &[1, 3, 4, 4])])
        .expect("a zero roi is the same resize as an omitted one");
    assert_eq!(imported.graph().node_count(), 1);
    assert_eq!(
        imported.graph().nodes()[0].op,
        fastnn::inference::OpId::Resize
    );
    let plan = Plan::from_imported(&imported).expect("the bound resize plans");
    let slot = plan.output_slot("y").expect("the output is in the plan");
    let slot = plan.slot(slot).expect("the slot exists");
    assert_eq!(
        slot.ty.dims(),
        &[
            DimExpr::constant(1),
            DimExpr::constant(3),
            DimExpr::constant(8),
            DimExpr::constant(8)
        ]
    );
}
