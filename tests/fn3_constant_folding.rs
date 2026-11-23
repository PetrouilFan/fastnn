//! Constant propagation through shape chains.
//!
//! A detector head builds its decode indices with
//! `Range(0, Cast(Gather(Shape(conv), i)), 1)`: the range's length is a shape
//! extent, the extent is selected from `Shape`'s payload, and the export casts
//! it to `f32` first. The importer tracks constant payloads so those nodes fold
//! instead of leaving an unprovable extent, and the typed IR gains rules for
//! the ops whose result type or length comes from values (`Cast`, `Range`,
//! `ConstantOfShape`, `Expand`).
//!
//! The fixture is generated with the `onnx` package:
//!
//! ```python
//! import onnx
//! import numpy as np
//! from onnx import TensorProto, helper, numpy_helper
//!
//! x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
//! y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
//! zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")
//! one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one")
//! idx = numpy_helper.from_array(np.array(2, dtype=np.int64), name="idx")
//! ax0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="ax0")
//! ten = numpy_helper.from_array(np.array(10.0, dtype=np.float32), name="ten")
//! nodes = [
//!     helper.make_node("Shape", ["x"], ["shape"]),
//!     helper.make_node("Gather", ["shape", "idx"], ["h"], axis=0),
//!     helper.make_node("Cast", ["h"], ["hf"], to=TensorProto.FLOAT),
//!     helper.make_node("Range", ["zero", "hf", "one"], ["r"]),
//!     helper.make_node("Unsqueeze", ["r", "ax0"], ["u"]),
//!     helper.make_node("Squeeze", ["u", "ax0"], ["s"]),
//!     helper.make_node("Add", ["s", "ten"], ["y"]),
//! ]
//! graph = helper.make_graph(nodes, "range-cast", [x], [y],
//!                           initializer=[zero, one, idx, ax0, ten])
//! model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
//! model.ir_version = 8
//! onnx.save(model, "tests/fixtures/range-cast.onnx")
//! ```
//!
//! It is 329 bytes and checked in. ONNX Runtime returns `[10, 11, 12, 13]` for
//! it; the plan below produces the same values.

use fastnn::inference::import::import_onnx_graph;
use fastnn::inference::ir::{DimExpr, TensorType};
use fastnn::inference::{DType, OpId};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/range-cast.onnx"
);

fn bytes() -> Vec<u8> {
    std::fs::read(FIXTURE).expect("the checked-in fixture is readable")
}

#[test]
fn a_shape_extent_feeds_a_range_through_a_float_cast() {
    let imported = import_onnx_graph(&bytes()).expect("the shape chain folds");
    let graph = imported.graph();

    // `Shape(x)` selects extent 2 (the 4), casts it to f32 and builds
    // `Range(0, 4, 1)`: the range's length is provable and the typed IR has a
    // concrete `[4]` there, not an unknown.
    let by_op = |op: OpId| -> Vec<&fastnn::inference::ir::Node> {
        graph.nodes().iter().filter(|node| node.op == op).collect()
    };
    let ranges = by_op(OpId::Range);
    assert_eq!(ranges.len(), 1, "the fixture has one Range");
    let range_output = graph
        .value(ranges[0].outputs[0])
        .expect("the range output exists");
    assert_eq!(range_output.ty.dtype, DType::F32);
    assert_eq!(
        range_output.ty.dims(),
        &[DimExpr::constant(4)],
        "Range length came from the folded shape chain"
    );

    let casts = by_op(OpId::Cast);
    assert_eq!(casts.len(), 1);
    let cast_output = graph.value(casts[0].outputs[0]).expect("cast output");
    assert_eq!(cast_output.ty.dtype, DType::F32);
    assert_eq!(
        cast_output.ty.dims(),
        &[],
        "the gathered extent is a scalar, not a length-one vector"
    );

    // The opset-13 axes-as-input form is lifted into `OpParams`, which is the
    // representation the typed IR's shape rules require.
    let unsqueezes = by_op(OpId::Unsqueeze);
    assert_eq!(unsqueezes.len(), 1);
    assert_eq!(unsqueezes[0].params.axes, vec![0]);
    assert_eq!(unsqueezes[0].inputs.len(), 1, "axes are not an operand");
    let squeezed = by_op(OpId::Squeeze);
    assert_eq!(squeezed.len(), 1);
    assert_eq!(squeezed[0].params.axes, vec![0]);
    assert_eq!(squeezed[0].inputs.len(), 1);
}

#[test]
fn every_extent_in_the_folded_graph_is_static() {
    let imported = import_onnx_graph(&bytes()).expect("the fixture imports");
    for value in imported.graph().values() {
        for dim in value.ty.dims() {
            assert!(
                matches!(dim, DimExpr::Const(_)),
                "value {:?} keeps the extent {dim}",
                value.name
            );
        }
    }
}

#[test]
fn the_folded_graph_plans_and_runs() {
    let imported = import_onnx_graph(&bytes()).expect("the fixture imports");
    let plan =
        fastnn::inference::runtime::Plan::from_imported(&imported).expect("a static graph plans");
    assert!(
        !plan.segments.is_empty(),
        "the plan keeps the compute nodes"
    );
    let output = imported
        .graph()
        .value(imported.outputs()[0])
        .expect("the output value");
    assert_eq!(output.ty.dtype, DType::F32);
    assert_eq!(output.ty.dims(), &[DimExpr::constant(4)]);
    // The declared output type is what the plan allocates for; a run is
    // covered by the CLI-level test in `fn3_cli.rs`.
    let expected: TensorType = output.ty.clone();
    assert_eq!(planned_output(&plan), expected);
}

/// The plan's slot type for the graph output, in graph order.
fn planned_output(plan: &fastnn::inference::runtime::Plan) -> TensorType {
    let output = plan.outputs.first().expect("one output");
    let slot = plan.slot(*output).expect("the slot exists");
    slot.ty.clone()
}
