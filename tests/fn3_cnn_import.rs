//! A real Ultralytics YOLOv8 graph imports into the typed IR, fully shaped.
//!
//! The two files are outside this repository, so a test that cannot find one
//! prints why and returns: a missing checkout is not a pass, but it is also not
//! a failure of the code under test. There is deliberately no `#[ignore]`, so
//! the test runs in every environment that has the model.
//!
//! The expected shapes and node counts below were read from the files with
//! `onnx.load` / `onnx.shape_inference.infer_shapes` (opset 12, IR 7):
//!
//! | file | input | output | nodes |
//! |---|---|---|---|
//! | `yolov8n.onnx` | `[1, 3, 320, 320]` | `[1, 84, 2100]` | 262 |
//! | `yolov8s.onnx` | `[1, 3, 640, 640]` | `[1, 84, 8400]` | 262 |

use std::path::Path;

use fastnn::inference::import::{import_onnx_graph, ImportedGraph};
use fastnn::inference::OpId;

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";
const YOLOV8S: &str = "/home/petrouil/Downloads/yolov8s.onnx";

/// The ONNX node census of both files.
const ONNX_CENSUS: [(&str, usize); 17] = [
    ("Conv", 64),
    ("Mul", 60),
    ("Sigmoid", 58),
    ("Constant", 22),
    ("Concat", 17),
    ("Add", 9),
    ("Split", 8),
    ("Reshape", 8),
    ("MaxPool", 3),
    ("Resize", 2),
    ("Transpose", 2),
    ("Div", 2),
    ("Slice", 2),
    ("Sub", 2),
    ("Softmax", 1),
    ("Shape", 1),
    ("Gather", 1),
];

fn load(path: &str) -> Option<ImportedGraph> {
    if !Path::new(path).exists() {
        eprintln!("{path} is not present on this machine; skipping the CNN import check");
        return None;
    }
    let bytes = std::fs::read(path).expect("the model is readable");
    match import_onnx_graph(&bytes) {
        Ok(imported) => Some(imported),
        Err(error) => panic!("{path} must import into the typed IR: {error}"),
    }
}

/// Assert everything both YOLO files must satisfy, then the file-specific size.
fn check(imported: &ImportedGraph, side: u64) -> (usize, usize) {
    let summary = imported.summary();
    assert_eq!(summary.onnx_nodes, 262, "ONNX node count");
    // Every Constant is folded into a weight, so it emits no typed node.
    let constant_nodes = summary.count_onnx("Constant");
    assert_eq!(constant_nodes, 22, "Constant nodes");
    assert_eq!(summary.initializers, 127, "initializers");
    assert_eq!(
        imported.weights().len(),
        127 + constant_nodes,
        "weight table"
    );
    assert_eq!(summary.fn3_nodes, 262 - constant_nodes, "typed node count");
    assert_eq!(
        imported.graph().node_count(),
        summary.fn3_nodes,
        "the typed graph holds every imported node"
    );
    for (op_type, count) in ONNX_CENSUS {
        assert_eq!(summary.count_onnx(op_type), count, "ONNX {op_type} count");
    }
    // The folded constants are still values, and every value must be shaped:
    // that is the typed IR's contract and the point of this phase.
    for value in imported.graph().values() {
        assert!(
            value.ty.is_static(),
            "value {:?} is not fully shaped: {}",
            value.name,
            value.ty
        );
    }
    // A graph output keeps its source name and its ONNX-reported type.
    let output = imported.lookup("output0").expect("output0 is a value");
    let ty = &imported.graph().value(output).expect("the id is valid").ty;
    assert_eq!(ty.dtype.to_string(), "f32");
    let dims: Vec<u64> = ty
        .dims()
        .iter()
        .map(|dim| dim.as_const().expect("the output is static"))
        .collect();
    assert_eq!(dims, vec![1, 84, side], "output0 shape");
    assert_eq!(imported.outputs(), &[output], "the graph has one output");
    assert_eq!(summary.count(OpId::Conv), 64);
    assert_eq!(summary.count(OpId::MaxPool), 3);
    assert_eq!(summary.count(OpId::Resize), 2);
    assert_eq!(summary.count(OpId::Sigmoid), 58);
    assert_eq!(summary.count(OpId::Split), 8);
    assert_eq!(summary.count(OpId::Slice), 2);
    // The 22 folded Constants are weights, not nodes; their payload is byte
    // accounted with the initializers.
    assert_eq!(summary.attribute_constants, 22);
    (summary.fn3_nodes, imported.graph().value_count())
}

#[test]
fn yolov8n_imports_into_the_typed_ir_with_static_shapes() {
    let Some(imported) = load(YOLOV8N) else {
        return;
    };
    let (nodes, values) = check(&imported, 2100);
    println!("[fn3cnn][measure] yolov8n: {nodes} typed nodes, {values} values");
}

#[test]
fn yolov8s_imports_into_the_typed_ir_with_static_shapes() {
    let Some(imported) = load(YOLOV8S) else {
        return;
    };
    let (nodes, values) = check(&imported, 8400);
    println!("[fn3cnn][measure] yolov8s: {nodes} typed nodes, {values} values");
}
