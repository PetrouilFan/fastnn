import numpy as np
import pytest

import fastnn as fnn


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper


def _run_ort_and_fastnn(tmp_path, nodes, initializers, input_array, output_names):
    graph = helper.make_graph(
        nodes,
        "shape_tensor_inputs",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_array.shape))],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in output_names],
        initializer=[numpy_helper.from_array(value, name=name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "model.onnx"
    fnn_path = tmp_path / "model.fnn"
    onnx.save(model, onnx_path)

    expected = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        output_names, {"X": input_array}
    )
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward(
        {"X": fnn.tensor(input_array, list(input_array.shape))}
    )
    return expected, [actual[name].numpy() for name in output_names]


def test_runtime_shape_uses_each_input_extent(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Shape", ["X"], ["Y"], name="shape")],
        "runtime_shape",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-shape.onnx"
    fnn_path = tmp_path / "runtime-shape.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))

    for tokens in (2, 5):
        x = np.arange(tokens * 4, dtype=np.float32).reshape(tokens, 4)
        expected = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        ).run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual.astype(np.int64), expected)


def test_runtime_constant_of_shape_matches_onnxruntime(tmp_path):
    fill = numpy_helper.from_array(np.asarray([2.5], dtype=np.float32))
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["X"], ["shape"], name="shape"),
            helper.make_node(
                "ConstantOfShape",
                ["shape"],
                ["Y"],
                name="constant_of_shape",
                value=fill,
            ),
        ],
        "runtime_constant_of_shape",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-constant-of-shape.onnx"
    fnn_path = tmp_path / "runtime-constant-of-shape.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for tokens in (2, 5):
        x = np.arange(tokens * 4, dtype=np.float32).reshape(tokens, 4)
        expected = session.run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_slice_constant_tensor_inputs_match_onnxruntime(tmp_path):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    initializers = {
        "starts": np.asarray([-2], dtype=np.int64),
        "ends": np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
        "axes": np.asarray([1], dtype=np.int64),
        "steps": np.asarray([1], dtype=np.int64),
    }
    nodes = [
        helper.make_node(
            "Slice",
            ["X", "starts", "ends", "axes", "steps"],
            ["Y"],
            name="slice",
        )
    ]

    expected, actual = _run_ort_and_fastnn(tmp_path, nodes, initializers, x, ["Y"])
    np.testing.assert_array_equal(actual[0], expected[0])


def test_slice_multiple_axes_fails_explicitly(tmp_path):
    x_shape = [2, 3, 4]
    initializers = {
        "starts": np.asarray([0, 1], dtype=np.int64),
        "ends": np.asarray([2, 3], dtype=np.int64),
        "axes": np.asarray([0, 1], dtype=np.int64),
        "steps": np.asarray([1, 1], dtype=np.int64),
    }
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["X", "starts", "ends", "axes", "steps"],
                ["Y"],
                name="slice",
            )
        ],
        "multi_axis_slice",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(value, name=name)
            for name, value in initializers.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "multi-axis.onnx"
    fnn_path = tmp_path / "multi-axis.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))

    with pytest.raises(RuntimeError, match="requires exactly one start/end/axis/step tuple"):
        fnn.build_model_from_fnn(str(fnn_path))


def test_squeeze_unsqueeze_constant_axes_match_onnxruntime(tmp_path):
    x = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    initializers = {
        "squeeze_axes": np.asarray([1], dtype=np.int64),
        "unsqueeze_axes": np.asarray([0, 2], dtype=np.int64),
    }
    nodes = [
        helper.make_node("Squeeze", ["X", "squeeze_axes"], ["squeezed"], name="squeeze"),
        helper.make_node(
            "Unsqueeze",
            ["squeezed", "unsqueeze_axes"],
            ["Y"],
            name="unsqueeze",
        ),
    ]

    expected, actual = _run_ort_and_fastnn(tmp_path, nodes, initializers, x, ["Y"])
    np.testing.assert_array_equal(actual[0], expected[0])


def test_split_constant_sizes_match_onnxruntime(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    initializers = {"split_sizes": np.asarray([2, 4], dtype=np.int64)}
    nodes = [
        helper.make_node(
            "Split",
            ["X", "split_sizes"],
            ["left", "right"],
            name="split",
            axis=1,
        )
    ]

    expected, actual = _run_ort_and_fastnn(
        tmp_path, nodes, initializers, x, ["left", "right"]
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
