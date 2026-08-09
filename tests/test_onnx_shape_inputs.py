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


def test_scalar_shape_gather_unsqueeze_concat_matches_onnxruntime(tmp_path):
    """TinyLlama's RoPE shape chain must keep scalar Gather rank zero."""
    initializers = [
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "index"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "one"),
        numpy_helper.from_array(np.asarray([64], dtype=np.int64), "head_dim"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["X"], ["shape"], name="shape"),
            helper.make_node("Gather", ["shape", "index"], ["length"], name="gather", axis=0),
            helper.make_node("Unsqueeze", ["length", "axes"], ["length_1d"], name="unsqueeze"),
            helper.make_node("Concat", ["length_1d", "one", "head_dim"], ["Y"], name="concat", axis=0),
        ],
        "scalar_shape_gather_unsqueeze_concat",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", "tokens"])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [3])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "scalar-shape-chain.onnx"
    fnn_path = tmp_path / "scalar-shape-chain.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"batch": 1, "tokens": 16}
    )
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for tokens in (1, 5):
        x = np.zeros((1, tokens), dtype=np.float32)
        expected = ort_session.run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual.astype(np.int64), expected)


def test_unsqueeze_axis_two_then_expand_matches_onnxruntime(tmp_path):
    axes = numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axes")
    target = numpy_helper.from_array(
        np.asarray([1, 3, 3, 1, 4], dtype=np.int64), "target"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Unsqueeze", ["X", "axes"], ["expanded_rank"]),
            helper.make_node("Expand", ["expanded_rank", "target"], ["Y"]),
        ],
        "gqa_unsqueeze_expand",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 3, 1, 4])],
        initializer=[axes, target],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "gqa-unsqueeze-expand.onnx"
    fnn_path = tmp_path / "gqa-unsqueeze-expand.fnn"
    onnx.save(model, onnx_path)
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 4)
    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["Y"], {"X": values})[0]
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward({"X": fnn.tensor(values, list(values.shape))})["Y"].numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -2, -3])
def test_flatten_axes_match_onnxruntime(tmp_path, axis):
    rank = 3
    normalized = axis if axis >= 0 else axis + rank
    outer = int(np.prod((2, 3, 4)[:normalized], dtype=np.int64))
    inner = int(np.prod((2, 3, 4)[normalized:], dtype=np.int64))
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["X"], ["Y"], axis=axis)],
        f"flatten_axis_{axis}",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [outer, inner])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / f"flatten-axis-{axis}.onnx"
    fnn_path = tmp_path / f"flatten-axis-{axis}.fnn"
    onnx.save(model, onnx_path)
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["Y"], {"X": values})[0]
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward({"X": fnn.tensor(values, list(values.shape))})["Y"].numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", [4, -4])
def test_flatten_rejects_out_of_range_axis(tmp_path, axis):
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["X"], ["Y"], axis=axis)],
        "flatten_invalid_axis",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / f"flatten-invalid-axis-{axis}.onnx"
    fnn_path = tmp_path / f"flatten-invalid-axis-{axis}.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    with pytest.raises((RuntimeError, ValueError), match="[Ff]latten|axis"):
        fnn.build_model_from_fnn(str(fnn_path))


def test_bool_gather_matches_onnxruntime(tmp_path):
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero")
    indices = numpy_helper.from_array(np.asarray([1, 0], dtype=np.int64), "indices")
    graph = helper.make_graph(
        [
            helper.make_node("Less", ["X", "zero"], ["mask"], name="mask"),
            helper.make_node("Flatten", ["mask"], ["flat_mask"], name="flatten", axis=1),
            helper.make_node(
                "Gather", ["flat_mask", "indices"], ["Y"], name="bool_gather", axis=0
            ),
        ],
        "bool_gather",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [2, 2])],
        initializer=[zero, indices],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "bool-gather.onnx"
    fnn_path = tmp_path / "bool-gather.fnn"
    onnx.save(model, onnx_path)
    values = np.asarray([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["Y"], {"X": values})[0]
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward({"X": fnn.tensor(values, list(values.shape))})["Y"].numpy()
    np.testing.assert_array_equal(actual.astype(bool), expected)


def test_comparison_boolean_and_isnan_lowering_matches_onnxruntime(tmp_path):
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero")
    true_scalar = numpy_helper.from_array(np.asarray(True, dtype=np.bool_), "true_scalar")
    output_names = [
        "less_equal",
        "greater_equal",
        "both",
        "either",
        "scalar_and",
        "is_nan",
    ]
    graph = helper.make_graph(
        [
            helper.make_node("LessOrEqual", ["X", "zero"], ["less_equal"]),
            helper.make_node("GreaterOrEqual", ["X", "zero"], ["greater_equal"]),
            helper.make_node("And", ["less_equal", "greater_equal"], ["both"]),
            helper.make_node("Or", ["less_equal", "greater_equal"], ["either"]),
            helper.make_node("And", ["true_scalar", "less_equal"], ["scalar_and"]),
            helper.make_node("IsNaN", ["X"], ["is_nan"]),
        ],
        "comparison_boolean_isnan",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])],
        [helper.make_tensor_value_info(name, TensorProto.BOOL, [5]) for name in output_names],
        initializer=[zero, true_scalar],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "comparison-boolean-isnan.onnx"
    fnn_path = tmp_path / "comparison-boolean-isnan.fnn"
    onnx.save(model, onnx_path)
    values = np.asarray([-np.inf, -1.0, 0.0, np.inf, np.nan], dtype=np.float32)
    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(output_names, {"X": values})
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward({"X": fnn.tensor(values, [5])})
    for name, reference in zip(output_names, expected):
        np.testing.assert_array_equal(actual[name].numpy() != 0, reference)


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


def test_int64_max_slice_end_preserves_live_symbolic_extent(tmp_path):
    starts = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.asarray([np.iinfo(np.int64).max], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes")
    steps = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "steps")
    graph = helper.make_graph(
        [helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"], name="tail")],
        "slice_to_end",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", "tokens", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", "tail_tokens", 4])],
        initializer=[starts, ends, axes, steps],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "slice-to-end.onnx"
    fnn_path = tmp_path / "slice-to-end.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for tokens in (2, 5):
        executor = fnn.build_model_from_fnn(
            str(fnn_path),
            symbolic_dim_bounds={"batch": 1, "tokens": 8, "tail_tokens": 7},
        )
        values = np.arange(tokens * 4, dtype=np.float32).reshape(1, tokens, 4)
        expected = session.run(["Y"], {"X": values})[0]
        actual = executor.forward({"X": fnn.tensor(values, list(values.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_shape_tail_slice_with_negative_start_has_exact_extent(tmp_path):
    starts = numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(
        np.asarray([np.iinfo(np.int64).max], dtype=np.int64), "ends"
    )
    axes = numpy_helper.from_array(np.asarray([0], dtype=np.int64), "axes")
    steps = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "steps")
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["X"], ["shape"], name="shape"),
            helper.make_node(
                "Slice",
                ["shape", "starts", "ends", "axes", "steps"],
                ["Y"],
                name="last_extent",
            ),
        ],
        "shape_tail_slice",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 4, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [1])],
        initializer=[starts, ends, axes, steps],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "shape-tail-slice.onnx"
    fnn_path = tmp_path / "shape-tail-slice.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"tokens": 8}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for tokens in (1, 5):
        values = np.zeros((tokens, 4, 5), dtype=np.float32)
        expected = session.run(["Y"], {"X": values})[0]
        actual = executor.forward(
            {"X": fnn.tensor(values, list(values.shape))}
        )["Y"].numpy()
        np.testing.assert_array_equal(actual.astype(np.int64), expected)


def test_scalar_reduce_mean_keepdims_does_not_modulo_zero(tmp_path):
    nodes = [
        helper.make_node(
            "ReduceMean", ["X"], ["Y"], name="reduce_scalar", keepdims=1
        )
    ]
    expected, actual = _run_ort_and_fastnn(
        tmp_path, nodes, {}, np.asarray(3.5, dtype=np.float32), ["Y"]
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)


def test_bounded_batch_flatten_reshape_matches_onnxruntime(tmp_path):
    target = numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "target")
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["X", "target"], ["Y"], name="flatten")],
        "multisymbol_flatten",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", "tokens"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch_times_tokens"])],
        initializer=[target],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "multisymbol-flatten.onnx"
    fnn_path = tmp_path / "multisymbol-flatten.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"batch": 1, "tokens": 8, "batch_times_tokens": 8}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for shape in ((1, 3), (1, 4)):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        expected = session.run(["Y"], {"X": values})[0]
        actual = executor.forward({"X": fnn.tensor(values, list(shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_constant_inferred_reshape_preserves_live_symbolic_extent(tmp_path):
    target = numpy_helper.from_array(np.asarray([-1, 1], dtype=np.int64), "target")
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["X", "target"], ["Y"], name="reshape")],
        "constant_inferred_reshape",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 1])],
        initializer=[target],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "constant-inferred-reshape.onnx"
    fnn_path = tmp_path / "constant-inferred-reshape.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"tokens": 16}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for tokens in (1, 5):
        x = np.arange(tokens, dtype=np.float32)
        expected = session.run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, [tokens])})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (tokens, 1)


def test_symbolic_runtime_reshape_capacity_is_derived_from_shape_values():
    from fastnn.io.graph_builder import _reshape_descriptors

    assert _reshape_descriptors(
        ["Bounded(batch;1)", "Bounded(tokens;8)"],
        [-1, "Bounded(tokens;8)"],
        {"batch": 1, "tokens": 8},
    ) == ["Bounded((batch);1)", "Bounded(tokens;8)"]


def test_runtime_reshape_matches_onnxruntime_across_live_extents(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["X", "target"], ["Y"], name="reshape")],
        "runtime_reshape",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 4]),
            helper.make_tensor_value_info("target", TensorProto.INT64, [3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-reshape.onnx"
    fnn_path = tmp_path / "runtime-reshape.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for tokens in (2, 5):
        x = np.arange(tokens * 4, dtype=np.float32).reshape(tokens, 4)
        target = np.asarray([tokens, 2, 2], dtype=np.int64)
        expected = session.run(["Y"], {"X": x, "target": target})[0]
        actual = executor.forward(
            {
                "X": fnn.tensor(x, list(x.shape)),
                "target": fnn.tensor(target.astype(np.float32), [3]),
            }
        )["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_static_multi_axis_slice_matches_onnxruntime(tmp_path):
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["X"],
                ["Y"],
                starts=[1, 1],
                ends=[3, 4],
                axes=[0, 2],
            )
        ],
        "static_multi_axis_slice",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    onnx_path = tmp_path / "multi-axis-slice.onnx"
    fnn_path = tmp_path / "multi-axis-slice.fnn"
    onnx.save(model, onnx_path)
    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["Y"], {"X": data})[0]
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward({"X": fnn.tensor(data, list(data.shape))})["Y"].numpy()
    np.testing.assert_array_equal(actual, expected)


def _build_runtime_range_executor(tmp_path, capacity=16):
    graph = helper.make_graph(
        [helper.make_node("Range", ["start", "limit", "step"], ["Y"])],
        "runtime_range",
        [
            helper.make_tensor_value_info("start", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("limit", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("step", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["range_length"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-range.onnx"
    fnn_path = tmp_path / "runtime-range.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    return fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"range_length": capacity}
    )


def test_runtime_range_negative_step_empty_and_zero_rejection(tmp_path):
    executor = _build_runtime_range_executor(tmp_path)

    def run(start, limit, step):
        values = {
            "start": fnn.tensor([float(start)], []),
            "limit": fnn.tensor([float(limit)], []),
            "step": fnn.tensor([float(step)], []),
        }
        return executor.forward(values)["Y"]

    np.testing.assert_array_equal(
        run(5, -1, -2).numpy(), np.asarray([5, 3, 1], np.float32)
    )
    np.testing.assert_array_equal(run(1, 1, 1).numpy(), np.asarray([], np.float32))
    np.testing.assert_array_equal(run(1, 5, -1).numpy(), np.asarray([], np.float32))
    with pytest.raises(RuntimeError, match="nonzero step"):
        run(0, 4, 0)
    with pytest.raises(RuntimeError, match="capacity|bound|maximum|exceeds|overflows"):
        run(0, 17, 1)


def test_runtime_range_resolves_derived_shape_scalars(tmp_path):
    initializers = {
        "shape_index": np.asarray(0, dtype=np.int64),
        "start": np.asarray(0.0, dtype=np.float32),
        "step": np.asarray(1.0, dtype=np.float32),
    }
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Gather", ["shape", "shape_index"], ["extent_i64"], axis=0),
        helper.make_node("Cast", ["extent_i64"], ["extent"], to=TensorProto.FLOAT),
        helper.make_node("Range", ["start", "extent", "step"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "derived_runtime_range",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["range_length"])],
        initializer=[numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "derived-runtime-range.onnx"
    fnn_path = tmp_path / "derived-runtime-range.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"tokens": 8, "range_length": 8}
    )
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for tokens in (1, 5):
        x = np.zeros((tokens, 2), dtype=np.float32)
        expected = ort_session.run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_runtime_slice_bounds_match_onnxruntime(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["X", "starts", "ends", "axes", "steps"],
                ["Y"],
                name="runtime_slice",
            )
        ],
        "runtime_slice",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 6]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-slice.onnx"
    fnn_path = tmp_path / "runtime-slice.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for raw_start, raw_end in ((1, 4), (-4, -1)):
        ort_bounds = {
            "starts": np.asarray([raw_start], dtype=np.int64),
            "ends": np.asarray([raw_end], dtype=np.int64),
            "axes": np.asarray([1], dtype=np.int64),
            "steps": np.asarray([1], dtype=np.int64),
        }
        fastnn_bounds = {
            name: fnn.tensor(value.astype(np.float32), [1])
            for name, value in ort_bounds.items()
        }
        for tokens in (2, 5):
            x = np.arange(tokens * 6, dtype=np.float32).reshape(tokens, 6)
            expected = session.run(["Y"], {"X": x, **ort_bounds})[0]
            actual = executor.forward(
                {"X": fnn.tensor(x, list(x.shape)), **fastnn_bounds}
            )["Y"].numpy()
            np.testing.assert_array_equal(actual, expected)


def test_nd_symbolic_broadcast_matches_onnxruntime(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Add", ["X", "mask"], ["Y"], name="broadcast_add")],
        "nd_symbolic_broadcast",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, "tokens", 4]),
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, [2, 1, "tokens", 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, "tokens", 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "nd-symbolic-broadcast.onnx"
    fnn_path = tmp_path / "nd-symbolic-broadcast.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for tokens in (2, 5):
        x = np.arange(2 * 3 * tokens * 4, dtype=np.float32).reshape(2, 3, tokens, 4)
        mask = np.arange(2 * tokens * 4, dtype=np.float32).reshape(2, 1, tokens, 4)
        expected = session.run(["Y"], {"X": x, "mask": mask})[0]
        actual = executor.forward(
            {
                "X": fnn.tensor(x, list(x.shape)),
                "mask": fnn.tensor(mask, list(mask.shape)),
            }
        )["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_f32_bool_cast_roundtrip_matches_onnxruntime(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["X"], ["condition"], name="to_bool", to=TensorProto.BOOL),
            helper.make_node("Cast", ["condition"], ["Y"], name="to_float", to=TensorProto.FLOAT),
        ],
        "bool_cast_roundtrip",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "bool-cast-roundtrip.onnx"
    fnn_path = tmp_path / "bool-cast-roundtrip.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for tokens in (2, 5):
        x = np.asarray([0.0, -2.0, 3.5, np.nan] * tokens, dtype=np.float32).reshape(tokens, 4)
        expected = session.run(["Y"], {"X": x})[0]
        actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_per_symbol_capacities_allow_smaller_live_inputs(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"], name="identity")],
        "bounded_live_input",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", "past", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", "past", 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "bounded-live-input.onnx"
    fnn_path = tmp_path / "bounded-live-input.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"batch": 1, "past": 8}
    )
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    actual = executor.forward({"X": fnn.tensor(x, list(x.shape))})["Y"].numpy()
    np.testing.assert_array_equal(actual, x)


def test_runtime_expand_matches_onnxruntime_at_multiple_live_extents(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Expand", ["X", "shape"], ["Y"], name="runtime_expand")],
        "runtime_expand",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["tokens", 1]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["tokens", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-expand.onnx"
    fnn_path = tmp_path / "runtime-expand.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"tokens": 8}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for tokens in (2, 5):
        x = np.arange(tokens, dtype=np.float32).reshape(tokens, 1)
        shape = np.array([tokens, 3], dtype=np.int64)
        expected = session.run(None, {"X": x, "shape": shape})[0]
        actual = executor.forward(
            {
                "X": fnn.tensor(x, list(x.shape)),
                "shape": fnn.tensor(shape.astype(np.float32), list(shape.shape)),
            }
        )["Y"].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_sin_cos_match_onnxruntime(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Sin", ["X"], ["sin_x"], name="sin"),
            helper.make_node("Cos", ["X"], ["cos_x"], name="cos"),
        ],
        "sin_cos",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [
            helper.make_tensor_value_info("sin_x", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("cos_x", TensorProto.FLOAT, [2, 4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "sin-cos.onnx"
    fnn_path = tmp_path / "sin-cos.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.array(
        [[-100.0, -np.pi, -0.0, 0.25], [1.0, np.pi / 2, np.pi, 100.0]],
        dtype=np.float32,
    )
    expected_sin, expected_cos = session.run(None, {"X": x})
    outputs = executor.forward({"X": fnn.tensor(x, list(x.shape))})
    np.testing.assert_allclose(outputs["sin_x"].numpy(), expected_sin, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(outputs["cos_x"].numpy(), expected_cos, rtol=1e-6, atol=1e-7)


def test_trilu_matches_onnxruntime_for_batched_matrices(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Trilu", ["X", "k_neg"], ["upper"], name="upper"),
            helper.make_node(
                "Trilu", ["X", "k_pos"], ["lower"], name="lower", upper=0
            ),
        ],
        "trilu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, "cols"])],
        [
            helper.make_tensor_value_info("upper", TensorProto.FLOAT, [1, 3, "cols"]),
            helper.make_tensor_value_info("lower", TensorProto.FLOAT, [1, 3, "cols"]),
        ],
        initializer=[
            numpy_helper.from_array(np.array(-1, dtype=np.int64), name="k_neg"),
            numpy_helper.from_array(np.array(1, dtype=np.int64), name="k_pos"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "trilu.onnx"
    fnn_path = tmp_path / "trilu.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path),
        symbolic_dim_bounds={"cols": 8},
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected_upper, expected_lower = session.run(None, {"X": x})
    outputs = executor.forward({"X": fnn.tensor(x, list(x.shape))})
    np.testing.assert_array_equal(outputs["upper"].numpy(), expected_upper)
    np.testing.assert_array_equal(outputs["lower"].numpy(), expected_lower)


def test_concat_uses_live_symbolic_outer_extent(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Concat", ["A", "B"], ["Y"], name="concat", axis=2)],
        "concat_live_outer",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, "outer", 2]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, "outer", 1]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "outer", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "concat-live.onnx"
    fnn_path = tmp_path / "concat-live.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"outer": 8}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    a = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    b = np.arange(100, 103, dtype=np.float32).reshape(1, 3, 1)
    expected = session.run(None, {"A": a, "B": b})[0]
    output = executor.forward(
        {
            "A": fnn.tensor(a, list(a.shape)),
            "B": fnn.tensor(b, list(b.shape)),
        }
    )["Y"].numpy()
    np.testing.assert_array_equal(output, expected)


def test_aot_runtime_owned_state_reset_and_isolation(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Add", ["state", "delta"], ["next_state"], name="advance")],
        "persistent_state",
        [
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("next_state", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "persistent-state.onnx"
    fnn_path = tmp_path / "persistent-state.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))

    model = fnn.build_model_from_fnn(str(fnn_path))
    model.configure_state(
        {"state": "next_state"},
        {"state": fnn.tensor([0.0], [1])},
    )
    first = model.create_session()
    second = model.create_session()
    assert model.state_descriptors() == [
        {
            "input": "state",
            "output": "next_state",
            "update": "replace",
            "capacity_bytes": "4",
        }
    ]
    assert first.forward_stateful({"delta": fnn.tensor([0.0], [1])}) == {}
    assert first.output_buffer_capacities() == [4]
    first.reset_state()
    out1 = first.forward_stateful({"delta": fnn.tensor([1.0], [1])}, True)
    out2 = first.forward_stateful({"delta": fnn.tensor([2.0], [1])}, True)
    isolated = second.forward_stateful({"delta": fnn.tensor([1.0], [1])}, True)
    np.testing.assert_array_equal(out1["next_state"].numpy(), np.array([1.0], np.float32))
    np.testing.assert_array_equal(out2["next_state"].numpy(), np.array([3.0], np.float32))
    np.testing.assert_array_equal(
        isolated["next_state"].numpy(), np.array([1.0], np.float32)
    )

    with pytest.raises(ValueError, match="runtime-owned"):
        first.forward_stateful(
            {
                "state": fnn.tensor([100.0], [1]),
                "delta": fnn.tensor([1.0], [1]),
            }
        )
    first.reset_state()
    reset = first.forward_stateful({"delta": fnn.tensor([2.0], [1])}, True)
    np.testing.assert_array_equal(reset["next_state"].numpy(), np.array([2.0], np.float32))

    first.reset_state()
    assert first.session_steps == 0
    assert not first.session_initialized
    with pytest.raises(RuntimeError, match="prefill before decode"):
        first.decode({"delta": fnn.tensor([1.0], [1])})
    prefill = first.prefill({"delta": fnn.tensor([1.0], [1])}, True)
    np.testing.assert_array_equal(
        prefill["next_state"].numpy(), np.array([1.0], np.float32)
    )
    assert first.session_steps == 1
    assert first.session_initialized
    with pytest.raises(RuntimeError, match="reset before prefill"):
        first.prefill({"delta": fnn.tensor([1.0], [1])})
    decoded = first.decode({"delta": fnn.tensor([2.0], [1])}, True)
    np.testing.assert_array_equal(
        decoded["next_state"].numpy(), np.array([3.0], np.float32)
    )
    assert first.session_steps == 2


def test_aot_runtime_owned_state_append_non_innermost_axis(tmp_path):
    state = helper.make_tensor_value_info(
        "state", TensorProto.FLOAT, [2, "state_length", 2]
    )
    delta = helper.make_tensor_value_info("delta", TensorProto.FLOAT, [2, 1, 2])
    current = helper.make_tensor_value_info(
        "current", TensorProto.FLOAT, [2, "state_length", 2]
    )
    append_chunk = helper.make_tensor_value_info(
        "append_chunk", TensorProto.FLOAT, [2, 1, 2]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["state"], ["current"]),
            helper.make_node("Identity", ["delta"], ["append_chunk"]),
        ],
        "append_state",
        [state, delta],
        [current, append_chunk],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = tmp_path / "append-state.onnx"
    fnn_path = tmp_path / "append-state.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))

    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"state_length": 4}
    )
    executor.configure_state(
        {"state": "append_chunk"},
        {"state": fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 1, 2])},
        {"state": 1},
    )
    assert executor.state_descriptors()[0]["update"] == "append"
    assert executor.state_descriptors()[0]["axis"] == "1"

    first = executor.forward_stateful(
        {"delta": fnn.tensor([5.0, 6.0, 7.0, 8.0], [2, 1, 2])}
    )
    np.testing.assert_array_equal(
        first["current"].numpy(),
        np.array([[[1.0, 2.0]], [[3.0, 4.0]]], np.float32),
    )
    second = executor.forward_stateful(
        {"delta": fnn.tensor([9.0, 10.0, 11.0, 12.0], [2, 1, 2])}
    )
    np.testing.assert_array_equal(
        second["current"].numpy(),
        np.array(
            [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]],
            np.float32,
        ),
    )
    executor.forward_stateful(
        {"delta": fnn.tensor([13.0, 14.0, 15.0, 16.0], [2, 1, 2])}
    )
    with pytest.raises(RuntimeError, match="exceeding capacity"):
        executor.forward_stateful(
            {"delta": fnn.tensor([17.0, 18.0, 19.0, 20.0], [2, 1, 2])}
        )
    assert executor.state_sizes()["state"] == 2 * 4 * 2 * 4
    executor.reset_state()
    assert executor.state_sizes()["state"] == 2 * 1 * 2 * 4


def test_aot_runtime_owned_state_rejects_capacity_overflow_atomically(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Concat", ["state", "delta"], ["next_state"], axis=0)],
        "bounded_persistent_state",
        [
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("next_state", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "bounded-persistent-state.onnx"
    fnn_path = tmp_path / "bounded-persistent-state.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(str(fnn_path))
    executor.configure_state(
        {"state": "next_state"}, {"state": fnn.tensor([0.0], [1])}
    )
    assert executor.state_sizes() == {"state": 4}
    with pytest.raises(RuntimeError, match="exceeding capacity"):
        executor.forward_stateful({"delta": fnn.tensor([1.0], [1])})
    assert executor.state_sizes() == {"state": 4}
    executor.reset_state()
    assert executor.state_sizes() == {"state": 4}

def test_dynamic_w4a8_accepts_rank3_matmul_activations(tmp_path):
    activation = helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, 3, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4])
    weights = np.linspace(-1.0, 1.0, 32 * 4, dtype=np.float32).reshape(32, 4)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["activation", "weights"], ["output"])],
        "rank3_w4a8",
        [activation],
        [output],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "rank3-w4a8.onnx"
    fnn_path = tmp_path / "rank3-w4a8.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))

    executor = fnn.build_model_from_fnn(str(fnn_path), quantize="w4a8-g32")
    values = np.linspace(-0.75, 0.75, 1 * 3 * 32, dtype=np.float32).reshape(1, 3, 32)
    actual = executor.forward({"activation": fnn.tensor(values, list(values.shape))})[
        "output"
    ].numpy()
    expected = np.matmul(values, weights)

    assert actual.shape == (1, 3, 4)
    np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.08)


def test_runtime_where_bool_broadcast_matches_onnxruntime(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["condition_source"], ["condition"], to=TensorProto.BOOL),
            helper.make_node("Where", ["condition", "X", "Y"], ["Z"], name="runtime_where"),
        ],
        "runtime_where",
        [
            helper.make_tensor_value_info("condition_source", TensorProto.FLOAT, ["tokens", 1]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1]),
        ],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["tokens", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "runtime-where.onnx"
    fnn_path = tmp_path / "runtime-where.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"tokens": 8}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    y = np.array([[-1.0]], dtype=np.float32)

    for tokens in (2, 5):
        condition = (np.arange(tokens) % 2).astype(np.float32).reshape(tokens, 1)
        feeds = {"condition_source": condition, "X": x, "Y": y}
        expected = session.run(None, feeds)[0]
        actual = executor.forward(
            {name: fnn.tensor(value, list(value.shape)) for name, value in feeds.items()}
        )["Z"].numpy()
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


def test_slice_multiple_axes_tensor_inputs_match_onnxruntime(tmp_path):
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

    expected = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["Y"], {"X": np.arange(24, dtype=np.float32).reshape(x_shape)})[0]
    executor = fnn.build_model_from_fnn(str(fnn_path))
    actual = executor.forward(
        {"X": fnn.tensor(np.arange(24, dtype=np.float32).reshape(x_shape), x_shape)}
    )["Y"].numpy()
    np.testing.assert_array_equal(actual, expected)


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


def test_diagnostic_outputs_retain_intermediate_values(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["X"], ["hidden"], name="relu"),
            helper.make_node("Identity", ["hidden"], ["Y"], name="identity"),
        ],
        "diagnostic_outputs",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "diagnostic.onnx"
    fnn_path = tmp_path / "diagnostic.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))

    x = np.asarray([-2.0, 3.0], dtype=np.float32)
    executor = fnn.build_model_from_fnn(
        str(fnn_path), diagnostic_outputs=["hidden"]
    )
    outputs = executor.forward({"X": fnn.tensor(x, [2])})
    np.testing.assert_array_equal(outputs["hidden"].numpy(), [0.0, 3.0])
    np.testing.assert_array_equal(outputs["Y"].numpy(), [0.0, 3.0])

    with pytest.raises(ValueError, match="not produced by the graph"):
        fnn.build_model_from_fnn(
            str(fnn_path), diagnostic_outputs=["missing"]
        )


def test_batched_matmul_keeps_rhs_batches_separate(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["Q", "K"], ["Y"], name="attention_scores")],
        "batched_attention_matmul",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 3, 1, 4]),
            helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 3, 4, "past"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 1, "past"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "batched-attention-matmul.onnx"
    fnn_path = tmp_path / "batched-attention-matmul.fnn"
    onnx.save(model, onnx_path)
    fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    executor = fnn.build_model_from_fnn(
        str(fnn_path), symbolic_dim_bounds={"past": 5}
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    q = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 4) / 7.0
    for past in (1, 2, 5):
        k = (
            np.arange(3 * 4 * past, dtype=np.float32).reshape(1, 3, 4, past)
            + np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1) * 100.0
        )
        expected = session.run(["Y"], {"Q": q, "K": k})[0]
        actual = executor.forward(
            {
                "Q": fnn.tensor(q, list(q.shape)),
                "K": fnn.tensor(k, list(k.shape)),
            }
        )["Y"].numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
