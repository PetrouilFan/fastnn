import os
import tempfile

import numpy as np

import fastnn as fnn


def test_save_load_model():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[8], output_dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.fastnn")
        fnn.io.save(model, path)

        assert os.path.exists(path)
        loaded = fnn.io.load(path)
        assert isinstance(loaded, dict)
        assert len(loaded) > 0


def test_dlpack_roundtrip():
    t = fnn.tensor([1.0, 2.0, 3.0], [3])

    assert t.shape == [3]


def test_model_state_dict():
    model = fnn.Linear(10, 5)
    params = model.named_parameters()

    assert len(params) > 0
    for name, param in params:
        assert name in ["weight", "bias"]


def test_v3_plain_scalar_tensor_roundtrip(tmp_path):
    from fastnn.io import (
        DTYPE_BOOL,
        DTYPE_F32,
        DTYPE_I32,
        DTYPE_I64,
        MODEL_MAGIC,
        read_fnn_header,
        read_fnn_parameters,
        write_fnn_file_v3,
    )

    expected = {
        "f32": (np.asarray([[1.25, -2.5]], dtype=np.float32), DTYPE_F32),
        "i64": (np.asarray([2, -1, 7], dtype=np.int64), DTYPE_I64),
        "i32": (np.asarray([-3, 4], dtype=np.int32), DTYPE_I32),
        "bool": (np.asarray([[True, False]], dtype=np.bool_), DTYPE_BOOL),
    }
    path = tmp_path / "typed.fnn"
    params = [
        (name, array, dtype, [], [], list(array.shape))
        for name, (array, dtype) in expected.items()
    ]
    with path.open("wb") as f:
        write_fnn_file_v3(f, {"graph": {}}, params, magic=MODEL_MAGIC, version=3)

    with path.open("rb") as f:
        _, version, _, count = read_fnn_header(f)
        actual = read_fnn_parameters(f, count, version=version)

    assert count == len(expected)
    for name, (expected_array, expected_dtype) in expected.items():
        data, dtype, scales, zeros, shape = actual[name]
        assert dtype == expected_dtype
        assert shape == list(expected_array.shape)
        assert data.dtype == expected_array.dtype
        np.testing.assert_array_equal(data, expected_array)
        assert scales == []
        assert zeros == []


def test_onnx_mixed_initializer_roundtrip(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from fastnn.io import DTYPE_BOOL, DTYPE_F32, DTYPE_I32, DTYPE_I64
    from fastnn.io import read_fnn_header, read_fnn_parameters
    from fastnn.io.onnx import import_onnx

    initializers = {
        "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "shape": np.asarray([1, 2, 3], dtype=np.int64),
        "axes": np.asarray([0, 2], dtype=np.int32),
        "mask": np.asarray([True, False, True], dtype=np.bool_),
    }
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"], name="identity")],
        "typed_initializers",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(value, name=name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "typed.onnx"
    fnn_path = tmp_path / "typed.fnn"
    onnx.save(model, onnx_path)

    import_onnx(str(onnx_path), str(fnn_path))

    with fnn_path.open("rb") as f:
        _, version, _, count = read_fnn_header(f)
        actual = read_fnn_parameters(f, count, version=version)

    expected_tags = {
        "weight": DTYPE_F32,
        "shape": DTYPE_I64,
        "axes": DTYPE_I32,
        "mask": DTYPE_BOOL,
    }
    assert set(initializers).issubset(actual)
    for name, expected_array in initializers.items():
        data, dtype, _, _, shape = actual[name]
        assert dtype == expected_tags[name]
        assert shape == list(expected_array.shape)
        assert data.dtype == expected_array.dtype
        np.testing.assert_array_equal(data, expected_array)


def test_onnx_symbolic_dimension_names_are_preserved(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from fastnn.io.onnx import import_onnx_to_compute_graph

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="identity")],
        "symbolic_shapes",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "sequence"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "sequence"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    path = tmp_path / "symbolic.onnx"
    onnx.save(model, path)

    converted = import_onnx_to_compute_graph(str(path))
    input_node = next(node for node in converted["nodes"] if node["opcode"] == "Input")
    assert input_node["output_shape"]["shape"] == ["Symbol(batch)", "Symbol(sequence)"]


def test_onnx_external_initializer_roundtrip(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from fastnn.io import read_fnn_header, read_fnn_parameters
    from fastnn.io.onnx import import_onnx

    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"], name="identity")],
        "external_weight",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])],
        initializer=[numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_path = tmp_path / "external.onnx"
    fnn_path = tmp_path / "external.fnn"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.data",
        size_threshold=0,
    )

    import_onnx(str(onnx_path), str(fnn_path))
    with fnn_path.open("rb") as f:
        _, version, _, count = read_fnn_header(f)
        params = read_fnn_parameters(f, count, version=version)
    np.testing.assert_array_equal(params["weight"][0], weight)


def test_onnx_external_initializer_cannot_escape_model_directory(tmp_path):
    import onnx
    from onnx import TensorProto, external_data_helper, helper, numpy_helper

    from fastnn.io.onnx import import_onnx_to_compute_graph

    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(np.asarray([1.0], dtype=np.float32).tobytes())
    tensor = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="weight")
    external_data_helper.set_external_data(tensor, location="../outside.bin")
    tensor.ClearField("raw_data")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"], name="identity")],
        "escaping_weight",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=[tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    path = tmp_path / "escaping.onnx"
    path.write_bytes(model.SerializeToString())

    try:
        import_onnx_to_compute_graph(str(path))
    except ValueError as error:
        assert "escapes the model directory" in str(error)
    else:
        raise AssertionError("external data path traversal must be rejected")
