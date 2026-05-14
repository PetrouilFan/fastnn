"""Tests for ONNX model import and DAG execution pipeline."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Skip all tests if onnx is not installed
pytest.importorskip("onnx")

# These tests require onnx to generate test models
try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except (ImportError, AttributeError):
    onnx = None


def _make_onnx_model(graph_def, model_name="test_model.onnx"):
    """Helper to create an ONNX model from a graph_def and save to temp file."""
    model = helper.make_model(graph_def)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, model_name)
    onnx.save(model, path)
    return path, tmpdir


# ---- Fixtures ----
@pytest.fixture
def simple_relu_model():
    """Create a minimal ONNX model: input -> Relu -> output."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu1")

    graph = helper.make_graph(
        [node],
        "test_graph",
        [X],
        [Y],
    )
    return _make_onnx_model(graph, "simple_relu.onnx")


@pytest.fixture
def simple_conv_model():
    """Create a simple Conv model: input -> Conv -> Relu -> output."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6, 8, 8])

    W = helper.make_tensor("W", TensorProto.FLOAT, [6, 3, 3, 3],
                           np.random.randn(6, 3, 3, 3).flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, [6],
                           np.random.randn(6).flatten().tolist())

    conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["conv_out"], name="conv1",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["Y"], name="relu1")

    graph = helper.make_graph(
        [conv, relu],
        "test_graph",
        [X],
        [Y],
        [W, B],
    )
    return _make_onnx_model(graph, "simple_conv.onnx")


@pytest.fixture
def simple_gemm_model():
    """Create a simple Gemm (Linear) model."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])

    W = helper.make_tensor("W", TensorProto.FLOAT, [8, 16],
                           np.random.randn(8, 16).flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, [8],
                           np.random.randn(8).flatten().tolist())

    gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], name="gemm1",
                            alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [gemm],
        "test_graph",
        [X],
        [Y],
        [W, B],
    )
    return _make_onnx_model(graph, "simple_gemm.onnx")


@pytest.fixture
def yolo_c2f_model():
    """Create a model resembling YOLO C2f (Split -> Conv -> Concat)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 128, 8, 8])

    # Weights for two convs
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [64, 64, 3, 3],
                            np.random.randn(64, 64, 3, 3).flatten().tolist())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [64, 64, 3, 3],
                            np.random.randn(64, 64, 3, 3).flatten().tolist())

    split = helper.make_node("Split", inputs=["X"], outputs=["split_1", "split_2"],
                             name="split1", axis=1)
    conv1 = helper.make_node("Conv", inputs=["split_1", "W1"], outputs=["conv1_out"],
                             name="conv_c2f_1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    conv2 = helper.make_node("Conv", inputs=["split_2", "W2"], outputs=["conv2_out"],
                             name="conv_c2f_2", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    concat = helper.make_node("Concat", inputs=["conv1_out", "conv2_out"], outputs=["Y"],
                              name="concat1", axis=1)

    graph = helper.make_graph(
        [split, conv1, conv2, concat],
        "test_graph",
        [X],
        [Y],
        [W1, W2],
    )
    return _make_onnx_model(graph, "yolo_c2f.onnx")


@pytest.fixture
def multi_op_model():
    """Model with multiple ops: Conv -> BatchNorm -> Relu -> MaxPool -> AvgPool -> Add."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6, 8, 8])

    # Conv params
    W = helper.make_tensor("W", TensorProto.FLOAT, [6, 3, 3, 3],
                           np.random.randn(6, 3, 3, 3).flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, [6],
                           np.random.randn(6).flatten().tolist())

    # BN params
    BN_W = helper.make_tensor("BN_W", TensorProto.FLOAT, [6],
                              np.ones(6, dtype=np.float32).tolist())
    BN_B = helper.make_tensor("BN_B", TensorProto.FLOAT, [6],
                              np.zeros(6, dtype=np.float32).tolist())
    BN_M = helper.make_tensor("BN_M", TensorProto.FLOAT, [6],
                              np.zeros(6, dtype=np.float32).tolist())
    BN_V = helper.make_tensor("BN_V", TensorProto.FLOAT, [6],
                              np.ones(6, dtype=np.float32).tolist())

    conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["conv_out"], name="conv1",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    bn = helper.make_node("BatchNormalization", inputs=["conv_out", "BN_W", "BN_B", "BN_M", "BN_V"],
                          outputs=["bn_out"], name="bn1", epsilon=1e-5)
    relu = helper.make_node("Relu", inputs=["bn_out"], outputs=["relu_out"], name="relu1")
    pool = helper.make_node("MaxPool", inputs=["relu_out"], outputs=["pool_out"],
                            name="pool1", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    avg = helper.make_node("AveragePool", inputs=["pool_out"], outputs=["avg_out"],
                           name="avg1", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    # Create a separate path for Add (add identity output)
    identity = helper.make_node("Identity", inputs=["pool_out"], outputs=["identity_out"], name="identity1")
    add = helper.make_node("Add", inputs=["avg_out", "identity_out"], outputs=["Y"], name="add1")

    graph = helper.make_graph(
        [conv, bn, relu, pool, avg, identity, add],
        "test_graph",
        [X],
        [Y],
        [W, B, BN_W, BN_B, BN_M, BN_V],
    )
    return _make_onnx_model(graph, "multi_op.onnx")


# ---- Tests ----

class TestOnnxImport:
    """Test the ONNX import pipeline."""

    def test_import_simple_relu(self, simple_relu_model):
        """Test importing a simple Relu model."""
        path, tmpdir = simple_relu_model
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            assert "layers" in result
            assert "parameters" in result
            assert len(result["layers"]) == 1
            assert result["layers"][0]["type"] == "Relu"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_import_simple_conv(self, simple_conv_model):
        """Test importing a Conv model with parameters."""
        path, tmpdir = simple_conv_model
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            assert len(result["layers"]) == 2
            assert result["layers"][0]["type"] == "Conv"
            assert result["layers"][0]["kernel_size"] == 3
            assert result["layers"][1]["type"] == "Relu"
            assert result["parameters"] > 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_import_simple_gemm(self, simple_gemm_model):
        """Test importing a Gemm (Linear) model."""
        path, tmpdir = simple_gemm_model
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            assert len(result["layers"]) == 1
            assert result["layers"][0]["type"] == "Linear"
            assert result["layers"][0]["in_features"] == 16
            assert result["layers"][0]["out_features"] == 8
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_import_yolo_c2f(self, yolo_c2f_model):
        """Test importing YOLO C2f (Split + Conv + Concat)."""
        path, tmpdir = yolo_c2f_model
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            layers = result["layers"]
            # Should have: Split, Conv, Conv, Concat
            assert len(layers) == 4
            assert layers[0]["type"] == "Split"
            assert layers[0]["axis"] == 1
            assert layers[1]["type"] == "Conv"
            assert layers[2]["type"] == "Conv"
            assert layers[3]["type"] == "Concat"
            assert layers[3]["axis"] == 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_import_multi_op(self, multi_op_model):
        """Test importing a model with multiple op types."""
        path, tmpdir = multi_op_model
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            layers = result["layers"]
            types = [l["type"] for l in layers]
            assert "Conv" in types
            assert "BatchNorm2d" in types
            assert "Relu" in types
            assert "MaxPool" in types
            assert "Add" in types or "ElementwiseAdd" in types or types == ["ElementwiseAdd"]
            # Check graph topology is stored
            with open(os.path.join(tmpdir, "out.fnn"), "rb") as f:
                from fastnn.io import read_fnn_header
                magic, ver, header, num_params = read_fnn_header(f)
                assert "graph" in header
                assert "nodes" in header["graph"]
                assert len(header["graph"]["nodes"]) >= 6
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_fnn_file_format(self, simple_conv_model):
        """Test that the output .fnn file has correct format."""
        path, tmpdir = simple_conv_model
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)

            # Verify file structure
            with open(out_path, "rb") as f:
                from fastnn.io import read_fnn_header, read_fnn_parameters
                magic, ver, header, num_params = read_fnn_header(f)
                assert ver in (2, 3), f"Expected version 2 or 3, got {ver}"
                assert "layers" in header
                params = read_fnn_parameters(f, num_params, version=ver)
                assert len(params) > 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_graph_topology(self, multi_op_model):
        """Test that graph topology is correctly stored."""
        path, tmpdir = multi_op_model
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)

            with open(out_path, "rb") as f:
                from fastnn.io import read_fnn_header
                magic, ver, header, num_params = read_fnn_header(f)

            graph = header.get("graph", {})
            assert "nodes" in graph
            assert "inputs" in graph
            assert "outputs" in graph

            # Verify node structure
            for node in graph["nodes"]:
                assert "name" in node
                assert "op_type" in node
                assert "inputs" in node
                assert "outputs" in node

            # Verify inputs
            for inp in graph["inputs"]:
                assert "name" in inp
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestDAGModel:
    """Test the DAG model execution."""

    def test_dag_model_relu(self, simple_relu_model):
        """Test DAGModel execution with a simple relu."""
        pytest.importorskip("onnx")
        path, tmpdir = simple_relu_model
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)

            # Try to load and run DAGModel
            try:
                from fastnn.io.dag_model import DAGModel
                from fastnn.io import read_fnn_header, read_fnn_parameters

                with open(out_path, "rb") as f:
                    magic, ver, header, num_params = read_fnn_header(f)

                with open(out_path, "rb") as f:
                    _, _, _, num_params = read_fnn_header(f)
                    params = read_fnn_parameters(f, num_params)

                model = DAGModel.from_header(header, params)

                x = np.random.randn(1, 3, 64, 64).astype(np.float32)
                import fastnn as fnn
                x_tensor = fnn.tensor(x, list(x.shape))
                outputs = model.forward({"X": x_tensor})
                assert len(outputs) > 0
                # Check output shape
                for name, out in outputs.items():
                    assert out.shape == [1, 3, 64, 64]
            except (ImportError, AttributeError):
                pytest.skip("DAGModel not available")
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_dag_model_conv(self, simple_conv_model):
        """Test DAGModel execution with a convolution."""
        path, tmpdir = simple_conv_model
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)

            try:
                from fastnn.io.dag_model import DAGModel
                from fastnn.io import read_fnn_header, read_fnn_parameters

                with open(out_path, "rb") as f:
                    _, ver, header, num_params = read_fnn_header(f)

                with open(out_path, "rb") as f:
                    _, _, _, num_params = read_fnn_header(f)
                    params = read_fnn_parameters(f, num_params, version=ver)

                model = DAGModel.from_header(header, params)

                x = np.random.randn(1, 3, 8, 8).astype(np.float32)
                import fastnn as fnn
                x_tensor = fnn.tensor(x, list(x.shape))
                outputs = model.forward({"X": x_tensor})
                assert len(outputs) > 0
                for name, out in outputs.items():
                    assert len(out.shape) == 4
            except (ImportError, AttributeError):
                pytest.skip("DAGModel not available")
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestOpHandlers:
    """Test specific ONNX op handlers."""

    def _import_and_check(self, model_fixture, expected_types):
        """Helper: import model and check layer types."""
        path, tmpdir = model_fixture
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            result = import_onnx(path, out_path)
            types = [l["type"] for l in result["layers"]]
            for expected in expected_types:
                assert expected in types, f"Expected {expected} in layer types, got {types}"
            return result
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_split_handler(self):
        """Test Split op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])
        Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, 4, 4])
        Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4, 4, 4])
        split = helper.make_node("Split", inputs=["X"], outputs=["Y1", "Y2"],
                                 name="split1", axis=1)
        graph = helper.make_graph([split], "test", [X], [Y1, Y2])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "split.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "Split" in types, f"Split not in types: {types}"
            split_layer = result["layers"][types.index("Split")]
            assert split_layer["axis"] == 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_shape_cast_gather_handlers(self):
        """Test Shape -> Cast -> Gather pattern (used in YOLO dynamic reshape)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 64, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64])

        shape = helper.make_node("Shape", inputs=["X"], outputs=["shape_out"], name="shape1")
        cast = helper.make_node("Cast", inputs=["shape_out"], outputs=["cast_out"],
                                name="cast1", to=1)  # to=1 is FLOAT
        # Gather: indices=0, axis=0 to get batch dim
        indices = helper.make_tensor("indices", TensorProto.INT64, [1], [0])
        gather = helper.make_node("Gather", inputs=["cast_out", "indices"], outputs=["gather_out"],
                                  name="gather1", axis=0)
        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"], name="identity1")

        graph = helper.make_graph(
            [shape, cast, gather, identity],
            "test",
            [X],
            [Y],
            [indices],
        )
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "shape_cast.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "ShapeOp" in types, f"ShapeOp not in {types}"
            assert "CastOp" in types, f"CastOp not in {types}"
            assert "GatherOp" in types, f"GatherOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_sub_handler(self):
        """Test Sub op handler (elementwise and bias)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])

        # Elementwise Sub (two non-constant inputs)
        sub = helper.make_node("Sub", inputs=["X", "X"], outputs=["Y"], name="sub1")

        graph = helper.make_graph([sub], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sub.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "ElementwiseSub" in types, f"ElementwiseSub not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_pooling_handlers(self):
        """Test AveragePool, GlobalAveragePool handlers."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 7, 7])

        avg = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"],
                               name="avg1", kernel_shape=[3, 3], strides=[2, 2],
                               pads=[0, 0, 0, 0])

        graph = helper.make_graph([avg], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "avg.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "AvgPool" in types
            avg_layer = result["layers"][types.index("AvgPool")]
            assert avg_layer["kernel_size"] == 3
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_resize_handler(self):
        """Test Resize op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 16, 16])

        scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
        resize = helper.make_node("Resize", inputs=["X", "", "scales"], outputs=["Y"],
                                  name="resize1", mode="nearest")

        graph = helper.make_graph([resize], "test", [X], [Y], [scales])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "resize.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "Resize" in types
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_reduce_mean_handler(self):
        """Test ReduceMean handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 1, 1])

        reduce = helper.make_node("ReduceMean", inputs=["X"], outputs=["Y"],
                                  name="reduce1", axes=[2, 3], keepdims=1)

        graph = helper.make_graph([reduce], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "reduce.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "ReduceMean" in types
            rl = result["layers"][types.index("ReduceMean")]
            assert rl["axes"] == [2, 3]
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestYoloOps:
    """Test YOLO-specific operation patterns."""

    def test_silu_handler(self):
        """Test SiLU op (used in YOLOv8+ backbone)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])

        # YOLO uses SiLU activation extensively
        node = helper.make_node("Sigmoid", inputs=["X"], outputs=["sig_out"], name="sig1")
        mul = helper.make_node("Mul", inputs=["sig_out", "X"], outputs=["Y"], name="mul1")

        graph = helper.make_graph([node, mul], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "silu.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "Sigmoid" in types
            assert "ElementwiseMul" in types or "Mul" in types
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_concat_handling(self):
        """Test Concat operation (used in C2f neck)."""
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 64, 8, 8])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 64, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 128, 8, 8])

        concat = helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"],
                                  name="concat1", axis=1)

        graph = helper.make_graph([concat], "test", [X1, X2], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "concat.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "Concat" in types
            concat_layer = result["layers"][0]
            assert concat_layer["axis"] == 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestNewOps:
    """Test newly added ONNX op handlers."""

    # ------------------------------------------------------------------ #
    # 1. Simple Math ops
    # ------------------------------------------------------------------ #

    def test_ceil_handler(self):
        """Test Ceil op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Ceil", inputs=["X"], outputs=["Y"], name="ceil1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ceil.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "CeilOp" in types, f"CeilOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_floor_handler(self):
        """Test Floor op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Floor", inputs=["X"], outputs=["Y"], name="floor1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "floor.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "FloorOp" in types, f"FloorOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_round_handler(self):
        """Test Round op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Round", inputs=["X"], outputs=["Y"], name="round1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "round.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "RoundOp" in types, f"RoundOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_sign_handler(self):
        """Test Sign op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Sign", inputs=["X"], outputs=["Y"], name="sign1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sign.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "SignOp" in types, f"SignOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_reciprocal_handler(self):
        """Test Reciprocal op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Reciprocal", inputs=["X"], outputs=["Y"], name="recip1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "recip.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "ReciprocalOp" in types, f"ReciprocalOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_isnan_handler(self):
        """Test IsNaN op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("IsNaN", inputs=["X"], outputs=["Y"], name="isnan1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "isnan.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "IsNaNOp" in types, f"IsNaNOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_isinf_handler(self):
        """Test IsInf op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("IsInf", inputs=["X"], outputs=["Y"], name="isinf1",
                                detect_positive=1, detect_negative=1)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "isinf.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "IsInfOp" in types, f"IsInfOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------ #
    # 2. Logical / Comparison ops
    # ------------------------------------------------------------------ #

    def test_not_handler(self):
        """Test Not op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.BOOL, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Not", inputs=["X"], outputs=["Y"], name="not1")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "not.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "NotOp" in types, f"NotOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_and_handler(self):
        """Test And op handler (binary logical op)."""
        X = helper.make_tensor_value_info("X", TensorProto.BOOL, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("And", inputs=["X", "Y"], outputs=["Z"], name="and1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "and.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "AndOp" in types, f"AndOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_or_handler(self):
        """Test Or op handler (binary logical op)."""
        X = helper.make_tensor_value_info("X", TensorProto.BOOL, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Or", inputs=["X", "Y"], outputs=["Z"], name="or1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "or.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "OrOp" in types, f"OrOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_xor_handler(self):
        """Test Xor op handler (binary logical op)."""
        X = helper.make_tensor_value_info("X", TensorProto.BOOL, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Xor", inputs=["X", "Y"], outputs=["Z"], name="xor1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "xor.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "XorOp" in types, f"XorOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_less_handler(self):
        """Test Less op handler (binary comparison)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Less", inputs=["X", "Y"], outputs=["Z"], name="less1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "less.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "LessOp" in types, f"LessOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_greater_handler(self):
        """Test Greater op handler (binary comparison)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Greater", inputs=["X", "Y"], outputs=["Z"], name="greater1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "greater.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "GreaterOp" in types, f"GreaterOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_equal_handler(self):
        """Test Equal op handler (binary comparison)."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [1, 3, 4, 4])
        node = helper.make_node("Equal", inputs=["X", "Y"], outputs=["Z"], name="equal1")
        graph = helper.make_graph([node], "test", [X, Y], [Z])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "equal.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "EqualOp" in types, f"EqualOp not in {types}"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------ #
    # 3. Tensor Manipulation ops
    # ------------------------------------------------------------------ #

    def test_cumsum_handler(self):
        """Test CumSum op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("CumSum", inputs=["X"], outputs=["Y"],
                                name="cumsum1", axis=1)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "cumsum.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "CumSum" in types, f"CumSum not in {types}"
            cumsum_layer = result["layers"][types.index("CumSum")]
            assert cumsum_layer["axis"] == 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_onehot_handler(self):
        """Test OneHot op handler."""
        indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 5])
        depth = helper.make_tensor("depth", TensorProto.INT64, [1], [10])
        values = helper.make_tensor("values", TensorProto.FLOAT, [2], [0.0, 1.0])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5, 10])
        node = helper.make_node("OneHot", inputs=["indices", "depth", "values"],
                                outputs=["Y"], name="onehot1", axis=-1)
        graph = helper.make_graph([node], "test", [indices], [Y], [depth, values])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "onehot.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "OneHot" in types, f"OneHot not in {types}"
            onehot_layer = result["layers"][types.index("OneHot")]
            assert onehot_layer["axis"] == -1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_gathernd_handler(self):
        """Test GatherND op handler (import only)."""
        data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])
        indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("GatherND", inputs=["data", "indices"],
                                outputs=["Y"], name="gathernd1", batch_dims=0)
        graph = helper.make_graph([node], "test", [data, indices], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "gathernd.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "GatherND" in types, f"GatherND not in {types}"
            gnd = result["layers"][types.index("GatherND")]
            assert gnd["batch_dims"] == 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_depthtospace_handler(self):
        """Test DepthToSpace op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 12, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
        node = helper.make_node("DepthToSpace", inputs=["X"], outputs=["Y"],
                                name="d2s1", blocksize=2, mode="DCR")
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "depthtospace.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "DepthToSpace" in types, f"DepthToSpace not in {types}"
            d2s = result["layers"][types.index("DepthToSpace")]
            assert d2s["blocksize"] == 2
            assert d2s["mode"] == "DCR"
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_spacetodepth_handler(self):
        """Test SpaceToDepth op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 12, 4, 4])
        node = helper.make_node("SpaceToDepth", inputs=["X"], outputs=["Y"],
                                name="s2d1", blocksize=2)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "spacetodepth.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "SpaceToDepth" in types, f"SpaceToDepth not in {types}"
            s2d = result["layers"][types.index("SpaceToDepth")]
            assert s2d["blocksize"] == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_eyelike_handler(self):
        """Test EyeLike op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 5])
        node = helper.make_node("EyeLike", inputs=["X"], outputs=["Y"],
                                name="eyelike1", dtype=1, k=0)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "eyelike.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "EyeLike" in types, f"EyeLike not in {types}"
            el = result["layers"][types.index("EyeLike")]
            assert el["k"] == 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------ #
    # 4. NN ops
    # ------------------------------------------------------------------ #

    def test_convtranspose_handler(self):
        """Test ConvTranspose op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6, 16, 16])

        W = helper.make_tensor("W", TensorProto.FLOAT, [6, 3, 3, 3],
                               np.random.randn(6, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("B", TensorProto.FLOAT, [6],
                               np.zeros(6, dtype=np.float32).tolist())

        node = helper.make_node("ConvTranspose", inputs=["X", "W", "B"],
                                outputs=["Y"], name="convtrans1",
                                kernel_shape=[3, 3], strides=[2, 2],
                                pads=[1, 1, 1, 1], output_padding=[1, 1])
        graph = helper.make_graph([node], "test", [X], [Y], [W, B])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "convtrans.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "ConvTranspose" in types, f"ConvTranspose not in {types}"
            ct = result["layers"][types.index("ConvTranspose")]
            assert ct["in_channels"] == 3
            assert ct["out_channels"] == 6
            assert ct["kernel_size"] == 3
            assert ct["bias"] is True
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_instancenorm_handler(self):
        """Test InstanceNormalization op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8, 8])

        scale = helper.make_tensor("scale", TensorProto.FLOAT, [4],
                                   np.ones(4, dtype=np.float32).tolist())
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [4],
                                  np.zeros(4, dtype=np.float32).tolist())

        node = helper.make_node("InstanceNormalization", inputs=["X", "scale", "bias"],
                                outputs=["Y"], name="in1", epsilon=1e-5)
        graph = helper.make_graph([node], "test", [X], [Y], [scale, bias])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "instancenorm.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "InstanceNorm" in types, f"InstanceNorm not in {types}"
            ins = result["layers"][types.index("InstanceNorm")]
            assert ins["num_features"] == 4
            assert ins["eps"] == pytest.approx(1e-5, rel=1e-6)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_logsoftmax_handler(self):
        """Test LogSoftmax op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
        node = helper.make_node("LogSoftmax", inputs=["X"], outputs=["Y"],
                                name="logsoftmax1", axis=1)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "logsoftmax.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "LogSoftmax" in types, f"LogSoftmax not in {types}"
            lsm = result["layers"][types.index("LogSoftmax")]
            assert lsm["axis"] == 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_selu_handler(self):
        """Test Selu op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("Selu", inputs=["X"], outputs=["Y"],
                                name="selu1", alpha=1.67326, gamma=1.0507)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "selu.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "Selu" in types, f"Selu not in {types}"
            selu = result["layers"][types.index("Selu")]
            assert selu["alpha"] == pytest.approx(1.67326, rel=1e-5)
            assert selu["gamma"] == pytest.approx(1.0507, rel=1e-5)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_rmsnorm_handler(self):
        """Test RMSNormalization op handler."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("RMSNormalization", inputs=["X"], outputs=["Y"],
                                name="rmsnorm1", epsilon=1e-5)
        graph = helper.make_graph([node], "test", [X], [Y])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "rmsnorm.onnx")
        onnx.save(helper.make_model(graph), path)
        try:
            from fastnn.io.onnx import import_onnx
            result = import_onnx(path, os.path.join(tmpdir, "out.fnn"))
            types = [l["type"] for l in result["layers"]]
            assert "RMSNorm" in types, f"RMSNorm not in {types}"
            rn = result["layers"][types.index("RMSNorm")]
            assert rn["eps"] == pytest.approx(1e-5, rel=1e-6)
        finally:
            import shutil
            shutil.rmtree(tmpdir)
