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
except ImportError:
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
            result = import_onnx(path, out_path)

            # Verify file structure
            with open(out_path, "rb") as f:
                from fastnn.io import read_fnn_header, read_fnn_parameters
                magic, ver, header, num_params = read_fnn_header(f)
                assert magic == b"FNN\x00"
                assert ver == 2
                assert "layers" in header
                params = read_fnn_parameters(f, num_params)
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
            result = import_onnx(path, out_path)

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
            except ImportError:
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
                    _, _, header, num_params = read_fnn_header(f)

                with open(out_path, "rb") as f:
                    _, _, _, num_params = read_fnn_header(f)
                    params = read_fnn_parameters(f, num_params)

                model = DAGModel.from_header(header, params)

                x = np.random.randn(1, 3, 8, 8).astype(np.float32)
                import fastnn as fnn
                x_tensor = fnn.tensor(x, list(x.shape))
                outputs = model.forward({"X": x_tensor})
                assert len(outputs) > 0
                for name, out in outputs.items():
                    assert len(out.shape) == 4
            except ImportError:
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
