"""Integration tests for the YOLO ONNX inference pipeline.

These tests verify the full pipeline: ONNX import -> model building -> 
inference -> post-processing. Some tests require downloading model files
and are marked as 'slow' or 'integration'.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir():
    """Create a temporary directory for model files, cleaned up after the test.

    Yields:
        Path string to a temporary directory that is automatically removed
        (via shutil.rmtree) after the test completes.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _create_onnx_model(
    input_names_shapes=None,
    output_names_shapes=None,
    nodes=None,
    initializers=None,
    graph_name="test",
    model_dir=None,
    filename="model.onnx",
):
    """Create and save a minimal ONNX model for testing.

    Builds an ONNX model from the given graph components and saves it to
    disk.  Any test class that needs a synthetic ONNX model calls this
    instead of duplicating the ``make_tensor_value_info`` /
    ``make_graph`` / ``make_model`` / ``onnx.save`` boilerplate.

    Args:
        input_names_shapes: List of ``(name, shape)`` tuples for model
            inputs.  Default: ``[("images", [1, 3, 64, 64])]``.
        output_names_shapes: List of ``(name, shape)`` tuples for model
            outputs.  Default: ``[("output0", [1, 16, 32, 32])]``.
        nodes: Sequence of ``onnx.NodeProto`` nodes in the graph.
        initializers: List of initializer ``TensorProto`` objects.
        graph_name: Name for the ONNX graph.
        model_dir: Directory to save the model in.  If ``None`` a
            temporary directory is created and must be cleaned up by the
            caller.
        filename: Filename for the saved ``.onnx`` file.

    Returns:
        Tuple of ``(path_to_onnx_file, model_dir)``.
    """
    pytest.importorskip("onnx")
    import onnx
    from onnx import helper, TensorProto

    if input_names_shapes is None:
        input_names_shapes = [("images", [1, 3, 64, 64])]
    if output_names_shapes is None:
        output_names_shapes = [("output0", [1, 16, 32, 32])]
    if nodes is None:
        nodes = []
    if initializers is None:
        initializers = []

    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in input_names_shapes
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in output_names_shapes
    ]

    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializers)
    model = helper.make_model(graph)

    if model_dir is None:
        model_dir = tempfile.mkdtemp()

    path = os.path.join(model_dir, filename)
    onnx.save(model, path)
    return path, model_dir


def _run_pipeline(onnx_path, fnn_dir=None):
    """Run the ONNX import step of the pipeline.

    Imports an ONNX file and writes the corresponding ``.fnn`` file.

    Args:
        onnx_path: Path to the input ``.onnx`` file.
        fnn_dir: Directory for the output ``.fnn`` file.  Defaults to
            the parent directory of ``onnx_path``.

    Returns:
        Tuple of ``(import_result, fnn_path)`` where ``import_result``
        is the dict returned by ``import_onnx`` and ``fnn_path`` is
        the path to the saved ``.fnn`` file.
    """
    from fastnn.io.onnx import import_onnx

    if fnn_dir is None:
        fnn_dir = os.path.dirname(onnx_path)

    fnn_path = os.path.join(fnn_dir, "out.fnn")
    result = import_onnx(onnx_path, fnn_path)
    return result, fnn_path


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestYoloPipeline:
    """Test the YOLO inference pipeline end-to-end."""

    def _make_conv_relu_nodes(self):
        """Create Conv + Relu ONNX nodes and initializers for a minimal model."""
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto
        import numpy as np

        W = helper.make_tensor("conv1.weight", TensorProto.FLOAT, [16, 3, 3, 3],
                               np.random.randn(16, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("conv1.bias", TensorProto.FLOAT, [16],
                               np.random.randn(16).flatten().tolist())

        conv = helper.make_node("Conv", inputs=["images", "conv1.weight", "conv1.bias"],
                                outputs=["conv_out"], name="conv1",
                                kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        relu = helper.make_node("Relu", inputs=["conv_out"],
                                outputs=["output0"], name="relu1")
        return [conv, relu], [W, B]

    @pytest.mark.parametrize("scenario", [
        pytest.param("import_check", id="onnx_import_minimal"),
        pytest.param("build_check", id="build_model_from_fnn"),
        pytest.param("forward_check", id="dag_model_forward"),
    ])
    def test_pipeline_scenarios(self, scenario, tmp_model_dir):
        """Run parametrized pipeline tests: import, build, or forward."""
        nodes, initializers = self._make_conv_relu_nodes()
        path, _ = _create_onnx_model(
            nodes=nodes, initializers=initializers,
            graph_name="yolo_test", model_dir=tmp_model_dir,
        )
        result, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        if scenario == "import_check":
            assert "layers" in result
            assert "graph" in result or b"graph" in open(fnn_path, "rb").read()

            with open(fnn_path, "rb") as f:
                from fastnn.io import read_fnn_header
                _, _, header, _ = read_fnn_header(f)
                assert "graph" in header
                assert len(header["graph"]["nodes"]) >= 2

        elif scenario == "build_check":
            from fastnn.io.graph_builder import build_model_from_fnn
            model = build_model_from_fnn(fnn_path)
            assert model is not None
            assert hasattr(model, "forward") or hasattr(model, "__call__")

        elif scenario == "forward_check":
            from fastnn.io.dag_model import DAGModel
            from fastnn.io import read_fnn_header, read_fnn_parameters

            with open(fnn_path, "rb") as f:
                _, _, header, num_params = read_fnn_header(f)
            with open(fnn_path, "rb") as f:
                _, _, _, num_params = read_fnn_header(f)
                params = read_fnn_parameters(f, num_params)

            model_dag = DAGModel.from_header(header, params)

            import fastnn as fnn
            x = fnn.tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), [1, 3, 64, 64])
            outputs = model_dag.forward({"images": x})

            assert len(outputs) > 0
            for name, out in outputs.items():
                assert out is not None

    def test_nms_utility(self):
        """Test the NMS post-processing utility."""
        from fastnn.utils.nms import nms, xywh2xyxy, yolo_decode

        # Test xywh2xyxy
        boxes_xywh = np.array([[50, 50, 100, 100], [200, 200, 50, 50]], dtype=np.float32)
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        assert boxes_xyxy.shape == (2, 4)
        np.testing.assert_array_almost_equal(boxes_xyxy[0], [0, 0, 100, 100])

        # Test NMS
        boxes = np.array([
            [10, 10, 100, 100],
            [12, 12, 98, 98],   # high IoU with first
            [200, 200, 300, 300],  # far away
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        indices = nms(boxes, scores, iou_threshold=0.5)
        # Should keep first and third (second suppressed due to high IoU with first)
        assert len(indices) == 2

        # Test yolo_decode
        output = np.zeros((1, 10, 84), dtype=np.float32)
        output[0, 0, :4] = [50, 50, 100, 100]  # xywh box
        output[0, 0, 4] = 0.9  # class score
        output[0, 0, 5] = 0.8  # background
        output[0, 0, 4 + 5] = 0.95  # class 5 score
        detections = yolo_decode(output, conf_threshold=0.5)
        assert len(detections) == 1  # one image
        assert len(detections[0]) > 0  # has detections

    def test_scale_boxes(self):
        """Test box scaling utility."""
        from fastnn.utils.nms import scale_boxes

        boxes = np.array([
            [100, 50, 500, 350],
        ], dtype=np.float32)

        # Model input: 640x640, original: 1280x720
        scaled = scale_boxes((640, 640), boxes, (720, 1280))
        assert scaled.shape == (1, 4)
        # Boxes should be scaled to original image coordinates


class TestYoloDetectionModel:
    """End-to-end YOLO detection pipeline tests.

    These tests create small ONNX models that mimic YOLO output structure.
    """

    def _make_conv_nodes(self):
        """Create a single Conv ONNX node and its initializers."""
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto
        import numpy as np

        W = helper.make_tensor("conv1.weight", TensorProto.FLOAT, [16, 3, 3, 3],
                               np.random.randn(16, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("conv1.bias", TensorProto.FLOAT, [16],
                               np.random.randn(16).flatten().tolist())

        conv = helper.make_node("Conv", inputs=["images", "conv1.weight", "conv1.bias"],
                                outputs=["output0"], name="conv1",
                                kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        return [conv], [W, B]

    @pytest.mark.parametrize("scenario", [
        pytest.param("import_check", id="yolo_v8_import"),
        pytest.param("forward_check", id="yolo_v8_forward"),
    ])
    def test_yolo_v8_pipeline(self, scenario, tmp_model_dir):
        """Test importing and running inference on a YOLOv8-like model."""
        nodes, initializers = self._make_conv_nodes()
        path, _ = _create_onnx_model(
            output_names_shapes=[("output0", [1, 16, 32, 32])],
            nodes=nodes, initializers=initializers,
            graph_name="yolov8_test", model_dir=tmp_model_dir,
        )
        result, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        if scenario == "import_check":
            layers = result["layers"]
            types = [l["type"] for l in layers]
            assert "Conv" in types

            with open(fnn_path, "rb") as f:
                from fastnn.io import read_fnn_header
                _, _, header, _ = read_fnn_header(f)
                assert len(header["graph"]["nodes"]) >= 1

        elif scenario == "forward_check":
            from fastnn.io.dag_model import DAGModel
            from fastnn.io import read_fnn_header, read_fnn_parameters

            with open(fnn_path, "rb") as f:
                _, _, header, num_params = read_fnn_header(f)
            with open(fnn_path, "rb") as f:
                _, _, _, num_params = read_fnn_header(f)
                params = read_fnn_parameters(f, num_params)

            model = DAGModel.from_header(header, params)

            import fastnn as fnn
            x = fnn.tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), [1, 3, 64, 64])
            outputs = model.forward({"images": x})
            assert len(outputs) > 0

    def test_yolo_decode_with_realistic_output(self):
        """Test YOLO decode with realistic simulated output."""
        from fastnn.utils.nms import yolo_decode, yolo_dfl_decode

        num_dets = 100
        num_classes = 80

        # Standard YOLO output (v5-like)
        output = np.random.randn(1, num_dets, 4 + num_classes).astype(np.float32)
        # Make some detections have high confidence
        output[0, 0, 4] = 0.9  # class 0 score
        output[0, 0, 5] = 0.1
        output[0, 1, 10] = 0.85  # class 6 score
        output[0, 2, 20] = 0.75  # class 16 score

        detections = yolo_decode(output, conf_threshold=0.5)
        assert len(detections) == 1
        assert len(detections[0]) > 0

        # DFL output (v8-like)
        reg_max = 16
        dfl_output = np.random.randn(1, num_dets, 4 * reg_max + num_classes).astype(np.float32)
        dfl_output[0, 0, 4 * reg_max + 5] = 0.9  # class 5 score

        dfl_detections = yolo_dfl_decode(dfl_output, conf_threshold=0.5, reg_max=reg_max)
        assert len(dfl_detections) == 1


@pytest.mark.slow
class TestBenchmarks:
    """Benchmark tests for ONNX pipeline performance.

    These are marked as 'slow' and won't run by default.
    Run with: pytest tests/test_yolo_pipeline.py -v -m slow
    """

    def test_onnx_import_speed(self, tmp_model_dir):
        """Benchmark ONNX import speed."""
        import time
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto

        nodes = []
        for i in range(10):
            nodes.append(helper.make_node(
                "Relu", inputs=["X" if i == 0 else f"out_{i-1}"],
                outputs=[f"out_{i}" if i < 9 else "Y"],
                name=f"relu_{i}",
            ))

        path, _ = _create_onnx_model(
            input_names_shapes=[("X", [1, 3, 64, 64])],
            output_names_shapes=[("Y", [1, 3, 64, 64])],
            nodes=nodes, graph_name="bench", model_dir=tmp_model_dir,
        )

        from fastnn.io.onnx import import_onnx
        out_path = os.path.join(tmp_model_dir, "bench.fnn")

        # Warmup
        import_onnx(path, out_path)

        # Benchmark
        start = time.time()
        num_iters = 20
        for _ in range(num_iters):
            import_onnx(path, out_path)
        elapsed = time.time() - start
        avg_ms = (elapsed / num_iters) * 1000

        print(f"\nONNX import speed: {avg_ms:.2f}ms per iteration ({num_iters} iters)")

    def test_dag_model_forward_speed(self, tmp_model_dir):
        """Benchmark DAGModel forward pass speed."""
        import time
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto
        import numpy as np

        W = helper.make_tensor("W", TensorProto.FLOAT, [16, 3, 3, 3],
                               np.random.randn(16, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("B", TensorProto.FLOAT, [16],
                               np.random.randn(16).flatten().tolist())

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["conv_out"],
                                name="conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
        relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["Y"], name="relu1")

        path, _ = _create_onnx_model(
            input_names_shapes=[("X", [1, 3, 224, 224])],
            output_names_shapes=[("Y", [1, 3, 224, 224])],
            nodes=[conv, relu], initializers=[W, B],
            graph_name="bench", model_dir=tmp_model_dir,
        )

        _, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        from fastnn.io.dag_model import DAGModel
        from fastnn.io import read_fnn_header, read_fnn_parameters

        with open(fnn_path, "rb") as f:
            _, _, header, num_params = read_fnn_header(f)
        with open(fnn_path, "rb") as f:
            _, _, _, num_params = read_fnn_header(f)
            params = read_fnn_parameters(f, num_params)

        model = DAGModel.from_header(header, params)

        import fastnn as fnn
        x = fnn.tensor(np.random.randn(1, 3, 224, 224).astype(np.float32), [1, 3, 224, 224])

        # Warmup
        model.forward({"X": x})

        # Benchmark
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            model.forward({"X": x})
        elapsed = time.time() - start
        avg_ms = (elapsed / num_iters) * 1000

        print(f"\nDAGModel forward speed: {avg_ms:.2f}ms per iteration ({num_iters} iters)")


class TestFullPipelineIntegration:
    """Full pipeline integration tests using synthetic models.

    These tests exercise the complete ONNX import -> model build ->
    inference path without requiring real model weights.
    """

    def _make_multi_conv_nodes(self):
        """Create ONNX nodes and initializers for a multi-output model."""
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto
        import numpy as np

        W1 = helper.make_tensor("conv1.weight", TensorProto.FLOAT, [16, 3, 3, 3],
                                np.random.randn(16, 3, 3, 3).flatten().tolist())
        W2 = helper.make_tensor("conv2.weight", TensorProto.FLOAT, [8, 3, 3, 3],
                                np.random.randn(8, 3, 3, 3).flatten().tolist())

        conv1 = helper.make_node("Conv", inputs=["images", "conv1.weight"],
                                 outputs=["conv1_out"], name="conv1",
                                 kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["output0"], name="relu1")

        conv2 = helper.make_node("Conv", inputs=["images", "conv2.weight"],
                                 outputs=["output1"], name="conv2",
                                 kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])

        return [conv1, relu1, conv2], [W1, W2]

    def test_multi_output_import_and_build(self, tmp_model_dir):
        """Test importing a multi-output model and building the executor."""
        nodes, initializers = self._make_multi_conv_nodes()
        path, _ = _create_onnx_model(
            output_names_shapes=[("output0", [1, 16, 32, 32]), ("output1", [1, 8, 64, 64])],
            nodes=nodes, initializers=initializers,
            graph_name="multi_output", model_dir=tmp_model_dir,
        )
        result, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        output_ids = result["graph"]["outputs"]
        assert len(output_ids) == 2
        nodes_by_id = {node["id"]: node for node in result["graph"]["nodes"]}
        output_shapes = [nodes_by_id[node_id]["output_shape"]["shape"] for node_id in output_ids]
        assert output_shapes == [
            ["Known(1)", "Known(16)", "Known(32)", "Known(32)"],
            ["Known(1)", "Known(8)", "Known(64)", "Known(64)"],
        ]

        from fastnn.io.graph_builder import build_model_from_fnn
        model = build_model_from_fnn(fnn_path)
        assert model is not None

        import fastnn as fnn
        x = fnn.tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), [1, 3, 64, 64])

        if hasattr(model, "forward"):
            outputs = model.forward({"images": x})
        else:
            outputs = model(x)

        assert outputs is not None
        if isinstance(outputs, dict):
            assert len(outputs) > 0

    def test_compute_graph_through_pipeline(self, tmp_model_dir):
        """Test that the full compute graph survives import->build->infer."""
        pytest.importorskip("onnx")
        from onnx import helper, TensorProto

        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"], name="id1")
        path, _ = _create_onnx_model(
            input_names_shapes=[("X", [1, 3, 4, 4])],
            output_names_shapes=[("Y", [1, 3, 4, 4])],
            nodes=[identity], graph_name="test", model_dir=tmp_model_dir,
        )
        _, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        from fastnn.io.graph_builder import build_model_from_fnn
        model = build_model_from_fnn(fnn_path)

        import fastnn as fnn
        x = fnn.tensor(np.random.randn(1, 3, 4, 4).astype(np.float32), [1, 3, 4, 4])

        if hasattr(model, "forward"):
            outputs = model.forward({"X": x})
        else:
            outputs = model(x)

        assert outputs is not None

    def test_yolo_wrapper_creation(self, tmp_model_dir):
        """Test YOLO wrapper creates model from synthetic ONNX."""
        nodes, initializers = self._make_multi_conv_nodes()
        path, _ = _create_onnx_model(
            output_names_shapes=[("output0", [1, 16, 32, 32]), ("output1", [1, 8, 64, 64])],
            nodes=nodes, initializers=initializers,
            graph_name="multi_output", model_dir=tmp_model_dir,
        )
        _, fnn_path = _run_pipeline(path, fnn_dir=tmp_model_dir)

        from fastnn.io.graph_builder import build_model_from_fnn
        model = build_model_from_fnn(fnn_path)
        assert model is not None, "Model should be built successfully"

    def test_onnx_op_count(self):
        """Test that op handler count is reasonable."""
        from fastnn.io.onnx import import_onnx
        import inspect
        source = inspect.getsource(import_onnx) + inspect.getsource(import_onnx.__globals__["import_onnx_to_compute_graph"])
        op_count = source.count('op_type == "') + len(import_onnx.__globals__["ONNX_TO_IR_OP"])
        assert op_count >= 70, f"Expected >= 70 op handlers, found {op_count}"
        print(f"\nTotal ONNX op handlers: {op_count}")
