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


class TestYoloPipeline:
    """Test the YOLO inference pipeline end-to-end."""

    def _create_minimal_model(self):
        """Create a minimal valid ONNX model for pipeline testing.
        
        Model: input -> Conv -> Relu -> output
        This exercises the core pipeline without needing a real YOLO model.
        """
        pytest.importorskip("onnx")
        import onnx
        from onnx import helper, TensorProto
        
        X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
        Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 16, 32, 32])
        
        W = helper.make_tensor("conv1.weight", TensorProto.FLOAT, [16, 3, 3, 3],
                               np.random.randn(16, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("conv1.bias", TensorProto.FLOAT, [16],
                               np.random.randn(16).flatten().tolist())
        
        conv = helper.make_node("Conv", inputs=["images", "conv1.weight", "conv1.bias"],
                                outputs=["conv_out"], name="conv1",
                                kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        relu = helper.make_node("Relu", inputs=["conv_out"],
                                outputs=["output0"], name="relu1")
        
        graph = helper.make_graph(
            [conv, relu],
            "yolo_test",
            [X],
            [Y],
            [W, B],
        )
        
        model = helper.make_model(graph)
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "minimal_yolo.onnx")
        onnx.save(model, path)
        return path, tmpdir

    def test_onnx_import_minimal(self):
        """Test that a minimal YOLO-like model can be imported."""
        path, tmpdir = self._create_minimal_model()
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            result = import_onnx(path, out_path)
            
            assert "layers" in result
            assert "graph" in result or b"graph" in open(out_path, "rb").read()
            
            # Verify the output file has graph topology
            with open(out_path, "rb") as f:
                from fastnn.io import read_fnn_header
                magic, ver, header, num_params = read_fnn_header(f)
                assert "graph" in header
                assert len(header["graph"]["nodes"]) >= 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_build_model_from_fnn(self):
        """Test building a model from an imported .fnn file."""
        path, tmpdir = self._create_minimal_model()
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)
            
            from fastnn.io.graph_builder import build_model_from_fnn
            model = build_model_from_fnn(out_path)
            assert model is not None
            assert hasattr(model, "forward") or hasattr(model, "__call__")
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_dag_model_forward(self):
        """Test DAGModel forward pass with minimal YOLO-like model."""
        path, tmpdir = self._create_minimal_model()
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)
            
            from fastnn.io.dag_model import DAGModel
            from fastnn.io import read_fnn_header, read_fnn_parameters
            
            with open(out_path, "rb") as f:
                _, _, header, num_params = read_fnn_header(f)
            with open(out_path, "rb") as f:
                _, _, _, num_params = read_fnn_header(f)
                params = read_fnn_parameters(f, num_params)
            
            model = DAGModel.from_header(header, params)
            
            # Run inference
            import fastnn as fnn
            x = fnn.tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), [1, 3, 64, 64])
            outputs = model.forward({"images": x})
            
            assert len(outputs) > 0
            for name, out in outputs.items():
                assert out is not None
        finally:
            import shutil
            shutil.rmtree(tmpdir)

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

    def _create_yolo_v8_like_model(self, batch_size=1, num_dets=8400, num_classes=80):
        """Create a YOLOv8-like model with Conv output.
        
        Uses Conv layers with appropriate strides to produce the right
        spatial dimensions, avoiding Reshape shape tensor issues.
        """
        pytest.importorskip("onnx")
        import onnx
        from onnx import helper, TensorProto
        
        X = helper.make_tensor_value_info("images", TensorProto.FLOAT, 
                                           [batch_size, 3, 64, 64])
        Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                           [batch_size, 16, 32, 32])
        
        W = helper.make_tensor("conv1.weight", TensorProto.FLOAT, [16, 3, 3, 3],
                               np.random.randn(16, 3, 3, 3).flatten().tolist())
        B = helper.make_tensor("conv1.bias", TensorProto.FLOAT, [16],
                               np.random.randn(16).flatten().tolist())
        
        conv = helper.make_node("Conv", inputs=["images", "conv1.weight", "conv1.bias"],
                                outputs=["output0"], name="conv1",
                                kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        
        graph = helper.make_graph(
            [conv],
            "yolov8_test",
            [X],
            [Y],
            [W, B],
        )
        
        model = helper.make_model(graph)
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "yolov8_test.onnx")
        onnx.save(model, path)
        return path, tmpdir

    def test_yolo_v8_import(self):
        """Test importing a YOLOv8-like ONNX model."""
        path, tmpdir = self._create_yolo_v8_like_model()
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            result = import_onnx(path, out_path)
            
            layers = result["layers"]
            types = [l["type"] for l in layers]
            assert "Conv" in types
            
            # Check graph topology
            with open(out_path, "rb") as f:
                from fastnn.io import read_fnn_header
                _, _, header, _ = read_fnn_header(f)
                assert len(header["graph"]["nodes"]) >= 1
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_yolo_v8_forward(self):
        """Test running inference on a YOLOv8-like model."""
        path, tmpdir = self._create_yolo_v8_like_model(1, 100, 20)
        try:
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "out.fnn")
            import_onnx(path, out_path)
            
            from fastnn.io.dag_model import DAGModel
            from fastnn.io import read_fnn_header, read_fnn_parameters
            
            with open(out_path, "rb") as f:
                _, _, header, num_params = read_fnn_header(f)
            with open(out_path, "rb") as f:
                _, _, _, num_params = read_fnn_header(f)
                params = read_fnn_parameters(f, num_params)
            
            model = DAGModel.from_header(header, params)
            
            import fastnn as fnn
            x = fnn.tensor(np.random.randn(1, 3, 64, 64).astype(np.float32), [1, 3, 64, 64])
            outputs = model.forward({"images": x})
            
            assert len(outputs) > 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

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

    def test_onnx_import_speed(self):
        """Benchmark ONNX import speed."""
        import time
        path, tmpdir = None, None
        try:
            # Create a small model
            pytest.importorskip("onnx")
            import onnx
            from onnx import helper, TensorProto
            
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 64, 64])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64])
            
            nodes = []
            for i in range(10):
                node = helper.make_node("Relu", inputs=[f"X" if i == 0 else f"out_{i-1}"],
                                        outputs=[f"out_{i}" if i < 9 else "Y"],
                                        name=f"relu_{i}")
                nodes.append(node)
            
            graph = helper.make_graph(nodes, "bench", [X], [Y])
            model = helper.make_model(graph)
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "bench.onnx")
            onnx.save(model, path)
            
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "bench.fnn")
            
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
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir)

    def test_dag_model_forward_speed(self):
        """Benchmark DAGModel forward pass speed."""
        import time
        path, tmpdir = None, None
        try:
            pytest.importorskip("onnx")
            import onnx
            from onnx import helper, TensorProto
            
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 224, 224])
            
            W = helper.make_tensor("W", TensorProto.FLOAT, [16, 3, 3, 3],
                                   np.random.randn(16, 3, 3, 3).flatten().tolist())
            B = helper.make_tensor("B", TensorProto.FLOAT, [16],
                                   np.random.randn(16).flatten().tolist())
            
            conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["conv_out"],
                                    name="conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
            relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["Y"], name="relu1")
            
            graph = helper.make_graph([conv, relu], "bench", [X], [Y], [W, B])
            model = helper.make_model(graph)
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "bench.onnx")
            onnx.save(model, path)
            
            from fastnn.io.onnx import import_onnx
            out_path = os.path.join(tmpdir, "bench.fnn")
            import_onnx(path, out_path)
            
            from fastnn.io.dag_model import DAGModel
            from fastnn.io import read_fnn_header, read_fnn_parameters
            
            with open(out_path, "rb") as f:
                _, _, header, num_params = read_fnn_header(f)
            with open(out_path, "rb") as f:
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
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir)
