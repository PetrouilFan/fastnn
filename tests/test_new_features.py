"""Tests for missing features: AvgPool2d, PReLU, DAGExecutor, graph optimizer, shape inference."""

import pytest
import numpy as np
import fastnn as fnn


class TestAvgPool2d:
    def test_avgpool2d_forward(self):
        """Test AvgPool2d forward pass produces correct output shape."""
        pool = fnn.AvgPool2d(kernel_size=2, stride=2)
        x = fnn.ones([1, 3, 4, 4])
        y = pool(x)
        assert y.shape == [1, 3, 2, 2], f"Expected [1, 3, 2, 2], got {y.shape}"

    def test_avgpool2d_forward_values(self):
        """Test AvgPool2d computes correct average values."""
        pool = fnn.AvgPool2d(kernel_size=2, stride=2)
        data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
        x = fnn.tensor(data, list(data.shape))
        y = pool(x)
        result = y.numpy()
        expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_avgpool2d_with_stride(self):
        """Test AvgPool2d with different stride and kernel."""
        pool = fnn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        x = fnn.ones([1, 1, 5, 5])
        y = pool(x)
        assert y.shape == [1, 1, 5, 5], f"Expected [1, 1, 5, 5], got {y.shape}"

    @pytest.mark.xfail(reason="PyTensor does not support writable requires_grad attribute yet", strict=False)
    def test_avgpool2d_gradient(self):
        """Test AvgPool2d backward pass (gradient flow)."""
        pool = fnn.AvgPool2d(kernel_size=2, stride=2)
        x = fnn.ones([1, 1, 4, 4])
        x.requires_grad = True
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should not be None"
        assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} should match input shape {x.shape}"


class TestPReLU:
    def test_prelu_forward(self):
        """Test PReLU forward pass."""
        prelu = fnn.PReLU(num_parameters=1)
        x = fnn.tensor(np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32), [1, 4])
        y = prelu(x)
        result = y.numpy()
        expected = np.array([[1.0, -0.25, 0.5, -0.125]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_prelu_with_custom_alpha(self):
        """Test PReLU with custom alpha parameter."""
        prelu = fnn.PReLU(num_parameters=3)
        params = prelu.parameters()
        assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"

    @pytest.mark.xfail(reason="PyTensor does not support writable requires_grad attribute yet", strict=False)
    def test_prelu_gradient(self):
        """Test PReLU backward pass."""
        prelu = fnn.PReLU(num_parameters=1)
        x = fnn.tensor(np.array([[1.0, -1.0]], dtype=np.float32), [1, 2])
        x.requires_grad = True
        y = prelu(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should not be None"


class TestDAGExecutor:
    """Test the Rust DAGExecutor directly."""

    def test_simple_relu_graph(self):
        """Test DAGExecutor with a simple Relu node."""
        nodes = [
            {
                "name": "relu1",
                "op_type": "Relu",
                "inputs": "X",
                "outputs": "Y",
            }
        ]
        params = {}
        input_names = ["X"]
        output_names = ["Y"]

        executor = fnn.DAGExecutor(nodes, params, input_names, output_names)

        x = fnn.tensor(np.array([[-1.0, 0.0, 1.0, -2.0]], dtype=np.float32), [1, 4])
        outputs = executor.forward({"X": x})

        assert "Y" in outputs, f"Expected 'Y' in outputs, got {list(outputs.keys())}"
        result = outputs["Y"].numpy()
        expected = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_graph(self):
        """Test DAGExecutor with Add node."""
        nodes = [
            {
                "name": "add1",
                "op_type": "Add",
                "inputs": "A, B",
                "outputs": "C",
            }
        ]
        params = {}
        input_names = ["A", "B"]
        output_names = ["C"]

        executor = fnn.DAGExecutor(nodes, params, input_names, output_names)

        a = fnn.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), [3])
        b = fnn.tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32), [3])
        outputs = executor.forward({"A": a, "B": b})

        assert "C" in outputs
        result = outputs["C"].numpy()
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_conv_relu_graph(self):
        """Test DAGExecutor with Conv -> Relu."""
        in_c, out_c = 1, 2
        weight_data = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
        bias_data = np.random.randn(out_c).astype(np.float32)

        nodes = [
            {
                "name": "conv1",
                "op_type": "Conv",
                "inputs": "X, conv1.weight, conv1.bias",
                "outputs": "conv_out",
                "stride": "1",
                "padding": "1",
            },
            {
                "name": "relu1",
                "op_type": "Relu",
                "inputs": "conv_out",
                "outputs": "Y",
            }
        ]

        params = {
            "conv1.weight": fnn.tensor(weight_data, list(weight_data.shape)),
            "conv1.bias": fnn.tensor(bias_data, list(bias_data.shape)),
        }

        executor = fnn.DAGExecutor(nodes, params, ["X"], ["Y"])

        x = fnn.tensor(np.random.randn(1, in_c, 8, 8).astype(np.float32), [1, in_c, 8, 8])
        outputs = executor.forward({"X": x})

        assert "Y" in outputs
        result = outputs["Y"].numpy()
        assert result.shape == (1, out_c, 8, 8), f"Expected shape (1, {out_c}, 8, 8), got {result.shape}"
        assert np.all(result >= -1e-5), "Relu output should be non-negative"

    def test_mul_graph(self):
        """Test DAGExecutor with elementwise Mul."""
        nodes = [
            {
                "name": "mul1",
                "op_type": "Mul",
                "inputs": "A, B",
                "outputs": "C",
            }
        ]
        executor = fnn.DAGExecutor(nodes, {}, ["A", "B"], ["C"])

        a = fnn.tensor(np.array([2.0, 3.0], dtype=np.float32), [2])
        b = fnn.tensor(np.array([4.0, 5.0], dtype=np.float32), [2])
        outputs = executor.forward({"A": a, "B": b})

        result = outputs["C"].numpy()
        expected = np.array([8.0, 15.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sigmoid_graph(self):
        """Test DAGExecutor with Sigmoid."""
        nodes = [{"name": "sig1", "op_type": "Sigmoid", "inputs": "X", "outputs": "Y"}]
        executor = fnn.DAGExecutor(nodes, {}, ["X"], ["Y"])

        x = fnn.tensor(np.array([0.0, 1.0, -1.0], dtype=np.float32), [3])
        outputs = executor.forward({"X": x})

        result = outputs["Y"].numpy()
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0])))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_reshape_graph(self):
        """Test DAGExecutor with Reshape."""
        nodes = [
            {
                "name": "reshape1",
                "op_type": "Reshape",
                "inputs": "X, shape",
                "outputs": "Y",
            }
        ]
        params = {
            "shape": fnn.tensor(np.array([2, 4], dtype=np.float32), [2]),
        }
        executor = fnn.DAGExecutor(nodes, params, ["X"], ["Y"])

        x = fnn.tensor(np.random.randn(2, 4).astype(np.float32), [2, 4])
        outputs = executor.forward({"X": x})

        assert "Y" in outputs
        result = outputs["Y"].numpy()
        assert result.shape == (2, 4), f"Expected (2, 4), got {result.shape}"

    def test_concat_graph(self):
        """Test DAGExecutor with Concat."""
        nodes = [
            {
                "name": "concat1",
                "op_type": "Concat",
                "inputs": "A, B",
                "outputs": "C",
                "axis": "0",
            }
        ]
        executor = fnn.DAGExecutor(nodes, {}, ["A", "B"], ["C"])

        a = fnn.tensor(np.array([[1.0], [2.0]], dtype=np.float32), [2, 1])
        b = fnn.tensor(np.array([[3.0], [4.0]], dtype=np.float32), [2, 1])
        outputs = executor.forward({"A": a, "B": b})

        result = outputs["C"].numpy()
        assert result.shape == (4, 1), f"Expected (4, 1), got {result.shape}"

    def test_multi_node_graph(self):
        """Test DAGExecutor with a multi-node graph (X -> Relu -> Sig -> Y)."""
        nodes = [
            {"name": "relu1", "op_type": "Relu", "inputs": "X", "outputs": "r1_out"},
            {"name": "sig1", "op_type": "Sigmoid", "inputs": "r1_out", "outputs": "Y"},
        ]
        executor = fnn.DAGExecutor(nodes, {}, ["X"], ["Y"])

        x = fnn.tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), [5])
        outputs = executor.forward({"X": x})

        result = outputs["Y"].numpy()
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 0.0, 0.0, 1.0, 2.0])))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestGraphOptimizerIntegration:
    def test_conv_bn_fusion_with_params(self):
        """Test Conv+BN fusion with actual parameter computation."""
        from fastnn.io.graph_optimizer import fuse_conv_bn

        nodes = [
            {"name": "conv1", "op_type": "Conv",
             "inputs": ["X", "conv1.weight", "conv1.bias"],
             "outputs": ["conv1_out"],
             "kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
            {"name": "bn1", "op_type": "BatchNormalization",
             "inputs": ["conv1_out", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"],
             "outputs": ["bn1_out"], "epsilon": 1e-5},
            {"name": "relu1", "op_type": "Relu", "inputs": ["bn1_out"], "outputs": ["Y"]},
        ]

        result = fuse_conv_bn(nodes)
        assert len(result) == 2, f"Expected 2 nodes after fusion, got {len(result)}"
        assert result[0]["op_type"] == "FusedConvBn", f"Expected FusedConvBn, got {result[0]['op_type']}"
        assert result[1]["op_type"] == "Relu"

    def test_full_optimize_pipeline(self):
        """Test the full optimize_graph pipeline."""
        from fastnn.io.graph_optimizer import optimize_graph

        header = {
            "graph": {
                "nodes": [
                    {"name": "conv1", "op_type": "Conv",
                     "inputs": ["X", "conv1.weight", "conv1.bias"],
                     "outputs": ["conv1_out"]},
                    {"name": "dead_node", "op_type": "Identity",
                     "inputs": ["X"], "outputs": ["dead_out"]},
                    {"name": "relu1", "op_type": "Relu",
                     "inputs": ["conv1_out"], "outputs": ["Y"]},
                ],
                "outputs": [{"name": "Y"}],
                "inputs": [{"name": "X", "shape": [1, 3, 8, 8]}],
            }
        }
        result = optimize_graph(header)
        nodes = result["graph"]["nodes"]
        assert len(nodes) == 2, f"Expected 2 nodes after optimization, got {len(nodes)}"
        names = [n["name"] for n in nodes]
        assert "dead_node" not in names, "dead_node should have been eliminated"


class TestShapeInferenceIntegration:
    def test_yolo_like_shapes(self):
        """Test shape inference for a YOLO-like computation path."""
        from fastnn.io.shape_inference import infer_shape

        shapes = infer_shape("Conv", [[1, 64, 80, 80]], {
            "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1
        })
        assert shapes == [[1, 128, 40, 40]]

        shapes = infer_shape("BatchNormalization", [[1, 128, 40, 40]], {
            "num_features": 128
        })
        assert shapes == [[1, 128, 40, 40]]

        shapes = infer_shape("MaxPool", [[1, 256, 20, 20]], {
            "kernel_size": 5, "stride": 1, "padding": 2
        })
        assert shapes == [[1, 256, 20, 20]]

        shapes = infer_shape("Concat", [[1, 256, 20, 20], [1, 256, 20, 20]], {"axis": 1})
        assert shapes == [[1, 512, 20, 20]]

    def test_upsample_shapes(self):
        """Test shape inference for upsampling."""
        from fastnn.io.shape_inference import infer_shape

        shapes = infer_shape("Resize", [[1, 128, 20, 20]], {"scales": [1.0, 1.0, 2.0, 2.0]})
        assert shapes == [[1, 128, 40, 40]]

    def test_split_shapes(self):
        """Test shape inference for Split."""
        from fastnn.io.shape_inference import infer_shape

        shapes = infer_shape("Split", [[1, 128, 40, 40]], {"axis": 1, "split": [64, 64]}, num_outputs=2)
        assert len(shapes) == 2
        assert shapes[0] == [1, 64, 40, 40]
        assert shapes[1] == [1, 64, 40, 40]


class TestDAGExecutorModule:
    def test_parameters_method(self):
        """Test DAGExecutor.parameters() returns correct params."""
        params = {
            "conv1.weight": fnn.ones([3, 1, 3, 3]),
            "conv1.bias": fnn.zeros([3]),
        }
        nodes = [{"name": "conv1", "op_type": "Conv", "inputs": "X, conv1.weight, conv1.bias", "outputs": "Y", "stride": "1", "padding": "1"}]
        executor = fnn.DAGExecutor(nodes, params, ["X"], ["Y"])

        p = executor.parameters()
        assert len(p) == 2, f"Expected 2 parameters, got {len(p)}"

    def test_named_parameters(self):
        """Test DAGExecutor.named_parameters()."""
        params = {
            "conv1.weight": fnn.ones([3, 1, 3, 3]),
        }
        nodes = [{"name": "conv1", "op_type": "Conv", "inputs": "X, conv1.weight", "outputs": "Y", "stride": "1", "padding": "1"}]
        executor = fnn.DAGExecutor(nodes, params, ["X"], ["Y"])

        named = executor.named_parameters()
        assert len(named) == 1
        name, _ = named[0]
        assert "weight" in name

    def test_zero_grad(self):
        """Test DAGExecutor.zero_grad() doesn't crash."""
        params = {
            "conv1.weight": fnn.ones([3, 1, 3, 3]),
        }
        nodes = [{"name": "conv1", "op_type": "Conv", "inputs": "X, conv1.weight", "outputs": "Y", "stride": "1", "padding": "1"}]
        executor = fnn.DAGExecutor(nodes, params, ["X"], ["Y"])
        executor.zero_grad()
