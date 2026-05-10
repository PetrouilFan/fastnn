"""Tests for shape inference module."""

from fastnn.io.shape_inference import infer_shape


class TestShapeInference:
    def test_relu(self):
        assert infer_shape("Relu", [[1, 3, 64, 64]], {}) == [[1, 3, 64, 64]]

    def test_conv(self):
        shapes = infer_shape("Conv", [[1, 3, 8, 8]], {
            "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1
        })
        assert shapes == [[1, 6, 8, 8]]

    def test_conv_stride2(self):
        shapes = infer_shape("Conv", [[1, 3, 8, 8]], {
            "out_channels": 6, "kernel_size": 3, "stride": 2, "padding": 1
        })
        assert shapes == [[1, 6, 4, 4]]

    def test_gemm(self):
        shapes = infer_shape("Gemm", [[1, 16]], {"in_features": 16, "out_features": 8, "transB": 1})
        assert shapes == [[1, 8]]

    def test_maxpool(self):
        shapes = infer_shape("MaxPool", [[1, 3, 16, 16]], {"kernel_size": 3, "stride": 2, "padding": 1})
        assert shapes == [[1, 3, 8, 8]]

    def test_avgpool(self):
        shapes = infer_shape("AveragePool", [[1, 3, 16, 16]], {"kernel_size": 3, "stride": 2, "padding": 0})
        assert shapes == [[1, 3, 7, 7]]

    def test_global_avg_pool(self):
        shapes = infer_shape("GlobalAveragePool", [[1, 3, 16, 16]], {})
        assert shapes == [[1, 3, 1, 1]]

    def test_reshape(self):
        shapes = infer_shape("Reshape", [[1, 3, 8, 8]], {"shape": [1, 192]})
        assert shapes == [[1, 192]]

    def test_flatten(self):
        shapes = infer_shape("Flatten", [[1, 3, 8, 8]], {"axis": 1})
        assert shapes == [[1, 192]]

    def test_transpose(self):
        shapes = infer_shape("Transpose", [[1, 3, 8, 8]], {"perm": [0, 2, 3, 1]})
        assert shapes == [[1, 8, 8, 3]]

    def test_concat(self):
        shapes = infer_shape("Concat", [[1, 3, 8, 8], [1, 6, 8, 8]], {"axis": 1})
        assert shapes == [[1, 9, 8, 8]]

    def test_split(self):
        shapes = infer_shape("Split", [[1, 64, 8, 8]], {"axis": 1, "split": [32, 32]}, num_outputs=2)
        assert len(shapes) == 2
        assert shapes[0] == [1, 32, 8, 8]
        assert shapes[1] == [1, 32, 8, 8]

    def test_add_broadcast(self):
        shapes = infer_shape("Add", [[1, 3, 8, 8], [1, 3, 8, 1]], {})
        assert shapes == [[1, 3, 8, 8]]

    def test_matmul(self):
        shapes = infer_shape("MatMul", [[1, 16], [16, 8]], {})
        assert shapes == [[1, 8]]

    def test_reduce_mean(self):
        shapes = infer_shape("ReduceMean", [[1, 3, 8, 8]], {"axes": [2, 3], "keepdims": True})
        assert shapes == [[1, 3, 1, 1]]

    def test_shape(self):
        shapes = infer_shape("Shape", [[1, 3, 64, 64]], {})
        assert shapes == [[4]]  # 4-dim shape

    def test_squeeze(self):
        shapes = infer_shape("Squeeze", [[1, 3, 1, 64]], {"axes": [0, 2]})
        assert shapes == [[3, 64]]

    def test_unsqueeze(self):
        shapes = infer_shape("Unsqueeze", [[3, 64]], {"axes": [0, 2]})
        assert shapes == [[1, 3, 1, 64]]

    def test_slice(self):
        shapes = infer_shape("Slice", [[1, 3, 64, 64]], {"starts": [0, 0, 0, 0], "ends": [1, 3, 32, 32], "axes": [0, 1, 2, 3], "steps": [1, 1, 1, 1]})
        assert shapes == [[1, 3, 32, 32]]

    def test_resize(self):
        shapes = infer_shape("Resize", [[1, 3, 8, 8]], {"scales": [1.0, 1.0, 2.0, 2.0]})
        assert shapes == [[1, 3, 16, 16]]

    def test_pad(self):
        shapes = infer_shape("Pad", [[1, 3, 8, 8]], {"pads": [0, 0, 1, 1, 0, 0, 1, 1]})
        assert shapes == [[1, 3, 10, 10]]

    def test_where(self):
        shapes = infer_shape("Where", [[1, 3, 8, 8], [1, 3, 8, 8], [1, 3, 8, 8]], {})
        assert shapes == [[1, 3, 8, 8]]
