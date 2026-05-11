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


class TestExtendedOps:
    def test_erf(self):
        assert infer_shape("Erf", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_cumsum(self):
        assert infer_shape("CumSum", [[1, 3, 8, 8]], {"axis": 1}) == [[1, 3, 8, 8]]

    def test_onehot(self):
        assert infer_shape("OneHot", [[2, 3], [1], [1]], {"axis": -1}) == [[2, 3, 1]]

    def test_gathernd(self):
        assert infer_shape("GatherND", [[1, 3, 8, 8], [1, 1]], {}) == [[1, 3, 8, 8]]

    def test_depthtospace(self):
        assert infer_shape("DepthToSpace", [[1, 12, 8, 8]], {"blocksize": 2}) == [[1, 3, 16, 16]]

    def test_spacetodepth(self):
        assert infer_shape("SpaceToDepth", [[1, 3, 16, 16]], {"blocksize": 2}) == [[1, 12, 8, 8]]

    def test_eyelike(self):
        assert infer_shape("EyeLike", [[5, 5]], {}) == [[5, 5]]

    def test_convtranspose(self):
        assert infer_shape("ConvTranspose", [[1, 3, 8, 8]], {"out_channels": 6, "kernel_size": 3, "stride": 2}) == [[1, 6, 17, 17]]

    def test_instancenormalization(self):
        assert infer_shape("InstanceNormalization", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_logsoftmax(self):
        assert infer_shape("LogSoftmax", [[1, 3, 8, 8]], {"axis": 1}) == [[1, 3, 8, 8]]

    def test_selu(self):
        assert infer_shape("Selu", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_rmsnorm(self):
        assert infer_shape("RmsNormalization", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_loop(self):
        assert infer_shape("Loop", [[1], [1]], {}, num_outputs=2) == [[1], [1]]

    def test_if(self):
        assert infer_shape("If", [[1]], {}, num_outputs=1) == [[1]]

    def test_nonmaxsuppression(self):
        assert infer_shape("NonMaxSuppression", [[100, 4]], {}) == [[1, 3]]

    def test_topk(self):
        assert infer_shape("TopK", [[1, 3, 8, 8]], {"k": 5, "axis": -1}) == [[1, 3, 8, 5], [1, 3, 8, 5]]

    def test_hardsigmoid(self):
        assert infer_shape("HardSigmoid", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_prelu(self):
        assert infer_shape("PRelu", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_constant(self):
        assert infer_shape("Constant", [], {"dims": [3, 4]}) == [[3, 4]]

    def test_compress(self):
        assert infer_shape("Compress", [[1, 3, 8, 8]], {"axis": 0}) == [[None, 3, 8, 8]]


class TestDelegation:
    def test_and_via_delegation(self):
        assert infer_shape("And", [[8], [8]], {}) == [[8]]

    def test_ceil_via_delegation(self):
        assert infer_shape("Ceil", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_equal_via_delegation(self):
        assert infer_shape("Equal", [[8], [8]], {}) == [[8]]

    def test_floor_via_delegation(self):
        assert infer_shape("Floor", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_greater_via_delegation(self):
        assert infer_shape("Greater", [[8], [8]], {}) == [[8]]

    def test_isinf_via_delegation(self):
        assert infer_shape("IsInf", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_isnan_via_delegation(self):
        assert infer_shape("IsNan", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_less_via_delegation(self):
        assert infer_shape("Less", [[8], [8]], {}) == [[8]]

    def test_not_via_delegation(self):
        assert infer_shape("Not", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_or_via_delegation(self):
        assert infer_shape("Or", [[8], [8]], {}) == [[8]]

    def test_range_via_delegation(self):
        assert infer_shape("Range", [[1]], {}) == [[None]]

    def test_reciprocal_via_delegation(self):
        assert infer_shape("Reciprocal", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_round_via_delegation(self):
        assert infer_shape("Round", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_sign_via_delegation(self):
        assert infer_shape("Sign", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]

    def test_xor_via_delegation(self):
        assert infer_shape("Xor", [[8], [8]], {}) == [[8]]

    def test_layernormalization_via_delegation(self):
        assert infer_shape("LayerNormalization", [[1, 3, 8, 8]], {}) == [[1, 3, 8, 8]]
