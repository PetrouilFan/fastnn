"""Tests for the graph optimizer module."""

from fastnn.io.graph_optimizer import (
    optimize_graph,
    eliminate_dead_nodes,
    fuse_conv_bn,
)


class TestDeadNodeElimination:
    def test_no_dead_nodes(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X"], "outputs": ["conv1_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["conv1_out"], "outputs": ["Y"]},
        ]
        graph = {"outputs": [{"name": "Y"}]}
        result = eliminate_dead_nodes(nodes, graph)
        assert len(result) == 2

    def test_eliminate_dead_node(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X"], "outputs": ["conv1_out"]},
            {"name": "dead_conv", "op_type": "Conv", "inputs": ["X"], "outputs": ["dead_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["conv1_out"], "outputs": ["Y"]},
        ]
        graph = {"outputs": [{"name": "Y"}]}
        result = eliminate_dead_nodes(nodes, graph)
        assert len(result) == 2  # dead_conv removed
        names = [n["name"] for n in result]
        assert "dead_conv" not in names

    def test_keep_model_outputs(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X"], "outputs": ["conv1_out"]},
            {"name": "branched_out", "op_type": "Identity", "inputs": ["conv1_out"], "outputs": ["aux_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["conv1_out"], "outputs": ["Y"]},
        ]
        graph = {"outputs": [{"name": "Y"}, {"name": "aux_out"}]}
        result = eliminate_dead_nodes(nodes, graph)
        assert len(result) == 3  # All kept since aux_out is a model output


class TestConvBnFusion:
    def test_fuse_conv_bn(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X", "W", "B"],
             "outputs": ["conv1_out"], "attrs": {"kernel_shape": [3, 3]}},
            {"name": "bn1", "op_type": "BatchNormalization",
             "inputs": ["conv1_out", "BN_W", "BN_B", "BN_M", "BN_V"],
             "outputs": ["bn1_out"], "attrs": {"epsilon": 1e-5}},
            {"name": "relu1", "op_type": "Relu", "inputs": ["bn1_out"], "outputs": ["Y"]},
        ]
        result = fuse_conv_bn(nodes)
        # Conv fused into FusedConvBn, BN removed → 2 nodes: FusedConvBn + Relu
        assert len(result) == 2
        # The first node should be FusedConvBn
        assert result[0]["op_type"] == "FusedConvBn"
        # Its output should be bn1_out (same as original BN)
        assert result[0]["outputs"] == ["bn1_out"]

    def test_no_fusion_without_bn(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X", "W"],
             "outputs": ["conv1_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["conv1_out"], "outputs": ["Y"]},
        ]
        result = fuse_conv_bn(nodes)
        assert len(result) == 2
        assert result[0]["op_type"] == "Conv"

    def test_fuse_with_batchnorm2d(self):
        nodes = [
            {"name": "conv1", "op_type": "Conv", "inputs": ["X", "W", "B"],
             "outputs": ["conv1_out"]},
            {"name": "bn1", "op_type": "BatchNorm2d",
             "inputs": ["conv1_out", "BN_W", "BN_B", "BN_M", "BN_V"],
             "outputs": ["bn1_out"]},
        ]
        result = fuse_conv_bn(nodes)
        assert len(result) == 1
        assert result[0]["op_type"] == "FusedConvBn"


class TestOptimizeGraph:
    def test_optimize_without_graph(self):
        header = {"layers": [{"name": "conv1", "type": "Conv"}]}
        result = optimize_graph(header)
        assert result == header

    def test_optimize_full_pipeline(self):
        header = {
            "graph": {
                "nodes": [
                    {"name": "conv1", "op_type": "Conv", 
                     "inputs": ["X", "W", "B"], "outputs": ["conv1_out"]},
                    {"name": "bn1", "op_type": "BatchNormalization",
                     "inputs": ["conv1_out", "BN_W", "BN_B", "BN_M", "BN_V"],
                     "outputs": ["bn1_out"]},
                    {"name": "dead_node", "op_type": "Identity",
                     "inputs": ["X"], "outputs": ["dead_out"]},
                    {"name": "relu1", "op_type": "Relu",
                     "inputs": ["bn1_out"], "outputs": ["Y"]},
                ],
                "outputs": [{"name": "Y"}],
                "inputs": [{"name": "X", "shape": [1, 3, 8, 8]}],
            }
        }
        result = optimize_graph(header)
        nodes = result["graph"]["nodes"]
        # Should have 2 nodes: FusedConvBn, Relu (dead_node eliminated)
        assert len(nodes) == 2
        assert nodes[0]["op_type"] == "FusedConvBn"
        assert nodes[1]["op_type"] == "Relu"
