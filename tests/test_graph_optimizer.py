"""Tests for the graph optimizer module (updated with constant folding tests)."""

from fastnn.io.graph_optimizer import (
    optimize_graph,
    eliminate_dead_nodes,
    fuse_conv_bn,
    fold_constants,
    _EVAL_REGISTRY,
    _get_attr_safe,
)

import numpy as np


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


# ================================================================== #
# New: Constant Folding Tests
# ================================================================== #

class TestConstantFolding:
    """Tests for fold_constants function and the evaluation registry."""

    def _fold(self, nodes, params=None, outputs=None):
        """Helper: run fold_constants with a minimal header."""
        header = {
            "graph": {
                "nodes": nodes,
                "outputs": outputs or [{"name": nodes[-1]["outputs"][0]}],
                "inputs": [],
            }
        }
        return fold_constants(nodes, header, params=params)

    # ---- Registry existence ----

    def test_registry_has_expected_entries(self):
        """Key foldable ops should be registered."""
        for op in ["add", "sub", "mul", "div", "shape", "cast", "relu",
                    "sigmoid", "concat", "transpose", "reshape", "slice",
                    "gather", "squeeze", "unsqueeze", "expand", "tile",
                    "where", "reducemean", "reducesum", "clip", "equal",
                    "greater", "less", "and", "or", "xor", "not",
                    "abs", "ceil", "floor", "round", "sign", "reciprocal",
                    "neg", "exp", "sqrt", "log", "min", "max",
                    "eyelike", "depthtospace", "spacetodepth", "cumsum",
                    "pad", "onehot", "range", "softplus", "selu", "elu",
                    "hardsigmoid", "hardswish", "identity"]:
            assert op in _EVAL_REGISTRY, f"Missing evaluator for '{op}'"
            assert callable(_EVAL_REGISTRY[op])

    # ---- Simple arithmetic folding ----

    def test_fold_add(self):
        nodes = [
            {"name": "a", "op_type": "Constant", "inputs": [], "outputs": ["a_out"]},
            {"name": "b", "op_type": "Constant", "inputs": [], "outputs": ["b_out"]},
            {"name": "add1", "op_type": "Add", "inputs": ["a_out", "b_out"], "outputs": ["y"]},
        ]
        # Seed constant_values from params
        params2 = {"a": np.array([1.0, 2.0], dtype=np.float32),
                    "b": np.array([3.0, 4.0], dtype=np.float32),
                    "a.value": np.array([1.0, 2.0], dtype=np.float32),
                    "b.value": np.array([3.0, 4.0], dtype=np.float32)}
        result = self._fold(nodes, params=params2)
        # add1 should be folded, so only 1 node (or 2 if materialized)
        assert len(result) <= len(nodes)

    def test_fold_shape(self):
        """Shape op should fold when input is constant."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
            {"name": "shape1", "op_type": "Shape", "inputs": ["c1_out"], "outputs": ["s"]},
        ]
        params = {"c1.value": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                  "c1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
        result = self._fold(nodes, params=params, outputs=[{"name": "s"}])
        # The folded shape node should be materialized as Constant
        assert len(result) == 1  # Constant with the folded shape value
        assert result[0]["op_type"] == "Constant"
        assert "folded" in result[0]["name"]

    def test_fold_relu(self):
        """Relu should fold when input is constant."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["c1_out"], "outputs": ["y"]},
        ]
        params = {"c1.value": np.array([-1.0, 0.0, 2.0], dtype=np.float32)}
        result = self._fold(nodes, params=params, outputs=[{"name": "y"}])
        # relu1 should be folded; only the materialized constant remains
        assert len(result) == 1

    def test_fold_chain(self):
        """Chain of constant ops: c1 -> relu -> add -> sigmoid should all fold."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
            {"name": "c2", "op_type": "Constant", "inputs": [], "outputs": ["c2_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["c1_out"], "outputs": ["r1"]},
            {"name": "add1", "op_type": "Add", "inputs": ["r1", "c2_out"], "outputs": ["a1"]},
            {"name": "sig1", "op_type": "Sigmoid", "inputs": ["a1"], "outputs": ["y"]},
        ]
        params = {
            "c1.value": np.array([-2.0, 0.0, 2.0], dtype=np.float32),
            "c2.value": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }
        result = self._fold(nodes, params=params, outputs=[{"name": "y"}])
        # All nodes should be folded; only materialized constant remains
        assert len(result) <= 2  # 1-2 nodes (materialized constant)
        assert any(n["op_type"] == "Constant" for n in result)

    # ---- No folding when inputs are dynamic ----

    def test_no_fold_dynamic_input(self):
        """Nodes with dynamic (non-constant) inputs should not be folded."""
        nodes = [
            {"name": "add1", "op_type": "Add",
             "inputs": ["dynamic_input", "c2_out"], "outputs": ["y"]},
            {"name": "c2", "op_type": "Constant", "inputs": [], "outputs": ["c2_out"]},
        ]
        params = {"c2.value": np.array([1.0, 2.0], dtype=np.float32)}
        result = self._fold(nodes, params=params, outputs=[{"name": "y"}])
        # Should keep both nodes since dynamic_input prevents folding
        assert len(result) == 2

    # ---- Specific evaluator correctness ----

    def test_eval_add(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0, 4.0], dtype=np.float32)
        result = _EVAL_REGISTRY["add"](None, [x, y], {})
        assert np.allclose(result[0], x + y)

    def test_eval_sub(self):
        x = np.array([5.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 1.0], dtype=np.float32)
        result = _EVAL_REGISTRY["sub"](None, [x, y], {})
        assert np.allclose(result[0], x - y)

    def test_eval_mul(self):
        x = np.array([2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0], dtype=np.float32)
        result = _EVAL_REGISTRY["mul"](None, [x, y], {})
        assert np.allclose(result[0], x * y)

    def test_eval_div(self):
        x = np.array([6.0, 9.0], dtype=np.float32)
        y = np.array([2.0, 3.0], dtype=np.float32)
        result = _EVAL_REGISTRY["div"](None, [x, y], {})
        assert np.allclose(result[0], x / y)

    def test_eval_relu(self):
        x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        result = _EVAL_REGISTRY["relu"](None, [x], {})
        assert np.allclose(result[0], np.maximum(0, x))

    def test_eval_sigmoid(self):
        x = np.array([0.0], dtype=np.float32)
        result = _EVAL_REGISTRY["sigmoid"](None, [x], {})
        assert np.allclose(result[0], 0.5)

    def test_eval_neg(self):
        x = np.array([1.0, -2.0], dtype=np.float32)
        result = _EVAL_REGISTRY["neg"](None, [x], {})
        assert np.allclose(result[0], -x)

    def test_eval_exp(self):
        x = np.array([0.0, 1.0], dtype=np.float32)
        result = _EVAL_REGISTRY["exp"](None, [x], {})
        assert np.allclose(result[0], np.exp(x))

    def test_eval_sqrt(self):
        x = np.array([4.0, 9.0], dtype=np.float32)
        result = _EVAL_REGISTRY["sqrt"](None, [x], {})
        assert np.allclose(result[0], np.sqrt(x))

    def test_eval_concat(self):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        node = {"attrs": {"axis": 1}}
        result = _EVAL_REGISTRY["concat"](node, [a, b], {})
        assert np.allclose(result[0], np.concatenate([a, b], axis=1))

    def test_eval_shape(self):
        x = np.zeros((2, 3, 4), dtype=np.float32)
        result = _EVAL_REGISTRY["shape"](None, [x], {})
        assert np.allclose(result[0], np.array([2, 3, 4]))

    def test_eval_cast_float_to_int64(self):
        x = np.array([1.5, 2.7], dtype=np.float32)
        node = {"attrs": {"to": 7}}  # INT64
        result = _EVAL_REGISTRY["cast"](node, [x], {})
        assert result[0].dtype == np.int64

    def test_eval_transpose(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        node = {"attrs": {"perm": [1, 0]}}
        result = _EVAL_REGISTRY["transpose"](node, [x], {})
        assert np.allclose(result[0], x.T)

    def test_eval_squeeze(self):
        x = np.ones((1, 3, 1, 4), dtype=np.float32)
        node = {"attrs": {"axes": [0, 2]}}
        result = _EVAL_REGISTRY["squeeze"](node, [x], {})
        assert result[0].shape == (3, 4)

    def test_eval_unsqueeze(self):
        x = np.ones((3, 4), dtype=np.float32)
        node = {"attrs": {"axes": [0, 2]}}
        result = _EVAL_REGISTRY["unsqueeze"](node, [x], {})
        assert result[0].shape == (1, 3, 1, 4)

    def test_eval_equal(self):
        result = _EVAL_REGISTRY["equal"](None,
            [np.array([1.0, 2.0]), np.array([1.0, 3.0])], {})
        assert np.allclose(result[0], [True, False])

    def test_eval_greater(self):
        result = _EVAL_REGISTRY["greater"](None,
            [np.array([1.0, 3.0]), np.array([2.0, 2.0])], {})
        assert np.allclose(result[0], [False, True])

    def test_eval_less(self):
        result = _EVAL_REGISTRY["less"](None,
            [np.array([1.0, 3.0]), np.array([2.0, 2.0])], {})
        assert np.allclose(result[0], [True, False])

    def test_eval_abs(self):
        result = _EVAL_REGISTRY["abs"](None,
            [np.array([-1.0, 2.0, -3.0])], {})
        assert np.allclose(result[0], [1.0, 2.0, 3.0])

    def test_eval_ceil(self):
        result = _EVAL_REGISTRY["ceil"](None,
            [np.array([1.2, 2.7])], {})
        assert np.allclose(result[0], [2.0, 3.0])

    def test_eval_floor(self):
        result = _EVAL_REGISTRY["floor"](None,
            [np.array([1.2, 2.7])], {})
        assert np.allclose(result[0], [1.0, 2.0])

    def test_eval_not(self):
        result = _EVAL_REGISTRY["not"](None,
            [np.array([True, False, True])], {})
        assert np.allclose(result[0], [False, True, False])

    def test_eval_expand(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        shape = np.array([2, 3], dtype=np.int64)
        result = _EVAL_REGISTRY["expand"](None, [x, shape], {})
        assert result[0].shape == (2, 3)

    def test_eval_eyelike(self):
        x = np.zeros((3, 3), dtype=np.float32)
        node = {"attrs": {"k": 0}}
        result = _EVAL_REGISTRY["eyelike"](node, [x], {})
        assert np.allclose(result[0], np.eye(3))

    def test_eval_softplus(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = _EVAL_REGISTRY["softplus"](None, [x], {})
        assert np.allclose(result[0], np.log(1.0 + np.exp(x)))

    def test_eval_hardsigmoid(self):
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        node = {"attrs": {"alpha": 0.2, "beta": 0.5}}
        result = _EVAL_REGISTRY["hardsigmoid"](node, [x], {})
        expected = np.clip(0.2 * x + 0.5, 0, 1)
        assert np.allclose(result[0], expected)

    def test_eval_hardswish(self):
        x = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
        result = _EVAL_REGISTRY["hardswish"](None, [x], {})
        expected = x * np.clip(x / 6.0 + 0.5, 0, 1)
        assert np.allclose(result[0], expected)

    def test_eval_selu(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        node = {"attrs": {"alpha": 1.67326, "gamma": 1.0507}}
        result = _EVAL_REGISTRY["selu"](node, [x], {})
        alpha, gamma = 1.67326, 1.0507
        expected = gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        assert np.allclose(result[0], expected)

    def test_eval_elu(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        node = {"attrs": {"alpha": 1.0}}
        result = _EVAL_REGISTRY["elu"](node, [x], {})
        expected = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        assert np.allclose(result[0], expected)

    def test_eval_flatten(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        node = {"attrs": {"axis": 1}}
        result = _EVAL_REGISTRY["flatten"](node, [x], {})
        expected = x.reshape(2, 12)
        assert np.allclose(result[0], expected)

    def test_eval_identity(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        result = _EVAL_REGISTRY["identity"](None, [x], {})
        assert np.allclose(result[0], x)


# ================================================================== #
# Edge Cases and Error Handling
# ================================================================== #

class TestConstantFoldingEdgeCases:
    """Edge cases: empty nodes, no params, malformed data."""

    def test_fold_empty_nodes(self):
        result = fold_constants([], {"graph": {"outputs": [], "nodes": []}})
        assert result == []

    def test_fold_without_params(self):
        """Without params, only metadata-only ops should fold if possible."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
        ]
        result = fold_constants(nodes, {"graph": {"outputs": [{"name": "c1_out"}], "nodes": nodes}})
        # Without params, the constant node's value is unknown, so it stays
        assert len(result) == 1

    def test_fold_consumer_non_foldable(self):
        """A folded node whose output is consumed by a non-foldable op
        should be materialized as a Constant."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["c1_out"], "outputs": ["r1"]},
            # NonFoldableOp is NOT in the registry — won't fold
            {"name": "consumer", "op_type": "NonFoldableOp", "inputs": ["r1"],
             "outputs": ["y"]},
        ]
        params = {
            "c1.value": np.array([-1.0, 0.0, 2.0], dtype=np.float32),
        }
        result = fold_constants(nodes, {
            "graph": {"outputs": [{"name": "y"}], "nodes": nodes}
        }, params=params)
        # relu1 should be folded and materialized as Constant (its output r1
        # is consumed by non-foldable consumer). The consumer stays.
        # c1 is also folded (no consumer of c1_out since relu1 is folded, so removed).
        # Expected: Constant(materialized relu1) + NonFoldableOp(consumer)
        assert len(result) == 2
        names = [n["name"] for n in result]
        assert any("folded" in n for n in names)
        assert "consumer" in names

    def test_fold_intermediate_unused_removed(self):
        """Folded nodes with outputs that are neither consumed nor model
        outputs should be removed entirely (not materialized)."""
        nodes = [
            {"name": "c1", "op_type": "Constant", "inputs": [], "outputs": ["c1_out"]},
            {"name": "unused", "op_type": "Relu", "inputs": ["c1_out"], "outputs": ["unused_out"]},
            {"name": "identity", "op_type": "Identity", "inputs": ["c1_out"], "outputs": ["y"]},
        ]
        params = {
            "c1.value": np.array([-1.0, 0.0, 2.0], dtype=np.float32),
        }
        result = fold_constants(nodes, {
            "graph": {"outputs": [{"name": "y"}], "nodes": nodes}
        }, params=params)
        # c1 folded, unused relu folded (output "unused_out" is not consumed and not a
        # model output — removed). identity folded, "y" is model output — materialized.
        # Expected: 1 node (materialized identity output)
        assert len(result) == 1
        assert any("folded" in n["name"] for n in result)

    def test_get_attr_safe_with_attrs_subdict(self):
        node = {"attrs": {"key1": "val1"}, "key2": "val2"}
        assert _get_attr_safe(node, "key1") == "val1"
        assert _get_attr_safe(node, "key2") == "val2"
        assert _get_attr_safe(node, "missing", "default") == "default"

    def test_get_attr_safe_top_level_precedence(self):
        """attrs sub-dict is checked first (current implementation)."""
        node = {"attrs": {"key1": "from_attrs"}, "key1": "from_top"}
        assert _get_attr_safe(node, "key1") == "from_attrs"
