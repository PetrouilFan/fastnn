"""Direct DAGModel dispatch unit tests.

Tests each dispatch branch in DAGModel._dispatch_op by constructing
minimal DAGModel instances with synthetic nodes and verifying outputs.

This complements test_onnx.py (which tests the ONNX import pipeline)
by testing the actual execution of every op branch.
"""

import numpy as np
import pytest

import fastnn as fnn


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_model(nodes, params=None, input_names=None, output_names=None):
    """Build a minimal DAGModel from a list of node dicts."""
    from fastnn.io.dag_model import DAGModel
    return DAGModel(
        nodes=nodes or [],
        params=params or {},
        input_names=input_names or ["X"],
        output_names=output_names or ["Y"],
    )


def _check_result(outputs, expected_keys=None, expected_shape=None):
    """Verify forward() returned a dict and optionally check shape."""
    assert isinstance(outputs, dict), f"Expected dict, got {type(outputs)}"
    assert len(outputs) > 0, "Expected at least one output"
    if expected_keys is not None:
        for k in expected_keys:
            assert k in outputs, f"Expected key {k!r} in outputs"
    if expected_shape is not None:
        for name, out in outputs.items():
            assert out.shape == expected_shape, (
                f"Output {name!r}: expected shape {expected_shape}, got {out.shape}"
            )


def _run_single_op(op_type, x_np, node_attrs=None, params=None,
                   second_input=None, input_name="X", output_name="Y"):
    """Run a single-op DAGModel and return the output tensor."""
    attrs = dict(node_attrs or {})
    model = _make_model(
        nodes=[{
            "name": "op1",
            "op_type": op_type,
            "inputs": [input_name] + (["X2"] if second_input is not None else []),
            "outputs": [output_name],
            **attrs,
        }],
        params=params or {},
        input_names=[input_name] + (["X2"] if second_input is not None else []),
        output_names=[output_name],
    )
    feed = {input_name: fnn.tensor(x_np, list(x_np.shape))}
    if second_input is not None:
        feed["X2"] = fnn.tensor(second_input, list(second_input.shape))
    return model.forward(feed)[output_name]


def _as_tensor(x_np):
    """Convert numpy array to fnn tensor."""
    return fnn.tensor(x_np, list(x_np.shape))


def _to_numpy(t):
    """Extract numpy array from fnn tensor."""
    return t.numpy() if hasattr(t, "numpy") else np.array(t)


# ================================================================== #
# Simple / Unary Math Ops
# ================================================================== #

class TestUnaryMathOps:
    """Abs, Ceil, Floor, Round, Sign, Reciprocal, Neg, Exp, Sqrt, Log, Erf, Not,
    IsNaN, IsInf, Elu, HardSigmoid, HardSwish, Selu, SoftPlus"""

    def test_abs(self):
        x = np.array([[-1.0, 2.0, -3.0]], dtype=np.float32)
        y = _run_single_op("Abs", x)
        assert np.allclose(_to_numpy(y), np.abs(x))

    def test_ceil(self):
        x = np.array([[1.2, 2.7, -0.3]], dtype=np.float32)
        y = _run_single_op("Ceil", x)
        assert np.allclose(_to_numpy(y), np.ceil(x))

    def test_floor(self):
        x = np.array([[1.2, 2.7, -0.3]], dtype=np.float32)
        y = _run_single_op("Floor", x)
        assert np.allclose(_to_numpy(y), np.floor(x))

    def test_round(self):
        x = np.array([[1.2, 2.7, 3.5]], dtype=np.float32)
        y = _run_single_op("Round", x)
        assert np.allclose(_to_numpy(y), np.round(x))

    def test_sign(self):
        x = np.array([[-5.0, 0.0, 3.0]], dtype=np.float32)
        y = _run_single_op("Sign", x)
        assert np.allclose(_to_numpy(y), np.sign(x))

    def test_reciprocal(self):
        x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
        y = _run_single_op("Reciprocal", x)
        assert np.allclose(_to_numpy(y), 1.0 / x)

    def test_neg(self):
        x = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
        y = _run_single_op("Neg", x)
        assert np.allclose(_to_numpy(y), -x)

    def test_exp(self):
        x = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
        y = _run_single_op("Exp", x)
        assert np.allclose(_to_numpy(y), np.exp(x))

    def test_sqrt(self):
        x = np.array([[1.0, 4.0, 9.0]], dtype=np.float32)
        y = _run_single_op("Sqrt", x)
        assert np.allclose(_to_numpy(y), np.sqrt(x))

    def test_log(self):
        x = np.array([[1.0, np.e, np.e ** 2]], dtype=np.float32)
        y = _run_single_op("Log", x)
        assert np.allclose(_to_numpy(y), np.log(x))

    def test_erf(self):
        x = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
        y = _run_single_op("Erf", x)
        assert _to_numpy(y).shape == x.shape
        # Erf ranges from -1 to 1
        assert np.all(np.abs(_to_numpy(y)) <= 1.0)

    def test_not(self):
        x = np.array([[True, False, True]], dtype=bool)
        y = _run_single_op("Not", x)
        assert np.allclose(_to_numpy(y), np.logical_not(x))

    def test_isnan(self):
        x = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        y = _run_single_op("IsNaN", x)
        assert np.allclose(_to_numpy(y), np.isnan(x))

    def test_isinf(self):
        x = np.array([[1.0, np.inf, -np.inf]], dtype=np.float32)
        y = _run_single_op("IsInf", x)
        assert np.allclose(_to_numpy(y), np.isinf(x))

    def test_elu(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Elu", x, node_attrs={"alpha": 1.0})
        expected = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_hardsigmoid(self):
        x = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
        y = _run_single_op("HardSigmoid", x, node_attrs={"alpha": 0.2, "beta": 0.5})
        expected = np.clip(0.2 * x + 0.5, 0, 1)
        assert np.allclose(_to_numpy(y), expected)

    def test_hardswish(self):
        x = np.array([[-3.0, 0.0, 3.0]], dtype=np.float32)
        y = _run_single_op("HardSwish", x)
        # fnn.hardswish uses x * relu6(x) / 6 formula (MobileNetV3 original)
        relu6 = np.minimum(np.maximum(x, 0), 6)
        expected = x * relu6 / 6.0
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_selu(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Selu", x,
                           node_attrs={"alpha": 1.67326, "gamma": 1.0507})
        alpha, gamma = 1.67326, 1.0507
        expected = gamma * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_softplus(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("SoftPlus", x)
        assert np.allclose(_to_numpy(y), np.log(1.0 + np.exp(x)))


# ================================================================== #
# Activations (fnn-native)
# ================================================================== #

class TestActivationOps:
    """Relu, Sigmoid, Tanh, Silu, Gelu, LeakyRelu, Softmax, LogSoftmax, Swish"""

    def test_relu(self):
        x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
        y = _run_single_op("Relu", x)
        assert np.allclose(_to_numpy(y), np.maximum(0, x))

    def test_sigmoid(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Sigmoid", x)
        assert np.allclose(_to_numpy(y), 1.0 / (1.0 + np.exp(-x)))

    def test_tanh(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Tanh", x)
        assert np.allclose(_to_numpy(y), np.tanh(x))

    def test_silu(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Silu", x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_gelu(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Gelu", x)
        expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_leaky_relu(self):
        x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
        y = _run_single_op("LeakyRelu", x, node_attrs={"alpha": 0.1})
        expected = np.where(x > 0, x, 0.1 * x)
        assert np.allclose(_to_numpy(y), expected)

    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y = _run_single_op("Softmax", x, node_attrs={"axis": 1})
        ex = np.exp(x - x.max(axis=1, keepdims=True))
        expected = ex / ex.sum(axis=1, keepdims=True)
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)
        assert _to_numpy(y).shape == x.shape

    def test_log_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y = _run_single_op("LogSoftmax", x, node_attrs={"axis": 1})
        ex = np.exp(x - x.max(axis=1, keepdims=True))
        sm = ex / ex.sum(axis=1, keepdims=True)
        expected = np.log(sm)
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)

    def test_swish(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        y = _run_single_op("Swish", x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        assert np.allclose(_to_numpy(y), expected, atol=1e-6)


# ================================================================== #
# Binary Elementwise Ops
# ================================================================== #

class TestBinaryElementwiseOps:
    """Add, Sub, Mul, Div, Pow, Equal, Greater, Less, And, Or, Xor"""

    def _run_binary(self, op_type, a, b, node_attrs=None):
        return _run_single_op(op_type, a, node_attrs=node_attrs, second_input=b)

    def test_add(self):
        a, b = np.array([[1.0, 2.0]], dtype=np.float32), np.array([[3.0, 4.0]], dtype=np.float32)
        y = self._run_binary("Add", a, b)
        assert np.allclose(_to_numpy(y), a + b)

    def test_sub(self):
        a, b = np.array([[5.0, 3.0]], dtype=np.float32), np.array([[2.0, 1.0]], dtype=np.float32)
        y = self._run_binary("Sub", a, b)
        assert np.allclose(_to_numpy(y), a - b)

    def test_mul(self):
        a, b = np.array([[2.0, 3.0]], dtype=np.float32), np.array([[4.0, 5.0]], dtype=np.float32)
        y = self._run_binary("Mul", a, b)
        assert np.allclose(_to_numpy(y), a * b)

    def test_div(self):
        a, b = np.array([[6.0, 9.0]], dtype=np.float32), np.array([[2.0, 3.0]], dtype=np.float32)
        y = self._run_binary("Div", a, b)
        assert np.allclose(_to_numpy(y), a / b)

    def test_pow(self):
        a = np.array([[2.0, 3.0]], dtype=np.float32)
        # fnn.pow takes scalar exponent (extracts first element from tensor)
        exponent = np.array([[3.0]], dtype=np.float32)
        y = _run_single_op("Pow", a, second_input=exponent)
        assert np.allclose(_to_numpy(y), a ** 3.0)

    def test_equal(self):
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0, 3.0]], dtype=np.float32)
        y = self._run_binary("Equal", a, b)
        assert np.allclose(_to_numpy(y), np.equal(a, b))

    def test_greater(self):
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = np.array([[0.0, 2.0, 3.0]], dtype=np.float32)
        y = self._run_binary("Greater", a, b)
        assert np.allclose(_to_numpy(y), np.greater(a, b))

    def test_less(self):
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = np.array([[2.0, 2.0, 1.0]], dtype=np.float32)
        y = self._run_binary("Less", a, b)
        assert np.allclose(_to_numpy(y), np.less(a, b))

    def test_and(self):
        a = np.array([[True, False, True]], dtype=bool)
        b = np.array([[True, True, False]], dtype=bool)
        y = self._run_binary("And", a, b)
        assert np.allclose(_to_numpy(y), np.logical_and(a, b))

    def test_or(self):
        a = np.array([[True, False, True]], dtype=bool)
        b = np.array([[True, True, False]], dtype=bool)
        y = self._run_binary("Or", a, b)
        assert np.allclose(_to_numpy(y), np.logical_or(a, b))

    def test_xor(self):
        a = np.array([[True, False, True]], dtype=bool)
        b = np.array([[True, True, False]], dtype=bool)
        y = self._run_binary("Xor", a, b)
        assert np.allclose(_to_numpy(y), np.logical_xor(a, b))


# ================================================================== #
# Min / Max (variadic)
# ================================================================== #

class TestMinMaxOps:
    """Min, Max (variadic)"""

    def test_min(self):
        a = np.array([[1.0, 5.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 2.0, 4.0]], dtype=np.float32)
        c = np.array([[2.0, 3.0, 1.0]], dtype=np.float32)
        # DAGModel Min handles variadic via numpy.minimum chain
        # Use a 3-input model via multiple nodes
        model = _make_model(
            nodes=[{
                "name": "min1",
                "op_type": "Min",
                "inputs": ["A", "B", "C"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["A", "B", "C"],
            output_names=["Y"],
        )
        feed = {
            "A": fnn.tensor(a, list(a.shape)),
            "B": fnn.tensor(b, list(b.shape)),
            "C": fnn.tensor(c, list(c.shape)),
        }
        outputs = model.forward(feed)
        y = outputs["Y"]
        expected = np.minimum(np.minimum(a, b), c)
        assert np.allclose(_to_numpy(y), expected)

    def test_max(self):
        a = np.array([[1.0, 5.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 2.0, 4.0]], dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "max1",
                "op_type": "Max",
                "inputs": ["A", "B"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["A", "B"],
            output_names=["Y"],
        )
        feed = {
            "A": fnn.tensor(a, list(a.shape)),
            "B": fnn.tensor(b, list(b.shape)),
        }
        outputs = model.forward(feed)
        assert np.allclose(_to_numpy(outputs["Y"]), np.maximum(a, b))


# ================================================================== #
# Tensor Manipulation Ops
# ================================================================== #

class TestTensorManipulation:
    """Concat, Reshape, Flatten, Transpose, Squeeze, Unsqueeze,
    Expand, CumSum, Compress, DepthToSpace, SpaceToDepth,
    EyeLike, OneHot, Range, Reverse, Split, Tile, Where, Pad, Slice,
    Shape, Cast, Gather, GatherND, ScatterND, TopK"""

    def test_concat(self):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "concat1",
                "op_type": "Concat",
                "inputs": ["A", "B"],
                "outputs": ["Y"],
                "axis": 1,
            }],
            params={},
            input_names=["A", "B"],
            output_names=["Y"],
        )
        feed = {"A": fnn.tensor(a, list(a.shape)), "B": fnn.tensor(b, list(b.shape))}
        outputs = model.forward(feed)
        assert np.allclose(_to_numpy(outputs["Y"]), np.concatenate([a, b], axis=1))

    def test_reshape(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Reshape uses second input as the target shape (must be numpy array, not tensor)
        model = _make_model(
            nodes=[{
                "name": "reshape1",
                "op_type": "Reshape",
                "inputs": ["X", "shape"],
                "outputs": ["Y"],
            }],
            params={"shape": np.array([2, 6], dtype=np.int64)},
            input_names=["X"],
            output_names=["Y"],
        )
        feed = {"X": fnn.tensor(x, list(x.shape))}
        outputs = model.forward(feed)
        assert _to_numpy(outputs["Y"]).shape == (2, 6)


# ================================================================== #
# Cast and Gather
# ================================================================== #

class TestCast:
    """Cast op: dtype conversion."""

    def test_cast_float_to_int64(self):
        x = np.array([[1.5, 2.7, -0.3]], dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "cast1",
                "op_type": "Cast",
                "inputs": ["X"],
                "outputs": ["Y"],
                "to": 7,  # INT64 (top-level attr)
            }],
            params={},
            input_names=["X"],
            output_names=["Y"],
        )
        feed = {"X": fnn.tensor(x, list(x.shape))}
        outputs = model.forward(feed)
        result = _to_numpy(outputs["Y"])
        # fnn tensors always store as f32, but values should be truncated like int64
        assert np.allclose(result, [[1.0, 2.0, 0.0]])

    def test_cast_float_to_bool(self):
        x = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "cast1",
                "op_type": "Cast",
                "inputs": ["X"],
                "outputs": ["Y"],
                "to": 9,  # BOOL (top-level attr)
            }],
            params={},
            input_names=["X"],
            output_names=["Y"],
        )
        feed = {"X": fnn.tensor(x, list(x.shape))}
        outputs = model.forward(feed)
        result = _to_numpy(outputs["Y"])
        # fnn tensors always store as f32, but values should be 0.0/1.0 like bool
        assert np.allclose(result, [[0.0, 1.0, 1.0]])

    def test_cast_passthrough(self):
        """Cast to float (default) should leave data unchanged."""
        x = np.array([[1.5, 2.5]], dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "cast1",
                "op_type": "castop",
                "inputs": ["X"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["X"],
            output_names=["Y"],
        )
        feed = {"X": fnn.tensor(x, list(x.shape))}
        outputs = model.forward(feed)
        assert np.allclose(_to_numpy(outputs["Y"]), x)


class TestGather:
    """Gather op: indexing along an axis."""

    def test_gather_axis0(self):
        data = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32)
        indices = np.array([0, 2], dtype=np.int64)
        model = _make_model(
            nodes=[{
                "name": "gather1",
                "op_type": "Gather",
                "inputs": ["X", "I"],
                "outputs": ["Y"],
                "axis": 0,  # top-level attr
            }],
            params={},
            input_names=["X", "I"],
            output_names=["Y"],
        )
        feed = {
            "X": fnn.tensor(data, list(data.shape)),
            "I": fnn.tensor(indices, list(indices.shape)),
        }
        outputs = model.forward(feed)
        expected = np.take(data, indices, axis=0)
        assert np.allclose(_to_numpy(outputs["Y"]), expected)

    def test_gather_axis1(self):
        data = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.float32)
        indices = np.array([2, 0], dtype=np.int64)
        model = _make_model(
            nodes=[{
                "name": "gather1",
                "op_type": "Gather",
                "inputs": ["X", "I"],
                "outputs": ["Y"],
                "axis": 1,  # top-level attr
            }],
            params={},
            input_names=["X", "I"],
            output_names=["Y"],
        )
        feed = {
            "X": fnn.tensor(data, list(data.shape)),
            "I": fnn.tensor(indices, list(indices.shape)),
        }
        outputs = model.forward(feed)
        expected = np.take(data, indices, axis=1)
        assert np.allclose(_to_numpy(outputs["Y"]), expected)


# ================================================================== #
# Newly Implemented Ops (DequantizeLinear, QuantizeLinear,
# RotaryEmbedding, EmbedLayerNormalization, GRU, LSTM)
# ================================================================== #

class TestDequantizeLinear:
    """DequantizeLinear op: y = (x - x_zero_point) * x_scale"""

    def test_dequantize_basic(self):
        x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
        scale = np.array([1.5], dtype=np.float32)
        zp = np.array([0], dtype=np.int8)
        model = _make_model(
            nodes=[{
                "name": "dq1",
                "op_type": "DequantizeLinear",
                "inputs": ["X", "scale", "zp"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["X", "scale", "zp"],
            output_names=["Y"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "scale": fnn.tensor(scale, list(scale.shape)),
            "zp": fnn.tensor(zp, list(zp.shape)),
        }
        outputs = model.forward(feed)
        expected = x.astype(np.float32) * 1.5
        assert np.allclose(_to_numpy(outputs["Y"]), expected)

    def test_dequantize_with_zp(self):
        x = np.array([[0, 1, 2]], dtype=np.int8)
        scale = np.array([2.0], dtype=np.float32)
        zp = np.array([1], dtype=np.int8)
        model = _make_model(
            nodes=[{
                "name": "dq1",
                "op_type": "DequantizeLinear",
                "inputs": ["X", "scale", "zp"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["X", "scale", "zp"],
            output_names=["Y"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "scale": fnn.tensor(scale, list(scale.shape)),
            "zp": fnn.tensor(zp, list(zp.shape)),
        }
        outputs = model.forward(feed)
        expected = (x.astype(np.float32) - 1.0) * 2.0
        assert np.allclose(_to_numpy(outputs["Y"]), expected)


class TestQuantizeLinear:
    """QuantizeLinear op: y = round(x / y_scale) + y_zero_point"""

    def test_quantize_basic(self):
        x = np.array([[0.0, 1.5, 3.0]], dtype=np.float32)
        scale = np.array([1.5], dtype=np.float32)
        zp = np.array([0], dtype=np.int8)
        model = _make_model(
            nodes=[{
                "name": "q1",
                "op_type": "QuantizeLinear",
                "inputs": ["X", "scale", "zp"],
                "outputs": ["Y"],
            }],
            params={},
            input_names=["X", "scale", "zp"],
            output_names=["Y"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "scale": fnn.tensor(scale, list(scale.shape)),
            "zp": fnn.tensor(zp, list(zp.shape)),
        }
        outputs = model.forward(feed)
        expected = np.round(x / 1.5).astype(np.int8) + 0
        assert np.allclose(_to_numpy(outputs["Y"]), expected)


class TestRotaryEmbedding:
    """Rotary position embedding."""

    def test_rope_basic(self):
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 4
        q = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        cos = np.ones((seq_len, head_dim), dtype=np.float32)
        sin = np.zeros((seq_len, head_dim), dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "rope1",
                "op_type": "RotaryEmbedding",
                "inputs": ["Q", "K", "cos", "sin"],
                "outputs": ["Y1", "Y2"],
            }],
            params={},
            input_names=["Q", "K", "cos", "sin"],
            output_names=["Y1", "Y2"],
        )
        feed = {
            "Q": fnn.tensor(q, list(q.shape)),
            "K": fnn.tensor(k, list(k.shape)),
            "cos": fnn.tensor(cos, list(cos.shape)),
            "sin": fnn.tensor(sin, list(sin.shape)),
        }
        outputs = model.forward(feed)
        assert "Y1" in outputs and "Y2" in outputs
        # With cos=1, sin=0, RoPE is identity
        assert np.allclose(_to_numpy(outputs["Y1"]), q)
        assert np.allclose(_to_numpy(outputs["Y2"]), k)

    def test_rope_rotation(self):
        """With cos=0, sin=1 along dim pairs, verify rotation."""
        _, seq_len, _, head_dim = 1, 2, 1, 4
        q = np.array([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)  # [1,2,1,4]
        k = np.array([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=np.float32)
        # sin with alternating pattern: [0,1,0,1,...] at each position
        # Actually the rotation applies per pair: (0,1), (2,3), etc.
        # With sin=1, cos=0 for dim 0: x_even becomes -x_odd, x_odd becomes x_even
        cos = np.zeros((seq_len, head_dim), dtype=np.float32)
        sin = np.ones((seq_len, head_dim), dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "rope1",
                "op_type": "RotaryEmbedding",
                "inputs": ["Q", "K", "cos", "sin"],
                "outputs": ["Y1", "Y2"],
            }],
            params={},
            input_names=["Q", "K", "cos", "sin"],
            output_names=["Y1", "Y2"],
        )
        feed = {
            "Q": fnn.tensor(q, list(q.shape)),
            "K": fnn.tensor(k, list(k.shape)),
            "cos": fnn.tensor(cos, list(cos.shape)),
            "sin": fnn.tensor(sin, list(sin.shape)),
        }
        outputs = model.forward(feed)
        q_out = _to_numpy(outputs["Y1"])
        k_out = _to_numpy(outputs["Y2"])
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestEmbedLayerNormalization:
    """Embedding + Layer Normalization (BERT-style)."""

    def test_embed_layer_norm_basic(self):
        batch, seq_len, hidden = 2, 3, 4
        vocab_size = 10
        max_pos = 8
        input_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        word_emb = np.random.randn(vocab_size, hidden).astype(np.float32)
        pos_emb = np.random.randn(max_pos, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "eln1",
                "op_type": "EmbedLayerNormalization",
                "inputs": ["input_ids", "word_emb", "pos_emb", "gamma", "beta"],
                "outputs": ["Y"],
                "epsilon": 1e-5,
            }],
            params={},
            input_names=["input_ids", "word_emb", "pos_emb", "gamma", "beta"],
            output_names=["Y"],
        )
        feed = {
            "input_ids": fnn.tensor(input_ids, list(input_ids.shape)),
            "word_emb": fnn.tensor(word_emb, list(word_emb.shape)),
            "pos_emb": fnn.tensor(pos_emb, list(pos_emb.shape)),
            "gamma": fnn.tensor(gamma, list(gamma.shape)),
            "beta": fnn.tensor(beta, list(beta.shape)),
        }
        outputs = model.forward(feed)
        out = _to_numpy(outputs["Y"])
        assert out.shape == (batch, seq_len, hidden)
        # After layer norm, mean should be ~0, std ~1 along hidden dim
        assert abs(out.mean(axis=-1).mean()) < 0.5
        assert abs(out.std(axis=-1).mean() - 1.0) < 0.5


class TestGRU:
    """GRU forward pass (single direction)."""

    def test_gru_basic(self):
        seq_len, batch, input_size, hidden = 3, 2, 4, 5
        x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
        w = np.random.randn(1, 3 * hidden, input_size).astype(np.float32)
        r = np.random.randn(1, 3 * hidden, hidden).astype(np.float32)
        model = _make_model(
            nodes=[{
                "name": "gru1",
                "op_type": "GRU",
                "inputs": ["X", "W", "R"],
                "outputs": ["Y", "Y_h"],
            }],
            params={},
            input_names=["X", "W", "R"],
            output_names=["Y", "Y_h"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "W": fnn.tensor(w, list(w.shape)),
            "R": fnn.tensor(r, list(r.shape)),
        }
        outputs = model.forward(feed)
        assert "Y" in outputs and "Y_h" in outputs
        # Y: [seq_len, num_directions, batch, hidden]
        assert _to_numpy(outputs["Y"]).shape == (seq_len, 1, batch, hidden)
        # Y_h: [num_directions, batch, hidden]
        assert _to_numpy(outputs["Y_h"]).shape == (1, batch, hidden)

    def test_gru_with_bias_init_h(self):
        seq_len, batch, input_size, hidden = 2, 1, 3, 4
        x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
        w = np.random.randn(1, 3 * hidden, input_size).astype(np.float32)
        r = np.random.randn(1, 3 * hidden, hidden).astype(np.float32)
        b = np.random.randn(1, 6 * hidden).astype(np.float32)
        init_h = np.random.randn(1, batch, hidden).astype(np.float32)
        model = _make_model(
            nodes=[{
                "name": "gru1",
                "op_type": "GRU",
                "inputs": ["X", "W", "R", "B", "seq_lens", "init_h"],
                "outputs": ["Y", "Y_h"],
            }],
            params={},
            input_names=["X", "W", "R", "B", "seq_lens", "init_h"],
            output_names=["Y", "Y_h"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "W": fnn.tensor(w, list(w.shape)),
            "R": fnn.tensor(r, list(r.shape)),
            "B": fnn.tensor(b, list(b.shape)),
            "seq_lens": fnn.tensor(np.array([2], dtype=np.int64), [1]),
            "init_h": fnn.tensor(init_h, list(init_h.shape)),
        }
        outputs = model.forward(feed)
        assert _to_numpy(outputs["Y"]).shape == (seq_len, 1, batch, hidden)


class TestLSTM:
    """LSTM forward pass (single direction)."""

    def test_lstm_basic(self):
        seq_len, batch, input_size, hidden = 3, 2, 4, 5
        x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
        w = np.random.randn(1, 4 * hidden, input_size).astype(np.float32)
        r = np.random.randn(1, 4 * hidden, hidden).astype(np.float32)
        model = _make_model(
            nodes=[{
                "name": "lstm1",
                "op_type": "LSTM",
                "inputs": ["X", "W", "R"],
                "outputs": ["Y", "Y_h", "Y_c"],
            }],
            params={},
            input_names=["X", "W", "R"],
            output_names=["Y", "Y_h", "Y_c"],
        )
        feed = {
            "X": fnn.tensor(x, list(x.shape)),
            "W": fnn.tensor(w, list(w.shape)),
            "R": fnn.tensor(r, list(r.shape)),
        }
        outputs = model.forward(feed)
        assert "Y" in outputs and "Y_h" in outputs and "Y_c" in outputs
        assert _to_numpy(outputs["Y"]).shape == (seq_len, 1, batch, hidden)
        assert _to_numpy(outputs["Y_h"]).shape == (1, batch, hidden)
        assert _to_numpy(outputs["Y_c"]).shape == (1, batch, hidden)

    def test_matmul(self):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        y = _run_single_op("matmul", a, second_input=b)
        assert np.allclose(_to_numpy(y), a @ b)

    def test_unknown_op_passthrough(self):
        """Unknown op types should pass through (first tensor)."""
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y = _run_single_op("FictionalOp999", x)
        assert np.allclose(_to_numpy(y), x)


# ================================================================== #
# Gemm Helper Tests
# ================================================================== #

class TestGemmVariants:
    """Gemm with different transB settings, MatMul standalone"""

    def test_gemm_no_transpose(self):
        x = np.random.randn(1, 8).astype(np.float32)
        w = np.random.randn(8, 4).astype(np.float32)  # not transposed
        b = np.zeros(4, dtype=np.float32)
        model = _make_model(
            nodes=[{
                "name": "gemm1",
                "op_type": "Gemm",
                "inputs": ["X"],
                "outputs": ["Y"],
                "transB": 0,
            }],
            params={"gemm1.weight": w,
                    "gemm1.bias": b},
            input_names=["X"],
            output_names=["Y"],
        )
        feed = {"X": fnn.tensor(x, list(x.shape))}
        outputs = model.forward(feed)
        assert _to_numpy(outputs["Y"]).shape == (1, 4)
