"""Behavioural tests for `AotExecutor.forward_prepared_fallback()`.

Mission 017 introduces an opt-in prepared-execution fallback path on
`AotExecutor`. The contract is: the new method accepts the same input
shape as `forward`, validates that the prepared plan is consistent
with the underlying executable plan, then dispatches through the
existing executor machinery. The result must be **byte-identical** to
`forward` on the same inputs.

These tests build tiny ONNX-style graphs with `AotExecutor`, run
`forward` and `forward_prepared_fallback` side-by-side, and assert
that both produce the same output tensor(s). They require the
`prepared-plan` Rust feature to be enabled at build time.
"""

import numpy as np
import pytest


def _relu_graph():
    """Single Relu graph: y = relu(x)."""
    import fastnn as fnn

    return fnn.AotExecutor(
        [
            {
                "name": "relu1",
                "op_type": "Relu",
                "inputs": "x",
                "outputs": "y",
            }
        ],
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )


def test_forward_prepared_fallback_matches_forward_relu():
    """Both paths produce identical Relu outputs on a 1x4 input."""
    import fastnn as fnn

    executor = _relu_graph()
    x = fnn.tensor(np.asarray([[-1.0, 0.0, 2.0, 3.0]], dtype=np.float32), [1, 4])

    expected = executor.forward({"x": x})
    fallback = executor.forward_prepared_fallback({"x": x})

    assert set(expected.keys()) == set(fallback.keys()) == {"y"}
    np.testing.assert_array_equal(
        expected["y"].numpy(), fallback["y"].numpy()
    )
    np.testing.assert_allclose(
        fallback["y"].numpy(), [[0.0, 0.0, 2.0, 3.0]]
    )


def test_forward_prepared_fallback_is_repeatable():
    """Repeated calls to the fallback must return identical outputs."""
    import fastnn as fnn

    executor = _relu_graph()
    x = fnn.tensor(np.asarray([[-2.0, -1.0, 0.5, 1.5]], dtype=np.float32), [1, 4])

    first = executor.forward_prepared_fallback({"x": x})
    second = executor.forward_prepared_fallback({"x": x})
    np.testing.assert_array_equal(
        first["y"].numpy(), second["y"].numpy()
    )


def test_forward_prepared_fallback_chain_relu_then_neg():
    """Multi-instruction plan: y = relu(x) then z = neg(y).

    Exercises the order-mapping check on a plan with two `CallKernel`
    instructions and no `WriteConst` constant producer.
    """
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        },
        {
            "name": "neg1",
            "op_type": "Neg",
            "inputs": "y",
            "outputs": "z",
        },
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["z"],
        input_shapes={"x": [1, 4]},
    )
    x = fnn.tensor(
        np.asarray([[-1.0, 0.0, 2.0, -3.0]], dtype=np.float32), [1, 4]
    )

    expected = executor.forward({"x": x})
    fallback = executor.forward_prepared_fallback({"x": x})

    np.testing.assert_array_equal(
        expected["z"].numpy(), fallback["z"].numpy()
    )
    # relu([-1, 0, 2, -3]) = [0, 0, 2, 0]; neg = [0, 0, -2, 0]
    np.testing.assert_allclose(
        fallback["z"].numpy(), [[0.0, 0.0, -2.0, 0.0]]
    )


def test_forward_prepared_fallback_matches_forward_with_constant_input():
    """Graph with a Constant input (=> WriteConst in the compiled plan)
    must still match across the two paths. This is the exact plan
    shape the order-mapping validation is meant to cover.
    """
    import fastnn as fnn

    # y = relu(x + b) where b is a graph constant.
    bias = fnn.tensor(
        np.asarray([0.5, -0.5, 0.25, -0.25], dtype=np.float32), [1, 4]
    )
    nodes = [
        {
            "name": "add1",
            "op_type": "Add",
            "inputs": "x,b",
            "outputs": "s",
        },
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "s",
            "outputs": "y",
        },
    ]
    executor = fnn.AotExecutor(
        nodes,
        {"b": bias},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )
    x = fnn.tensor(
        np.asarray([[-1.0, 0.0, 2.0, 3.0]], dtype=np.float32), [1, 4]
    )

    expected = executor.forward({"x": x})
    fallback = executor.forward_prepared_fallback({"x": x})

    np.testing.assert_array_equal(
        expected["y"].numpy(), fallback["y"].numpy()
    )
    # y = relu([-0.5, -0.5, 2.25, 2.75]) = [0, 0, 2.25, 2.75]
    np.testing.assert_allclose(
        fallback["y"].numpy(), [[0.0, 0.0, 2.25, 2.75]]
    )


def test_forward_prepared_fallback_method_exists():
    """Smoke: the new method must be exposed on the AotExecutor pyclass.

    Acts as a guard against accidental removal during a future refactor.
    """
    import fastnn as fnn

    executor = _relu_graph()
    assert hasattr(executor, "forward_prepared_fallback")
    assert callable(executor.forward_prepared_fallback)
