"""Regression tests for ReduceMean / GlobalAveragePool + Reshape shape inference.

Background
----------
MobileNetV2 emits ``ReduceMean(axes=[-1,-2], keepdims=1)`` followed by
``Reshape`` to ``[1, C]``.  MobileNetV3-small and EfficientNet-B0 emit
``GlobalAveragePool`` (decomposed into two chained ``ReduceMean`` calls
with keepdims) followed by ``Flatten``.

Before the fix the Rust shape-inference pass overwrote the
``keepdim=true`` output shape with the keepdim=false variant, producing:

  - ``[1, 1280, 1, 1]`` collapsed to ``[1280, 7, 7]`` for MobileNetV2,
    which then mismatches the Reshape target ``[1, 1280]`` (element
    count 62720 vs 1280).
  - ``[1, C, 1, 1]`` with ``W`` left un-reduced for
    MobileNetV3-small / EfficientNet-B0, which then collapses to
    ``[1, C*W]`` after Flatten, breaking a downstream 1x1 Conv2d that
    requires 4D input.

These tests build tiny synthetic graphs that mirror those failure
patterns and assert the compiled output shape matches the ONNX
reference.
"""

import pytest
pytest.importorskip("onnx")

import numpy as np
import onnx
from onnx import TensorProto, helper


def _reduce_mean_with_reshape_graph():
    """NCHW ReduceMean(keepdims=1) over the two trailing axes -> Reshape.

    Mirrors the MobileNetV2 head: input ``[1, 1280, 7, 7]``,
    ReduceMean(axes=[-1, -2], keepdims=1) -> ``[1, 1280, 1, 1]``,
    Reshape to ``[1, 1280]``.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1280, 7, 7])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1280])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], vals=[-1, -2])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], vals=[1, 1280])
    reduce = helper.make_node(
        "ReduceMean",
        inputs=["X", "axes"],
        outputs=["mean"],
        name="reduce1",
        keepdims=1,
    )
    reshape = helper.make_node(
        "Reshape", inputs=["mean", "shape"], outputs=["Y"], name="reshape1"
    )
    graph = helper.make_graph([reduce, reshape], "reduce_reshape", [X], [Y], [axes, shape])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _global_average_pool_then_flatten_graph():
    """GlobalAveragePool then Flatten, mirroring SE block output and head.

    The reduction must fully collapse the two spatial dims so the
    result is ``[1, C, 1, 1]`` and the subsequent Flatten produces
    ``[1, C]`` (which is the canonical pre-FC shape used by torchvision
    MobileNetV3 and EfficientNet heads).
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])
    pool = helper.make_node(
        "GlobalAveragePool", inputs=["X"], outputs=["pooled"], name="gap1"
    )
    flat = helper.make_node(
        "Flatten", inputs=["pooled"], outputs=["Y"], name="flat1", axis=1
    )
    graph = helper.make_graph([pool, flat], "gap_flatten", [X], [Y])
    return helper.make_model(graph)


def _format_attr_value(value) -> str:
    """Render an attribute value as a string for the AotExecutor node dict.

    Mirrors the formatting used by the model zoo script and the
    ONNX importer: int / float / string → ``str``, list / ndarray →
    comma-joined.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _import_and_run(model) -> tuple[np.ndarray, tuple[int, ...]]:
    """Run an ONNX model through fastnn's AotExecutor and return the
    first output's numpy value + shape.
    """
    import fastnn as fnn

    onnx_model = model
    graph = onnx_model.graph
    initializer_names = {init.name for init in graph.initializer}
    input_names = [
        i.name for i in graph.input if i.name not in initializer_names
    ]
    output_names = [o.name for o in graph.output]
    input_shapes: dict[str, list[int]] = {}
    for i in graph.input:
        if i.name in input_names:
            tt = i.type.tensor_type
            input_shapes[i.name] = [
                int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1
                for d in tt.shape.dim
            ]
    params: dict[str, fnn.tensor] = {}
    for init in graph.initializer:
        arr = onnx.numpy_helper.to_array(init).astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))
    nodes: list[dict] = []
    for idx, n in enumerate(graph.node):
        item: dict = {
            "name": n.name or f"{n.op_type}_{idx}",
            "op_type": n.op_type,
            "inputs": ",".join(n.input),
            "outputs": ",".join(n.output),
        }
        for a in n.attribute:
            item[a.name] = _format_attr_value(onnx.helper.get_attribute_value(a))
        nodes.append(item)
    executor = fnn.AotExecutor(
        nodes, params, input_names, output_names, input_shapes=input_shapes
    )
    feed = {
        name: fnn.tensor(
            np.ones(input_shapes[name], dtype=np.float32), input_shapes[name]
        )
        for name in input_names
    }
    out = executor.forward(feed)
    first = out[output_names[0]]
    arr = first.numpy()
    return arr, tuple(arr.shape)


def test_reduce_mean_keepdim_reshape_output_shape():
    """ReduceMean(keepdims=1) on a 4D NCHW tensor followed by Reshape.

    Regression for MobileNetV2 AOT import.  The compiled plan must
    produce a 2D ``[1, 1280]`` output (not a 3D ``[1280, 7, 7]`` which
    would fail the Reshape element-count check).
    """
    model = _reduce_mean_with_reshape_graph()
    arr, shape = _import_and_run(model)
    assert shape == (1, 1280), (
        f"expected ReduceMean(keepdims=1)->Reshape output [1, 1280], got {shape}"
    )
    # All-ones input -> spatial mean is 1.0 everywhere.
    np.testing.assert_allclose(arr, np.ones((1, 1280), dtype=np.float32))


def test_global_average_pool_output_shape_is_4d():
    """GlobalAveragePool must produce ``[N, C, 1, 1]``.

    Regression for MobileNetV3-small / EfficientNet-B0 AOT import
    where the previous decomposition reduced dim 2 twice, leaving the
    W dimension un-reduced.  That cascaded into a 2D output after
    Flatten and broke the downstream 1x1 Conv2d in SE blocks.
    """
    model = _global_average_pool_then_flatten_graph()
    arr, shape = _import_and_run(model)
    assert shape == (1, 16), (
        f"expected GlobalAveragePool->Flatten output [1, 16], got {shape}"
    )
    np.testing.assert_allclose(arr, np.ones((1, 16), dtype=np.float32))


def test_reduce_mean_negative_axes_kept_by_shape_inference():
    """ONNX axes like ``[-1, -2]`` must be normalized before parsing.

    The previous converter parsed axes as ``usize`` which silently
    dropped negative entries, falling back to a default reduce of
    dim 0 (the batch).  With the fix, both negative and positive axes
    are normalized against the input rank so the reduce targets the
    intended spatial dims.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], vals=[-1, -2])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], vals=[1, 3])
    reduce = helper.make_node(
        "ReduceMean",
        inputs=["X", "axes"],
        outputs=["mean"],
        name="reduce_neg",
        keepdims=1,
    )
    reshape = helper.make_node(
        "Reshape", inputs=["mean", "shape"], outputs=["Y"], name="reshape_neg"
    )
    graph = helper.make_graph(
        [reduce, reshape], "neg_axes", [X], [Y], [axes, shape]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    arr, shape_out = _import_and_run(model)
    assert shape_out == (1, 3)
    np.testing.assert_allclose(arr, np.ones((1, 3), dtype=np.float32))
