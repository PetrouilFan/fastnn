import numpy as np
import pytest


def test_aot_executor_profile_reports_per_kernel_timings():
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        }
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )
    x = fnn.tensor(np.asarray([[-1.0, 0.0, 2.0, 3.0]], dtype=np.float32), [1, 4])

    result = executor.profile({"x": x})

    assert "outputs" in result
    assert "profile" in result
    np.testing.assert_allclose(result["outputs"]["y"].numpy(), [[0.0, 0.0, 2.0, 3.0]])
    entries = result["profile"]
    assert len(entries) >= 1
    relu_entries = [entry for entry in entries if entry["kernel_name"] == "relu_f32"]
    assert relu_entries
    assert relu_entries[0]["elapsed_ns"] > 0
    assert relu_entries[0]["node_name"] == "relu1"


def test_aot_executor_preserves_dynamic_runtime_shapes_in_forward_and_profile():
    import fastnn as fnn

    nodes = [{"name": "relu1", "op_type": "Relu", "inputs": "x", "outputs": "y"}]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [-1, 4]},
    )

    for batch in (2, 3):
        values = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4) - 2.0
        x = fnn.tensor(values, [batch, 4])
        forward = executor.forward({"x": x})["y"].numpy()
        profiled = executor.profile({"x": x})["outputs"]["y"].numpy()
        expected = np.maximum(values, 0.0)
        assert forward.shape == (batch, 4)
        assert profiled.shape == (batch, 4)
        np.testing.assert_allclose(forward, expected)
        np.testing.assert_allclose(profiled, expected)


def test_aot_executor_rejects_duplicate_output_names():
    import fastnn as fnn

    nodes = [{"name": "relu1", "op_type": "Relu", "inputs": "x", "outputs": "y"}]
    with pytest.raises(ValueError, match="unique"):
        fnn.AotExecutor(
            nodes,
            {},
            ["x"],
            ["y", "y"],
            input_shapes={"x": [1, 4]},
        )


def test_memory_stats_reports_instruction_level_static_traffic():
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        }
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )

    stats = executor.memory_stats()

    rows = stats["top_instructions_by_static_bytes"]
    assert rows, "memory_stats should expose instruction-level traffic rows"
    relu_rows = [row for row in rows if row["kernel_name"] == "relu_f32"]
    assert relu_rows
    row = relu_rows[0]
    assert row["instruction_index"] >= 0
    assert row["kind"] == "call_kernel"
    assert row["read_bytes"] == 16
    assert row["write_bytes"] == 16
    assert row["static_bytes"] == 32


def test_memory_stats_instruction_rows_include_graph_io_shapes_for_layout_work():
    import fastnn as fnn

    nodes = [
        {"name": "concat1", "op_type": "Concat", "inputs": "a,b", "outputs": "y", "axis": "1"}
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["a", "b"],
        ["y"],
        input_shapes={"a": [1, 2, 4], "b": [1, 3, 4]},
    )

    stats = executor.memory_stats()
    rows = [row for row in stats["top_instructions_by_static_bytes"] if row["kernel_name"] == "concat"]
    assert rows, "concat should appear as a profiled instruction row"
    row = rows[0]
    assert row["node_id"] >= 0
    assert row["node_name"] == "concat1"
    assert row["op_type"] == "Concat"
    assert row["input_node_ids"] == row["input_nodes"]
    assert len(row["input_nodes"]) == 2
    assert row["input_shapes"] == [[1, 2, 4], [1, 3, 4]]
    assert row["output_shape"] == [1, 5, 4]


def test_memory_stats_reports_write_const_rows_for_persistent_constant_work():
    import fastnn as fnn

    bias = fnn.tensor(
        np.asarray([0.5, -0.5, 0.25, -0.25], dtype=np.float32), [1, 4]
    )
    nodes = [
        {"name": "add1", "op_type": "Add", "inputs": "x,b", "outputs": "s"},
        {"name": "relu1", "op_type": "Relu", "inputs": "s", "outputs": "y"},
    ]
    executor = fnn.AotExecutor(
        nodes,
        {"b": bias},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )

    stats = executor.memory_stats()

    rows = stats["top_write_consts_by_size"]
    assert rows, "memory_stats should expose largest WriteConst instructions"
    row = rows[0]
    assert row["instruction_index"] >= 0
    assert row["write_bytes"] == 16
    assert row["dst_offset"] >= 0
    assert row["dst_size"] == 16
    assert row["data_len"] == 16


def test_memory_stats_cross_references_prepared_static_weight_write_consts():
    import fastnn as fnn

    weight = fnn.tensor(np.asarray([[[[2.0]]]], dtype=np.float32), [1, 1, 1, 1])
    bias = fnn.tensor(np.asarray([0.25], dtype=np.float32), [1])
    nodes = [
        {
            "name": "conv1",
            "op_type": "Conv",
            "inputs": "x,w,b",
            "outputs": "y",
            "stride": "1",
            "padding": "0",
            "dilation": "1",
            "group": "1",
        }
    ]
    executor = fnn.AotExecutor(
        nodes,
        {"w": weight, "b": bias},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 1, 2, 2]},
    )

    rows = executor.memory_stats()["top_write_consts_by_size"]

    prepared_rows = [row for row in rows if row.get("prepared_static_role")]
    assert prepared_rows, "Conv WriteConst rows should identify prepared static bindings"
    roles = {row["prepared_static_role"] for row in prepared_rows}
    assert roles == {"conv_weight", "conv_bias"}
    for row in prepared_rows:
        assert row["prepared_consumer_instruction_index"] >= 0
        assert row["prepared_input_index"] in {1, 2}
        assert row["prepared_constant_index"] >= 0
        assert row["prepared_constant_name"].startswith("conv_")


def test_aot_executor_rejects_minimum_i64_input_dimension():
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        }
    ]
    with pytest.raises(ValueError, match="unsupported dimension"):
        fnn.AotExecutor(
            nodes,
            {},
            ["x"],
            ["y"],
            input_shapes={"x": [-(2**63), 4]},
        )


def test_aot_executor_rejects_noncontiguous_input_without_panicking():
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        }
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [2, 2]},
    )
    noncontiguous = fnn.zeros([2, 2]).transpose(0, 1)
    with pytest.raises(ValueError, match="cannot be passed to AOT execution"):
        executor.forward({"x": noncontiguous})


@pytest.mark.parametrize(
    ("op_type", "inputs", "minimum"),
    [("Relu", "", 1), ("Add", "x", 2)],
)
def test_aot_executor_rejects_missing_node_inputs(op_type, inputs, minimum):
    import fastnn as fnn

    nodes = [
        {
            "name": "malformed",
            "op_type": op_type,
            "inputs": inputs,
            "outputs": "y",
        }
    ]
    with pytest.raises(RuntimeError, match=rf"requires at least {minimum} input"):
        fnn.AotExecutor(
            nodes,
            {},
            ["x"],
            ["y"],
            input_shapes={"x": [1]},
        )


@pytest.mark.parametrize(
    ("dimension", "message"),
    [
        (-1.0, "invalid dimension"),
        (float("nan"), "invalid dimension"),
        (1.5, "invalid dimension"),
        (1_000_000_000.0, "import budget"),
    ],
)
def test_constant_of_shape_rejects_hostile_dimensions(dimension, message):
    import fastnn as fnn

    nodes = [
        {
            "name": "constant_shape",
            "op_type": "ConstantOfShape",
            "inputs": "shape",
            "outputs": "y",
        }
    ]
    shape = fnn.tensor(np.asarray([dimension], dtype=np.float32), [1])
    with pytest.raises(RuntimeError, match=message):
        fnn.AotExecutor(nodes, {"shape": shape}, [], ["y"])
