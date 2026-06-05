import numpy as np


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
