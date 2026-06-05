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
