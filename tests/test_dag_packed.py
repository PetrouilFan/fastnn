"""Tests for packed dispatch in DAG executor."""
import numpy as np
import fastnn as fnn


def test_matmul_packed_dispatch():
    """Test that MatMul with packed weights dispatches correctly."""
    w_f32 = np.random.randn(4, 4).astype(np.float32)
    w_shape = list(w_f32.shape)
    packed = fnn.PackedTensor8.from_f32_auto(w_f32.ravel().tolist(), w_shape)

    nodes = [
        {"name": "matmul", "op_type": "MatMul", "inputs": "input", "outputs": "output"}
    ]
    input_names = ["input"]
    output_names = ["output"]

    params = {}
    packed_params = {"matmul.weight": (
        bytes(packed.to_bytes()), w_shape, 2, packed.scales(), packed.zeros()
    )}

    executor = fnn.DAGExecutor(nodes, params, input_names, output_names,
                               packed_params=packed_params)

    x = np.random.randn(1, 4).astype(np.float32)
    result = executor.forward({"input": fnn.Tensor(x.ravel().tolist(), list(x.shape))})

    output = result["output"]
    assert output.shape == [1, 4], f"Expected [1,4] got {output.shape}"
    print(f"test_matmul_packed_dispatch: OK, output shape={output.shape}")


def test_conv_transpose_packed_dispatch():
    """Test ConvTranspose with packed weights dispatches correctly."""
    w_shape = [2, 2, 3, 3]
    w_f32 = np.random.randn(*w_shape).astype(np.float32)

    packed = fnn.PackedTensor8.from_f32_auto(w_f32.ravel().tolist(), w_shape)

    nodes = [
        {"name": "conv_transpose", "op_type": "ConvTranspose",
         "inputs": "input", "outputs": "output",
         "stride": "2", "padding": "1", "kernel_shape": "3"}
    ]
    input_names = ["input"]
    output_names = ["output"]

    params = {}
    packed_params = {"conv_transpose.weight": (
        bytes(packed.to_bytes()), w_shape, 2, packed.scales(), packed.zeros()
    )}

    executor = fnn.DAGExecutor(nodes, params, input_names, output_names,
                               packed_params=packed_params)

    x = np.random.randn(1, 2, 8, 8).astype(np.float32)
    result = executor.forward({"input": fnn.Tensor(x.ravel().tolist(), list(x.shape))})

    output = result["output"]
    assert len(output.shape) == 4, f"Expected 4D output, got {output.shape}"
    print(f"test_conv_transpose_packed_dispatch: OK, output shape={output.shape}")


def test_quantize_dequantize_linear():
    """Test QuantizeLinear/DequantizeLinear ops."""
    nodes = [
        {"name": "dq", "op_type": "DequantizeLinear",
         "inputs": "input, scale, zp", "outputs": "output"}
    ]
    input_names = ["input", "scale", "zp"]
    output_names = ["output"]

    q_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale = np.array([0.1], dtype=np.float32)
    zp = np.array([0.0], dtype=np.float32)

    params = {
        "input": fnn.Tensor(q_input.ravel().tolist(), [4]),
        "scale": fnn.Tensor(scale.ravel().tolist(), [1]),
        "zp": fnn.Tensor(zp.ravel().tolist(), [1]),
    }

    executor = fnn.DAGExecutor(nodes, params, input_names, output_names)
    result = executor.forward(params)

    output = result["output"]
    expected = q_input * 0.1
    assert output.shape == [4], f"Expected [4] got {output.shape}"
    print(f"test_quantize_dequantize_linear: OK")


def test_embedding_packed_dispatch():
    """Test Embedding with packed weights.

    The packed weight shape is [embedding_dim, num_embeddings].
    Note: num_embeddings must be a multiple of 4 for U8x4 packing.
    """
    embedding_dim = 8
    num_embeddings = 8
    table = np.random.randn(embedding_dim, num_embeddings).astype(np.float32)
    w_shape = [embedding_dim, num_embeddings]

    packed = fnn.PackedTensor8.from_f32_auto(table.ravel().tolist(), w_shape)

    nodes = [
        {"name": "embed", "op_type": "Embedding",
         "inputs": "input", "outputs": "output"}
    ]
    input_names = ["input"]
    output_names = ["output"]

    params = {}
    packed_params = {"embed.weight": (
        bytes(packed.to_bytes()), w_shape, 2, packed.scales(), packed.zeros()
    )}

    executor = fnn.DAGExecutor(nodes, params, input_names, output_names,
                               packed_params=packed_params)

    indices = fnn.Tensor([2.0, 5.0, 1.0], [3])
    result = executor.forward({"input": indices})
    output = result["output"]
    assert output.shape == [3, embedding_dim], f"Expected [3,8] got {output.shape}"
    print(f"test_embedding_packed_dispatch: OK")


if __name__ == "__main__":
    test_matmul_packed_dispatch()
    test_conv_transpose_packed_dispatch()
    test_quantize_dequantize_linear()
    test_embedding_packed_dispatch()
    print("All DAG packed dispatch tests passed!")
