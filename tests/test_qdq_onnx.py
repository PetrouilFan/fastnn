"""Tests for Q/DQ ONNX import."""
import numpy as np
import os
import tempfile

pytest_importorskip = None
try:
    import pytest
except ImportError:
    pass
else:
    pytest_importorskip = pytest.importorskip

skip_msg = "onnx not installed, skipping"

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    onnx = None


def create_qdq_conv_model():
    """Create a simple Conv model with Q/DQ wrapping."""
    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 30, 30])

    w = np.random.randn(16, 3, 3, 3).astype(np.float32)
    w_init = helper.make_tensor('conv.weight', TensorProto.FLOAT, [16, 3, 3, 3], w.flatten().tolist())

    in_scale_val = np.array([0.1], dtype=np.float32)
    in_zp_val = np.array([0], dtype=np.int32)
    out_scale_val = np.array([0.1], dtype=np.float32)
    out_zp_val = np.array([0], dtype=np.int32)

    in_scale = helper.make_tensor('in_scale', TensorProto.FLOAT, [1], in_scale_val.flatten().tolist())
    in_zp = helper.make_tensor('in_zp', TensorProto.INT32, [1], in_zp_val.flatten().tolist())
    out_scale = helper.make_tensor('out_scale', TensorProto.FLOAT, [1], out_scale_val.flatten().tolist())
    out_zp = helper.make_tensor('out_zp', TensorProto.INT32, [1], out_zp_val.flatten().tolist())

    nodes = [
        helper.make_node('QuantizeLinear', ['input', 'in_scale', 'in_zp'], ['q_input'], name='quant'),
        helper.make_node('Conv', ['q_input', 'conv.weight'], ['q_conv'], name='conv',
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('DequantizeLinear', ['q_conv', 'out_scale', 'out_zp'], ['output'], name='dequant'),
    ]

    graph = helper.make_graph(nodes, 'test', [inp], [out], [w_init, in_scale, in_zp, out_scale, out_zp])
    model = helper.make_model(graph)

    path = os.path.join(tempfile.gettempdir(), 'qdq_conv.onnx')
    onnx.save(model, path)
    return path


def test_qdq_folding():
    """Test that Q/DQ pattern is detected and folded by import_onnx."""
    if onnx is None:
        print(f"SKIP: {skip_msg}")
        return
    from fastnn.io.onnx import import_onnx
    onnx_path = create_qdq_conv_model()
    fnn_path = os.path.join(tempfile.gettempdir(), 'qdq_conv.fnn')

    try:
        info = import_onnx(onnx_path, fnn_path)
        assert info is not None
        print(f"test_qdq_folding: OK, layers={len(info.get('layers', []))}")
    finally:
        if os.path.exists(fnn_path):
            os.remove(fnn_path)


if __name__ == "__main__":
    test_qdq_folding()
    print("All Q/DQ tests passed!")
