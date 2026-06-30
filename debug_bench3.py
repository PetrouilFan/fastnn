"""Same as benchmark's _run_fastnn but with timing."""
import time, numpy as np

def run_test():
    onnx_path = type(sys.path[0])(r'/tmp/fastnn-yolo-verify/yolo11n/model.onnx')
    import onnx
    from pathlib import Path
    onnx_path = Path('/tmp/fastnn-yolo-verify/yolo11n/model.onnx')
    
    # Exact copy of _run_fastnn
    import fastnn as fnn
    from scripts.yolo_compare_fastnn_pytorch import _make_fastnn_executor
    
    executor, input_name, output_name = _make_fastnn_executor(onnx_path, 'f32')
    
    x = np.random.randn(1, 3, 640, 640).astype(np.float32)
    fx = fnn.tensor(x, [1, 3, 640, 640])
    
    t0 = time.perf_counter()
    fy = executor.forward({input_name: fx})[output_name].numpy()
    t1 = time.perf_counter()
    print(f"First forward: {(t1-t0)*1000:.0f}ms")
    
    for i in range(3):
        t0 = time.perf_counter()
        executor.forward({input_name: fx})
        t1 = time.perf_counter()
        print(f"  Forward {i}: {(t1-t0)*1000:.0f}ms")

import sys
run_test()
