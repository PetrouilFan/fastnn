"""Profile each conv2d_silu layer individually in yolo11n."""
import fastnn as fnn
import onnx, numpy as np
from pathlib import Path
import time, json

onnx_path = Path('/tmp/fastnn-yolo-verify/yolo11n/model.onnx')
model = onnx.load(str(onnx_path))

executor = fnn.AotExecutor.from_onnx(str(onnx_path))
x = np.random.randn(1, 3, 640, 640).astype(np.float32)
fx = fnn.tensor(x, [1, 3, 640, 640])

# Warmup
_ = executor.forward({'images': fx})

# Profile
prof = executor.profile({'images': fx})
entries = sorted(prof['profile'], key=lambda e: -e['elapsed_ns'])

# Group by kernel name
from collections import defaultdict
by_kernel = defaultdict(list)
for e in entries:
    by_kernel[e['kernel_name']].append(e['elapsed_ns'] / 1e6)

print(f"{'Kernel':30s} {'Count':6s} {'Total(ms)':10s} {'Mean(ms)':10s}")
print('-' * 60)
for k, times in sorted(by_kernel.items(), key=lambda x: -sum(x[1])):
    total = sum(times)
    cnt = len(times)
    mean = total / cnt
    print(f'{k:30s} {cnt:6d} {total:10.2f} {mean:10.2f}')

# Print individual conv2d_silu times
silu_times = [(e['kernel_name'], e['elapsed_ns'] / 1e6) for e in entries if 'silu' in e['kernel_name']]
print(f'\nIndividual conv2d_silu times:')
for i, (name, ms) in enumerate(sorted(silu_times, key=lambda x: -x[1])):
    print(f'  [{i:2d}] {ms:9.2f}ms')
