"""Export YOLO11n to ONNX, then convert to FastNN .fnn format."""
from ultralytics import YOLO
import fastnn
import onnx
import logging

logging.basicConfig(level=logging.INFO)

# Step 1: Download and export to ONNX
print("Loading YOLO11n model...")
model = YOLO("yolo11n.pt")
print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, opset=11, dynamic=False, simplify=True)
print("Output: yolo11n.onnx")

# Step 2: Check ONNX ops
print("\n=== Checking ONNX Operators ===")
onnx_model = onnx.load("yolo11n.onnx")
ops = {}
for node in onnx_model.graph.node:
    op = node.op_type
    ops[op] = ops.get(op, 0) + 1

for op, count in sorted(ops.items()):
    print(f"  {op:<30s} x{count}")

# Step 3: Convert ONNX → FastNN .fnn
print("\n=== Converting to FastNN .fnn ===")
info = fastnn.import_onnx("yolo11n.onnx", "yolo11n.fnn")
print(f"Converted: {info['parameters']} parameter tensors")
print(f"Input shape:  {info['input_shape']}")
print(f"Output shape: {info['output_shape']}")
print(f"\nLayers ({len(info['layers'])}):")
for l in info['layers']:
    print(f"  {l['type']:30s} {l['name']}")

# Log unsupported ops — critical for correctness check
unsupported = [l for l in info['layers'] if l['type'].startswith('Unsupported')]
if unsupported:
    print(f"\n⚠ UNSUPPORTED OPS ({len(unsupported)}):")
    for l in unsupported:
        print(f"  {l['type']}")
else:
    print("\n✓ All ops supported!")
