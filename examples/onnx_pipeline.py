#!/usr/bin/env python3
"""Example: Low-level ONNX pipeline usage with fastnn.

This demonstrates the building blocks of the ONNX pipeline:
1. Import an ONNX model to .fnn format
2. Build a model from the .fnn file
3. Run inference with raw tensors
4. Use the shape inference and graph optimizer directly

Usage:
    python examples/onnx_pipeline.py --model yolov8n.onnx
"""

import argparse
import json
import time
import tempfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX pipeline example")
    parser.add_argument("--model", "-m", required=True, help="Path to ONNX model")
    parser.add_argument("--inspect", action="store_true", help="Inspect model structure")
    parser.add_argument("--optimize", action="store_true", help="Apply graph optimizations")
    return parser.parse_args()


def main():
    args = parse_args()
    import fastnn as fnn

    onnx_path = str(args.model)
    tmpdir = tempfile.mkdtemp()
    fnn_path = Path(tmpdir) / "model.fnn"

    print("=" * 60)
    print("FastNN ONNX Pipeline Demo")
    print("=" * 60)

    # Step 1: Import ONNX model
    print("\n1. Importing ONNX model...")
    start = time.time()
    info = fnn.convert_from_onnx(onnx_path, str(fnn_path))
    elapsed = time.time() - start
    print(f"   Imported in {elapsed*1000:.1f}ms")
    print(f"   Layers: {len(info.get('layers', []))}")
    print(f"   Parameters: {info.get('parameters', 0)}")
    print(f"   Input shape: {info.get('input_shape')}")
    print(f"   Output shape: {info.get('output_shape')}")

    # Step 2: Inspect model structure
    if args.inspect:
        print("\n2. Model structure:")
        from fastnn.io import read_fnn_header
        with open(fnn_path, "rb") as f:
            _, _, header, _ = read_fnn_header(f)

        graph = header.get("graph", {})
        nodes = graph.get("nodes", [])
        print(f"   Total nodes: {len(nodes)}")
        print("\n   Node types:")
        type_counts = {}
        for node in nodes:
            op = node.get("op_type", "Unknown")
            type_counts[op] = type_counts.get(op, 0) + 1
        for op, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"     {op}: {count}")

        print("\n   Model inputs:")
        for inp in graph.get("inputs", []):
            name = inp.get("name", "")
            shape = inp.get("shape")
            print(f"     {name}: {shape}")

        print("\n   Model outputs:")
        for out in graph.get("outputs", []):
            name = out.get("name", "")
            shape = out.get("shape")
            print(f"     {name}: {shape}")

    # Step 3: Apply graph optimizations
    if args.optimize:
        print("\n3. Optimizing graph...")
        from fastnn.io.graph_optimizer import optimize_graph
        from fastnn.io import read_fnn_header

        with open(fnn_path, "rb") as f:
            _, _, header, _ = read_fnn_header(f)

        optimized = optimize_graph(header)
        opt_nodes = optimized["graph"]["nodes"]
        orig_nodes = header["graph"]["nodes"]
        print(f"   Nodes: {len(orig_nodes)} -> {len(opt_nodes)}")

        fused = [n for n in opt_nodes if n.get("fused")]
        if fused:
            print(f"   Fused {len(fused)} Conv+BN pairs")

    # Step 4: Build model
    print("\n4. Building model...")
    start = time.time()
    model = fnn.build_model_from_fnn(str(fnn_path))
    elapsed = time.time() - start
    print(f"   Built in {elapsed*1000:.1f}ms")
    print(f"   Type: {type(model).__name__}")

    # Step 5: Run inference
    print("\n5. Running inference...")
    input_shape = info.get("input_shape", [1, 3, 640, 640])
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    x = fnn.tensor(dummy_input, list(dummy_input.shape))

    start = time.time()
    if hasattr(model, "forward"):
        outputs = model.forward({"images": x} if "images" in str(info.get("graph", {}).get("inputs", [])) else {"input": x})
    else:
        outputs = model(x)
    elapsed = time.time() - start
    print(f"   Forward pass: {elapsed*1000:.1f}ms")

    if isinstance(outputs, dict):
        for name, tensor in outputs.items():
            if hasattr(tensor, "shape"):
                print(f"   Output '{name}': {tensor.shape}")
            elif hasattr(tensor, "numpy"):
                out_array = tensor.numpy()
                print(f"   Output '{name}': {out_array.shape}")
    else:
        print(f"   Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
