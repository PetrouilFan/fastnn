#!/usr/bin/env python3
"""Microbenchmark exact YOLO Conv+SiLU shapes in fastnn vs PyTorch.

This isolates the conv backend from the full YOLO graph. For each shape, the
script generates a one-layer ONNX model:

    input -> Conv(weight,bias) -> Sigmoid -> Mul

fastnn should fuse this to conv2d_silu, while PyTorch runs
`torch.nn.functional.conv2d` followed by `torch.nn.functional.silu` on the same
random input/weights/bias.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from yolo_compare_fastnn_pytorch import _make_fastnn_executor  # noqa: E402


YOLO_TOP_SHAPES: list[dict[str, Any]] = [
    # name, input NCHW, weight F,C,KH,KW, stride, padding
    {"name": "yolo_1x1_f128_c192_sp400", "input": [1, 192, 20, 20], "weight": [128, 192, 1, 1], "stride": 1, "padding": 0},
    {"name": "yolo_3x3_f64_c64_sp400", "input": [1, 64, 20, 20], "weight": [64, 64, 3, 3], "stride": 1, "padding": 1},
    {"name": "yolo_3x3_f32_c32_sp1600", "input": [1, 32, 40, 40], "weight": [32, 32, 3, 3], "stride": 1, "padding": 1},
    {"name": "yolo_3x3_f16_c16_sp6400", "input": [1, 16, 80, 80], "weight": [16, 16, 3, 3], "stride": 1, "padding": 1},
    {"name": "yolo_stem_3x3_f16_c3_sp25600", "input": [1, 3, 320, 320], "weight": [16, 3, 3, 3], "stride": 2, "padding": 1},
    {"name": "yolo_3x3_f32_c16_sp6400", "input": [1, 16, 160, 160], "weight": [32, 16, 3, 3], "stride": 2, "padding": 1},
    {"name": "yolo_1x1_f64_c64_sp1600", "input": [1, 64, 40, 40], "weight": [64, 64, 1, 1], "stride": 1, "padding": 0},
    {"name": "yolo_3x3_f128_c128_sp100", "input": [1, 128, 10, 10], "weight": [128, 128, 3, 3], "stride": 1, "padding": 1},
]


def timer(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "iters": iters,
        "mean_ms": float(statistics.mean(samples)),
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def make_conv_silu_onnx(path: Path, shape: dict[str, Any], weight: np.ndarray, bias: np.ndarray) -> str:
    input_shape = list(map(int, shape["input"]))
    weight_shape = list(map(int, shape["weight"]))
    stride = int(shape["stride"])
    pad = int(shape["padding"])
    n, _c, h, w = input_shape
    f, _wc, kh, kw = weight_shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    output_shape = [n, f, oh, ow]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "weight", "bias"],
            ["conv_out"],
            name="conv",
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("Sigmoid", ["conv_out"], ["sigmoid_out"], name="sigmoid"),
        helper.make_node("Mul", ["conv_out", "sigmoid_out"], ["output"], name="mul_silu"),
    ]
    graph = helper.make_graph(
        nodes,
        f"{shape['name']}_conv_silu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(weight.astype(np.float32), "weight"), numpy_helper.from_array(bias.astype(np.float32), "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return "output"


def bench_shape(shape: dict[str, Any], out_dir: Path, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    import fastnn as fnn
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape["input"], dtype=np.float32)
    w = rng.standard_normal(shape["weight"], dtype=np.float32) * 0.05
    b = rng.standard_normal((shape["weight"][0],), dtype=np.float32) * 0.01

    onnx_path = out_dir / f"{shape['name']}.onnx"
    output_name = make_conv_silu_onnx(onnx_path, shape, w, b)
    executor, input_name, fastnn_output_name = _make_fastnn_executor(onnx_path)
    assert fastnn_output_name == output_name

    fx = fnn.tensor(x, list(x.shape))
    tx = torch.from_numpy(x)
    tw = torch.from_numpy(w)
    tb = torch.from_numpy(b)
    stride = int(shape["stride"])
    padding = int(shape["padding"])

    with torch.no_grad():
        y_torch = F.silu(F.conv2d(tx, tw, tb, stride=stride, padding=padding)).detach().cpu().numpy()
    y_fastnn = executor.forward({input_name: fx})[output_name].numpy()
    diff = np.abs(y_fastnn - y_torch)

    pytorch_speed = timer(lambda: F.silu(F.conv2d(tx, tw, tb, stride=stride, padding=padding)), warmup, iters)
    fastnn_speed = timer(lambda: executor.forward({input_name: fx}), warmup, iters)
    profile = executor.profile({input_name: fx})["profile"]
    conv_entries = [e for e in profile if e.get("kernel_name") in {"conv2d_silu", "conv2d", "conv2d_relu", "conv2d_gelu"}]

    n, c, h, ww = shape["input"]
    f, _c, kh, kw = shape["weight"]
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (ww + 2 * padding - kw) // stride + 1
    gemm_m = f
    gemm_k = c * kh * kw
    gemm_n = oh * ow
    flops = 2 * gemm_m * gemm_k * gemm_n
    fast_gflops = (flops / 1e9) / (fastnn_speed["mean_ms"] / 1000.0)
    torch_gflops = (flops / 1e9) / (pytorch_speed["mean_ms"] / 1000.0)

    return {
        "name": shape["name"],
        "input": shape["input"],
        "weight": shape["weight"],
        "stride": stride,
        "padding": padding,
        "output": [n, f, oh, ow],
        "gemm": [gemm_m, gemm_k, gemm_n],
        "flops": flops,
        "accuracy": {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())},
        "pytorch_speed": pytorch_speed,
        "fastnn_speed": fastnn_speed,
        "speed_ratio_fastnn_over_pytorch": fastnn_speed["mean_ms"] / pytorch_speed["mean_ms"],
        "fastnn_gflops_est": fast_gflops,
        "pytorch_gflops_est": torch_gflops,
        "fastnn_conv_profile": conv_entries,
        "onnx_path": str(onnx_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/fastnn-conv-shape-microbench"))
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    rows = [bench_shape(shape, args.out_dir, args.warmup, args.iters, args.seed) for shape in YOLO_TOP_SHAPES]
    rows.sort(key=lambda r: r["speed_ratio_fastnn_over_pytorch"], reverse=True)
    result = {"threads": args.threads, "warmup": args.warmup, "iters": args.iters, "rows": rows}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"threads={args.threads} warmup={args.warmup} iters={args.iters}")
    for r in rows:
        print(
            f"{r['name']:34s} ratio={r['speed_ratio_fastnn_over_pytorch']:.2f} "
            f"fastnn={r['fastnn_speed']['mean_ms']:.3f}ms torch={r['pytorch_speed']['mean_ms']:.3f}ms "
            f"gf/s fast={r['fastnn_gflops_est']:.1f} torch={r['pytorch_gflops_est']:.1f} "
            f"max_abs={r['accuracy']['max_abs']:.2e} gemm={tuple(r['gemm'])}"
        )
        for e in r["fastnn_conv_profile"]:
            print(f"  profile {e.get('kernel_name')} {float(e.get('elapsed_ns', 0))/1e6:.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
