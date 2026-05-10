"""Merge Python (results.json) + Rust (packed_results.json) into CI step summary."""

import json
import sys
import os


def load(path: str):
    with open(path) as f:
        return json.load(f)


def fmt_ms(ms: float) -> str:
    if ms < 10:
        return f"{ms:.3f}"
    return f"{ms:.2f}"


def memory_label(size: int) -> str:
    """Memory for an M×K weight matrix in f32."""
    mb = size * size * 4.0 / (1024.0 * 1024.0)
    return f"{mb:.0f} MB" if mb >= 1 else f"{mb:.1f} MB"


def main():
    py_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    rust_path = sys.argv[2] if len(sys.argv) > 2 else "packed_results.json"

    if not os.path.exists(py_path):
        print(f"::warning::Missing {py_path}", file=sys.stderr)
        return
    if not os.path.exists(rust_path):
        print(f"::warning::Missing {rust_path}", file=sys.stderr)
        return

    py_data = load(py_path)
    rust_data = load(rust_path)

    py_results = py_data.get("results", [])
    packed_results = rust_data.get("results", [])

    # ------------------------------------------------------------------
    # GEMV table — all implementations at all 4 sizes
    # ------------------------------------------------------------------
    sizes = [256, 512, 1024, 4096]

    # Build lookup: (impl, size) -> ms
    gemv_ms: dict = {}

    # Python: fastnn f32 + PyTorch f32
    for r in py_results:
        op = r.get("op", "")
        if op.startswith("gemv_"):
            for size in sizes:
                if op == f"gemv_{size}x{size}":
                    gemv_ms[("fastnn f32", size)] = r["fastnn_ms"]
                    gemv_ms[("PyTorch f32", size)] = r["torch_ms"]

    # Rust: packed precisions
    for r in packed_results:
        precision = r["precision"]
        size = r["size"]
        gemv_ms[(f"fastnn {precision}", size)] = r["ms"]

    implementations = [
        "PyTorch f32",
        "fastnn f32",
        "fastnn F16x2",
        "fastnn U8x4",
        "fastnn U4x8",
    ]

    md = "## ⚡ fastnn vs PyTorch — Latency Comparison\n\n"
    md += "### GEMV (matrix × vector) — Latency (ms)\n\n"
    cols = "| Implementation | " + " | ".join(f"  {s}×{s}" for s in sizes) + " | Memory  |"
    md += cols + "\n"
    md += "|" + "|".join("-" * max(len(s), 10) for s in cols.split("|")[1:-1]) + "|\n"

    for impl in implementations:
        cells = [impl]
        for size in sizes:
            key = (impl, size)
            if key in gemv_ms:
                cells.append(fmt_ms(gemv_ms[key]))
            else:
                cells.append("—")
        # Memory column
        if "U4x8" in impl:
            cells.append("8 MB")
        elif "U8x4" in impl:
            cells.append("16 MB")
        elif "F16x2" in impl:
            cells.append("32 MB")
        else:
            cells.append("64 MB")
        md += "| " + " | ".join(f"{c:>10}" for c in cells) + " |\n"

    # ------------------------------------------------------------------
    # Other operations table
    # ------------------------------------------------------------------
    skip_ops = {"gemv"}
    other_rows = []
    for r in py_results:
        op = r.get("op", "")
        parts = op.split("_")
        if parts and parts[0] in skip_ops:
            continue
        other_rows.append(r)

    if other_rows:
        md += "\n### Other Operations — Latency (ms)\n\n"
        md += "| Operation | PyTorch f32 | fastnn f32 |\n"
        md += "|---|---|---|\n"
        for r in other_rows:
            op = r["op"]
            th = fmt_ms(r["torch_ms"])
            fn = fmt_ms(r["fastnn_ms"])
            md += f"| {op} | {th:>10} | {fn:>10} |\n"

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print(md)


if __name__ == "__main__":
    main()
