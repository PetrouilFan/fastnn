#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def benchmark_entries(criterion_dir: Path):
    for estimates_path in sorted(criterion_dir.rglob("new/estimates.json")):
        benchmark_dir = estimates_path.parent.parent
        relative_parts = benchmark_dir.relative_to(criterion_dir).parts
        if len(relative_parts) < 2:
            continue

        group = relative_parts[0]
        benchmark = "/".join(relative_parts[1:])

        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)

        yield {
            "group": group,
            "benchmark": benchmark,
            "mean_ns": estimates["mean"]["point_estimate"],
            "median_ns": estimates["median"]["point_estimate"],
            "std_dev_ns": estimates["std_dev"]["point_estimate"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Criterion benchmark output into one JSON file.")
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSON summary to write",
    )
    args = parser.parse_args()

    criterion_dir = Path(args.criterion_dir)
    output_path = Path(args.output)

    if not criterion_dir.exists():
        raise SystemExit(f"criterion directory does not exist: {criterion_dir}")

    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterion_dir": str(criterion_dir),
        "benchmarks": list(benchmark_entries(criterion_dir)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
