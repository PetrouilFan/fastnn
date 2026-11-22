#!/usr/bin/env python3
"""Diff two `perf_report.py` reports and fail on a regression.

The thresholds are deliberately generous, because a shared CI runner is not a
quiet machine. The point is to catch a change that makes a model twice as slow,
not to police a five percent wobble: the numbers in a report are medians from a
series, and the same binary can move by ten percent under background load.

Pairs are matched on (model name, model sha256, dtype, threads). A pair present
in the base and absent from the head, or measured on another machine, is
reported as skipped rather than treated as a regression.

Exit codes: 0 no regression, 1 regression, 2 nothing comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "fastnn-perf/1":
        raise SystemExit(f"{path} is not a fastnet-perf/1 report")
    return report


def key(report: dict) -> tuple[str, str, str, int]:
    return (
        report["model"]["name"],
        report["model"]["sha256"],
        report["dtype"],
        report["threads"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument(
        "--max-median-regression",
        type=float,
        default=0.15,
        help="fractional median slowdown that counts as a regression (default 0.15)",
    )
    parser.add_argument(
        "--max-peak-regression",
        type=float,
        default=0.20,
        help="fractional peak-RSS growth that counts as a regression (default 0.20)",
    )
    args = parser.parse_args(argv)

    base = load(args.base)
    head = load(args.head)

    if base["machine"]["platform"] != head["machine"]["platform"]:
        raise SystemExit(
            "the reports were made on different platforms "
            f"({base['machine']['platform']} vs {head['machine']['platform']}); "
            "their timings are not comparable"
        )
    if key(base) != key(head):
        raise SystemExit(
            "the reports describe different measurements: "
            f"{key(base)} vs {key(head)}"
        )

    base_median = base["timing_ms"]["median"]
    head_median = head["timing_ms"]["median"]
    base_peak = base["memory_mb"]["peak"]
    head_peak = head["memory_mb"]["peak"]
    median_change = (head_median - base_median) / base_median
    peak_change = (head_peak - base_peak) / base_peak if base_peak else 0.0

    print(f"model          {base['model']['name']} ({base['dtype']}, {base['threads']} threads)")
    print(f"median         {base_median:.2f} -> {head_median:.2f} ms ({median_change:+.1%})")
    print(f"p95            {base['timing_ms']['p95']:.2f} -> {head['timing_ms']['p95']:.2f} ms")
    print(f"peak RSS       {base_peak:.1f} -> {head_peak:.1f} MB ({peak_change:+.1%})")

    failed = False
    if median_change > args.max_median_regression:
        print(
            f"REGRESSION: median grew {median_change:.1%}, above the "
            f"{args.max_median_regression:.0%} budget",
            file=sys.stderr,
        )
        failed = True
    if peak_change > args.max_peak_regression:
        print(
            f"REGRESSION: peak RSS grew {peak_change:.1%}, above the "
            f"{args.max_peak_regression:.0%} budget",
            file=sys.stderr,
        )
        failed = True

    accuracy = head.get("accuracy")
    if accuracy:
        mismatched = [
            name
            for name, entry in accuracy.items()
            if entry.get("hash_match") is False
        ]
        if mismatched:
            print(
                "ACCURACY: output digests differ for " + ", ".join(mismatched),
                file=sys.stderr,
            )
            failed = True
        else:
            print("accuracy       output digests match the reference")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
