#!/usr/bin/env python3
"""Analyze cProfile output for the DDP benchmark."""

import pstats
import sys


def analyze_profile(profile_file, num_functions=30):
    """Analyze the profile and print top functions."""
    p = pstats.Stats(profile_file)
    p.strip_dirs()  # Remove directory names for cleaner output
    p.sort_stats("cumulative")  # Sort by cumulative time
    print(f"Top {num_functions} functions by cumulative time:")
    p.print_stats(num_functions)
    print("\n" + "=" * 80 + "\n")

    p.sort_stats("time")  # Sort by own time
    print(f"Top {num_functions} functions by own time:")
    p.print_stats(num_functions)


if __name__ == "__main__":
    profile_file = sys.argv[1] if len(sys.argv) > 1 else "bench_ddp.prof"
    analyze_profile(profile_file, num_functions=20)
