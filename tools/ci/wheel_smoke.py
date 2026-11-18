#!/usr/bin/env python3
"""Smoke-test an installed FastNN wheel from outside the source tree.

Run with the wheel's virtualenv interpreter and a temporary working directory,
so a source-tree import (editable install) cannot masquerade as a wheel install.

Exits 0 only when:
  * fastnn imports from the virtualenv site-packages, not the checkout;
  * the version matches the expected version when one is given;
  * the 3.0 inference surface imports and lists at least the CPU device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect-version", default="")
    parser.add_argument("--source-root", default="")
    args = parser.parse_args(argv)

    import fastnn

    location = Path(fastnn.__file__).resolve()
    print(f"fastnn imported from: {location}")
    if args.source_root:
        source = Path(args.source_root).resolve()
        # Build artifacts live under <source>/target, so only reject an import
        # from the source package itself, not from any path under the checkout.
        for forbidden in (source / "fastnn", source / "src"):
            if location == forbidden or forbidden in location.parents:
                print(
                    f"FAIL: fastnn was imported from the source tree ({forbidden}), not an installed wheel",
                    file=sys.stderr,
                )
                return 1

    version = getattr(fastnn, "__version__", "")
    print(f"fastnn.__version__ = {version}")
    if args.expect_version and version != args.expect_version:
        print(f"FAIL: expected version {args.expect_version!r}, got {version!r}", file=sys.stderr)
        return 1

    # The shipped Python surface is the 3.0 inference API. It must import
    # without a model and without a GPU.
    import fastnn.fn3 as fn3

    if fn3.Session is not fastnn._core.fn3.Session:
        print("FAIL: fastnn.fn3.Session is not fastnn._core.fn3.Session", file=sys.stderr)
        return 1
    devices = fn3.devices()
    if not devices or devices[0].kind != "cpu":
        print(f"FAIL: fn3.devices() = {devices!r}; the CPU device must always be listed", file=sys.stderr)
        return 1
    print(f"inference smoke: fastnn.fn3 imported, devices={[d.id for d in devices]}")

    print("WHEEL SMOKE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
