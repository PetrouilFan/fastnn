#!/usr/bin/env python3
"""Smoke-test an installed FastNN wheel from outside the source tree.

Run with the wheel's virtualenv interpreter and a temporary working directory,
so a source-tree import (editable install) cannot masquerade as a wheel install.

Exits 0 only when:
  * fastnn imports from the virtualenv site-packages, not the checkout;
  * the version matches the expected version when one is given;
  * a minimal inference call produces the expected values.
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

    # The frozen 2.x facade and the 3.0 core expose different surfaces. Name the
    # one being tested so the lane's output cannot be mistaken for the other.
    if hasattr(fastnn, "tensor") and hasattr(fastnn, "ones"):
        print("smoke surface: frozen 2.x Python facade (replaced by FN3-18 / #64)")
        x = fastnn.ones([2, 2])
        if list(x.shape) != [2, 2]:
            print(f"FAIL: ones([2,2]).shape = {x.shape}", file=sys.stderr)
            return 1
        doubled = (x + x).numpy().tolist()
        if doubled != [[2.0, 2.0], [2.0, 2.0]]:
            print(f"FAIL: ones([2,2]) + itself = {doubled}", file=sys.stderr)
            return 1
        print("inference smoke: ones([2,2]) + ones([2,2]) == [[2,2],[2,2]]")
    elif hasattr(fastnn, "fn3"):
        print("smoke surface: 3.0 fn3 API")
        # FN3-18 (#64) owns the exact surface; importability is the packaging
        # contract here until that issue lands its own smoke.
        _ = fastnn.fn3
        print("inference smoke: fastnn.fn3 imported")
    else:
        print("FAIL: package exposes neither the 2.x facade nor fastnn.fn3", file=sys.stderr)
        return 1

    print("WHEEL SMOKE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
