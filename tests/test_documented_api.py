"""Gate the Python facade names that the documentation advertises.

`scripts/ci/check-doc-truthfulness.py` verifies that local markdown links and
anchors resolve. It cannot see a claim like "`fnn.RNN(input_size, hidden_size)`
is the RNN layer" when that class has never existed, which is what
`docs/reference/api-reference.md` said for RNN, LSTM and GRU. A reader following
the reference then writes code that cannot import.

Every back-ticked `fnn.Something` in the repository's markdown is a promise
about this package, so assert it resolves. Dotted names descend (`fnn.io.convert`
checks both hops), which also covers the submodules and the `fastnn-convert`
entry point.
"""

from __future__ import annotations

import re
from pathlib import Path

import fastnn

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "target", ".venv", "node_modules", "site-packages"}
BACKTICKED_FACADE = re.compile(r"`fnn\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def _documented_facade_names() -> dict[str, list[str]]:
    """Map each documented `fnn.*` name to the markdown files that mention it."""
    seen: dict[str, list[str]] = {}
    for markdown in sorted(ROOT.rglob("*.md")):
        if SKIP_PARTS.intersection(markdown.parts):
            continue
        text = markdown.read_text(encoding="utf-8", errors="ignore")
        for match in BACKTICKED_FACADE.finditer(text):
            seen.setdefault(match.group(1), []).append(
                markdown.relative_to(ROOT).as_posix()
            )
    return seen


def _resolves(name: str) -> bool:
    current: object = fastnn
    for part in name.split("."):
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    return True


def test_every_documented_facade_name_exists() -> None:
    documented = _documented_facade_names()
    # A guard against the regex silently matching nothing after a docs move.
    assert len(documented) > 20, f"only found {len(documented)} documented names"

    missing = {
        name: sorted(set(paths))
        for name, paths in documented.items()
        if not _resolves(name)
    }
    assert not missing, "\n".join(
        f"docs promise `fnn.{name}` but the package has no such attribute "
        f"(mentioned in {', '.join(paths)})"
        for name, paths in sorted(missing.items())
    )
