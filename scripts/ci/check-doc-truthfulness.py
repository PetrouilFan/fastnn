import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def training_doc_qualifies_data_parallel_as_experimental() -> None:
    training_path = ROOT / "docs" / "training.md"
    if not training_path.exists():
        # docs/training.md was removed in the v2.5.0 docs rewrite; skip
        return
    text = training_path.read_text(encoding="utf-8")

    section = text.split("## Distributed Data Parallel", 1)[1].split("## Inference", 1)[0]

    assert "experimental" in section.lower()
    assert "CPU gradient aggregation" in section
    assert "hardware-dependent" in section.lower()


def _is_local_link(target: str) -> bool:
    if target.startswith(("http://", "https://", "mailto:", "tel:")):
        return False
    path_without_anchor = target.split("#", 1)[0]
    return bool(path_without_anchor or target.startswith("#"))


def _link_path(markdown_path: Path, target: str) -> Path:
    path_without_anchor = target.split("#", 1)[0]
    if not path_without_anchor:
        return markdown_path
    candidate = (markdown_path.parent / path_without_anchor).resolve()
    if candidate.exists():
        return candidate
    markdown_candidate = candidate.with_suffix(".md")
    if markdown_candidate.exists():
        return markdown_candidate
    return candidate


def _github_heading_slug(heading: str) -> str:
    """Return the GitHub-style anchor slug for a markdown heading."""
    slug = heading.strip().lower()
    slug = re.sub(r"<[^>]+>", "", slug)
    slug = re.sub(r"[`*_~]", "", slug)
    slug = re.sub(r"[^a-z0-9 _-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _markdown_anchors(markdown_path: Path) -> set[str]:
    anchors: set[str] = set()
    text = markdown_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            anchors.add(_github_heading_slug(match.group(1)))
    return anchors


def _iter_markdown_links(text: str) -> list[str]:
    """Return inline markdown link targets outside fenced code blocks."""
    targets: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        targets.extend(match.group(1).strip() for match in LOCAL_LINK_RE.finditer(line))
    return targets


def docs_have_resolvable_local_markdown_links() -> None:
    failures: list[str] = []

    for markdown_path in sorted(ROOT.rglob("*.md")):
        if any(
            part in {".git", "target", ".venv", "node_modules", ".kilo"}
            for part in markdown_path.parts
        ):
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for target in _iter_markdown_links(text):
            if not target or not _is_local_link(target):
                continue
            link_path = _link_path(markdown_path, target)
            if not link_path.exists():
                rel_path = markdown_path.relative_to(ROOT).as_posix()
                failures.append(f"{rel_path}: missing local link target {target}")
                continue

            if "#" in target:
                anchor = target.split("#", 1)[1]
                if anchor and anchor not in _markdown_anchors(link_path):
                    rel_path = markdown_path.relative_to(ROOT).as_posix()
                    failures.append(f"{rel_path}: missing local link anchor {target}")

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    checks = [
        training_doc_qualifies_data_parallel_as_experimental,
        docs_have_resolvable_local_markdown_links,
    ]
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
