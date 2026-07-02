import subprocess
import sys
from pathlib import Path

DOC_FILE = "docs/guides/training/training-basics.md"


def _run_doc_truthfulness(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/ci/check-doc-truthfulness.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )


def test_doc_truthfulness_script_rejects_missing_local_markdown_link():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / DOC_FILE
    original = doc_path.read_text(encoding="utf-8")

    try:
        doc_path.write_text(
            original + "\n[Broken local doc link](missing-doc-for-guardrail.md)\n",
            encoding="utf-8",
        )
        result = _run_doc_truthfulness(repo_root)
    finally:
        doc_path.write_text(original, encoding="utf-8")

    assert result.returncode != 0
    assert DOC_FILE in result.stderr
    assert "missing-doc-for-guardrail.md" in result.stderr


def test_doc_truthfulness_script_rejects_missing_local_markdown_anchor():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / DOC_FILE
    original = doc_path.read_text(encoding="utf-8")

    try:
        doc_path.write_text(
            original + "\n[Broken local doc anchor](getting-started.md#missing-anchor-for-guardrail)\n",
            encoding="utf-8",
        )
        result = _run_doc_truthfulness(repo_root)
    finally:
        doc_path.write_text(original, encoding="utf-8")

    assert result.returncode != 0
    assert DOC_FILE in result.stderr
    assert "getting-started.md#missing-anchor-for-guardrail" in result.stderr


def test_doc_truthfulness_script_rejects_missing_same_doc_anchor():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / DOC_FILE
    original = doc_path.read_text(encoding="utf-8")

    try:
        doc_path.write_text(
            original + "\n[Broken same-doc anchor](#missing-anchor-for-guardrail)\n",
            encoding="utf-8",
        )
        result = _run_doc_truthfulness(repo_root)
    finally:
        doc_path.write_text(original, encoding="utf-8")

    assert result.returncode != 0
    assert DOC_FILE in result.stderr
    assert "#missing-anchor-for-guardrail" in result.stderr


def test_doc_truthfulness_script_rejects_missing_local_image_link():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / DOC_FILE
    original = doc_path.read_text(encoding="utf-8")

    try:
        doc_path.write_text(
            original + "\n![Broken local image](missing-image-for-guardrail.png)\n",
            encoding="utf-8",
        )
        result = _run_doc_truthfulness(repo_root)
    finally:
        doc_path.write_text(original, encoding="utf-8")

    assert result.returncode != 0
    assert DOC_FILE in result.stderr
    assert "missing-image-for-guardrail.png" in result.stderr
