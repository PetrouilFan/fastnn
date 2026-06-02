from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def training_doc_qualifies_data_parallel_as_experimental() -> None:
    text = (ROOT / "docs" / "training.md").read_text(encoding="utf-8")

    section = text.split("## Distributed Data Parallel", 1)[1].split("## Inference", 1)[0]

    assert "experimental" in section.lower()
    assert "CPU gradient aggregation" in section
    assert "hardware-dependent" in section.lower()


if __name__ == "__main__":
    training_doc_qualifies_data_parallel_as_experimental()
