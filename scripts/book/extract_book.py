"""Extract the reference monograph into auditable, page-addressable chapters.

This is intentionally a deterministic extractor rather than a summarizer.  The
knowledge notes under ``knowledge_base/book`` are the human-readable distilled
layer; these files remain the evidence layer used to verify page-level claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "papers" / "extracted_text" / "book"


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    title: str
    book_start: int
    book_end: int
    pdf_start: int
    pdf_end: int
    sections: tuple[str, ...]


CHAPTERS = (
    Chapter(
        1,
        "overview",
        "Overview of Industrial Alarm Systems",
        1,
        47,
        13,
        59,
        (
            "1.1 Basic Concepts and Background Information",
            "1.2 Alarm Overloading and Its Main Causes",
            "1.3 Current Research Status in Literature",
            "1.4 Major Research Problems to Be Solved",
        ),
    ),
    Chapter(
        2,
        "univariate_design",
        "Optimal Design of Univariate Alarm Systems",
        49,
        127,
        60,
        138,
        (
            "2.1 Alarm Delay Timers for IID Process Variables",
            "2.2 Alarm Delay Timers for Non-IID Process Variables",
            "2.3 Alarm Deadbands for Non-IID Process Variables",
            "2.4 Alarm Thresholds Based on Alarm Probability Plots",
        ),
    ),
    Chapter(
        3,
        "multivariate_design",
        "Optimal Design of Multivariate Alarm Systems",
        129,
        220,
        139,
        230,
        (
            "3.1 Normal Operating Zone-Based Multivariate Alarm Systems",
            "3.2 Variation Direction-Based Multivariate Alarm Systems",
            "3.3 Multivariate Alarm Systems for Electrical Pumps",
            "3.4 Multivariate Alarm Systems for Condensers",
        ),
    ),
    Chapter(
        4,
        "root_cause",
        "Root-Cause Analysis of Alarm Events",
        221,
        301,
        231,
        311,
        (
            "4.1 Causality Inference for Alarm Variables",
            "4.2 Causality Inference for Process Variables",
            "4.3 Root-Cause Analysis for Alarm Variables",
            "4.4 Root-Cause Analysis for Process Variables",
        ),
    ),
    Chapter(
        5,
        "alarm_floods",
        "Analysis of Industrial Alarm Floods",
        303,
        379,
        312,
        388,
        (
            "5.1 Detection of Alarm Floods",
            "5.2 Similarity Analysis of Alarm Floods",
            "5.3 Pattern Mining of Alarm Floods",
            "5.4 Prediction of Alarm Floods",
        ),
    ),
    Chapter(
        6,
        "visual_analytics",
        "Alarm Visual Analytics and Applications",
        381,
        420,
        389,
        428,
        (
            "6.1 Overview of Alarm Data and Analytics",
            "6.2 Visual Analytics for Alarm System Performance",
            "6.3 Visual Analytics for Related Alarms and Events",
            "6.4 Visual Analytics for Alarm Flood Sequences",
        ),
    ),
)


LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ad": "",
    }
)


def normalize(text: str) -> str:
    text = text.translate(LIGATURES).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def find_default_pdf() -> Path:
    candidates = sorted(ROOT.glob("*.pdf"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one top-level reference PDF, found {len(candidates)}"
        )
    return candidates[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_range(reader: PdfReader, pdf_start: int, pdf_end: int) -> str:
    pages: list[str] = []
    for pdf_page in range(pdf_start, pdf_end + 1):
        text = normalize(reader.pages[pdf_page - 1].extract_text() or "")
        pages.append(f"--- PDF_PAGE={pdf_page} ---\n{text}")
    return "\n\n".join(pages) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pdf_path = (args.pdf or find_default_pdf()).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 433:
        raise ValueError(f"Expected 433 PDF pages, found {len(reader.pages)}")

    (output / "00_front_matter.txt").write_text(
        extract_range(reader, 1, 12), encoding="utf-8"
    )
    for chapter in CHAPTERS:
        filename = f"{chapter.number:02d}_{chapter.slug}.txt"
        (output / filename).write_text(
            extract_range(reader, chapter.pdf_start, chapter.pdf_end),
            encoding="utf-8",
        )

    manifest = {
        "title": "Intelligent Industrial Alarm Systems",
        "subtitle": "Advanced Analysis and Design Methods",
        "authors": ["Jiandong Wang", "Wenkai Hu", "Tongwen Chen"],
        "year": 2024,
        "doi": "10.1007/978-981-97-6516-4",
        "source_file": pdf_path.name,
        "source_sha256": sha256(pdf_path),
        "pdf_pages": len(reader.pages),
        "chapters": [asdict(chapter) for chapter in CHAPTERS],
        "notes": [
            "PDF page numbers are physical pages and are one-based.",
            "Book page numbers are the printed chapter pagination.",
            "The PDF-to-book page offset is not constant because blank leaves are omitted.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(CHAPTERS)} chapters to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
