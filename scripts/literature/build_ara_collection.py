"""Build one auditable ARA package per registered literature source."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "papers" / "literature" / "registry.json"
DOWNLOADS = ROOT / "papers" / "literature" / "download_manifest.json"
OUTPUT = ROOT / "papers" / "literature" / "ara"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def artifact_status(paper: dict, download: dict) -> str:
    if download.get("status") == "downloaded":
        if paper.get("official_artifact", {}).get("status") == "downloaded":
            return "source_and_code_acquired"
        return "source_acquired_code_pending"
    return "metadata_only"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    downloads = json.loads(args.downloads.read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in downloads["records"]}
    collection_rows = []

    for paper in registry["papers"]:
        paper_dir = args.output / paper["id"]
        download = by_id.get(paper["id"], {"status": "manifest_missing"})
        status = artifact_status(paper, download)
        metadata = {
            "id": paper["id"],
            "title": paper["title"],
            "year": paper["year"],
            "venue": paper["venue"],
            "doi": paper["doi"],
            "role": paper["role"],
            "access": paper["access"],
            "local_source": download,
            "official_artifact": paper.get("official_artifact"),
            "ara_status": status,
        }
        write(paper_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

        claims_md = "\n".join(
            f"- `{claim['id']}` [{claim['status']}] {claim['statement']}  \n"
            f"  Support locator: `{claim['support']}`"
            for claim in paper["claims"]
        )
        write(
            paper_dir / "PAPER.md",
            f"""# {paper['title']}

## Metadata

- Year: {paper['year']}
- Venue: {paper['venue']}
- DOI: [{paper['doi']}](https://doi.org/{paper['doi']})
- Benchmark role: {paper['role']}
- ARA status: `{status}`

## Executive summary

{paper['method_summary']}

## Claims and evidence

{claims_md}

## Reproduction status

Source acquisition is `{download.get('status')}`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
""",
        )
        write(
            paper_dir / "logic" / "concepts.md",
            f"""# Concepts

- Benchmark role: {paper['role']}
- Method summary: {paper['method_summary']}
- Access class: `{paper['access']}`
""",
        )
        write(
            paper_dir / "logic" / "problem.md",
            f"""# Problem

This source is included to support **{paper['role']}**. The exact input schema,
prediction target, split unit, and metrics must be frozen in an experiment config
before the method can be called reproduced.
""",
        )
        write(paper_dir / "logic" / "claims.md", f"# Claims\n\n{claims_md}")
        write(
            paper_dir / "logic" / "related_work.md",
            "# Related work\n\nSee `papers/literature/registry.json` and the book algorithm inventory for the method family and expansion cycle.",
        )
        write(
            paper_dir / "logic" / "experiments.md",
            "# Experiments\n\nNo paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.",
        )
        write(
            paper_dir / "logic" / "solution" / "method.md",
            f"# Method\n\n{paper['method_summary']}",
        )
        write(
            paper_dir / "logic" / "solution" / "constraints.md",
            "# Constraints\n\n- Do not substitute synthetic smoke data for the paper dataset.\n- Do not report metadata/code acquisition as score reproduction.\n- Preserve grouped episode/run splits and data provenance.",
        )
        write(
            paper_dir / "evidence" / "README.md",
            "# Evidence\n\n`source/` records source acquisition; `runs/` records local engineering validation; `tables/` and `figures/` contain derived, non-copyright evidence only.",
        )
        write(
            paper_dir / "evidence" / "source" / "source_overview.md",
            f"""# Source overview

- Landing page: {paper['landing_url']}
- PDF source: {paper.get('pdf_url', 'not registered')}
- Local status: `{download.get('status')}`
- Local path: `{download.get('path', 'none')}`
- SHA-256: `{download.get('sha256', 'unavailable')}`
- Pages: `{download.get('pages', 'unavailable')}`
- Official artifact: {paper.get('official_artifact', {}).get('url', 'not registered')}
""",
        )
        write(
            paper_dir / "evidence" / "runs" / "local_validation.md",
            "# Local validation\n\nStatus: **not run**.\n\nRequired fields: command, environment lock, dataset version/hash, split, random seeds, expected paper result, observed result, tolerance, and decision.",
        )
        write(paper_dir / "evidence" / "tables" / "README.md", "# Tables\n\nNo derived tables yet.")
        write(paper_dir / "evidence" / "figures" / "README.md", "# Figures\n\nNo derived figures yet.")
        write(
            paper_dir / "src" / "environment.md",
            "# Environment\n\nStatus: not locked. Record OS, Python version, dependency lock, hardware, and official artifact version before validation.",
        )
        write(paper_dir / "src" / "code" / "README.md", "# Code\n\nNo vendored code. Official artifacts remain source-linked and must be license-audited before import.")
        write(paper_dir / "src" / "configs" / "README.md", "# Configs\n\nNo frozen reproduction config yet.")
        trace = {
            "paper_id": paper["id"],
            "nodes": [
                {"id": "source", "kind": "evidence", "status": download.get("status")},
                {"id": "method", "kind": "extraction", "status": "registered"},
                {"id": "implementation", "kind": "code", "status": "pending"},
                {"id": "validation", "kind": "run", "status": "pending"},
            ],
            "edges": [
                {"from": "source", "to": "method", "support": "explicit_or_book_crosscheck"},
                {"from": "method", "to": "implementation", "support": "requires_equation_level_trace"},
                {"from": "implementation", "to": "validation", "support": "requires_frozen_run"},
            ],
        }
        write(
            paper_dir / "trace" / "exploration_tree.json",
            json.dumps(trace, indent=2, ensure_ascii=False),
        )
        collection_rows.append(
            {
                "id": paper["id"],
                "year": paper["year"],
                "doi": paper["doi"],
                "role": paper["role"],
                "source_status": download.get("status"),
                "ara_status": status,
            }
        )

    manifest = args.output / "collection_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collection_rows[0].keys())
        writer.writeheader()
        writer.writerows(collection_rows)
    write(
        args.output / "README.md",
        "# Literature ARA collection\n\nEach directory separates source evidence, method claims, code status, and local validation. `metadata_only` and `source_acquired_code_pending` are not reproductions.",
    )
    print(f"built {len(collection_rows)} ARA packages under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
