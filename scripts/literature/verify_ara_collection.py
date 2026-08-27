"""Verify literature ARA structure and locally acquired PDF evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "metadata.json",
    "PAPER.md",
    "logic/concepts.md",
    "logic/problem.md",
    "logic/claims.md",
    "logic/related_work.md",
    "logic/experiments.md",
    "logic/solution/method.md",
    "logic/solution/constraints.md",
    "evidence/README.md",
    "evidence/source/source_overview.md",
    "evidence/runs/local_validation.md",
    "src/environment.md",
    "src/code/README.md",
    "src/configs/README.md",
    "trace/exploration_tree.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "papers/literature/registry.json")
    parser.add_argument("--downloads", type=Path, default=ROOT / "papers/literature/download_manifest.json")
    parser.add_argument("--ara", type=Path, default=ROOT / "papers/literature/ara")
    parser.add_argument("--report", type=Path, default=ROOT / "papers/literature/ara/verification_report.json")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    downloads = json.loads(args.downloads.read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in downloads["records"]}
    checks: list[dict[str, object]] = []
    errors: list[str] = []
    for paper in registry["papers"]:
        paper_dir = args.ara / paper["id"]
        missing = [path for path in REQUIRED if not (paper_dir / path).is_file()]
        record = by_id.get(paper["id"])
        pdf_ok = None
        if record and record["status"] == "downloaded":
            pdf_path = ROOT / record["path"]
            pdf_ok = pdf_path.is_file() and sha256(pdf_path) == record["sha256"]
            if not pdf_ok:
                errors.append(f"{paper['id']}: PDF missing or checksum mismatch")
        if missing:
            errors.append(f"{paper['id']}: missing {', '.join(missing)}")
        checks.append(
            {
                "id": paper["id"],
                "structure_ok": not missing,
                "missing": missing,
                "download_status": record["status"] if record else "missing",
                "pdf_checksum_ok": pdf_ok,
            }
        )

    report = {
        "registered": len(registry["papers"]),
        "checks": checks,
        "errors": errors,
        "passed": not errors,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered": report["registered"], "errors": len(errors), "passed": report["passed"]}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
