"""Generate a strict machine-readable/current benchmark coverage audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    book = read("configs/algorithms/book_algorithms.json")["algorithms"]
    sota = read("configs/algorithms/sota_algorithms.json")["algorithms"]
    tasks = read("configs/tasks/downstream_tasks.json")["tasks"]
    sources = read("configs/datasets/public_sources.json")["sources"]
    data_audit = read("data/public_datasets/audit.json")
    data_by_id = {item["id"]: item for item in data_audit}
    paper_downloads = read("papers/literature/download_manifest.json")
    paper_access = read("papers/literature/access_audit.json")
    families = sorted({item["dataset_family"] for item in sources})
    usable_families = sorted(
        {
            item["dataset_family"]
            for item in sources
            if item["payload_role"] == "main"
            and data_by_id.get(item["id"], {}).get("valid", False)
        }
    )
    matrix_lines = (ROOT / "docs/algorithm_matrix.md").read_text(encoding="utf-8").splitlines()
    callable_families = sum(
        line.startswith("| ")
        and not line.startswith("| Family")
        and not line.startswith("|---")
        for line in matrix_lines
    )
    split_configs = [read(str(path.relative_to(ROOT)).replace("\\", "/")) for path in (ROOT / "configs/splits").glob("*.json")]
    report_payloads = [
        read(str(path.relative_to(ROOT)).replace("\\", "/"))
        for path in sorted((ROOT / "experiments/reports").glob("*.json"))
    ]
    validation_reports = [payload for payload in report_payloads if payload.get("config")]
    validation_configs = [
        read(payload["config"])
        for payload in validation_reports
    ]
    validated_tasks = sorted(
        {
            task
            for config in validation_configs
            for task in config.get("downstream_tasks", [])
        }
    )
    report = {
        "schema_version": 1,
        "cutoff_date": "2026-08-28",
        "algorithms": {
            "callable_method_families": callable_families,
            "model_configs": len(list((ROOT / "configs/models").glob("*.json"))),
            "book_deliverables": len(book),
            "book_callable": sum(bool(item.get("implementation")) for item in book),
            "book_by_status": {
                status: sum(item["status"] == status for item in book)
                for status in ("verified", "partial", "missing")
            },
            "selected_sota_deliverables": len(sota),
            "sota_callable": sum(bool(item.get("local_implementation")) for item in sota),
            "sota_by_status": {
                status: sum(item["status"] == status for item in sota)
                for status in ("verified", "partial", "missing")
            },
            "strict_score_closed": sum(item["status"] == "verified" for item in book + sota),
        },
        "datasets": {
            "registry_records": len(sources),
            "logical_public_families": len(families),
            "logical_public_family_ids": families,
            "main_payload_available_families": len(usable_families),
            "main_payload_available_ids": usable_families,
            "synthetic_smoke_datasets": len(list((ROOT / "configs/datasets").glob("synthetic_*.json"))),
        },
        "tasks": {
            "defined": len(tasks),
            "runnable_real_data": sum(
                item["status"].startswith("runnable_real_data") for item in tasks
            ),
            "protocol_ready_data_gated": sum(
                "gated" in item["status"]
                or "gated" in item.get("primary_payload_status", "")
                for item in tasks
            ),
            "task_status": {item["id"]: item["status"] for item in tasks},
        },
        "evidence": {
            "registered_papers": paper_downloads["summary"]["registered"],
            "downloaded_papers": paper_downloads["summary"]["downloaded"],
            "unpaywall_open_access_records": paper_access["summary"]["unpaywall_oa"],
            "unpaywall_direct_pdf_candidates": paper_access["summary"]["oa_pdf_candidates"],
            "leaderboard_eligible_splits": sum(bool(item.get("leaderboard_eligible")) for item in split_configs),
            "real_data_validation_reports": sum(
                str(config.get("task", "")).startswith("real_")
                for config in validation_configs
            ),
            "real_data_validated_tasks": validated_tasks,
            "ara_algorithm_validations_passed": next(
                (
                    payload["summary"]["passed"]
                    for payload in report_payloads
                    if payload.get("registry") == "papers/literature/registry.json"
                ),
                0,
            ),
        },
    }
    output = ROOT / "docs/status_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = f"""# Benchmark coverage audit

Cutoff: `{report['cutoff_date']}`. This page is generated by
`python scripts/audit_benchmark_coverage.py`; edit the JSON registries rather than
this report.

## Algorithms

| Scope | Registered | Callable | Verified | Partial | Missing |
|---|---:|---:|---:|---:|---:|
| Book deliverables | {report['algorithms']['book_deliverables']} | {report['algorithms']['book_callable']} | {report['algorithms']['book_by_status']['verified']} | {report['algorithms']['book_by_status']['partial']} | {report['algorithms']['book_by_status']['missing']} |
| Selected SOTA | {report['algorithms']['selected_sota_deliverables']} | {report['algorithms']['sota_callable']} | {report['algorithms']['sota_by_status']['verified']} | {report['algorithms']['sota_by_status']['partial']} | {report['algorithms']['sota_by_status']['missing']} |

- Callable method families in the algorithm matrix: **{report['algorithms']['callable_method_families']}**.
- Strict score-closed reproductions: **{report['algorithms']['strict_score_closed']}**.
- A callable implementation is not considered verified until the cited split,
  metric, seed protocol, and reference score have all been reproduced.

## Data and tasks

| Item | Count |
|---|---:|
| Public registry records | {report['datasets']['registry_records']} |
| Logical public dataset families | {report['datasets']['logical_public_families']} |
| Main payloads locally available | {report['datasets']['main_payload_available_families']} |
| Synthetic smoke datasets | {report['datasets']['synthetic_smoke_datasets']} |
| Downstream tasks defined | {report['tasks']['defined']} |
| Tasks runnable on available real data | {report['tasks']['runnable_real_data']} |
| Tasks with the primary payload gated | {report['tasks']['protocol_ready_data_gated']} |
| Leaderboard-eligible splits | {report['evidence']['leaderboard_eligible_splits']} |
| Real-data validation reports | {report['evidence']['real_data_validation_reports']} |

Locally available public main payloads: {', '.join(f'`{item}`' for item in report['datasets']['main_payload_available_ids'])}.

## Evidence

| Item | Count |
|---|---:|
| Papers registered | {report['evidence']['registered_papers']} |
| Paper PDFs downloaded | {report['evidence']['downloaded_papers']} |
| DOI records marked open by Unpaywall | {report['evidence']['unpaywall_open_access_records']} |
| Direct PDF candidates reported by Unpaywall | {report['evidence']['unpaywall_direct_pdf_candidates']} |
| Tasks with executed real-data validation | {', '.join(report['evidence']['real_data_validated_tasks']) or 'none'} |
| ARA paper packages with fresh local validation | {report['evidence']['ara_algorithm_validations_passed']} |

The detailed machine-readable report is in `docs/status_audit.json`.
"""
    (ROOT / "docs/status_audit.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
