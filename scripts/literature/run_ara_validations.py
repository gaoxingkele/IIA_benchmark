"""Execute every registered ARA local-validation command and record fresh evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "papers/literature/registry.json"
REPORT = ROOT / "experiments/reports/ara_algorithm_validation.json"


def command_argv(command: str) -> list[str]:
    argv = command.split()
    if argv and argv[0].lower() in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    papers = json.loads(args.registry.read_text(encoding="utf-8"))["papers"]
    command_results: dict[str, dict[str, object]] = {}
    for command in dict.fromkeys(
        paper.get("local_validation", {}).get("command") for paper in papers
    ):
        if not command:
            continue
        started = time.perf_counter()
        process = subprocess.run(
            command_argv(command),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (process.stdout + process.stderr).strip()
        command_results[command] = {
            "exit_code": process.returncode,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "output_tail": output.splitlines()[-8:],
        }
    records = []
    for paper in papers:
        validation = paper.get("local_validation", {})
        command = validation.get("command")
        execution = command_results.get(command) if command else None
        records.append(
            {
                "id": paper["id"],
                "doi": paper["doi"],
                "command": command,
                "scope": validation.get("scope"),
                "status": (
                    "passed"
                    if execution and execution["exit_code"] == 0
                    else "missing_command"
                    if not command
                    else "failed"
                ),
                "execution": execution,
            }
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": args.registry.relative_to(ROOT).as_posix(),
        "summary": {
            "registered_papers": len(records),
            "unique_commands": len(command_results),
            "passed": sum(record["status"] == "passed" for record in records),
            "failed": sum(record["status"] == "failed" for record in records),
            "missing_command": sum(
                record["status"] == "missing_command" for record in records
            ),
        },
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if not report["summary"]["failed"] and not report["summary"]["missing_command"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
