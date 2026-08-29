#!/usr/bin/env python3
"""Execute one registered NPP Alarm Wave-1 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.runner import run_experiment  # noqa: E402


RUN_PLAN = {
    "run_1": "configs/experiments/npp_alarm_ctfh_state_validation.json",
    "run_2": "configs/experiments/npp_alarm_ctfh_rising_edge_validation.json",
    "run_3": "configs/experiments/npp_alarm_hdam_state_validation.json",
    "run_4": "configs/experiments/npp_alarm_casim_state_validation.json",
    "run_5": "configs/experiments/npp_alarm_cone_ctfh_validation.json",
    "run_6": "configs/experiments/npp_alarm_cross_conformal_ctfh_validation.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    sources = [
        ROOT / "src/iia_benchmark/data/npp_alarm.py",
        ROOT / "src/iia_benchmark/models/flood_book.py",
        ROOT / "src/iia_benchmark/runner.py",
    ]
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    run_name = out_dir.name
    if run_name not in RUN_PLAN:
        raise ValueError(f"out_dir basename must be one of {sorted(RUN_PLAN)}")
    config = ROOT / RUN_PLAN[run_name]
    started = datetime.now(timezone.utc)
    execution_provenance = provenance()
    before = time.perf_counter()
    payload = run_experiment(config)
    duration = time.perf_counter() - before
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_info.json"
    final_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_name": run_name,
                "config": RUN_PLAN[run_name],
                "config_sha256": sha256_file(config),
                "git_revision": revision,
                "execution_provenance": execution_provenance,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_clock_seconds": duration,
                "result": payload,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = payload["result"]
    print(
        json.dumps(
            {
                "run_name": run_name,
                "wall_clock_seconds": duration,
                "experiment_id": payload["experiment_id"],
                "metrics": result["metrics"],
                "activation": result["activation"],
                "final_info": final_path.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
