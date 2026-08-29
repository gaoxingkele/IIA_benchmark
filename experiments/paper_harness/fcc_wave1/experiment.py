#!/usr/bin/env python3
"""Execute one pre-registered FCC Wave-1 run selected by its output directory."""

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
    "run_1": "configs/experiments/fcc_ctfh_state_validation.json",
    "run_2": "configs/experiments/fcc_ctfh_rising_edge_validation.json",
    "run_3": "configs/experiments/fcc_hdam_state_validation.json",
    "run_4": "configs/experiments/fcc_casim_state_validation.json",
    "run_5": "configs/experiments/fcc_cone_ctfh_uncertainty_validation.json",
    "run_6": "configs/experiments/fcc_cross_conformal_ctfh_uncertainty_validation.json",
    "run_7": "configs/experiments/fcc_accelerated_alignment_validation.json",
    "run_8": "configs/experiments/fcc_charm_patterns_validation.json",
    "run_9": "configs/experiments/fcc_max_entropy_next_alarm_validation.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_provenance() -> dict[str, object]:
    """Capture the exact mutable worktree state used by future runs."""
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
    ).stdout.encode("utf-8")
    source_paths = [
        ROOT / "src/iia_benchmark/data/fcc.py",
        ROOT / "src/iia_benchmark/models/flood_book.py",
        ROOT / "src/iia_benchmark/runner.py",
    ]
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in source_paths
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
    print(
        f"{run_name} tests whether {config.stem} activates and discriminates all 16 "
        "FCC abnormal situations under complete-run train/calibration/test separation."
    )
    provenance = execution_provenance()
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
    final = {
        "schema_version": 1,
        "run_name": run_name,
        "config": RUN_PLAN[run_name],
        "config_sha256": sha256_file(config),
        "git_revision": revision,
        "execution_provenance": provenance,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": duration,
        "result": payload,
    }
    (out_dir / "final_info.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "run_name": run_name,
        "wall_clock_seconds": duration,
        "experiment_id": payload["experiment_id"],
        "metrics": payload["result"]["metrics"],
        "activation": payload["result"]["activation"],
        "final_info": (out_dir / "final_info.json").relative_to(ROOT).as_posix(),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
