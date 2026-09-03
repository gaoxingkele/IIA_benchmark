#!/usr/bin/env python3
"""Run the configuration-driven univariate adaptation onboarding pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.adaptation import run_univariate_transfer  # noqa: E402
from iia_benchmark.data import load_univariate_transfer_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config, bundle = load_univariate_transfer_config(config_path, root=ROOT)
    seed = int(config["seed"])
    print(
        "This run tests whether a newly registered univariate dataset can be "
        "scored without held-out tuning, or must be explicitly denied."
    )
    result = run_univariate_transfer(bundle, config["adaptation"], seed=seed)
    result["input_files"] = [
        {
            "path": Path(source["path"]).relative_to(ROOT).as_posix(),
            "bytes": Path(source["path"]).stat().st_size,
            "sha256": sha256_file(Path(source["path"])),
        }
        for source in {
            item["path"]: item for item in bundle.partition_sources.values()
        }.values()
    ]
    payload = {
        "schema_version": 1,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "reporting_boundary": config["reporting_boundary"],
    }
    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
