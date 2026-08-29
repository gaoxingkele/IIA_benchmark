from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_paper_harness_matrix_is_closed() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/paper_harness.py", "status"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    audit = json.loads(completed.stdout)
    assert audit["issues"] == []
    assert audit["algorithms"] == {
        "registered": 30,
        "book": 20,
        "sota": 10,
        "matrix_covered": 30,
        "current_E2_or_higher": 10,
        "three_or_more_valid_dataset_targets": 30,
    }
    assert audit["algorithm_dataset_targets"] == {
        "all": 121,
        "M2_M3": 111,
        "M1_sentinels": 10,
        "adapter_runnable_all": 118,
        "adapter_runnable_M2_M3": 108,
        "adapter_pending": 3,
    }
    assert audit["references"]["registered_papers"] == 28
    assert audit["references"]["backlog_covered"] == 28
