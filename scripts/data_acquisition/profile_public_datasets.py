"""Create lightweight structural profiles without copying dataset contents."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public_datasets"


def main() -> int:
    profile: dict[str, object] = {}
    piade_raw = DATA / "piade" / "raw_data.csv"
    if piade_raw.exists():
        frame = pd.read_csv(piade_raw, usecols=["equipment_ID", "alarm", "type"])
        alarms = frame["alarm"].dropna().astype(str)
        profile["piade_raw"] = {
            "rows": len(frame),
            "equipment": int(frame["equipment_ID"].nunique()),
            "alarm_codes_including_sentinel": int(alarms.nunique()),
            "non_sentinel_alarm_rows": int((alarms != "A_000").sum()),
            "machine_states": sorted(frame["type"].dropna().astype(str).unique().tolist()),
        }
    piade_sequences = DATA / "piade" / "sequences_1h_data.csv"
    if piade_sequences.exists():
        frame = pd.read_csv(piade_sequences)
        profile["piade_sequences"] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "equipment": int(frame["equipment_ID"].nunique()),
        }
    skab_data = DATA / "skab" / "data"
    if skab_data.exists():
        files = sorted(skab_data.rglob("*.csv"))
        profile["skab"] = {"csv_experiments": len(files)}
    tep_data = DATA / "tep_classic" / "data"
    if tep_data.exists():
        files = sorted(tep_data.glob("d*.dat"))
        profile["tep_classic"] = {
            "run_files": len(files),
            "normal_files": sum(path.stem.startswith("d00") for path in files),
            "fault_files": sum(not path.stem.startswith("d00") for path in files),
            "features": 52,
        }
    output = DATA / "profile.json"
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
