#!/usr/bin/env python3
"""Generate the two registered FCC Wave-1 comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
DISPLAY_NAMES = {
    "run_1": "CTFH state",
    "run_2": "CTFH edge",
    "run_3": "HDAM",
    "run_4": "CASIM",
    "run_5": "ConE",
    "run_6": "Cross-Conformal",
    "run_7": "Alignment",
    "run_8": "CHARM",
    "run_9": "MaxEnt",
}


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(PROJECT.glob("run_*/final_info.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]["result"]
        runs.append(
            {
                "name": path.parent.name,
                "model": result["model_id"],
                "representation": result["alarm_representation"],
                "metrics": result["metrics"],
                "prefix": result.get("prefix_metrics", {}),
                "activation": result["activation"],
            }
        )
    if not runs:
        raise RuntimeError("no completed run_*/final_info.json files")
    return runs


def main() -> int:
    runs = load_runs()
    point = [run for run in runs if "balanced_accuracy" in run["metrics"]]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(point))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [run["metrics"]["balanced_accuracy"] for run in point],
        width,
        label="Balanced accuracy",
    )
    axis.bar(
        positions + width / 2,
        [run["metrics"]["macro_f1"] for run in point],
        width,
        label="Macro-F1",
    )
    axis.axhline(1 / 16, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, [DISPLAY_NAMES.get(r["name"], r["name"]) for r in point])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Score")
    axis.set_title("FCC complete-run classification")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_1.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for run in runs:
        if not run["prefix"]:
            continue
        lengths = sorted(map(int, run["prefix"]))
        rows = [run["prefix"][str(length)] for length in lengths]
        if "balanced_accuracy" in rows[0]:
            values = [row["balanced_accuracy"] for row in rows]
            label = f"{run['model']} {run['representation']} BA"
        else:
            class_count = run["activation"]["class_count"]
            values = [1 - row["average_set_size"] / class_count for row in rows]
            label = f"{run['model']} efficiency"
        axis.plot(lengths, values, marker="o", label=label)
    axis.set_xlabel("Observed prefix (minutes)")
    axis.set_ylabel("Balanced accuracy or normalized set efficiency")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("FCC online-prefix discrimination")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_2.png", dpi=220)
    plt.close(fig)
    print(PROJECT / "Figure_1.png")
    print(PROJECT / "Figure_2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
