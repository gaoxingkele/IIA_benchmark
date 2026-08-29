#!/usr/bin/env python3
"""Plot compact TEP Alarm Wave-1 discrimination and prefix-set summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DISPLAY = {
    "run_1": "CTFH state",
    "run_2": "CTFH edge",
    "run_3": "HDAM",
    "run_4": "CASIM",
}


def load(run_name: str) -> dict[str, object] | None:
    path = ROOT / run_name / "final_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


point_rows = []
for run_name, label in DISPLAY.items():
    payload = load(run_name)
    if payload is not None:
        metrics = payload["result"]["result"]["metrics"]
        point_rows.append((label, metrics["balanced_accuracy"], metrics["macro_f1"]))
if point_rows:
    labels, balanced, macro_f1 = zip(*point_rows)
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x - width / 2, balanced, width, label="Balanced accuracy")
    axis.bar(x + width / 2, macro_f1, width, label="Macro-F1")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("TEP Alarm five-class engineering validation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_1.png", dpi=180)
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
available = False
for run_name, label in (("run_5", "ConE+CTFH"), ("run_6", "Cross+CTFH")):
    payload = load(run_name)
    if payload is None:
        continue
    available = True
    prefix = payload["result"]["result"]["prefix_metrics"]
    lengths = np.asarray([int(value) for value in prefix])
    coverage = np.asarray([prefix[str(value)]["coverage"] for value in lengths])
    set_size = np.asarray([prefix[str(value)]["average_set_size"] for value in lengths])
    axes[0].plot(lengths, coverage, marker="o", label=label)
    axes[1].plot(lengths, set_size, marker="o", label=label)
axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1, label="target 0.90")
axes[0].set(title="Coverage by prefix", xlabel="Prefix (minutes)", ylabel="Coverage", ylim=(0, 1.05))
axes[1].axhline(5, color="black", linestyle=":", linewidth=1, label="full label set")
axes[1].set(title="Prediction-set size", xlabel="Prefix (minutes)", ylabel="Mean labels", ylim=(0, 5.25))
if available:
    for axis in axes:
        axis.legend()
fig.tight_layout()
fig.savefig(ROOT / "Figure_2.png", dpi=180)
plt.close(fig)
