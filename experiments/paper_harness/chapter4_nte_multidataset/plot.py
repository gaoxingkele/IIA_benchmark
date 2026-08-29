#!/usr/bin/env python3
"""Plot Chapter-4.1 NTE/NDTE multi-dataset structural results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DISPLAY = {"run_1": "FCC", "run_2": "TEP Alarm", "run_3": "NPP"}


rows = []
for run_name, label in DISPLAY.items():
    path = ROOT / run_name / "final_info.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text(encoding="utf-8"))["result"]["result"]
    rows.append((label, result["metrics"], result["activation"]))

if rows:
    labels = [row[0] for row in rows]
    activation = [row[1]["graph_activation_rate"] for row in rows]
    direct = [row[1]["direct_edge_fraction"] for row in rows]
    pruned = [row[1]["indirect_pruning_fraction"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - width, activation, width, label="Run activation")
    axis.bar(x, direct, width, label="Direct-edge fraction")
    axis.bar(x + width, pruned, width, label="NDTE pruning fraction")
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Fraction")
    axis.set_title("Book Chapter 4.1 NTE/NDTE structural transfer")
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_1.png", dpi=180)
    plt.close(fig)

    within = [row[1]["within_class_direct_edge_jaccard"] for row in rows]
    cross = [row[1]["cross_class_direct_edge_jaccard"] for row in rows]
    edges = [row[1]["mean_significant_edges_per_run"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].bar(x - 0.18, within, 0.36, label="Within class")
    axes[0].bar(x + 0.18, cross, 0.36, label="Cross class")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Direct-edge Jaccard", ylim=(0, 0.55))
    axes[0].legend()
    axes[1].bar(x, edges, 0.55)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Edges per run")
    axes[1].set_title("Surrogate-significant NTE edges")
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_2.png", dpi=180)
    plt.close(fig)
