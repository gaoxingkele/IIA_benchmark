#!/usr/bin/env python3
"""Plot Chapter 4.2 IGTE/IGDTE multi-dataset results."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = {
    "TEP IDV(1)": (1, 2, 3),
    "PRONTO": (4, 5, 6),
    "SKAB": (7, 8, 9),
}


payloads = {}
for label, numbers in RUNS.items():
    rows = []
    for number in numbers:
        path = ROOT / f"run_{number}" / "final_info.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    if rows:
        payloads[label] = rows

if payloads:
    labels = list(payloads)
    activation = [
        np.mean([row["metrics"]["graph_activation_rate"] for row in payloads[label]])
        for label in labels
    ]
    edges = [
        np.mean([row["metrics"]["mean_significant_edges"] for row in payloads[label]])
        for label in labels
    ]
    pruned = [
        np.mean([row["metrics"]["mean_pruned_edges"] for row in payloads[label]])
        for label in labels
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    axes[0].bar(x, activation, 0.58)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Episode activation")
    axes[0].set_title("Surrogate-significant IGTE")
    axes[1].bar(x - 0.18, edges, 0.36, label="IGTE edges")
    axes[1].bar(x + 0.18, pruned, 0.36, label="IGDTE pruned")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Mean edges per episode")
    axes[1].legend()
    fig.suptitle("Book Chapter 4.2 three-dataset, three-seed validation")
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_1.png", dpi=180)
    plt.close(fig)

    cross_seed, within, cross = [], [], []
    for label in labels:
        rows = payloads[label]
        seed_scores = []
        for left, right in combinations(rows, 2):
            for left_episode, right_episode in zip(
                left["episodes"], right["episodes"], strict=True
            ):
                a = {
                    (edge["source"], edge["target"])
                    for edge in left_episode["graph"]["edges"]
                    if edge["direct"]
                }
                b = {
                    (edge["source"], edge["target"])
                    for edge in right_episode["graph"]["edges"]
                    if edge["direct"]
                }
                seed_scores.append(len(a & b) / len(a | b) if a | b else 1.0)
        cross_seed.append(float(np.mean(seed_scores)))
        within_values = [
            row["metrics"]["within_group_direct_edge_jaccard"]
            for row in rows
            if row["metrics"]["within_group_direct_edge_jaccard"] is not None
        ]
        cross_values = [
            row["metrics"]["cross_group_direct_edge_jaccard"]
            for row in rows
            if row["metrics"]["cross_group_direct_edge_jaccard"] is not None
        ]
        within.append(float(np.mean(within_values)) if within_values else np.nan)
        cross.append(float(np.mean(cross_values)) if cross_values else np.nan)
    fig, axis = plt.subplots(figsize=(8.7, 4.7))
    axis.bar(x - 0.24, cross_seed, 0.24, label="Same episode / cross seed")
    axis.bar(x, within, 0.24, label="Within group")
    axis.bar(x + 0.24, cross, 0.24, label="Cross group")
    axis.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Direct-edge Jaccard",
        ylim=(0, 1.05),
        title="IGTE graph reproducibility and group structure",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_2.png", dpi=180)
    plt.close(fig)
