#!/usr/bin/env python3
"""Plot Book Chapter 2 exact and multi-dataset validation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = [
    json.loads((ROOT / f"run_{index}" / "final_info.json").read_text(encoding="utf-8"))["result"]
    for index in (1, 2, 3)
]
DATASETS = ("tep_classic", "pronto", "skab")
DATASET_LABELS = ("TEP", "PRONTO", "SKAB")
ALGORITHMS = (
    "book_2_1_iid_delay_timer",
    "book_2_2_non_iid_delay_timer",
    "book_2_3_non_iid_deadband",
    "book_2_4_alarm_probability_plot",
)
LABELS = ("IID delay", "non-IID delay", "Deadband", "APP")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)
for axis, dataset, dataset_label in zip(axes, DATASETS, DATASET_LABELS, strict=True):
    means, errors = [], []
    for algorithm in ALGORITHMS:
        values = [
            run["aggregate_metrics"][dataset][algorithm]["mean_f1"] for run in RUNS
        ]
        means.append(float(np.mean(values)))
        errors.append(float(np.std(values)))
    x = np.arange(len(ALGORITHMS))
    axis.bar(x, means, yerr=errors, color=COLORS, capsize=3)
    axis.set(xticks=x, xticklabels=LABELS, ylim=(0, 1.0), title=dataset_label)
    axis.tick_params(axis="x", rotation=25)
axes[0].set_ylabel("Held-out F1")
fig.suptitle("Book Chapter 2: three datasets and three bootstrap seeds")
fig.tight_layout()
fig.savefig(ROOT / "Figure_1.png", dpi=180)
plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharex=True, sharey=True)
for axis, dataset, dataset_label in zip(axes, DATASETS, DATASET_LABELS, strict=True):
    for algorithm, label, color in zip(ALGORITHMS, LABELS, COLORS, strict=True):
        far = np.mean(
            [
                run["aggregate_metrics"][dataset][algorithm]["mean_false_alarm_rate"]
                for run in RUNS
            ]
        )
        mar = np.mean(
            [
                run["aggregate_metrics"][dataset][algorithm]["mean_missed_alarm_rate"]
                for run in RUNS
            ]
        )
        axis.scatter(far, mar, s=70, color=color, label=label)
    axis.axvline(0.05, color="black", linestyle="--", linewidth=0.8)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=0.8)
    axis.set(title=dataset_label, xlim=(-0.02, 1.0), ylim=(-0.02, 1.0))
axes[0].set_ylabel("Missed alarm rate")
axes[1].set_xlabel("False alarm rate")
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("FAR/MAR transfer relative to the 0.05 design targets")
fig.tight_layout()
fig.savefig(ROOT / "Figure_2.png", dpi=180)
plt.close(fig)


fig, axis = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(3)
table_error = [
    max(row["maximum_absolute_error"] for row in run["exact_reproduction"]["table_vii"])
    for run in RUNS
]
example_error = [
    max(
        value
        for row in run["exact_reproduction"]["examples"].values()
        for value in row["local_absolute_theory_error"].values()
    )
    for run in RUNS
]
axis.bar(x - 0.18, table_error, 0.36, label="Table VII exact equations")
axis.bar(x + 0.18, example_error, 0.36, label="Examples 1-2 Monte Carlo")
axis.set(
    xticks=x,
    xticklabels=["1103", "2207", "3301"],
    yscale="log",
    ylabel="Maximum absolute error (log scale)",
    xlabel="Seed",
    title="Xu 2012 named-item reproduction",
)
axis.axhline(1e-4, color="black", linestyle="--", linewidth=0.8, label="Table tolerance")
axis.legend()
fig.tight_layout()
fig.savefig(ROOT / "Figure_3.png", dpi=180)
plt.close(fig)
