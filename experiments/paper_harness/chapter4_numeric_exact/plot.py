#!/usr/bin/env python3
"""Plot the Chapter 4.3/4.4 numerical validation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


plr = []
for run_name in ("run_1", "run_2", "run_3"):
    path = ROOT / run_name / "final_info.json"
    if path.exists():
        plr.append(json.loads(path.read_text(encoding="utf-8"))["result"])
if plr:
    factors = np.asarray(
        [[row["factors"] for row in payload["segments"]] for payload in plr]
    )
    mean, low, high = (
        np.mean(factors, axis=0),
        np.min(factors, axis=0),
        np.max(factors, axis=0),
    )
    x = np.arange(1, 33)
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for index, label in enumerate(("x1", "x2")):
        axis.plot(x, mean[:, index], label=f"{label} mean contribution")
        axis.fill_between(x, low[:, index], high[:, index], alpha=0.18)
    axis.axvline(16.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Published PLR segment",
        ylabel="Contribution factor",
        ylim=(0, 1.02),
        title="Book Chapter 4.4 three-seed numerical reproduction",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_1.png", dpi=180)
    plt.close(fig)

bayes_path = ROOT / "run_4" / "final_info.json"
if bayes_path.exists():
    metrics = json.loads(bayes_path.read_text(encoding="utf-8"))["result"]["metrics"]
    labels = ["Overall", "Nuisance samples"]
    recursive = [metrics["overall_online_accuracy"], metrics["nuisance_sample_accuracy"]]
    naive = [metrics["naive_overall_accuracy"], metrics["naive_nuisance_sample_accuracy"]]
    x = np.arange(2)
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.bar(x - 0.18, recursive, 0.36, label="Recursive Bayesian")
    axis.bar(x + 0.18, naive, 0.36, label="Instantaneous lookup")
    axis.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Accuracy against controlled event truth",
        ylim=(0, 1.05),
        title="Book Chapter 4.3 nuisance-alarm stress recreation",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_2.png", dpi=180)
    plt.close(fig)
