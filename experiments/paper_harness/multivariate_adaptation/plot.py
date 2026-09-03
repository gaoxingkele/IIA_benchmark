#!/usr/bin/env python3
"""Aggregate and plot multivariate adaptation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = ROOT / "experiments/reports/multivariate_distribution_adaptation_validation.json"
RUNS = [
    json.loads((HERE / f"run_{number}" / "final_info.json").read_text(encoding="utf-8"))
    for number in (1, 2, 3)
]
DATASETS = ("tep_classic", "pronto", "skab")
VARIANTS = ("M0", "M1", "M2", "M3")
LABELS = {
    "M0": "Classical",
    "M1": "Robust+shrinkage",
    "M2": "Block calibrated",
    "M3": "Auto/selective",
}


def units(dataset: str, variant: str) -> list[dict[str, object]]:
    return [
        episode["variants"][variant]
        for run in RUNS
        for episode in run["result"]["datasets"][dataset]
    ]


aggregate: dict[str, dict[str, object]] = {}
for dataset in DATASETS:
    aggregate[dataset] = {}
    for variant in VARIANTS:
        rows = [row for row in units(dataset, variant) if row["status"] == "scored"]
        aggregate[dataset][variant] = {
            "scored_units": len(rows),
            "denied_units": 9 - len(rows),
            "coverage": len(rows) / 9,
            "metrics": {
                name: {
                    "mean": float(np.mean([row["empirical"][name] for row in rows])),
                    "standard_deviation": float(
                        np.std([row["empirical"][name] for row in rows], ddof=1)
                    )
                    if len(rows) > 1
                    else 0.0,
                }
                for name in ("false_alarm_rate", "missed_alarm_rate", "f1")
            }
            if rows
            else None,
        }
    audits = [
        episode["distribution_audit"]["normal_train_to_evaluation"]
        for run in RUNS
        for episode in run["result"]["datasets"][dataset]
    ]
    aggregate[dataset]["distribution_summary"] = {
        name: {
            "median": float(np.median([row[name] for row in audits])),
            "minimum": float(np.min([row[name] for row in audits])),
            "maximum": float(np.max([row[name] for row in audits])),
        }
        for name in (
            "per_feature_ks_median",
            "per_feature_ks_maximum",
            "standardized_median_shift_maximum",
            "covariance_relative_frobenius_shift",
            "maximum_absolute_correlation_shift",
            "candidate_effective_rank",
            "candidate_absolute_lag_one_median",
        )
    }


payload = {
    "schema_version": 1,
    "experiment": "multivariate_distribution_adaptation",
    "runs": [run["run_name"] for run in RUNS],
    "seeds": [run["result"]["seed"] for run in RUNS],
    "datasets": aggregate,
    "stage_acceptance": {
        "same_chapter3_protocol": all(
            episode["source_chapter3_protocol_match"]
            for run in RUNS
            for dataset in DATASETS
            for episode in run["result"]["datasets"][dataset]
        ),
        "all_uncertainty_present": all(
            row["block_bootstrap"] is not None
            for dataset in DATASETS
            for variant in VARIANTS
            for row in units(dataset, variant)
            if row["status"] == "scored"
        ),
        "selective_reporting": "M3 metrics include scored units only; coverage and denial counts are mandatory",
    },
    "reporting_boundary": RUNS[0]["result"]["reporting_boundary"],
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)


plt.style.use("seaborn-v0_8-whitegrid")
colors = ("#4C78A8", "#72B7B2", "#F2CF5B", "#E45756")
x = np.arange(len(DATASETS))
width = 0.19
fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
for index, variant in enumerate(VARIANTS):
    values = [aggregate[d][variant]["metrics"]["f1"]["mean"] if aggregate[d][variant]["metrics"] else np.nan for d in DATASETS]
    bars = ax.bar(x + (index - 1.5) * width, values, width, label=LABELS[variant], color=colors[index])
    if variant == "M3":
        for bar, dataset in zip(bars, DATASETS, strict=True):
            coverage = aggregate[dataset][variant]["coverage"]
            if coverage < 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{coverage:.0%} coverage",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
ax.set_xticks(x, ("TEP", "PRONTO", "SKAB"))
ax.set_ylabel("Held-out F1")
ax.set_ylim(0, 1)
ax.set_title("Multivariate adaptation: F1 and selective coverage")
ax.legend(ncols=2, frameon=False)
fig.savefig(HERE / "Figure_1.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1), constrained_layout=True)
for ax, dataset in zip(axes, DATASETS, strict=True):
    for index, variant in enumerate(VARIANTS):
        metric = aggregate[dataset][variant]["metrics"]
        if metric:
            ax.scatter(metric["false_alarm_rate"]["mean"], metric["missed_alarm_rate"]["mean"], s=75, color=colors[index], label=LABELS[variant])
    ax.set_title(dataset.replace("_classic", "").upper())
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("FAR")
    ax.set_ylabel("MAR")
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(HERE / "Figure_2.png", dpi=180)
plt.close(fig)

audit_names = (
    "per_feature_ks_median",
    "standardized_median_shift_maximum",
    "covariance_relative_frobenius_shift",
    "maximum_absolute_correlation_shift",
    "candidate_absolute_lag_one_median",
)
audit_labels = ("KS", "Median shift", "Covariance shift", "Correlation shift", "Lag-1")
matrix = np.asarray([[aggregate[d]["distribution_summary"][name]["median"] for name in audit_names] for d in DATASETS])
fig, ax = plt.subplots(figsize=(8.8, 4.3), constrained_layout=True)
image = ax.imshow(np.log1p(matrix), cmap="YlOrRd", aspect="auto")
ax.set_xticks(np.arange(len(audit_labels)), audit_labels, rotation=20, ha="right")
ax.set_yticks(np.arange(3), ("TEP", "PRONTO", "SKAB"))
for row in range(3):
    for column in range(len(audit_names)):
        ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Normal train-to-evaluation distribution shift (median)")
fig.colorbar(image, ax=ax, label="log(1 + value)")
fig.savefig(HERE / "Figure_3.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
delta_f1 = [aggregate[d]["M2"]["metrics"]["f1"]["mean"] - aggregate[d]["M0"]["metrics"]["f1"]["mean"] for d in DATASETS]
delta_far = [aggregate[d]["M2"]["metrics"]["false_alarm_rate"]["mean"] - aggregate[d]["M0"]["metrics"]["false_alarm_rate"]["mean"] for d in DATASETS]
for ax, values, title in zip(axes, (delta_f1, delta_far), ("Delta F1: M2 - M0", "Delta FAR: M2 - M0"), strict=True):
    ax.bar(("TEP", "PRONTO", "SKAB"), values, color=["#4C78A8" if value >= 0 else "#E45756" for value in values])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
fig.savefig(HERE / "Figure_4.png", dpi=180)
plt.close(fig)

print(json.dumps(payload["stage_acceptance"], ensure_ascii=False, indent=2))
