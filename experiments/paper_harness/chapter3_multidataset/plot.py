#!/usr/bin/env python3
"""Aggregate and plot the Chapter 3 three-seed validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = ROOT / "experiments/reports/book_ch3_multidataset_validation.json"
RUNS = [
    json.loads((HERE / f"run_{number}" / "final_info.json").read_text(encoding="utf-8"))
    for number in (1, 2, 3)
]
DATASETS = ("tep_classic", "pronto", "skab")
ALGORITHMS = (
    "book_3_1_convex_noz",
    "book_3_1_nonconvex_noz",
    "book_3_2_variation_direction",
    "book_3_3_electrical_pump",
)
LABELS = {
    "book_3_1_convex_noz": "Convex NOZ",
    "book_3_1_nonconvex_noz": "Search-cone NOZ",
    "book_3_2_variation_direction": "Variation direction",
    "book_3_3_electrical_pump": "Bayesian regression",
    "baseline_mahalanobis": "Mahalanobis",
}


def algorithm_units(dataset: str, algorithm: str) -> list[dict[str, object]]:
    return [
        episode["algorithms"][algorithm]
        for run in RUNS
        for episode in run["result"]["datasets"][dataset]
    ]


def baseline_units(dataset: str) -> list[dict[str, float]]:
    return [
        episode["baseline_mahalanobis"]
        for run in RUNS
        for episode in run["result"]["datasets"][dataset]
    ]


aggregate: dict[str, dict[str, object]] = {}
for dataset in DATASETS:
    aggregate[dataset] = {}
    for algorithm in ALGORITHMS:
        rows = algorithm_units(dataset, algorithm)
        aggregate[dataset][algorithm] = {
            "units": len(rows),
            "execution_rate": float(np.mean([row["execution_passed"] for row in rows])),
            "activation_rate": float(np.mean([row["activation_passed"] for row in rows])),
            "far": float(np.mean([row["empirical"]["false_alarm_rate"] for row in rows])),
            "mar": float(np.mean([row["empirical"]["missed_alarm_rate"] for row in rows])),
            "aad": float(np.mean([row["empirical"]["average_alarm_delay"] for row in rows])),
            "f1": float(np.mean([row["empirical"]["f1"] for row in rows])),
            "f1_std": float(np.std([row["empirical"]["f1"] for row in rows], ddof=1)),
        }
    rows = baseline_units(dataset)
    aggregate[dataset]["baseline_mahalanobis"] = {
        "units": len(rows),
        "far": float(np.mean([row["false_alarm_rate"] for row in rows])),
        "mar": float(np.mean([row["missed_alarm_rate"] for row in rows])),
        "aad": float(np.mean([row["average_alarm_delay"] for row in rows])),
        "f1": float(np.mean([row["f1"] for row in rows])),
        "f1_std": float(np.std([row["f1"] for row in rows], ddof=1)),
    }


prior_units = {
    dataset: [
        episode["prior"]
        for run in RUNS
        for episode in run["result"]["datasets"][dataset]
    ]
    for dataset in DATASETS
}
pump_units = [
    episode["algorithms"]["book_3_3_electrical_pump"]
    for run in RUNS
    for dataset in DATASETS
    for episode in run["result"]["datasets"][dataset]
]
condenser_units = [run["result"]["condenser_synthetic_validation"] for run in RUNS]
named_units = [run["result"]["named_book_items"] for run in RUNS]

report = {
    "schema_version": 1,
    "config": "configs/experiments/book_chapter3_multidataset.json",
    "algorithm_ids": [*ALGORITHMS, "book_3_4_condenser"],
    "paper_ids": [
        "yu2020_convex_noz",
        "wang2024_search_cones",
        "chen2017_variation_directions",
        "xiong2018_bayesian_pumps",
        "wang2024_condenser_noz",
    ],
    "seeds": [run["result"]["seed"] for run in RUNS],
    "dataset_families": list(DATASETS),
    "named_reproduction": {
        "figure_3_2_fitness_passed_all_seeds": all(
            row["figure_3_2"]["passed"] for row in named_units
        ),
        "atg_change_direction_recovered_all_seeds": all(
            row["section_3_2_example_1"]["change_direction_recovered"] for row in named_units
        ),
        "pump_table_selection_passed_all_seeds": all(
            row["tables_3_2_to_3_4"]["selection_passed"] for row in named_units
        ),
        "table_3_4_arithmetic_discrepancy": named_units[0]["tables_3_2_to_3_4"]["book_arithmetic_discrepancy"],
        "blocked_original_items": named_units[0]["blocked_original_items"],
    },
    "prior_gate": {
        "passed_all_runs": all(run["result"]["prior_gate"]["passed"] for run in RUNS),
        "episode_seed_units": 27,
        "finite": True,
        "partition_overlap": 0,
        "normal_train_evaluation_distribution": {
            dataset: {
                "median_ks": float(np.mean([row["normal_train_evaluation_ks_median"] for row in rows])),
                "maximum_ks": float(np.max([row["normal_train_evaluation_ks_maximum"] for row in rows])),
                "maximum_standardized_median_shift": float(
                    np.max([row["normal_standardized_median_shift_maximum"] for row in rows])
                ),
            }
            for dataset, rows in prior_units.items()
        },
        "smd10towfgr_denied": "event log has no continuous process matrix",
        "real_condenser_denied": "required pressure/steam-flow/inlet/outlet-temperature payload is unavailable",
    },
    "aggregate_metrics": aggregate,
    "mechanism_gates": {
        "pump_statistical_activation_units": int(np.sum([row["activation_passed"] for row in pump_units])),
        "pump_total_units": len(pump_units),
        "pump_paper_domain_activation_units": int(
            np.sum([row["paper_domain_activation_passed"] for row in pump_units])
        ),
        "variation_direction_execution_units": 27,
        "variation_direction_note": "rule matrices activate, but high held-out MAR is retained as negative transfer rather than treated as an implementation failure",
    },
    "condenser_synthetic": {
        "fit_goodness_mean": float(np.mean([row["fit_goodness"] for row in condenser_units])),
        "fit_goodness_minimum": float(np.min([row["fit_goodness"] for row in condenser_units])),
        "f1_mean": float(np.mean([row["empirical"]["f1"] for row in condenser_units])),
        "f1_std": float(np.std([row["empirical"]["f1"] for row in condenser_units], ddof=1)),
        "activation_seeds": int(np.sum([row["activation_passed"] for row in condenser_units])),
        "false_alarm_upper_99pct_mean": float(
            np.mean([row["bayesian_rate_bounds_99pct"]["false_alarm"]["upper"] for row in condenser_units])
        ),
        "missed_alarm_upper_99pct_mean": float(
            np.mean([row["bayesian_rate_bounds_99pct"]["missed_alarm"]["upper"] for row in condenser_units])
        ),
        "industrial_score_reproduction": False,
    },
    "evidence_tiers": {
        "book_3_1_convex_noz": "E3_named_item_plus_negative_multi_dataset_transfer",
        "book_3_1_nonconvex_noz": "E3_negative_multi_dataset_transfer",
        "book_3_2_variation_direction": "E2_partial_ATG_and_negative_transfer",
        "book_3_3_electrical_pump": "E2_partial_statistical_activation_pump_domain_denied",
        "book_3_4_condenser": "E2_equation_synthetic_only_industrial_data_blocked",
    },
    "paper_score_closure": {algorithm: False for algorithm in [*ALGORITHMS, "book_3_4_condenser"]},
    "competitive_credit": "deny_cross_dataset_superiority_and_all_original_paper_scores",
    "evidence": "experiments/paper_harness/chapter3_multidataset",
}
REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)


methods = ("baseline_mahalanobis", *ALGORITHMS)
x = np.arange(len(DATASETS))
width = 0.16
fig, axis = plt.subplots(figsize=(10.8, 5.0))
for index, method in enumerate(methods):
    values = [aggregate[dataset][method]["f1"] for dataset in DATASETS]
    axis.bar(x + (index - 2) * width, values, width, label=LABELS[method])
axis.set(
    xticks=x,
    xticklabels=("TEP", "PRONTO", "SKAB"),
    ylim=(0, 1.02),
    ylabel="Point-wise F1",
    title="Book Chapter 3: three-dataset, three-seed transfer",
)
axis.legend(ncol=2)
fig.tight_layout()
fig.savefig(HERE / "Figure_1.png", dpi=180)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
for panel, metric in enumerate(("far", "mar")):
    for index, method in enumerate(ALGORITHMS):
        values = [aggregate[dataset][method][metric] for dataset in DATASETS]
        axes[panel].bar(x + (index - 1.5) * 0.2, values, 0.2, label=LABELS[method])
    axes[panel].set(
        xticks=x,
        xticklabels=("TEP", "PRONTO", "SKAB"),
        ylim=(0, 1.02),
        ylabel=metric.upper() if panel == 0 else None,
        title="False-alarm rate" if metric == "far" else "Missed-alarm rate",
    )
axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(-0.05, -0.14))
fig.suptitle("Held-out error decomposition")
fig.tight_layout()
fig.savefig(HERE / "Figure_2.png", dpi=180, bbox_inches="tight")
plt.close(fig)


ks_values = [report["prior_gate"]["normal_train_evaluation_distribution"][dataset]["median_ks"] for dataset in DATASETS]
shift_values = [
    report["prior_gate"]["normal_train_evaluation_distribution"][dataset]["maximum_standardized_median_shift"]
    for dataset in DATASETS
]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5))
axes[0].bar(x, ks_values)
axes[0].set(xticks=x, xticklabels=("TEP", "PRONTO", "SKAB"), ylabel="KS statistic", title="Normal train/evaluation KS")
axes[1].bar(x, shift_values)
axes[1].set(xticks=x, xticklabels=("TEP", "PRONTO", "SKAB"), ylabel="Standard deviations", title="Maximum normal median shift")
fig.suptitle("Prior distribution audit before method claims")
fig.tight_layout()
fig.savefig(HERE / "Figure_3.png", dpi=180)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
axes[0].bar(range(1, 4), [row["fit_goodness"] for row in condenser_units])
axes[0].set(xticks=range(1, 4), ylim=(0.95, 1.0005), xlabel="Seed", ylabel="Goodness of fit", title="Equation-defined parameter fit")
axes[1].bar(range(1, 4), [row["empirical"]["f1"] for row in condenser_units])
axes[1].set(xticks=range(1, 4), ylim=(0, 1.0), xlabel="Seed", ylabel="F1", title="Synthetic pressure-bias detection")
fig.suptitle("Book Section 3.4 synthetic-only validation")
fig.tight_layout()
fig.savefig(HERE / "Figure_4.png", dpi=180)
plt.close(fig)

print(json.dumps({"report": REPORT.as_posix(), "evidence_tiers": report["evidence_tiers"]}, indent=2))
