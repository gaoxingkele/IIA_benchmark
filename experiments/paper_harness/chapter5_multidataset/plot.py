#!/usr/bin/env python3
"""Aggregate and plot the three Chapter 5 multi-dataset runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
REPORT = ROOT / "experiments/reports/book_ch5_multidataset_validation.json"
RUNS = [
    json.loads((PROJECT / f"run_{index}/final_info.json").read_text(encoding="utf-8"))
    for index in (1, 2, 3)
]
DATASETS = ("tep_alarm_dataport", "npp_alarm_dataport", "fcc_alarm")
LABELS = ("TEP Alarm", "NPP Alarm", "FCC Alarm")
ALGORITHMS = (
    "book_5_1_flood_detection",
    "book_5_2_alarm_alignment",
    "book_5_3_closed_patterns",
    "book_5_4_max_entropy_prediction",
)


def values(dataset: str, algorithm: str, *keys: str) -> list[float]:
    output = []
    for run in RUNS:
        row: Any = run["result"]["datasets"][dataset][algorithm]
        for key in keys:
            row = row[key]
        output.append(float(row))
    return output


def summary(rows: list[float]) -> dict[str, float]:
    array = np.asarray(rows, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def activation_counts(kind: str) -> dict[str, dict[str, int | None]]:
    return {
        algorithm: {
            dataset: (
                None
                if RUNS[0]["result"][kind][algorithm][dataset] is None
                else sum(
                    bool(run["result"][kind][algorithm][dataset]) for run in RUNS
                )
            )
            for dataset in DATASETS
        }
        for algorithm in ALGORITHMS
    }


def build_report() -> dict[str, Any]:
    datasets = {}
    for dataset in DATASETS:
        first = RUNS[0]["result"]["datasets"][dataset]
        datasets[dataset] = {
            "match": first["match"],
            "protocol": first["protocol"],
            "boundary": first["boundary"],
            "prior_gate_passed_seeds": sum(
                run["result"]["datasets"][dataset]["prior_gate"]["passed"]
                for run in RUNS
            ),
            "source_runs": first["prior_gate"]["source_runs"],
            "trajectory_signatures": first["prior_gate"]["trajectory_signatures"],
            "source_cross_label_conflicting_signatures": first["prior_gate"]["cross_label_conflicting_signatures"],
            "used_cross_label_conflicting_signatures": first["prior_gate"]["used_cross_label_conflicting_signatures"],
            "activation_sequence_length": first["prior_gate"]["activation_sequence_length"],
            "criterion_c": {
                "candidate_episode_rate": summary(values(dataset, ALGORITHMS[0], "candidate_episode_rate")),
                "candidate_intervals": summary(values(dataset, ALGORITHMS[0], "candidate_intervals")),
                "maximum_attention_cardinality": summary(values(dataset, ALGORITHMS[0], "maximum_attention_cardinality")),
                "supervised_interval_score_reportable": False,
            },
            "alignment": {
                "balanced_accuracy": summary(values(dataset, ALGORITHMS[1], "metrics", "balanced_accuracy")),
                "macro_f1": summary(values(dataset, ALGORITHMS[1], "metrics", "macro_f1")),
                "set_jaccard_baseline_balanced_accuracy": summary(values(dataset, ALGORITHMS[1], "set_jaccard_baseline", "balanced_accuracy")),
                "mean_best_similarity": summary(values(dataset, ALGORITHMS[1], "mean_best_similarity")),
                "cells_evaluated": summary(values(dataset, ALGORITHMS[1], "cells_evaluated")),
            },
            "closed_patterns": {
                "balanced_accuracy": summary(values(dataset, ALGORITHMS[2], "metrics", "balanced_accuracy")),
                "macro_f1": summary(values(dataset, ALGORITHMS[2], "metrics", "macro_f1")),
                "class_core_jaccard_baseline_balanced_accuracy": summary(values(dataset, ALGORITHMS[2], "class_core_jaccard_baseline", "balanced_accuracy")),
                "total_closed_patterns": summary(values(dataset, ALGORITHMS[2], "total_closed_patterns")),
                "total_representative_patterns": summary(values(dataset, ALGORITHMS[2], "total_representative_patterns")),
                "compression_ratio": summary(values(dataset, ALGORITHMS[2], "compression_ratio")),
            },
            "maximum_entropy": {
                "top1_accuracy": summary(values(dataset, ALGORITHMS[3], "metrics", "top1_accuracy")),
                "top3_accuracy": summary(values(dataset, ALGORITHMS[3], "metrics", "top3_accuracy")),
                "macro_f1_eta_surrogate": summary(values(dataset, ALGORITHMS[3], "metrics", "macro_f1_eta_surrogate")),
                "negative_log_likelihood": summary(values(dataset, ALGORITHMS[3], "metrics", "negative_log_likelihood")),
                "global_frequency_baseline_top1": summary(values(dataset, ALGORITHMS[3], "global_frequency_baseline", "top1_accuracy")),
            },
            "method_wall_clock_seconds": {
                algorithm: summary([
                    run["result"]["datasets"][dataset]["method_wall_clock_seconds"][algorithm]
                    for run in RUNS
                ])
                for algorithm in ALGORITHMS
            },
        }
    named = RUNS[0]["result"]["named_book_items"]
    old_fcc_alignment = 0.8875
    new_fcc_alignment = datasets["fcc_alarm"]["alignment"]["balanced_accuracy"]["mean"]
    return {
        "schema_version": 1,
        "config": "configs/experiments/book_chapter5_multidataset.json",
        "seeds": [run["result"]["seed"] for run in RUNS],
        "algorithm_ids": list(ALGORITHMS),
        "paper_ids": [
            "wang2018_flood_detection",
            "hu2016_local_alignment",
            "hu2018_closed_alarm_patterns",
            "xu2021_max_entropy"
        ],
        "named_reproduction": {
            "all_passed_seeds": all(run["result"]["named_reproduction_passed"] for run in RUNS),
            "items": named,
        },
        "datasets": datasets,
        "mechanism_activation_seeds": activation_counts("mechanism_activation"),
        "performance_activation_seeds": activation_counts("performance_activation"),
        "competitive_credit_seeds": activation_counts("competitive_credit"),
        "evidence_tiers": {
            "book_5_1_flood_detection": "E2_real_data_descriptive_only_no_expert_intervals",
            "book_5_2_alarm_alignment": "E3_named_item_and_multi_dataset_transfer_with_negative_baseline_comparison",
            "book_5_3_closed_patterns": "E3_named_item_and_multi_dataset_transfer",
            "book_5_4_max_entropy_prediction": "E3_table_5_15_exact_and_negative_multi_dataset_transfer"
        },
        "fcc_duplicate_bias_audit": {
            "old_fixed_split_balanced_accuracy": old_fcc_alignment,
            "new_grouped_unique_three_seed_mean": new_fcc_alignment,
            "absolute_change": new_fcc_alignment - old_fcc_alignment,
            "interpretation": "the old FCC fixed split placed exact rising-edge duplicates across partitions and was optimistic"
        },
        "paper_score_closure": {algorithm: False for algorithm in ALGORITHMS},
        "strict_paper_score_closed_algorithms": 0,
        "original_data_blockers": RUNS[0]["result"]["original_data_blockers"],
        "reporting_boundary": RUNS[0]["result"]["reporting_boundary"],
    }


def plot_classification(report: dict[str, Any]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.2
    series = (
        ("Alignment", "alignment", "balanced_accuracy", "#377eb8"),
        ("Set Jaccard", "alignment", "set_jaccard_baseline_balanced_accuracy", "#a6cee3"),
        ("CHARM", "closed_patterns", "balanced_accuracy", "#e41a1c"),
        ("Core Jaccard", "closed_patterns", "class_core_jaccard_baseline_balanced_accuracy", "#fb9a99"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (label, section, metric, color) in enumerate(series):
        means = [report["datasets"][dataset][section][metric]["mean"] for dataset in DATASETS]
        stds = [report["datasets"][dataset][section][metric]["std"] for dataset in DATASETS]
        ax.bar(x + (index - 1.5) * width, means, width, yerr=stds, capsize=3, label=label, color=color)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Chapter 5 sequence classification (three grouped seeds)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_1.png", dpi=180)
    plt.close(fig)


def plot_entropy(report: dict[str, Any]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.25
    top1 = [report["datasets"][dataset]["maximum_entropy"]["top1_accuracy"]["mean"] for dataset in DATASETS]
    baseline = [report["datasets"][dataset]["maximum_entropy"]["global_frequency_baseline_top1"]["mean"] for dataset in DATASETS]
    eta = [report["datasets"][dataset]["maximum_entropy"]["macro_f1_eta_surrogate"]["mean"] for dataset in DATASETS]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, top1, width, label="MaxEnt Top-1", color="#4daf4a")
    ax.bar(x, baseline, width, label="Frequency Top-1", color="#b2df8a")
    ax.bar(x + width, eta, width, label="MaxEnt macro-F1 (eta surrogate)", color="#984ea3")
    ax.axhline(0.8, color="#e31a1c", linestyle="--", linewidth=1.5, label="Book effectiveness gate 0.8")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Maximum-entropy transfer remains below the book gate")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_2.png", dpi=180)
    plt.close(fig)


def plot_criterion(report: dict[str, Any]) -> None:
    rates = [report["datasets"][dataset]["criterion_c"]["candidate_episode_rate"]["mean"] for dataset in DATASETS]
    cardinality = [report["datasets"][dataset]["criterion_c"]["maximum_attention_cardinality"]["mean"] for dataset in DATASETS]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(DATASETS))
    bars = ax.bar(x, rates, color="#ff7f00", width=0.55)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of test runs with Criterion-C candidates")
    ax.set_title("Criterion C is descriptive without expert flood intervals")
    second = ax.twinx()
    second.plot(x, cardinality, color="#6a3d9a", marker="o", linewidth=2, label="Max attention cardinality")
    second.set_ylabel("Maximum attention-set cardinality")
    for bar, value in zip(bars, rates, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_3.png", dpi=180)
    plt.close(fig)


def plot_runtime(report: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    colors = ("#fdbf6f", "#1f78b4", "#e31a1c", "#33a02c")
    labels = ("Criterion C", "Alignment", "CHARM", "Maximum entropy")
    for algorithm, label, color in zip(ALGORITHMS, labels, colors, strict=True):
        rows = np.asarray([
            report["datasets"][dataset]["method_wall_clock_seconds"][algorithm]["mean"]
            for dataset in DATASETS
        ])
        ax.bar(x, rows, bottom=bottom, label=label, color=color)
        bottom += rows
    ax.set_xticks(x, LABELS)
    ax.set_yscale("log")
    ax.set_ylabel("Mean wall-clock seconds (log scale)")
    ax.set_title("Method-level runtime after harness optimization")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_4.png", dpi=180)
    plt.close(fig)


def main() -> int:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_classification(report)
    plot_entropy(report)
    plot_criterion(report)
    plot_runtime(report)
    print(json.dumps({
        "report": REPORT.as_posix(),
        "named_reproduction": report["named_reproduction"]["all_passed_seeds"],
        "performance_activation_seeds": report["performance_activation_seeds"],
        "strict_paper_score_closed_algorithms": report["strict_paper_score_closed_algorithms"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
