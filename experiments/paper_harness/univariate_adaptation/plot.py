#!/usr/bin/env python3
"""Aggregate and plot the univariate distribution-transfer audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
RUNS = ("run_1", "run_2", "run_3")
DATASETS = ("tep_classic", "pronto", "skab")
LABELS = {"tep_classic": "TEP", "pronto": "PRONTO", "skab": "SKAB"}
COLORS = {"tep_classic": "#3B6FB6", "pronto": "#D9822B", "skab": "#3D9970"}


def load_runs() -> list[dict[str, object]]:
    return [
        json.loads((PROJECT / run / "final_info.json").read_text(encoding="utf-8"))
        for run in RUNS
    ]


def records(runs: list[dict[str, object]], dataset: str) -> list[dict[str, object]]:
    return [
        row
        for run in runs
        for row in run["result"]["datasets"][dataset]
    ]


def summary(runs: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for dataset in DATASETS:
        rows = records(runs, dataset)
        audits = [row["held_out_posthoc_audit"] for row in rows]
        normal_ks = np.asarray(
            [row["normal_train_to_evaluation"]["ks"] for row in audits]
        )
        abnormal_ks = np.asarray(
            [row["abnormal_calibration_to_evaluation"]["ks"] for row in audits]
        )
        auc = np.asarray([row["evaluation_auc"] for row in audits])
        lag = np.asarray(
            [row["normal_evaluation_temporal"]["lag_one_autocorrelation"] for row in audits]
        )
        prevalence = np.asarray(
            [row["evaluation_abnormal_prevalence"] for row in audits]
        )
        threshold_far = np.asarray(
            [row["threshold_transfer"]["normal_evaluation_exceedance_rate"] for row in audits]
        )
        statuses = [row["calibration_applicability"]["status"] for row in rows]
        adapter_names = tuple(rows[0]["initial_adapter_results"])
        book_ids = tuple(rows[0]["adapted_book_suite"])
        router_rows = [row["automatic_router"] for row in rows]
        router_scored = [row for row in router_rows if row["empirical"] is not None]
        result[dataset] = {
            "records": len(rows),
            "features": sorted({row["feature"] for row in rows}),
            "normal_ks": {
                "median": float(np.median(normal_ks)),
                "minimum": float(np.min(normal_ks)),
                "maximum": float(np.max(normal_ks)),
            },
            "abnormal_ks": {
                "median": float(np.median(abnormal_ks)),
                "minimum": float(np.min(abnormal_ks)),
                "maximum": float(np.max(abnormal_ks)),
            },
            "evaluation_auc": {
                "median": float(np.median(auc)),
                "minimum": float(np.min(auc)),
                "maximum": float(np.max(auc)),
            },
            "normal_lag_one": {
                "median": float(np.median(lag)),
                "minimum": float(np.min(lag)),
                "maximum": float(np.max(lag)),
            },
            "evaluation_abnormal_prevalence": {
                "median": float(np.median(prevalence)),
                "minimum": float(np.min(prevalence)),
                "maximum": float(np.max(prevalence)),
            },
            "raw_threshold_normal_evaluation_exceedance": {
                "median": float(np.median(threshold_far)),
                "minimum": float(np.min(threshold_far)),
                "maximum": float(np.max(threshold_far)),
            },
            "calibration_gate_status_counts": {
                status: statuses.count(status)
                for status in ("static", "adapt", "reject_univariate")
            },
            "initial_adapter_metrics": {
                adapter: {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    row["initial_adapter_results"][adapter]["empirical"][metric]
                                    for row in rows
                                ]
                            )
                        ),
                        "standard_deviation_across_seeds_and_episodes": float(
                            np.std(
                                [
                                    row["initial_adapter_results"][adapter]["empirical"][metric]
                                    for row in rows
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in ("false_alarm_rate", "missed_alarm_rate", "f1")
                }
                for adapter in adapter_names
            },
            "adapted_book_suite_metrics": {
                algorithm_id: {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    row["adapted_book_suite"][algorithm_id]["empirical"][metric]
                                    for row in rows
                                ]
                            )
                        ),
                        "standard_deviation_across_seeds_and_episodes": float(
                            np.std(
                                [
                                    row["adapted_book_suite"][algorithm_id]["empirical"][metric]
                                    for row in rows
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in ("false_alarm_rate", "missed_alarm_rate", "f1")
                }
                for algorithm_id in book_ids
            },
            "automatic_router": {
                "status_counts": {
                    status: sum(row["status"] == status for row in router_rows)
                    for status in ("static", "adapt", "reject_univariate")
                },
                "scored_episodes": len(router_scored),
                "denied_episodes": len(router_rows) - len(router_scored),
                "scored_mean_metrics": {
                    metric: (
                        float(np.mean([row["empirical"][metric] for row in router_scored]))
                        if router_scored
                        else None
                    )
                    for metric in ("false_alarm_rate", "missed_alarm_rate", "f1")
                },
            },
        }
    return result


def plot_shift(summary_values: dict[str, object]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.34
    normal = [summary_values[item]["normal_ks"]["median"] for item in DATASETS]
    abnormal = [summary_values[item]["abnormal_ks"]["median"] for item in DATASETS]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(x - width / 2, normal, width, label="Normal train→evaluation", color="#4C78A8")
    axis.bar(x + width / 2, abnormal, width, label="Abnormal calibration→evaluation", color="#F58518")
    axis.axhline(0.2, color="#555555", linestyle="--", linewidth=1, label="Routing beacon 0.20")
    axis.set(
        xticks=x,
        xticklabels=[LABELS[item] for item in DATASETS],
        ylabel="Kolmogorov–Smirnov distance",
        ylim=(0, 1.02),
        title="Univariate distribution transfer differs by dataset mechanism",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(PROJECT / "Figure_1.png", dpi=180)
    plt.close(figure)


def plot_portability(runs: list[dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for dataset in DATASETS:
        rows = records(runs, dataset)
        x = [row["held_out_posthoc_audit"]["normal_train_to_evaluation"]["ks"] for row in rows]
        y = [row["held_out_posthoc_audit"]["evaluation_auc"] for row in rows]
        size = [
            80 + 420 * row["held_out_posthoc_audit"]["evaluation_abnormal_prevalence"]
            for row in rows
        ]
        axis.scatter(x, y, s=size, alpha=0.68, label=LABELS[dataset], color=COLORS[dataset], edgecolor="white", linewidth=0.7)
    axis.axvline(0.2, color="#777777", linestyle="--", linewidth=1)
    axis.axhline(0.6, color="#777777", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Normal train→evaluation KS distance",
        ylabel="Held-out oriented AUC",
        xlim=(-0.02, 0.55),
        ylim=(-0.02, 1.04),
        title="Baseline drift and anomaly separability are distinct failure axes",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(PROJECT / "Figure_2.png", dpi=180)
    plt.close(figure)


def write_notes(report: dict[str, object]) -> None:
    rows = [
        "Univariate adaptation distribution audit",
        "========================================",
        "",
        "This audit separates leakage-safe calibration routing from held-out post-hoc diagnostics.",
        "The three seeds repeat the registered Chapter 2 feature-selection protocol; seed records",
        "are not treated as independent physical episodes for inferential p-values.",
        "",
    ]
    for dataset in DATASETS:
        item = report["datasets"][dataset]
        rows.extend(
            [
                f"{LABELS[dataset]}: normal KS median={item['normal_ks']['median']:.4f}, "
                f"abnormal KS median={item['abnormal_ks']['median']:.4f}, "
                f"evaluation AUC median={item['evaluation_auc']['median']:.4f}, "
                f"lag-1 median={item['normal_lag_one']['median']:.4f}.",
                f"Calibration gate counts: {item['calibration_gate_status_counts']}.",
                "Initial ECDF adapter mean F1: "
                + ", ".join(
                    f"{name}={values['f1']['mean']:.4f}"
                    for name, values in item["initial_adapter_metrics"].items()
                )
                + ".",
                "Adapted Chapter 2 suite mean F1: "
                + ", ".join(
                    f"{name}={values['f1']['mean']:.4f}"
                    for name, values in item["adapted_book_suite_metrics"].items()
                )
                + ".",
                f"Automatic router: {item['automatic_router']}.",
                "",
            ]
        )
    rows.extend(
        [
            "Interpretation",
            "--------------",
            "TEP is predominantly a stationary-transfer case with d11 providing weak univariate separation.",
            "PRONTO is dominated by abnormal-phase drift and strong temporal dependence.",
            "SKAB has near-perfect abnormal ranking but severe normal-baseline shift and class imbalance.",
            "These are engineering transfer findings (M2/P1), not paper-exact industrial alarm scores.",
        ]
    )
    (PROJECT / "notes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    runs = load_runs()
    dataset_summary = summary(runs)
    baseline_path = ROOT / "experiments/reports/book_ch2_multidataset_validation.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_iid = {
        dataset: baseline["aggregate_metrics"][dataset]["book_2_1_iid_delay_timer"]
        for dataset in DATASETS
    }
    tep_ecdf = dataset_summary["tep_classic"]["initial_adapter_metrics"][
        "ecdf_two_sided"
    ]
    pronto_block = dataset_summary["pronto"]["initial_adapter_metrics"][
        "block_recent_two_sided"
    ]
    skab_block = dataset_summary["skab"]["initial_adapter_metrics"][
        "block_recent_one_sided"
    ]
    report = {
        "schema_version": 1,
        "experiment": "univariate_adaptation_distribution_audit",
        "runs": list(RUNS),
        "seeds": [int(run["result"]["seed"]) for run in runs],
        "datasets": dataset_summary,
        "frozen_book_chapter2_baseline_metrics": baseline["aggregate_metrics"],
        "mechanism_conclusions": {
            "tep_classic": "mostly stationary normal transfer; d11 is weakly separable in the selected univariate representation",
            "pronto": "abnormal phase drift and strong serial dependence dominate missed alarms",
            "skab": "normal baseline drift and low abnormal prevalence dominate false alarms despite perfect ranking",
        },
        "stage_acceptance": {
            "adapted_book_suite_coverage": {
                "algorithms": list(
                    dataset_summary["tep_classic"]["adapted_book_suite_metrics"]
                ),
                "datasets": list(DATASETS),
                "seeds": [1103, 2207, 3301],
                "passed": bool(
                    all(
                        len(dataset_summary[dataset]["adapted_book_suite_metrics"])
                        == 4
                        for dataset in DATASETS
                    )
                ),
            },
            "router_d11_univariate_denial": {
                "denied_tep_episode_seed_units": dataset_summary["tep_classic"][
                    "automatic_router"
                ]["denied_episodes"],
                "expected": 3,
                "passed": bool(
                    dataset_summary["tep_classic"]["automatic_router"][
                        "denied_episodes"
                    ]
                    == 3
                ),
            },
            "router_pronto_far_constraint": {
                "candidate_far": dataset_summary["pronto"]["automatic_router"][
                    "scored_mean_metrics"
                ]["false_alarm_rate"],
                "maximum_far": 0.15,
                "passed": bool(
                    dataset_summary["pronto"]["automatic_router"][
                        "scored_mean_metrics"
                    ]["false_alarm_rate"]
                    <= 0.15
                ),
                "limitation": "calibration-only routing cannot anticipate the observed PRONTO abnormal evaluation-phase drift; a future-event guarantee is not claimed",
            },
            "tep_performance_retention": {
                "candidate": "ecdf_two_sided",
                "baseline_f1": baseline_iid["tep_classic"]["f1"],
                "candidate_f1": tep_ecdf["f1"]["mean"],
                "passed": bool(
                    tep_ecdf["f1"]["mean"]
                    >= baseline_iid["tep_classic"]["f1"] - 0.02
                ),
            },
            "skab_false_alarm_reduction": {
                "candidate": "block_recent_one_sided",
                "baseline_far": baseline_iid["skab"]["far"],
                "candidate_far": skab_block["false_alarm_rate"]["mean"],
                "baseline_mar": baseline_iid["skab"]["mar"],
                "candidate_mar": skab_block["missed_alarm_rate"]["mean"],
                "passed": bool(
                    skab_block["false_alarm_rate"]["mean"]
                    <= 0.5 * baseline_iid["skab"]["far"]
                    and skab_block["missed_alarm_rate"]["mean"]
                    <= baseline_iid["skab"]["mar"] + 0.03
                ),
            },
            "pronto_missed_alarm_reduction_with_far_constraint": {
                "candidate": "block_recent_two_sided",
                "baseline_mar": baseline_iid["pronto"]["mar"],
                "candidate_mar": pronto_block["missed_alarm_rate"]["mean"],
                "candidate_far": pronto_block["false_alarm_rate"]["mean"],
                "maximum_far": 0.15,
                "passed": bool(
                    pronto_block["missed_alarm_rate"]["mean"]
                    <= 0.8 * baseline_iid["pronto"]["mar"]
                    and pronto_block["false_alarm_rate"]["mean"] <= 0.15
                ),
                "failure_interpretation": "recency/block adaptation recovers abnormal recall but produces unacceptable normal false alarms; do not promote without regime/change-point routing",
            },
            "safe_rolling_promoted": {
                "passed": False,
                "reason": "the contamination guard freezes after large baseline shifts and produces excessive held-out false alarms",
            },
        },
        "routing_boundary": "Only calibration_applicability may route adapters. held_out_posthoc_audit is explanatory evidence and cannot tune the benchmark.",
    }
    report_path = ROOT / "experiments/reports/univariate_distribution_audit_validation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_shift(dataset_summary)
    plot_portability(runs)
    write_notes(report)
    print(json.dumps(report["datasets"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
