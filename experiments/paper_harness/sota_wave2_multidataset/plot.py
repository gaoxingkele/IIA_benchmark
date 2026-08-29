#!/usr/bin/env python3
"""Aggregate SOTA Wave 2 runs and render validation figures."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
REPORT_PATH = ROOT / "experiments/reports/sota_wave2_multidataset_validation.json"
MODEL_IDS = (
    "jaccard_class_core",
    "ctfh_fingerprinting",
    "structured_hdam",
    "casim",
    "modified_tfidf_afc",
    "time_encoded_histogram_hybrid",
)
MODEL_LABELS = {
    "jaccard_class_core": "Jaccard core",
    "ctfh_fingerprinting": "CTFH",
    "structured_hdam": "HDAM",
    "casim": "CASIM",
    "modified_tfidf_afc": "Modified TF-IDF",
    "time_encoded_histogram_hybrid": "Time histogram",
}
DATASET_LABELS = {
    "tep_alarm_dataport": "TEP Alarm",
    "npp_alarm_dataport": "NPP Alarm",
    "fcc_alarm": "FCC Alarm",
}
COLORS = ("#667085", "#d1495b", "#00798c", "#2a9d8f", "#7b2cbf", "#f4a261")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runs() -> list[dict[str, Any]]:
    paths = sorted(PROJECT.glob("run_*/final_info.json"))
    if len(paths) != 3:
        raise RuntimeError(f"expected three complete runs, found {len(paths)}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [row["seed"] for row in runs] != [1103, 2207, 3301]:
        raise RuntimeError("run seeds are not the frozen 1103/2207/3301 sequence")
    return runs


def summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def directory_manifest(root: Path, pattern: str) -> dict[str, Any]:
    paths = sorted(root.rglob(pattern))
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    combined = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": root.relative_to(ROOT).as_posix(),
        "files": len(records),
        "bytes": int(sum(row["bytes"] for row in records)),
        "combined_sha256": combined,
    }


def provenance_gate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    config_path = ROOT / runs[0]["config"]
    current_config = sha256_file(config_path)
    config_valid = all(row["config_sha256"] == current_config for row in runs)
    source_maps = [row["execution_provenance"]["source_sha256"] for row in runs]
    common_paths = sorted(set.intersection(*(set(row) for row in source_maps)))
    identical = all(len({row[path] for row in source_maps}) == 1 for path in common_paths)
    current = {
        path: sha256_file(ROOT / path)
        for path in common_paths
        if (ROOT / path).is_file()
    }
    current_match = all(all(row[path] == current[path] for path in common_paths) for row in source_maps)
    npp_manifest = directory_manifest(
        ROOT / "data/public_datasets/npp_alarm_dataport/payload/alpha_050", "*.csv"
    )
    final_paths = sorted(PROJECT.glob("run_*/final_info.json"))
    return {
        "passed": bool(config_valid and identical and current_match),
        "config_sha256": current_config,
        "config_matches_all_runs": config_valid,
        "source_hashes_identical_across_runs": identical,
        "source_hashes_match_current_worktree": current_match,
        "source_files": len(common_paths),
        "npp_csv_manifest": npp_manifest,
        "final_info_sha256": {
            path.parent.name: sha256_file(path) for path in final_paths
        },
        "known_boundary": (
            "The NPP directory was not hashed inside the original run provenance. "
            "This aggregate records its complete current CSV manifest; deterministic "
            "adapter statistics and G0 trajectory counts agree across every run."
        ),
    }


def aggregate_classification(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASET_LABELS:
        output[dataset] = {}
        for model in MODEL_IDS:
            rows = [row["results"][dataset]["classification"][model] for row in runs]
            output[dataset][model] = {
                "balanced_accuracy": summary([item["metrics"]["balanced_accuracy"] for item in rows]),
                "macro_f1": summary([item["metrics"]["macro_f1"] for item in rows]),
                "fit_seconds": summary([item["diagnostics"]["fit_seconds"] for item in rows]),
                "activation_passes": sum(item["gates"]["activation"] for item in rows),
                "performance_passes": sum(item["gates"]["performance"] for item in rows),
                "competitive_passes": (
                    None
                    if model == "jaccard_class_core"
                    else sum(item["gates"]["competitive"] for item in rows)
                ),
                "selected_ngram_sizes": (
                    [item["diagnostics"]["selected_ngram_size"] for item in rows]
                    if model == "modified_tfidf_afc"
                    else None
                ),
            }
    return output


def aggregate_robustness(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASET_LABELS:
        output[dataset] = {}
        for model in MODEL_IDS:
            rows = [row["results"][dataset]["robustness"]["models"][model] for row in runs]
            perturbations = tuple(rows[0]["full_progress_auc_by_perturbation"])
            output[dataset][model] = {
                "full_progress_mean_normalized_auc": summary(
                    [item["full_progress_mean_normalized_auc"] for item in rows]
                ),
                "maximum_degradation": summary([item["maximum_degradation"] for item in rows]),
                "auc_by_perturbation": {
                    perturbation: summary(
                        [item["full_progress_auc_by_perturbation"][perturbation] for item in rows]
                    )
                    for perturbation in perturbations
                },
                "activation_passes": sum(item["gates"]["activation"] for item in rows),
                "performance_passes": sum(item["gates"]["performance"] for item in rows),
                "competitive_passes": (
                    None
                    if model == "jaccard_class_core"
                    else sum(item["gates"]["competitive"] for item in rows)
                ),
            }
    return output


def aggregate_conformal(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = ("cone_afc_2024", "cross_conformal_afc_2025")
    output: dict[str, Any] = {}
    for dataset in DATASET_LABELS:
        output[dataset] = {}
        for method in methods:
            rows = [row["results"][dataset]["conformal"][method] for row in runs]
            lengths = rows[0]["prefix_lengths"]
            if any(item["prefix_lengths"] != lengths for item in rows):
                raise RuntimeError("prefix grid changed between runs")
            by_prefix = {}
            for length in lengths:
                metrics = [item["metrics_by_prefix"][str(length)] for item in rows]
                by_prefix[str(length)] = {
                    key: summary([item[key] for item in metrics])
                    for key in (
                        "coverage",
                        "average_set_size",
                        "empty_rate",
                        "singleton_rate",
                        "singleton_accuracy",
                    )
                    if all(item[key] is not None for item in metrics)
                }
            output[dataset][method] = {
                "prefix_lengths": lengths,
                "by_prefix": by_prefix,
                "coverage_target_passes_at_full_prefix": sum(
                    item["gates"]["coverage_at_full_prefix"] for item in rows
                ),
                "efficiency_passes_at_full_prefix": sum(
                    item["gates"]["efficient_at_full_prefix"] for item in rows
                ),
                "activation_passes": sum(item["gates"]["activation"] for item in rows),
            }
    return output


def aggregate_uncertainty(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASET_LABELS:
        rows = [row["results"][dataset]["uncertainty_reduction"] for row in runs]
        if any(item["status"] != "executed" for item in rows):
            output[dataset] = {
                "statuses": [item["status"] for item in rows],
                "executed_runs": sum(item["status"] == "executed" for item in rows),
            }
            continue
        output[dataset] = {
            "statuses": [item["status"] for item in rows],
            "training_rows": [item["training_rows"] for item in rows],
            "test_rows": [item["test_rows"] for item in rows],
            "mean_absolute_error_minutes": summary(
                [item["metrics"]["mean_absolute_error_minutes"] for item in rows]
            ),
            "median_baseline_mae_minutes": summary(
                [item["metrics"]["median_baseline_mae_minutes"] for item in rows]
            ),
            "jackknife_plus_coverage": summary(
                [item["metrics"]["jackknife_plus_coverage"] for item in rows]
            ),
            "mean_interval_width_minutes": summary(
                [item["metrics"]["mean_interval_width_minutes"] for item in rows]
            ),
            "delay_timer_emissions": summary(
                [item["metrics"]["delay_timer_emissions"] for item in rows]
            ),
            "activation_passes": sum(item["gates"]["activation"] for item in rows),
            "performance_passes": sum(item["gates"]["performance"] for item in rows),
            "competitive_passes": sum(item["gates"]["competitive"] for item in rows),
        }
    return output


def aggregate_prior(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for dataset in DATASET_LABELS:
        rows = [row["results"][dataset]["prior_gate"] for row in runs]
        output[dataset] = {
            "passes": sum(item["passed"] for item in rows),
            "checks": {
                key: sum(item["checks"][key] for item in rows)
                for key in rows[0]["checks"]
            },
            "within_class_tag_jaccard": summary(
                [item["mean_within_class_tag_jaccard"] for item in rows]
            ),
            "cross_class_tag_jaccard": summary(
                [item["mean_cross_class_tag_jaccard"] for item in rows]
            ),
            "event_count_distribution": rows[0]["event_count_distribution"],
            "unique_tag_distribution": rows[0]["unique_tag_distribution"],
            "exclusions": rows[0]["exclusions"],
        }
    return output


def make_figures(report: dict[str, Any]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    datasets = tuple(DATASET_LABELS)
    x = np.arange(len(datasets), dtype=float)
    width = 0.13

    figure, axis = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    for index, (model, color) in enumerate(zip(MODEL_IDS, COLORS)):
        means = [report["classification"][dataset][model]["balanced_accuracy"]["mean"] for dataset in datasets]
        errors = [report["classification"][dataset][model]["balanced_accuracy"]["standard_deviation"] for dataset in datasets]
        axis.bar(x + (index - 2.5) * width, means, width, yerr=errors, capsize=2, label=MODEL_LABELS[model], color=color)
    axis.set(xticks=x, xticklabels=[DATASET_LABELS[item] for item in datasets], ylabel="Balanced accuracy", ylim=(0, 1.08), title="SOTA point classification: grouped three-seed validation")
    axis.axhline(0.2, color="#98a2b3", linestyle="--", linewidth=0.8, label="TEP chance (0.20)")
    axis.legend(ncol=3, frameon=False, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(PROJECT / "Figure_1.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    for index, (model, color) in enumerate(zip(MODEL_IDS, COLORS)):
        means = [report["robustness"][dataset][model]["full_progress_mean_normalized_auc"]["mean"] for dataset in datasets]
        errors = [report["robustness"][dataset][model]["full_progress_mean_normalized_auc"]["standard_deviation"] for dataset in datasets]
        axis.bar(x + (index - 2.5) * width, means, width, yerr=errors, capsize=2, label=MODEL_LABELS[model], color=color)
    axis.set(xticks=x, xticklabels=[DATASET_LABELS[item] for item in datasets], ylabel="Mean normalized robustness AUC", ylim=(0, 1.08), title="Missing/spurious/timing/delay/mixed robustness")
    axis.legend(ncol=3, frameon=False, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(PROJECT / "Figure_2.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        class_count = len(runs_global[0]["results"][dataset]["prior_gate"]["class_counts"]["test"])
        for method, color, marker in (
            ("cone_afc_2024", "#d1495b", "o"),
            ("cross_conformal_afc_2025", "#00798c", "s"),
        ):
            row = report["conformal"][dataset][method]
            lengths = row["prefix_lengths"]
            progress = np.asarray(lengths, dtype=float) / lengths[-1]
            coverage = [row["by_prefix"][str(length)]["coverage"]["mean"] for length in lengths]
            normalized_size = [row["by_prefix"][str(length)]["average_set_size"]["mean"] / class_count for length in lengths]
            label = "ConE" if method.startswith("cone") else "Cross-Conformal"
            axis.plot(progress, coverage, color=color, marker=marker, label=f"{label} coverage")
            axis.plot(progress, normalized_size, color=color, marker=marker, linestyle="--", alpha=0.75, label=f"{label} set size / K")
        axis.axhline(0.9, color="#98a2b3", linewidth=0.8, linestyle=":")
        axis.set(title=DATASET_LABELS[dataset], xlabel="Observed fraction", ylim=(0, 1.05))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Coverage or normalized set size")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Conformal coverage must be read with efficiency")
    figure.savefig(PROJECT / "Figure_3.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.8, 5.3), constrained_layout=True)
    predicted = [report["uncertainty_reduction"][dataset]["mean_absolute_error_minutes"]["mean"] for dataset in datasets]
    baseline = [report["uncertainty_reduction"][dataset]["median_baseline_mae_minutes"]["mean"] for dataset in datasets]
    coverage = [report["uncertainty_reduction"][dataset]["jackknife_plus_coverage"]["mean"] for dataset in datasets]
    axis.bar(x - 0.18, predicted, 0.36, label="Jackknife+ RF MAE", color="#2a9d8f")
    axis.bar(x + 0.18, baseline, 0.36, label="Median-time baseline MAE", color="#98a2b3")
    axis.set(xticks=x, xticklabels=[DATASET_LABELS[item] for item in datasets], ylabel="Minutes", title="Next uncertainty-reduction forecasting")
    axis.grid(axis="y", alpha=0.2)
    second = axis.twinx()
    second.plot(x, coverage, color="#d1495b", marker="o", linewidth=2, label="Interval coverage")
    second.set(ylabel="Jackknife+ coverage", ylim=(0, 1.05))
    handles, labels = axis.get_legend_handles_labels()
    handles_2, labels_2 = second.get_legend_handles_labels()
    axis.legend(handles + handles_2, labels + labels_2, frameon=False, loc="upper right")
    figure.savefig(PROJECT / "Figure_4.png", dpi=180)
    plt.close(figure)


def write_notes(report: dict[str, Any]) -> None:
    lines = [
        "SOTA Wave 2 multi-dataset experiment notes",
        "",
        "Protocol",
        "- Seeds: 1103, 2207, 3301.",
        "- TEP/NPP/FCC complete rising-edge trajectories are grouped before fitting.",
        "- Five point classifiers share one split and Jaccard class-core parent.",
        "- Robustness uses test-only missing/spurious/timing/delay/mixed corruptions.",
        "- ConE and Cross-Conformal report coverage together with set efficiency.",
        "- Uncertainty reduction predicts the next contraction of the ConE set.",
        "",
        "Classification (mean balanced accuracy +/- seed SD)",
    ]
    for dataset in DATASET_LABELS:
        lines.append(DATASET_LABELS[dataset])
        for model in MODEL_IDS:
            row = report["classification"][dataset][model]["balanced_accuracy"]
            lines.append(f"  {MODEL_LABELS[model]}: {row['mean']:.4f} +/- {row['standard_deviation']:.4f}")
    lines.extend(["", "Full-progress robustness (mean normalized AUC)"])
    for dataset in DATASET_LABELS:
        values = ", ".join(
            f"{MODEL_LABELS[model]}={report['robustness'][dataset][model]['full_progress_mean_normalized_auc']['mean']:.4f}"
            for model in MODEL_IDS
        )
        lines.append(f"- {DATASET_LABELS[dataset]}: {values}")
    lines.extend(["", "Conformal full-prefix result"])
    for dataset in DATASET_LABELS:
        for method in ("cone_afc_2024", "cross_conformal_afc_2025"):
            row = report["conformal"][dataset][method]
            last = str(row["prefix_lengths"][-1])
            metrics = row["by_prefix"][last]
            lines.append(
                f"- {DATASET_LABELS[dataset]} {method}: coverage={metrics['coverage']['mean']:.4f}, set_size={metrics['average_set_size']['mean']:.4f}"
            )
    lines.extend(["", "Uncertainty reduction"])
    for dataset in DATASET_LABELS:
        row = report["uncertainty_reduction"][dataset]
        lines.append(
            f"- {DATASET_LABELS[dataset]}: MAE={row['mean_absolute_error_minutes']['mean']:.4f}, baseline={row['median_baseline_mae_minutes']['mean']:.4f}, coverage={row['jackknife_plus_coverage']['mean']:.4f}, competitive_passes={row['competitive_passes']}/3"
        )
    lines.extend(
        [
            "",
            "Evidence boundaries",
            "- No cited paper score is closed: exact full texts/capsules, paper splits, and/or original datasets remain unavailable.",
            "- Modified TF-IDF KPCA isolation is non-applicable because all three payloads contain abnormal classes only and no normal-operation class.",
            "- The attempted packed-LSTM runtime patch was rejected: endpoint tests passed but end-to-end runtime did not materially improve.",
            "- Robustness Monte Carlo is bounded; three outer seeds yield six draws per model/scenario after aggregation.",
        ]
    )
    (PROJECT / "notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global runs_global
    runs_global = load_runs()
    report = {
        "schema_version": 1,
        "experiment_id": "sota_wave2_multidataset",
        "task": "real_sota_alarm_flood_multidataset_validation",
        "downstream_tasks": ["T4", "T5"],
        "real_data_validated_algorithms": [
            "casim_2024",
            "cone_afc_2024",
            "uncertainty_reduction_2025",
            "etfa_robustness_2025",
            "afc_robustbench_2026",
            "cross_conformal_afc_2025",
            "modified_tfidf_afc_2025",
            "ctfh_fingerprinting_2025",
            "structured_hdam_2026",
            "hybrid_histogram_afc_2026",
        ],
        "run_seeds": [row["seed"] for row in runs_global],
        "wall_clock_seconds": summary([row["wall_clock_seconds"] for row in runs_global]),
        "provenance_gate": provenance_gate(runs_global),
        "prior_validation": aggregate_prior(runs_global),
        "classification": aggregate_classification(runs_global),
        "robustness": aggregate_robustness(runs_global),
        "conformal": aggregate_conformal(runs_global),
        "uncertainty_reduction": aggregate_uncertainty(runs_global),
        "evidence_updates": [
            {"algorithm_id": "modified_tfidf_afc_2025", "from": "E0", "to": "E2"},
            {"algorithm_id": "hybrid_histogram_afc_2026", "from": "E0", "to": "E2"},
            {"algorithm_id": "uncertainty_reduction_2025", "from": "E0", "to": "E2"},
            {"algorithm_id": "etfa_robustness_2025", "from": "E0", "to": "E2"},
            {"algorithm_id": "afc_robustbench_2026", "from": "E0", "to": "E2"},
        ],
        "strict_paper_score_closure": {
            "closed": 0,
            "total": 9,
            "reason": "Exact paper data/splits/full text or official capsules remain unavailable.",
        },
        "runtime_gene_trial": {
            "where": "runtime",
            "why": "variable-length TF-IDF LSTM padding",
            "candidate": "packed recurrent sequence",
            "validity": "endpoint equality unit test passed",
            "activation": True,
            "gain": "no material end-to-end acceleration observed before ConE",
            "bank_decision": "reject_keep_parent",
        },
        "reporting_boundary": (
            "Engineering and cross-dataset transfer evidence only; no paper table is claimed reproduced."
        ),
    }
    if not report["provenance_gate"]["passed"]:
        raise RuntimeError(f"provenance gate failed: {report['provenance_gate']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figures(report)
    write_notes(report)
    print(
        json.dumps(
            {
                "report": REPORT_PATH.relative_to(ROOT).as_posix(),
                "figures": [f"Figure_{index}.png" for index in range(1, 5)],
                "provenance_passed": report["provenance_gate"]["passed"],
                "strict_paper_scores_closed": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
