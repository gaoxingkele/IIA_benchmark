#!/usr/bin/env python3
"""Aggregate and plot Chapter 4 gap-closure evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
REPORT = ROOT / "experiments/reports/book_ch4_gap_closure_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def load_runs() -> list[dict[str, Any]]:
    runs = []
    for index in range(1, 4):
        path = PROJECT / f"run_{index}" / "final_info.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def provenance_gate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    config_path = ROOT / runs[0]["config"]
    config_hash = sha256_file(config_path)
    source_maps = [run["execution_provenance"]["source_sha256"] for run in runs]
    identical = all(mapping == source_maps[0] for mapping in source_maps[1:])
    current = {
        relative: sha256_file(ROOT / relative) for relative in source_maps[0]
    }
    current_match = current == source_maps[0]
    ranges = []
    for run in runs:
        ranges.append(run["results"]["piade_plr"]["prior_gate"]["input_ranges"])
    cross_fold_disjoint = all(
        left["equipment"] != right["equipment"]
        or left["stop"] < right["start"]
        or right["stop"] < left["start"]
        for left_run in range(len(ranges))
        for right_run in range(left_run + 1, len(ranges))
        for left in ranges[left_run]
        for right in ranges[right_run]
    )
    config_match = all(run["config_sha256"] == config_hash for run in runs)
    mandatory = all(all(run["mandatory_gates"].values()) for run in runs)
    return {
        "passed": bool(
            config_match and identical and current_match and cross_fold_disjoint and mandatory
        ),
        "config_sha256": config_hash,
        "config_matches_all_runs": config_match,
        "source_hashes_identical_across_runs": identical,
        "source_hashes_match_current_worktree": current_match,
        "piade_cross_fold_windows_disjoint": cross_fold_disjoint,
        "mandatory_gates_pass_all_runs": mandatory,
        "source_files": len(source_maps[0]),
        "final_info_sha256": {
            run["run_name"]: sha256_file(
                PROJECT / run["run_name"] / "final_info.json"
            )
            for run in runs
        },
    }


def aggregate_igdte(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    controlled = [run["results"]["controlled_igdte"] for run in runs]
    imaks = [run["results"]["imaks_diagnostics"]["igdte"] for run in runs]
    prior_real = json.loads(
        (ROOT / "experiments/reports/book_ch4_igte_igdte_multidataset_validation.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "controlled_chain": {
            "igte": summary([row["metrics"]["igte"] for row in controlled]),
            "igdte": summary([row["metrics"]["igdte"] for row in controlled]),
            "surrogate_threshold": summary(
                [row["metrics"]["surrogate_threshold"] for row in controlled]
            ),
            "conditional_to_pairwise_ratio": summary(
                [row["metrics"]["conditional_to_pairwise_ratio"] for row in controlled]
            ),
            "mechanism_passes": sum(row["gates"]["mechanism"] for row in controlled),
        },
        "imaks_documented_edge": {
            "igte": summary([row["known_edge_igte"] for row in imaks]),
            "control_conditioned_igdte": summary(
                [row["control_conditioned_igdte"] for row in imaks]
            ),
            "surrogate_threshold": summary(
                [row["surrogate_threshold"] for row in imaks]
            ),
            "detection_passes": sum(row["known_edge_detected"] for row in imaks),
        },
        "existing_real_transfer": {
            "datasets": ["tep_classic", "pronto", "skab"],
            "episode_seed_evaluations": 21,
            "igdte_pruned_edges": 0,
            "activation": prior_real["activation"]["igdte"],
            "evidence": "experiments/reports/book_ch4_igte_igdte_multidataset_validation.json",
        },
        "evidence_decision": "E0_to_E1_controlled_activation; deny E2 because no acquired real/synthetic-dataset edge passes the frozen pruning/detection gate",
    }


def aggregate_bn(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    targets = {}
    for target in ("ME", "HE", "UE"):
        rows = [run["results"]["enas_recursive_bn"]["targets"][target] for run in runs]
        targets[target] = {
            "events": rows[0]["manual_annotation_rows"],
            "raw_impulse_nonempty_rate": summary(
                [row["raw_impulse"]["nonempty_decision_rate"] for row in rows]
            ),
            "persistent_nonempty_rate": summary(
                [row["persistent_adapter"]["nonempty_decision_rate"] for row in rows]
            ),
            "persistent_known_candidate_rate": summary(
                [row["persistent_adapter"]["known_candidate_decision_rate"] for row in rows]
            ),
            "persistent_unknown_rate": summary(
                [row["persistent_adapter"]["unknown_decision_rate"] for row in rows]
            ),
            "corrupted_known_candidate_rate": summary(
                [
                    row["corrupted_persistent_adapter"]["known_candidate_decision_rate"]
                    for row in rows
                ]
            ),
        }
    return {
        "enas": {
            "rows": runs[0]["results"]["enas_recursive_bn"]["prior_gate"]["rows"],
            "error_counts": runs[0]["results"]["enas_recursive_bn"]["prior_gate"]["error_counts"],
            "prior_passes": sum(
                run["results"]["enas_recursive_bn"]["prior_gate"]["passed"]
                for run in runs
            ),
            "mechanism_passes": sum(
                run["results"]["enas_recursive_bn"]["gates"]["mechanism"]
                for run in runs
            ),
            "targets": targets,
        },
        "imaks_documented_edge": {
            "source_cause_recall_during_target_alarm": summary(
                [
                    run["results"]["imaks_diagnostics"]["recursive_bn"][
                        "source_cause_recall_during_target_alarm"
                    ]
                    for run in runs
                ]
            ),
            "synthetic_only": True,
        },
        "evidence_decision": "E1_to_E2_real_EnAS_online_activation; exact tag-level root accuracy remains unavailable",
    }


def aggregate_plr(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    equipment = {}
    for name in ("s_1", "s_2", "s_3", "s_4", "s_5"):
        rows = [run["results"]["piade_plr"]["equipment"][name] for run in runs]
        equipment[name] = {
            "activation_rate": summary([row["activation_rate"] for row in rows]),
            "active_windows": sum(row["active_windows"] for row in rows),
            "evaluated_windows": sum(row["windows"] for row in rows),
        }
    return {
        "piade": {
            "chronological_folds": [run["chronological_fold"] for run in runs],
            "activation_rate": summary(
                [run["results"]["piade_plr"]["metrics"]["activation_rate"] for run in runs]
            ),
            "active_windows": sum(
                run["results"]["piade_plr"]["metrics"]["active_windows"] for run in runs
            ),
            "evaluated_windows": sum(
                run["results"]["piade_plr"]["metrics"]["evaluated_windows"] for run in runs
            ),
            "mechanism_passes": sum(
                run["results"]["piade_plr"]["gates"]["mechanism"] for run in runs
            ),
            "equipment": equipment,
        },
        "imaks_diagnostic": {
            "active_passes": sum(
                run["results"]["imaks_diagnostics"]["plr"]["active"] for run in runs
            ),
            "target_trends": runs[0]["results"]["imaks_diagnostics"]["plr"]["target_trends"],
            "recovered_lags": runs[0]["results"]["imaks_diagnostics"]["plr"]["lags"],
            "known_lag_samples": 180,
            "interpretation": runs[0]["results"]["imaks_diagnostics"]["plr"]["interpretation"],
        },
        "existing_book_numeric": {
            "evidence": "experiments/reports/book_ch4_plr_numeric_validation.json",
            "published_delays_recovered": [10, 8],
            "dominant_driver_accuracy": 1.0,
        },
        "evidence_decision": "E1_to_E2_real_PIADE_activation; causal Top-k/MRR and paper superiority remain unavailable",
    }


def make_figures(report: dict[str, Any]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    igdte = report["igdte"]
    labels = ["Controlled IGTE", "Controlled IGDTE", "iMAKS IGTE", "iMAKS IGDTE"]
    means = [
        igdte["controlled_chain"]["igte"]["mean"],
        igdte["controlled_chain"]["igdte"]["mean"],
        igdte["imaks_documented_edge"]["igte"]["mean"],
        igdte["imaks_documented_edge"]["control_conditioned_igdte"]["mean"],
    ]
    errors = [
        igdte["controlled_chain"]["igte"]["standard_deviation"],
        igdte["controlled_chain"]["igdte"]["standard_deviation"],
        igdte["imaks_documented_edge"]["igte"]["standard_deviation"],
        igdte["imaks_documented_edge"]["control_conditioned_igdte"]["standard_deviation"],
    ]
    fig, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    axis.bar(labels, means, yerr=errors, capsize=3, color=["#00798c", "#2a9d8f", "#d1495b", "#edae49"])
    axis.axhline(
        igdte["imaks_documented_edge"]["surrogate_threshold"]["mean"],
        color="#d1495b",
        linestyle="--",
        label="iMAKS surrogate threshold",
    )
    axis.set(ylabel="Information transfer", title="IGDTE activates on the controlled chain but misses the iMAKS edge")
    axis.tick_params(axis="x", rotation=15)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(PROJECT / "Figure_1.png", dpi=180)
    plt.close(fig)

    targets = ("ME", "HE", "UE")
    x = np.arange(len(targets))
    raw = [report["recursive_bn"]["enas"]["targets"][name]["raw_impulse_nonempty_rate"]["mean"] for name in targets]
    persistent = [report["recursive_bn"]["enas"]["targets"][name]["persistent_known_candidate_rate"]["mean"] for name in targets]
    unknown = [report["recursive_bn"]["enas"]["targets"][name]["persistent_unknown_rate"]["mean"] for name in targets]
    fig, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    axis.bar(x - 0.25, raw, 0.25, label="Raw impulse: nonempty", color="#98a2b3")
    axis.bar(x, persistent, 0.25, label="5-row adapter: known", color="#00798c")
    axis.bar(x + 0.25, unknown, 0.25, label="5-row adapter: unknown", color="#d1495b")
    axis.set(xticks=x, xticklabels=targets, ylabel="Event decision rate", ylim=(0, 1.05), title="EnAS recursive BN is representation-sensitive")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(PROJECT / "Figure_2.png", dpi=180)
    plt.close(fig)

    equipment = tuple(report["plr"]["piade"]["equipment"])
    x = np.arange(len(equipment))
    means = [report["plr"]["piade"]["equipment"][name]["activation_rate"]["mean"] for name in equipment]
    errors = [report["plr"]["piade"]["equipment"][name]["activation_rate"]["standard_deviation"] for name in equipment]
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    axis.bar(x, means, yerr=errors, capsize=3, color="#2a9d8f")
    axis.axhline(0.1, color="#d1495b", linestyle="--", label="Frozen activation gate")
    axis.set(xticks=x, xticklabels=equipment, ylabel="Nonzero-contribution window rate", ylim=(0, 1.0), title="PIADE PLR activation across disjoint chronological folds")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(PROJECT / "Figure_3.png", dpi=180)
    plt.close(fig)


def write_notes(report: dict[str, Any]) -> None:
    bn = report["recursive_bn"]["enas"]["targets"]
    lines = [
        "Chapter 4 gap-closure notes",
        "",
        "Outcome",
        "- IGDTE: E1 controlled mechanism activation; real/synthetic-dataset detection remains negative.",
        "- Recursive BN: E2 real EnAS online engineering activation; exact tag-level cause truth is absent.",
        "- PLR: E2 real PIADE grouped contribution activation; causal ranking accuracy is unavailable.",
        "",
        "IGDTE",
        f"- Controlled chain: IGTE={report['igdte']['controlled_chain']['igte']['mean']:.4f}, IGDTE={report['igdte']['controlled_chain']['igdte']['mean']:.4f}, passes={report['igdte']['controlled_chain']['mechanism_passes']}/3.",
        f"- iMAKS known edge: IGTE={report['igdte']['imaks_documented_edge']['igte']['mean']:.4f}, threshold={report['igdte']['imaks_documented_edge']['surrogate_threshold']['mean']:.4f}, detections={report['igdte']['imaks_documented_edge']['detection_passes']}/3.",
        "- Existing TEP/PRONTO/SKAB evidence: zero IGDTE prunes in 21 episode-seed evaluations.",
        "",
        "Recursive BN on EnAS",
    ]
    for target in ("ME", "HE", "UE"):
        lines.append(
            f"- {target}: raw nonempty={bn[target]['raw_impulse_nonempty_rate']['mean']:.4f}, persistent known={bn[target]['persistent_known_candidate_rate']['mean']:.4f}, persistent unknown={bn[target]['persistent_unknown_rate']['mean']:.4f}."
        )
    lines.extend(
        [
            "",
            "PLR on PIADE",
            f"- {report['plr']['piade']['active_windows']}/{report['plr']['piade']['evaluated_windows']} disjoint transition windows activate; mean fold rate={report['plr']['piade']['activation_rate']['mean']:.4f}.",
            "- The iMAKS correlated target is a sustained offset; only recovery activates and the stated 180-sample lag is not recovered.",
            "",
            "Boundaries",
            "- No unavailable paper table, thermal-plant result, or feature-level causal accuracy is claimed.",
            "- Synthetic iMAKS and generated-chain evidence cannot replace real industrial causal truth.",
            "- Negative activation and mismatch evidence is retained in the report.",
        ]
    )
    (PROJECT / "notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    runs = load_runs()
    report = {
        "schema_version": 1,
        "experiment_id": "chapter4_gap_closure",
        "task": "real_root_cause_gap_closure",
        "downstream_tasks": ["T3"],
        "real_data_validated_algorithms": [
            "book_4_3_recursive_bn",
            "book_4_4_plr_rca",
        ],
        "run_seeds": [run["seed"] for run in runs],
        "wall_clock_seconds": summary([run["wall_clock_seconds"] for run in runs]),
        "provenance_gate": provenance_gate(runs),
        "igdte": aggregate_igdte(runs),
        "recursive_bn": aggregate_bn(runs),
        "plr": aggregate_plr(runs),
        "evidence_updates": [
            {"algorithm_id": "book_4_2_igdte", "from": "E0", "to": "E1"},
            {"algorithm_id": "book_4_3_recursive_bn", "from": "E1", "to": "E2"},
            {"algorithm_id": "book_4_4_plr_rca", "from": "E1", "to": "E2"},
        ],
        "strict_paper_score_closure": {
            "closed": 0,
            "total": 3,
            "reason": "Exact paper plant payloads, causal truth and scoring protocols remain unavailable.",
        },
        "reporting_boundary": "Engineering activation and explicit negative evidence only; no cited paper score is reproduced.",
    }
    if not report["provenance_gate"]["passed"]:
        raise RuntimeError(f"provenance gate failed: {report['provenance_gate']}")
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figures(report)
    write_notes(report)
    print(
        json.dumps(
            {
                "report": REPORT.relative_to(ROOT).as_posix(),
                "provenance_passed": True,
                "evidence_updates": report["evidence_updates"],
                "strict_paper_scores_closed": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
