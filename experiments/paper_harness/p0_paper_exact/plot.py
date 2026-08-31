#!/usr/bin/env python3
"""Plot P0 transfer and Paper-Exact closure status."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    status = load(ROOT / "paper_harness/paper_exact/status.v1.json")
    casim = status["papers"][0]
    transfer = casim["transfer_result"]["datasets"]
    labels = ["Paper author\ndefault", "TEP transfer", "NPP transfer", "FCC transfer"]
    values = [
        casim["paper_exact_result"]["mean_balanced_accuracy"],
        transfer["tep_alarm_dataport"]["balanced_accuracy"],
        transfer["npp_alarm_dataport"]["balanced_accuracy"],
        transfer["fcc_alarm"]["balanced_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, values, color=["#2b6cb0", "#2f855a", "#d69e2e", "#805ad5"])
    ax.axhline(1.0, color="#4a5568", linewidth=0.8, linestyle="--")
    ax.set_ylim(0.75, 1.015)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("CASIM: author-default original domain vs frozen transfer results")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_1.png", dpi=180)
    plt.close(fig)

    names = ["CASIM", "ConE-AFC", "BiP-AFC"]
    acquired = [1, 1, 1]
    author_default = [1, 1, 1]
    full_paper_grid = [1, 1, 1]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = range(len(names))
    ax.barh(y, acquired, color="#90cdf4", label="Capsule acquired")
    ax.barh(y, author_default, color="#3182ce", label="Author default complete")
    ax.barh(y, full_paper_grid, color="#1a365d", label="Paper-grid compute progress")
    ax.set_yticks(list(y), names)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_title("P0 artifact and paper-grid compute coverage (not P3)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_2.png", dpi=180)
    plt.close(fig)

    open_set = load(PROJECT / "run_1/paper_grid/repetitions_10/summary.json")
    envelope = open_set["balanced_accuracy_random_instance_envelope"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.fill_between(
        open_set["thresholds"],
        envelope["minimum"],
        envelope["maximum"],
        color="#90cdf4",
        alpha=0.3,
        linewidth=0,
        label="10-instance range",
    )
    ax.fill_between(
        open_set["thresholds"],
        envelope["q25"],
        envelope["q75"],
        color="#4299e1",
        alpha=0.4,
        linewidth=0,
        label="10-instance IQR",
    )
    ax.plot(
        open_set["thresholds"],
        open_set["mean_balanced_accuracy"],
        color="#2b6cb0",
        linewidth=1.8,
        label="Local 10-instance mean",
    )
    ax.scatter(
        [open_set["maximum"]["threshold"]],
        [open_set["maximum"]["balanced_accuracy"]],
        color="#2b6cb0",
        zorder=3,
    )
    ax.scatter([0.324], [0.947], color="#c53030", marker="x", s=55, label="Paper Figure 13c peak")
    ax.axvline(0.324, color="#c53030", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Novelty threshold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("CASIM open-set threshold curve and random-instance envelope")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_3.png", dpi=180)
    plt.close(fig)

    bip = load(PROJECT / "run_3/paper_grid/summary.json")
    controlled = bip["numba_controlled_split_ablation"]
    dataset_labels = [row["dataset"].upper() for row in controlled]
    x = range(len(controlled))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    width = 0.34
    axes[0].bar(
        [value - width / 2 for value in x],
        [row["point_mae_delta"] for row in controlled],
        width,
        color="#2b6cb0",
        label="Point MAE delta",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [row["interval_width_delta"] for row in controlled],
        width,
        color="#d69e2e",
        label="Interval-width delta",
    )
    axes[0].axhline(0, color="#4a5568", linewidth=0.8)
    axes[0].set_xticks(list(x), dataset_labels)
    axes[0].set_ylabel("Disjoint minus overlap")
    axes[0].set_title("Pure RF-subset effect")
    axes[0].legend(fontsize=8)

    audits = [
        bip["controlled_paired_protocol_audit"],
        bip["numba_controlled_paired_protocol_audit"],
    ]
    audit_labels = ["Python seed", "Python + Numba seed"]
    values = [item["test_bifurcation_max_absolute_delta"] for item in audits]
    bars = axes[1].bar(audit_labels, values, color=["#c53030", "#2f855a"])
    axes[1].set_ylabel("Max paired test-bifurcation delta")
    axes[1].set_title("CASIM randomness confound")
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            str(value),
            ha="center",
            fontsize=9,
        )
    fig.suptitle("BiP CASIM deterministic split control")
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_4.png", dpi=180)
    plt.close(fig)

    cone_independent = load(
        PROJECT / "run_2/independent_same_fold/mbw_lr/summary.json"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    matched = cone_independent["comparison_rows_within_tolerance"]
    total = cone_independent["comparison_rows_total"]
    bars = axes[0].bar(
        ["Exact match", "Mismatch"],
        [matched, total - matched],
        color=["#2f855a", "#c53030"],
    )
    axes[0].set_ylabel("Same-fold metric rows")
    axes[0].set_title("Independent ConE conformal layer")
    axes[0].set_ylim(0, total * 1.1)
    for bar, value in zip(bars, [matched, total - matched]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.015,
            str(value),
            ha="center",
            fontsize=9,
        )

    gates = ["Author grid", "Conformal layer", "Base models", "Docker"]
    gate_values = [1, 1, 0, 0]
    axes[1].barh(
        gates,
        gate_values,
        color=["#2f855a", "#2f855a", "#a0aec0", "#a0aec0"],
    )
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xticks([0, 1], ["Open", "Closed"])
    axes[1].set_title("ConE P3 gate scope")
    fig.suptitle("ConE independent same-fold validation (MBW-LR scores)")
    fig.tight_layout()
    fig.savefig(PROJECT / "Figure_5.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
