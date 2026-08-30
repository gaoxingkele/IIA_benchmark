#!/usr/bin/env python3
"""Profile the official P0 Code Ocean alarm payloads before model evaluation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/reports/p0_codeocean_data_prior.json"


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
    }


def canonical_hash(values: np.ndarray) -> str:
    binary = np.asarray(values > 0, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(binary.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(binary.reshape(-1)).tobytes())
    return digest.hexdigest()


def profile(rows: Iterable[tuple[str, str, np.ndarray]], expected: dict) -> dict:
    labels: Counter[str] = Counter()
    lengths: list[float] = []
    active_cells: list[float] = []
    activation_edges: list[float] = []
    unique_alarm_channels: list[float] = []
    hashes: Counter[str] = Counter()
    hash_members: dict[str, list[dict[str, str]]] = {}
    total_cells = 0
    total_active = 0
    finite = True
    binary = True
    channel_counts: Counter[int] = Counter()
    empty_episodes = 0
    sample_count = 0
    for identifier, label, values in rows:
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("each alarm episode must be channels x time")
        sample_count += 1
        labels[str(label)] += 1
        channel_counts[int(array.shape[0])] += 1
        lengths.append(float(array.shape[1]))
        finite = finite and bool(np.isfinite(array).all())
        binary = binary and bool(np.isin(array, [0, 1]).all())
        state = np.asarray(array > 0, dtype=np.uint8)
        count = int(np.sum(state))
        edges = int(np.sum(np.diff(np.pad(state, ((0, 0), (1, 0))), axis=1) == 1))
        used = int(np.sum(np.any(state > 0, axis=1)))
        active_cells.append(float(count))
        activation_edges.append(float(edges))
        unique_alarm_channels.append(float(used))
        empty_episodes += int(count == 0)
        total_cells += state.size
        total_active += count
        trajectory_hash = canonical_hash(state)
        hashes[trajectory_hash] += 1
        hash_members.setdefault(trajectory_hash, []).append(
            {"id": identifier, "label": str(label)}
        )
    observed_balance = len(set(labels.values())) == 1
    balance_matches = observed_balance if expected.get("require_class_balance", True) else True
    checks = {
        "sample_count": sample_count == expected["samples"],
        "class_count": len(labels) == expected["classes"],
        "class_balance_matches_protocol": balance_matches,
        "finite": finite,
        "binary": binary,
        "expected_channel_count": set(channel_counts) == {expected["channels"]},
        "nonempty_episodes": empty_episodes == 0,
    }
    return {
        "expected": expected,
        "checks": checks,
        "passed": all(checks.values()),
        "samples": sample_count,
        "class_counts": dict(sorted(labels.items())),
        "observed_class_balance": observed_balance,
        "channel_counts": {str(key): value for key, value in sorted(channel_counts.items())},
        "length_samples": percentile_summary(lengths),
        "active_state_cells": percentile_summary(active_cells),
        "activation_edges": percentile_summary(activation_edges),
        "unique_active_alarm_channels": percentile_summary(unique_alarm_channels),
        "global_active_state_density": float(total_active / total_cells),
        "canonical_unique_trajectories": len(hashes),
        "duplicate_episodes_beyond_first": int(sum(value - 1 for value in hashes.values())),
        "largest_duplicate_group": int(max(hashes.values())),
        "duplicate_groups": [
            {"trajectory_sha256": key, "members": members}
            for key, members in sorted(hash_members.items())
            if len(members) > 1
        ],
        "empty_episodes": empty_episodes,
    }


def casim_rows() -> Iterable[tuple[str, str, np.ndarray]]:
    root = ROOT / "data/public_datasets/codeocean/casim_v1/export/data"
    labels = pd.read_csv(root / "labels.csv", dtype={"ID": str}).set_index("ID")["Label"]
    for path in sorted(root.glob("*_ALARMS_Binary.csv")):
        identifier = path.name.split("_ALARMS")[0]
        yield identifier, str(int(labels.loc[identifier])), pd.read_csv(path, header=None).to_numpy().T


def casim_official_split_audit(duplicate_groups: list[dict]) -> dict:
    root = ROOT / "data/public_datasets/codeocean/casim_v1/export/data"
    files = [
        name
        for name in root.iterdir()
        if name.is_file() and name.name not in {"labels.csv", "LICENSE"}
    ]
    identifiers = [path.name.split("_ALARMS")[0] for path in files]
    labels = pd.read_csv(root / "labels.csv", dtype={"ID": str}).set_index("ID")["Label"]
    y = np.asarray([int(labels.loc[identifier]) for identifier in identifiers])
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    fold_rows = []
    for fold, (train, test) in enumerate(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42).split(
            np.zeros(len(y)), y
        ),
        start=1,
    ):
        train_set = set(int(value) for value in train)
        test_set = set(int(value) for value in test)
        crossings = []
        for group in duplicate_groups:
            member_ids = [member["id"] for member in group["members"]]
            member_positions = [positions[identifier] for identifier in member_ids]
            if any(position in train_set for position in member_positions) and any(
                position in test_set for position in member_positions
            ):
                crossings.append(member_ids)
        fold_rows.append(
            {
                "fold": fold,
                "duplicate_groups_crossing_train_test": crossings,
                "crossing_group_count": len(crossings),
            }
        )
    return {
        "protocol": "official RepeatedStratifiedKFold(5, 1, random_state=42) and os.listdir file order",
        "folds": fold_rows,
        "folds_with_duplicate_leakage": [
            row["fold"] for row in fold_rows if row["crossing_group_count"]
        ],
        "crossing_group_instances": sum(row["crossing_group_count"] for row in fold_rows),
        "interpretation": "The author-default split is sample-stratified rather than trajectory-grouped; identical known-class trajectories occur across train/test in folds 3-5.",
    }


def headerless_class_rows(root: Path, class_names: tuple[str, ...]) -> Iterable[tuple[str, str, np.ndarray]]:
    for label in class_names:
        for path in sorted((root / label).glob("*.csv")):
            yield path.stem, label, pd.read_csv(path, header=None).to_numpy().T


def bip_tep_rows() -> Iterable[tuple[str, str, np.ndarray]]:
    root = ROOT / "data/public_datasets/codeocean/bip_afc_v3/export/data/tep"
    for label in tuple(f"class_{index}" for index in range(5)):
        for path in sorted((root / label).glob("*.csv")):
            frame = pd.read_csv(path)
            yield path.stem, label, frame.iloc[:, 2:].to_numpy().T


def main() -> None:
    cone_root = ROOT / "data/public_datasets/codeocean/cone_afc_v2/export/data"
    bip_root = ROOT / "data/public_datasets/codeocean/bip_afc_v3/export/data/synthetic"
    casim = profile(casim_rows(), {"samples": 310, "classes": 15, "channels": 76, "require_class_balance": False, "note": "14 known labels plus label -1 outliers; class imbalance is an intended paper property"})
    casim["official_split_duplicate_audit"] = casim_official_split_audit(
        casim["duplicate_groups"]
    )
    report = {
        "schema_version": 1,
        "generated_at": "2026-08-30",
        "protocol": "streaming read-only pre-model prior; raw files are not modified",
        "datasets": {
            "casim_tep": casim,
            "cone_synthetic": profile(
                headerless_class_rows(cone_root, ("AK1", "AK2", "AK3", "AK4", "AK5")),
                {"samples": 18750, "classes": 5, "channels": 10},
            ),
            "bip_synthetic": profile(
                headerless_class_rows(bip_root, tuple(f"class_{index}" for index in range(5))),
                {"samples": 1875, "classes": 5, "channels": 10},
            ),
            "bip_tep": profile(bip_tep_rows(), {"samples": 1000, "classes": 5, "channels": 50, "note": "25 process variables x high/low alarm channels"}),
        },
    }
    report["all_prior_gates_passed"] = all(row["passed"] for row in report["datasets"].values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
