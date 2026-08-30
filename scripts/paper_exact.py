#!/usr/bin/env python3
"""Validate, execute, and summarize official Paper-Exact P0 capsules."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/reproducibility/codeocean_capsules.v1.json"
CARD_ROOT = ROOT / "paper_harness/paper_exact"
PAPER_IDS = (
    "faulwasser2024_casim",
    "faulwasser2024_cone_afc",
    "faulwasser2025_uncertainty_reduction",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for item in sorted((row for row in path.rglob("*") if row.is_file()), key=lambda row: row.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        record = f"{relative}\t{sha256_file(item)}\t{size}\n".encode()
        digest.update(record)
        count += 1
        total_bytes += size
    return {"files": count, "bytes": total_bytes, "sha256": digest.hexdigest()}


def cards() -> dict[str, dict[str, Any]]:
    return {
        paper_id: read_json(CARD_ROOT / f"{paper_id}.v1.json")
        for paper_id in PAPER_IDS
    }


def capsules() -> dict[str, dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    return {row["paper_id"]: row for row in manifest["capsules"]}


def validate(require_local: bool, full_hash: bool) -> tuple[dict[str, Any], list[str]]:
    card_rows = cards()
    capsule_rows = capsules()
    required = set(read_json(CARD_ROOT / "schema.v1.json")["required"])
    issues: list[str] = []
    status: dict[str, Any] = {}
    if set(card_rows) != set(capsule_rows):
        issues.append("protocol cards and capsule manifest use different paper IDs")
    for paper_id in PAPER_IDS:
        card = card_rows[paper_id]
        capsule = capsule_rows[paper_id]
        missing = sorted(required - set(card))
        if missing:
            issues.append(f"{paper_id}: missing card fields {missing}")
        if card.get("paper_id") != paper_id:
            issues.append(f"{paper_id}: card ID mismatch")
        if card.get("official_artifact", {}).get("doi") != capsule.get("doi"):
            issues.append(f"{paper_id}: capsule DOI mismatch")
        pdf_path = ROOT / card["paper"]["local_pdf"]
        if not pdf_path.is_file() or sha256_file(pdf_path) != card["paper"]["pdf_sha256"]:
            issues.append(f"{paper_id}: PDF missing or hash mismatch")
        archive = ROOT / capsule["archive_path"]
        export = ROOT / capsule["export_root"]
        local = {
            "archive_present": archive.is_file(),
            "export_present": export.is_dir(),
            "archive_hash_match": None,
            "code_manifest_match": None,
            "data_manifest_match": None,
        }
        if archive.is_file():
            local["archive_hash_match"] = sha256_file(archive) == capsule["archive_sha256"]
            if not local["archive_hash_match"]:
                issues.append(f"{paper_id}: archive hash mismatch")
        elif require_local:
            issues.append(f"{paper_id}: local capsule archive is missing")
        if export.is_dir() and full_hash:
            local["code_manifest_match"] = directory_manifest(export / "code") == capsule["code_manifest"]
            local["data_manifest_match"] = directory_manifest(export / "data") == capsule["data_manifest"]
            if not local["code_manifest_match"]:
                issues.append(f"{paper_id}: code directory manifest mismatch")
            if not local["data_manifest_match"]:
                issues.append(f"{paper_id}: data directory manifest mismatch")
        elif require_local and not export.is_dir():
            issues.append(f"{paper_id}: extracted capsule is missing")
        status[paper_id] = local
    return {
        "schema_version": 1,
        "paper_exact_cards": len(card_rows),
        "capsules": status,
        "issues": issues,
    }, issues


def result_dir(card: dict[str, Any]) -> Path:
    target = ROOT / card["result_tracks"]["paper_exact_result"]
    return target.parent if target.suffix else target


def native_python(capsule: dict[str, Any]) -> Path:
    environment = ROOT / capsule["native_environment"]
    if os.name == "nt":
        return environment / "Scripts/python.exe"
    return environment / "bin/python"


def write_environment_snapshot(paper_id: str, engine: str) -> dict[str, Any]:
    card = cards()[paper_id]
    capsule = capsules()[paper_id]
    python = native_python(capsule)
    expected = card["official_artifact"]["environment"]
    package_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "sktime": "sktime",
        "imbalanced_learn": "imbalanced-learn",
        "mapie": "mapie",
    }
    query_names = [package_names[key] for key in expected if key != "python"]
    query = (
        "import importlib.metadata as m,json,platform,sys;"
        f"names={query_names!r};"
        "print(json.dumps({'python':platform.python_version(),"
        "'executable':sys.executable,'packages':{n:m.version(n) for n in names}}))"
    )
    observed = json.loads(
        subprocess.run(
            [str(python), "-c", query],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    )
    expected_packages = {
        package_names[key]: value for key, value in expected.items() if key != "python"
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    snapshot = {
        "schema_version": 1,
        "paper_id": paper_id,
        "engine": engine,
        "authoritative_engine": "docker",
        "native_compatibility_run": engine == "native-windows",
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "python": observed,
        "expected": {"python": expected["python"], "packages": expected_packages},
        "exact_dependency_match": observed["python"].startswith(expected["python"] + ".")
        and observed["packages"] == expected_packages,
        "docker_image": capsule["docker_image"],
        "dockerfile_hash_comment": capsule["dockerfile_hash_comment"],
        "git_revision_at_snapshot": revision,
        "capsule_archive_sha256": capsule["archive_sha256"],
    }
    output = result_dir(card)
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot


def link_or_copy(source: Path, destination: Path) -> bool:
    """Link a directory when permitted, otherwise copy it.

    Returns ``True`` for a live symlink and ``False`` for a detached copy.  The
    distinction matters for the writable ``/results`` mount on Windows: a
    detached staging copy must be synchronized back after the author process.
    """
    try:
        destination.symlink_to(source, target_is_directory=True)
        return True
    except OSError:
        shutil.copytree(source, destination)
        return False


def first_free_drive() -> str:
    for letter in "RSTUVWXYZ":
        candidate = f"{letter}:"
        if not Path(f"{candidate}\\").exists():
            return candidate
    raise RuntimeError("no free drive letter is available for native capsule execution")


def run_author(paper_id: str, engine: str) -> int:
    card = cards()[paper_id]
    capsule = capsules()[paper_id]
    export = ROOT / capsule["export_root"]
    output = result_dir(card)
    author_results = output / "author_results"
    if author_results.exists() and any(author_results.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty result directory: {author_results}")
    output.mkdir(parents=True, exist_ok=True)
    author_results.mkdir(exist_ok=True)
    log_path = output / "author_run.log"
    write_environment_snapshot(paper_id, engine)

    if engine == "docker":
        command = [
            "docker", "run", "--platform", "linux/amd64", "--rm",
            "--workdir", "/code",
            "--volume", f"{export / 'data'}:/data:ro",
            "--volume", f"{export / 'code'}:/code:ro",
            "--volume", f"{author_results}:/results",
            capsule["docker_image"], "bash", "run",
        ]
        cwd = ROOT
        drive = None
    elif engine == "native-windows":
        if os.name != "nt":
            raise RuntimeError("native-windows engine is available only on Windows")
        python = native_python(capsule)
        if not python.is_file():
            raise RuntimeError(f"exact native environment missing: {python}")
        staging = ROOT / "tmp/codeocean_runs" / f"{paper_id}_runner"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        link_or_copy(export / "code", staging / "code")
        link_or_copy(export / "data", staging / "data")
        results_linked = link_or_copy(author_results, staging / "results")
        drive = first_free_drive()
        subprocess.run(["subst", drive, str(staging)], check=True)
        cwd = Path(f"{drive}\\code")
        command = [str(python), "main.py"]
    else:
        raise ValueError(f"unsupported engine: {engine}")

    try:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8", newline="") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
    finally:
        if drive is not None:
            subprocess.run(["subst", drive, "/d"], check=False)
    if engine == "native-windows" and not results_linked:
        # Native Windows may reject directory symlinks. Preserve everything
        # the unchanged Capsule wrote to its staged absolute /results path.
        shutil.copytree(staging / "results", author_results, dirs_exist_ok=True)
    finished_at = datetime.now(timezone.utc)
    (output / "resource_usage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paper_id": paper_id,
                "engine": engine,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "wall_time_seconds": time.perf_counter() - started,
                "return_code": return_code,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if return_code:
        raise RuntimeError(f"author capsule failed with exit code {return_code}")
    print(f"[PASS] author run completed: {paper_id}")
    return 0


def csv_columns(path: Path) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if key and key != row.keys().__iter__().__next__() and value not in (None, ""):
                    output.setdefault(key, []).append(float(value.strip("[]")))
    return output


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(statistics.fmean(rows))


def numpy_vector_cells(path: Path) -> list[np.ndarray]:
    vectors = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            values = [value for key, value in row.items() if key and not key.startswith("Unnamed")]
            if len(values) != 1:
                raise ValueError(f"expected one data cell per row in {path}")
            vectors.append(np.fromstring(values[0].strip().strip("[]").replace("\n", " "), sep=" "))
    return vectors


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    return mean(float(np.mean(y_pred[y_true == label] == label)) for label in labels)


def casim_duplicate_excluded_sensitivity(author_results: Path) -> dict[str, Any]:
    from sklearn.model_selection import RepeatedStratifiedKFold

    data_root = ROOT / "data/public_datasets/codeocean/casim_v1/export/data"
    files = [
        name
        for name in os.listdir(data_root)
        if name not in {"labels.csv", "LICENSE"}
    ]
    identifiers = [name.split("_ALARMS")[0] for name in files]
    label_frame = {}
    with (data_root / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label_frame[str(row["ID"])] = int(row["Label"])
    labels = np.asarray([label_frame[identifier] for identifier in identifiers])
    predictions = numpy_vector_cells(author_results / "CASIM_predictions.csv")
    stored_truth = numpy_vector_cells(author_results / "CASIM_test_splits.csv")
    prior = read_json(ROOT / "experiments/reports/p0_codeocean_data_prior.json")
    groups = prior["datasets"]["casim_tep"]["duplicate_groups"]
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    folds = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    for fold, ((train, test), y_pred, y_saved) in enumerate(
        zip(splitter.split(np.zeros(len(labels)), labels), predictions, stored_truth),
        start=1,
    ):
        train_set = set(int(value) for value in train)
        test_known = [int(value) for value in test if labels[int(value)] != -1]
        y_true = labels[test_known]
        if not np.array_equal(y_true, y_saved.astype(int)):
            raise RuntimeError(f"stored CASIM test split differs from reconstructed fold {fold}")
        excluded_positions = []
        excluded_ids = []
        for offset, index in enumerate(test_known):
            identifier = identifiers[index]
            for group in groups:
                member_ids = [member["id"] for member in group["members"]]
                if identifier in member_ids and any(
                    positions[other] in train_set for other in member_ids if other != identifier
                ):
                    excluded_positions.append(offset)
                    excluded_ids.append(identifier)
        keep = np.ones(len(y_true), dtype=bool)
        keep[excluded_positions] = False
        folds.append(
            {
                "fold": fold,
                "excluded_test_ids_with_train_twin": excluded_ids,
                "original_balanced_accuracy": balanced_accuracy(y_true, y_pred.astype(int)),
                "duplicate_excluded_balanced_accuracy": balanced_accuracy(
                    y_true[keep], y_pred.astype(int)[keep]
                ),
            }
        )
    return {
        "interpretation": "post-hoc test sensitivity using the unchanged fitted author models; contaminated test rows are excluded, but models are not retrained",
        "folds": folds,
        "excluded_test_instances": sum(len(row["excluded_test_ids_with_train_twin"]) for row in folds),
        "mean_duplicate_excluded_balanced_accuracy": mean(
            row["duplicate_excluded_balanced_accuracy"] for row in folds
        ),
    }


def summarize_casim(author_results: Path) -> dict[str, Any]:
    rows = csv_columns(author_results / "CASIM_results.csv")
    values = next(iter(rows.values()))
    return {
        "capsule_default_protocol": "five-fold closed-set",
        "fold_balanced_accuracy": values,
        "mean_balanced_accuracy": mean(values),
        "median_balanced_accuracy": float(statistics.median(values)),
        "minimum_balanced_accuracy": min(values),
        "maximum_balanced_accuracy": max(values),
        "duplicate_excluded_test_sensitivity": casim_duplicate_excluded_sensitivity(
            author_results
        ),
    }


def summarize_cone(author_results: Path) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for model in ("WDI_1NN", "ACM_SVM", "CASIM", "EAC_1NN", "MBW_LR"):
        coverage = csv_columns(author_results / f"{model}_coverages.csv")
        set_size = csv_columns(author_results / f"{model}_avg_set_sizes.csv")
        coverage_avg = coverage["avg"]
        size_avg = set_size["avg"]
        models[model] = {
            "mean_coverage_all_prefixes": mean(coverage_avg),
            "mean_set_size_all_prefixes": mean(size_avg),
            "final_prefix_coverage": coverage_avg[-1],
            "final_prefix_set_size": size_avg[-1],
            "prefix_count": len(coverage_avg),
        }
    casim = models["CASIM"]
    return {
        "capsule_default_protocol": "five folds, alpha=0.05, calibration_per_class=22",
        "models": models,
        # Retain the original CASIM aliases for schema compatibility with the
        # first P0 summary and downstream report readers.
        **casim,
    }


def summarize_bip(author_results: Path, log_path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    for dataset in ("tep", "synthetic"):
        output[dataset] = {}
        for model in ("ACM_SVM", "EAC_1NN", "MBW_LR", "CASIM"):
            mae = csv_columns(author_results / dataset / f"{model}_mae_results.csv")
            coverage = csv_columns(author_results / dataset / f"{model}_coverage_results.csv")
            fold_keys = [key for key in mae if key.startswith("Fold_")]
            point_values = [value for key in fold_keys for value in mae[key]]
            coverage_errors: list[float] = []
            with (author_results / dataset / f"{model}_coverage_results.csv").open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    alpha = float(row["Alpha"])
                    coverage_errors.extend(abs(float(row[key]) - (1.0 - alpha)) for key in fold_keys)
            counts = [
                int(value)
                for value in re.findall(
                    rf"\[{dataset} - {model}\] Fold.*?Number of bifurcations in test data:\s+(\d+)",
                    log_text,
                    flags=re.DOTALL,
                )
            ]
            output[dataset][model] = {
                "mean_MAE_points": mean(point_values),
                "mean_MAE_coverage": mean(coverage_errors),
                "mean_test_bifurcations": mean(counts) if counts else None,
                "folds": len(fold_keys),
                "alpha_points": len(coverage.get("Fold_1", [])),
            }
    return output


def compare_to_paper(paper_id: str, card: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if paper_id == "faulwasser2024_casim":
        observed = {
            "CASIM median TPR": metrics["median_balanced_accuracy"],
            "CASIM TPR range": metrics["maximum_balanced_accuracy"] - metrics["minimum_balanced_accuracy"],
        }
        rows = []
        for target in card["paper_targets"]:
            value = observed.get(target["metric"])
            rows.append(
                {
                    "item": target["item"],
                    "metric": target["metric"],
                    "paper_value": target["value"],
                    "author_capsule_default": value,
                    "delta": None if value is None else value - target["value"],
                    "numeric_within_tolerance": None if value is None else abs(value - target["value"]) <= card["tolerances"]["classification_absolute"],
                    "protocol_match": False,
                    "closed": False,
                }
            )
        return rows
    if paper_id == "faulwasser2024_cone_afc":
        rows = []
        table_1 = card["reference_tables"]["Table_1_accuracy_and_coverage"]
        table_2 = card["reference_tables"]["Table_2_average_set_size"]
        for model, observed in metrics["models"].items():
            for item, metric, paper_pair in (
                ("Table 1", "coverage", table_1[model]["coverage"]["0.05/22"]),
                ("Table 2", "average_set_size", table_2[model]["0.05/22"]),
            ):
                value = observed[
                    "mean_coverage_all_prefixes"
                    if metric == "coverage"
                    else "mean_set_size_all_prefixes"
                ]
                delta = value - paper_pair[0]
                rows.append(
                    {
                        "item": item,
                        "model": model,
                        "metric": metric,
                        "alpha": 0.05,
                        "calibration_per_class": 22,
                        "paper_mean": paper_pair[0],
                        "paper_std": paper_pair[1],
                        "author_capsule_default": value,
                        "delta": delta,
                        "numeric_within_tolerance": abs(delta)
                        <= card["tolerances"]["mean_absolute"],
                        "protocol_match": False,
                        "closed": False,
                    }
                )
        return rows
    rows = []
    for target in card["paper_targets"]:
        dataset = target["dataset"]
        model = target["model"]
        observed = metrics[dataset][model]
        if target["metric"] == "MAE":
            for metric, target_key, tolerance_key in (
                ("MAE_coverage", "coverage_mean", "MAE_coverage_absolute"),
                ("MAE_points", "points_mean", "MAE_points_relative"),
            ):
                value = observed[f"mean_{metric}"]
                paper_value = target[target_key]
                delta = value - paper_value
                tolerance = card["tolerances"][tolerance_key]
                within = abs(delta) <= tolerance if metric == "MAE_coverage" else abs(delta) <= tolerance * paper_value
                rows.append(
                    {
                        "item": target["item"],
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "paper_mean": paper_value,
                        "author_capsule_default": value,
                        "delta": delta,
                        "numeric_within_tolerance": within,
                        "protocol_match": False,
                        "closed": False,
                    }
                )
        elif observed["mean_test_bifurcations"] is not None:
            value = observed["mean_test_bifurcations"]
            paper_value = target["test_mean"]
            delta = value - paper_value
            rows.append(
                {
                    "item": target["item"],
                    "dataset": dataset,
                    "model": model,
                    "metric": "test_bifurcations",
                    "paper_mean": paper_value,
                    "author_capsule_default": value,
                    "delta": delta,
                    "numeric_within_tolerance": abs(delta) <= card["tolerances"]["bifurcation_count_relative"] * paper_value,
                    "protocol_match": False,
                    "closed": False,
                }
            )
    return rows


def summarize(paper_id: str) -> dict[str, Any]:
    card = cards()[paper_id]
    capsule = capsules()[paper_id]
    card_path = CARD_ROOT / f"{paper_id}.v1.json"
    output = result_dir(card)
    author_results = output / "author_results"
    if paper_id == "faulwasser2024_casim":
        metrics = summarize_casim(author_results)
    elif paper_id == "faulwasser2024_cone_afc":
        metrics = summarize_cone(author_results)
    else:
        metrics = summarize_bip(author_results, output / "author_run.log")
    result = {
        "schema_version": 1,
        "paper_id": paper_id,
        "reproduction_level": "P2_author_capsule_default",
        "paper_exact_closed": False,
        "capsule": {
            "doi": capsule["doi"],
            "archive_sha256": capsule["archive_sha256"],
            "code_manifest": capsule["code_manifest"],
            "data_manifest": capsule["data_manifest"],
        },
        "protocol_card": str(card_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_card_sha256": sha256_file(card_path),
        "author_code_unchanged": True,
        "environment": {
            "path": str((output / "environment.json").relative_to(ROOT)).replace("\\", "/"),
            "exact_dependency_match": read_json(output / "environment.json")["exact_dependency_match"] if (output / "environment.json").is_file() else None,
            "native_compatibility_run": read_json(output / "environment.json")["native_compatibility_run"] if (output / "environment.json").is_file() else None,
        },
        "metrics": metrics,
        "paper_comparison": compare_to_paper(paper_id, card, metrics),
        "known_mismatches": card["known_mismatches"],
        "next_stage": "parameterize the complete paper grid without changing preprocessing or splits, then compare the independent implementation on identical folds",
    }
    target = output / "final_info.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def print_status() -> None:
    for paper_id, card in cards().items():
        final = result_dir(card) / "final_info.json"
        state = "author_default_summarized" if final.is_file() else "author_default_pending"
        print(f"{paper_id}: transfer=complete; paper_exact={state}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--require-local", action="store_true")
    check.add_argument("--full-hash", action="store_true")
    subparsers.add_parser("status")
    author = subparsers.add_parser("run-author")
    author.add_argument("--paper-id", required=True, choices=PAPER_IDS)
    author.add_argument("--engine", choices=("docker", "native-windows"), default="native-windows" if os.name == "nt" else "docker")
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--paper-id", required=True, choices=PAPER_IDS)
    environment = subparsers.add_parser("snapshot-environment")
    environment.add_argument("--paper-id", required=True, choices=PAPER_IDS)
    environment.add_argument("--engine", choices=("docker", "native-windows"), default="native-windows" if os.name == "nt" else "docker")
    args = parser.parse_args()

    if args.command == "check":
        audit, issues = validate(args.require_local, args.full_hash)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 1 if issues else 0
    if args.command == "status":
        print_status()
        return 0
    if args.command == "run-author":
        return run_author(args.paper_id, args.engine)
    if args.command == "snapshot-environment":
        print(json.dumps(write_environment_snapshot(args.paper_id, args.engine), ensure_ascii=False, indent=2))
        return 0
    summarize(args.paper_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
