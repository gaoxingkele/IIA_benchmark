"""Check core point/PA/AUC/AP contracts against pinned TAB function bodies.

Extracts only named functions to avoid importing TAB's complete training stack.
Random fixtures validate evaluator semantics, not benchmark performance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from iia_benchmark.evaluation.mtsad_metrics import point_adjust, pointwise_metrics


def load_functions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in names]
    if {item.name for item in functions} != set(names):
        raise ValueError("Pinned TAB function contract changed")
    namespace = {"np": np, "metrics": metrics}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproducibility/mtsad_protocol_sources.v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = ROOT / config["output_root"]
    source = base / "sources/tab/original/ts_benchmark/evaluation/metrics"
    label_file = source / "classification_metrics_label.py"
    score_file = source / "classification_metrics_score.py"
    labels = load_functions(label_file, {"f_score", "adjust_predicts", "adjust_f_score"})
    scores = load_functions(score_file, {"auc_roc", "auc_pr"})
    rng = np.random.default_rng(20261007)
    errors = {key: 0.0 for key in ("f_score", "adjust_f_score", "auc_roc", "auc_pr")}
    for _ in range(32):
        truth = rng.integers(0, 2, 200)
        raw = rng.normal(size=200)
        predicted = (raw > .5).astype(int)
        expected = {
            "f_score": pointwise_metrics(truth, predicted)["f1"],
            "adjust_f_score": pointwise_metrics(truth, point_adjust(truth, predicted))["f1"],
            "auc_roc": metrics.roc_auc_score(truth, raw),
            "auc_pr": metrics.average_precision_score(truth, raw),
        }
        actual = {key: (labels if key in labels else scores)[key](truth, predicted if key in labels else raw) for key in errors}
        for key in errors:
            errors[key] = max(errors[key], float(abs(actual[key] - expected[key])))
    if max(errors.values()) > 1e-12:
        raise AssertionError(errors)
    record = {
        "status": "passed", "fixture_count": 32, "seed": 20261007,
        "maximum_absolute_errors": errors,
        "tab_commit": next(p["author_commit"] for p in config["papers"] if p["id"] == "tab"),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (label_file, score_file)},
        "boundary": "Named original function bodies on randomized binary fixtures only; not full TAB execution, VUS/Affiliation verification or benchmark evidence. TAB auc_pr is average precision.",
    }
    target = base / "metric_contract_check.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
