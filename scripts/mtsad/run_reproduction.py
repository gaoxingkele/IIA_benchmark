"""Re-run the SWaT/MSL/SMAP/PSM/SMD benchmark under explicit protocol choices.

The script is deliberately explicit about the parts of the reference harnesses
that change the numbers: the window-to-timestamp layout, the threshold rule,
and whether point adjustment is applied.  Every run records which combination
was used so a comparison against a published table is never made by accident.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data.mtsad import (  # noqa: E402
    load_dataset_config,
    load_mtsad_split,
    standardise,
    stack_windows,
    window_starts,
)
from iia_benchmark.evaluation.mtsad_metrics import (  # noqa: E402
    evaluate_scores,
    percentile_threshold,
)
from iia_benchmark.models.mtsad_detectors import build_detector  # noqa: E402


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


def load_model_config(name: str) -> dict:
    path = ROOT / "configs" / "models" / f"mtsad_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing model config for {name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flattened_view(
    values: np.ndarray, window: int, step: int
) -> np.ndarray:
    """Replicate the Time-Series-Library concatenated-window layout."""

    starts = window_starts(len(values), window, step)
    index = starts[:, None] + np.arange(window)[None, :]
    return values[index].reshape(-1)


def window_scores_to_flat(scores: np.ndarray, window: int) -> np.ndarray:
    """Map window scores to the concatenated layout used by the reference harness.

    Detectors may score a whole window at once (one value per window, as USAD
    does) or every timestamp inside the window (as the Anomaly Transformer
    does, whose association-discrepancy energy is per position).
    """

    if scores.ndim == 1:
        return np.repeat(scores, window)
    if scores.ndim == 2 and scores.shape[1] == window:
        return scores.reshape(-1)
    raise ValueError(f"unexpected window-score shape {scores.shape}")


def align_scores_to_starts(scores: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Tolerate detectors that score fewer windows than the grid provides.

    GCAD cannot score the last ``pred_len`` windows because their forecast target
    lies beyond the series, so its scores correspond to the *first*
    ``len(scores)`` starts.  Refusing that would force a padding hack inside the
    detector; aligning explicitly keeps the correspondence honest.
    """

    if len(scores) == len(starts):
        return starts
    if len(scores) < len(starts):
        return starts[: len(scores)]
    raise ValueError("the detector returned more scores than there are windows")


def window_scores_to_per_point(
    scores: np.ndarray, window: int, step: int, length: int
) -> np.ndarray:
    if scores.ndim == 1:
        final = scores
    elif scores.ndim == 2 and scores.shape[1] == window:
        final = scores[:, -1]
    else:
        raise ValueError(f"unexpected window-score shape {scores.shape}")
    starts = align_scores_to_starts(final, window_starts(length, window, step))
    totals = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    np.add.at(totals, starts + window - 1, final)
    np.add.at(counts, starts + window - 1, 1)
    out = np.full(length, np.nan, dtype=np.float64)
    covered = counts > 0
    out[covered] = totals[covered] / counts[covered]
    return out


def per_point_view(scores: np.ndarray, window: int, step: int, length: int) -> np.ndarray:
    starts = window_starts(length, window, step)
    totals = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    np.add.at(totals, starts + window - 1, scores)
    np.add.at(counts, starts + window - 1, 1)
    out = np.full(length, np.nan, dtype=np.float64)
    covered = counts > 0
    out[covered] = totals[covered] / counts[covered]
    return out


def run_single(
    *,
    dataset: str,
    model: str,
    seed: int,
    protocols: dict,
    device: str,
    epoch_scale: float = 1.0,
    cache: dict | None = None,
    save_scores: bool = False,
) -> dict:
    cache = cache if cache is not None else {}
    if dataset not in cache:
        split = load_mtsad_split(dataset, root=ROOT / "data/public_datasets/mtsad")
        cache[dataset] = split
    split = cache[dataset]
    config = load_dataset_config(dataset, ROOT / "configs/datasets")
    model_config = load_model_config(model)
    # A detector may legitimately need a different window or alarm budget per
    # dataset (the reference DCdetector scripts use 105/90/60 windows and
    # per-dataset anomaly ratios).  The override is declared in the model config
    # and recorded on every run so the comparison stays attributable.
    overrides = (model_config.get("dataset_overrides") or {}).get(dataset, {})
    window = int(overrides.get("window", config["window"]))
    step = int(overrides.get("window_step", config["window_step"]))
    anomaly_ratio = float(overrides.get("anomaly_ratio", config["anomaly_ratio"]))
    train_z, test_z, stats = standardise(split)

    parameters = dict(model_config.get("parameters") or {})
    if "window" in parameters:
        parameters["window"] = window
    for key, value in overrides.items():
        if key in {"window", "window_step", "anomaly_ratio"}:
            continue
        if key in parameters:
            parameters[key] = value
    if "seed" in parameters:
        parameters["seed"] = seed
    if "device" not in parameters and model_config.get("kind") == "window":
        parameters["device"] = device
    if epoch_scale != 1.0 and "epochs" in parameters:
        parameters["epochs"] = max(1, int(round(parameters["epochs"] * epoch_scale)))

    detector = build_detector(model, parameters)
    started = time.time()
    if detector.kind == "window":
        # float32 keeps the materialised sliding-window tensors within a few GB
        # even for the 400k+ window test splits.
        train_windows = stack_windows(train_z.astype(np.float32), window, step)
        test_windows = stack_windows(test_z.astype(np.float32), window, step)
        detector.fit(train_windows)
        train_scores = detector.score(train_windows)
        test_scores = detector.score(test_windows)
        layouts = {
            "window_flatten": (
                window_scores_to_flat(train_scores, window),
                window_scores_to_flat(test_scores, window),
                flattened_view(split.test_labels.astype(np.float64), window, step) > 0.5,
            ),
            "last_point": (
                window_scores_to_per_point(train_scores, window, step, len(train_z)),
                window_scores_to_per_point(test_scores, window, step, len(test_z)),
                split.test_labels,
            ),
        }
        # A detector may score fewer windows than the grid provides (GCAD cannot
        # score the windows whose forecast target falls past the end of the
        # series).  Its scores then belong to the first windows, so the flattened
        # label stream is truncated to the same length instead of refusing the run.
        flat_train, flat_test, flat_truth = layouts["window_flatten"]
        if len(flat_truth) > len(flat_test):
            layouts["window_flatten"] = (
                flat_train[: len(flat_test)],
                flat_test,
                flat_truth[: len(flat_test)],
            )
    else:
        detector.fit(train_z)
        train_scores = detector.score(train_z)
        test_scores = detector.score(test_z)
        layouts = {
            "last_point": (train_scores, test_scores, split.test_labels),
            "window_flatten": (train_scores, test_scores, split.test_labels),
        }
    fit_seconds = time.time() - started

    results = {}
    for name, protocol in protocols.items():
        layout = protocol["aggregation"]
        if detector.kind == "point":
            layout = "last_point"
        if layout not in layouts:
            continue
        train_layout, test_layout, truth_layout = layouts[layout]
        covered = np.isfinite(test_layout)
        if not covered.all():
            test_used = np.where(covered, test_layout, -np.inf)
            truth_used = truth_layout[covered]
            scores_used = test_layout[covered]
        else:
            test_used = test_layout
            truth_used = truth_layout
            scores_used = test_layout
        if protocol["threshold"] == "percentile_combined":
            train_finite = train_layout[np.isfinite(train_layout)]
            threshold = percentile_threshold(train_finite, test_used, anomaly_ratio)
        elif protocol["threshold"] == "best_f1":
            from iia_benchmark.evaluation.mtsad_metrics import best_f1_threshold

            threshold = best_f1_threshold(truth_used, scores_used)["threshold"]
        else:
            raise ValueError(f"unknown threshold rule: {protocol['threshold']}")
        metrics = evaluate_scores(
            truth=truth_used,
            scores=scores_used,
            threshold=threshold,
            point_adjustment=protocol["point_adjustment"],
        )
        metrics.update(
            {
                "protocol": name,
                "layout": layout,
                "threshold_rule": protocol["threshold"],
                "point_adjustment": protocol["point_adjustment"],
                "scored_points": int(covered.sum()),
                "anomaly_rate_scored": float(truth_used.mean()),
            }
        )
        results[name] = metrics

    record = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "window": window,
        "window_step": step,
        "anomaly_ratio": anomaly_ratio,
        "window_override": {key: value for key, value in overrides.items()},
        "parameters": parameters,
        "train_shape": list(split.train.shape),
        "test_shape": list(split.test.shape),
        "test_anomaly_rate": split.anomaly_rate,
        "repairs": split.repairs or {},
        "degenerate_channels": int(np.sum(stats["degenerate_channels"])),
        "fit_seconds": fit_seconds,
        "device": device,
        "protocols": results,
    }
    if save_scores:
        record["_layouts"] = layouts
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "mtsad_reproduction.json",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--epoch-scale",
        type=float,
        default=1.0,
        help="Scale deep-model epochs (smoke runs use a fraction of the reference budget).",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Persist the per-layout train/test scores next to the run record so "
        "alternative thresholds can be re-derived without retraining.",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = args.datasets or experiment["datasets"]
    models = args.models or experiment["models"]
    seeds = args.seeds or experiment["seeds"]
    protocols = experiment["protocols"]
    out_dir = args.out_dir or ROOT / experiment["output_dir"]
    if args.tag:
        out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    revision = git_revision()
    cache: dict = {}
    records = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                started = time.time()
                record = run_single(
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    protocols=protocols,
                    device=args.device,
                    epoch_scale=args.epoch_scale,
                    cache=cache,
                    save_scores=args.save_scores,
                )
                layouts = record.pop("_layouts", None)
                record["wall_seconds"] = time.time() - started
                record["git_revision"] = revision
                record["python"] = platform.python_version()
                records.append(record)
                path = out_dir / f"{dataset}__{model}__seed{seed}.json"
                if layouts is not None:
                    arrays = {}
                    for layout_name, (train_layout, test_layout, truth_layout) in layouts.items():
                        arrays[f"{layout_name}__train"] = np.asarray(train_layout, dtype=np.float64)
                        arrays[f"{layout_name}__test"] = np.asarray(test_layout, dtype=np.float64)
                        arrays[f"{layout_name}__truth"] = np.asarray(truth_layout, dtype=bool)
                    np.savez_compressed(
                        out_dir / f"{dataset}__{model}__seed{seed}__scores.npz", **arrays
                    )
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary = record["protocols"].get(experiment["default_protocol"], {})
                print(
                    f"{dataset:<5} {model:<20} seed={seed} "
                    f"f1_pa={summary.get('adjusted_f1', float('nan')):.4f} "
                    f"f1_pt={summary.get('pointwise_f1', float('nan')):.4f} "
                    f"({record['fit_seconds']:.1f}s fit, {record['wall_seconds']:.1f}s wall)",
                    flush=True,
                )

    combined = {
        "schema_version": 1,
        "experiment": experiment["id"],
        "git_revision": revision,
        "device": args.device,
        "epoch_scale": args.epoch_scale,
        "protocols": protocols,
        "records": records,
    }
    (out_dir / "records.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'records.json'} ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
