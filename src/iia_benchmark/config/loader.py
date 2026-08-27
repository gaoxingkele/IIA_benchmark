from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENT_FIELDS = {
    "id",
    "task",
    "domain",
    "system",
    "dataset",
    "split",
    "model",
    "metrics",
    "outputs",
}


def load_json_reference(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config = load_json_reference(path)
    missing = REQUIRED_EXPERIMENT_FIELDS - set(config)
    if missing:
        raise ValueError(f"Experiment config missing fields: {sorted(missing)}")
    return config

