"""Validate config references, extracted-book boundaries, and source registry shape."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from iia_benchmark.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = (
        "README.md",
        "pyproject.toml",
        "configs/datasets/public_sources.json",
        "papers/extracted_text/book/manifest.json",
        "knowledge_base/book/README.md",
        "docs/three_round_expansion.md",
        "configs/algorithms/book_algorithms.json",
        "configs/algorithms/sota_algorithms.json",
        "configs/tasks/downstream_tasks.json",
    )
    failures: list[str] = []
    for relative in required:
        if not (ROOT / relative).exists():
            failures.append(f"missing: {relative}")

    for path in sorted((ROOT / "configs" / "experiments").glob("*.json")):
        try:
            experiment = load_experiment_config(path)
            for field in ("system", "dataset", "split", "model", "metrics"):
                target = ROOT / experiment[field]
                if not target.is_file():
                    failures.append(f"{path.name}: missing {field} reference {experiment[field]}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{path.name}: {error}")

    manifest_path = ROOT / "papers" / "extracted_text" / "book" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("pdf_pages") != 433 or len(manifest.get("chapters", [])) != 6:
            failures.append("book manifest must contain 433 pages and six chapters")

    registry = json.loads(
        (ROOT / "configs" / "datasets" / "public_sources.json").read_text(encoding="utf-8")
    )
    identifiers = [source["id"] for source in registry.get("sources", [])]
    if len(identifiers) != len(set(identifiers)):
        failures.append("public dataset ids must be unique")
    rounds = {source.get("round") for source in registry.get("sources", [])}
    if rounds != {1, 2, 3}:
        failures.append(f"public registry rounds must equal {{1,2,3}}, got {rounds}")
    valid_roles = {"main", "documentation", "metadata"}
    for source in registry.get("sources", []):
        if not source.get("dataset_family"):
            failures.append(f"{source['id']}: missing dataset_family")
        if source.get("payload_role") not in valid_roles:
            failures.append(f"{source['id']}: invalid payload_role {source.get('payload_role')}")

    task_registry = json.loads(
        (ROOT / "configs" / "tasks" / "downstream_tasks.json").read_text(encoding="utf-8")
    )
    tasks = task_registry.get("tasks", [])
    task_ids = [item.get("id") for item in tasks]
    if task_ids != [f"T{index}" for index in range(1, 7)]:
        failures.append(f"downstream tasks must be ordered T1..T6, got {task_ids}")

    book_registry = json.loads(
        (ROOT / "configs" / "algorithms" / "book_algorithms.json").read_text(
            encoding="utf-8"
        )
    )

    sota_registry = json.loads(
        (ROOT / "configs" / "algorithms" / "sota_algorithms.json").read_text(encoding="utf-8")
    )
    sota = sota_registry.get("algorithms", [])
    sota_ids = [item.get("id") for item in sota]
    if len(sota_ids) != len(set(sota_ids)):
        failures.append("SOTA algorithm ids must be unique")
    for item in sota:
        if item.get("status") not in {"verified", "partial", "missing"}:
            failures.append(f"{item.get('id')}: invalid SOTA status {item.get('status')}")
        if item.get("status") == "missing" and item.get("local_implementation"):
            failures.append(f"{item.get('id')}: missing SOTA cannot name an implementation")

    for item, field in [
        *((item, "implementation") for item in book_registry.get("algorithms", [])),
        *((item, "local_implementation") for item in sota),
    ]:
        value = item.get(field)
        paths = value if isinstance(value, list) else [value] if isinstance(value, str) else []
        if not paths:
            failures.append(f"{item.get('id')}: no callable implementation registered")
            continue
        for dotted_path in paths:
            try:
                module_name, attribute = dotted_path.rsplit(".", 1)
                candidate = getattr(importlib.import_module(module_name), attribute)
                if not callable(candidate):
                    failures.append(f"{item.get('id')}: {dotted_path} is not callable")
            except (ImportError, AttributeError, ValueError) as error:
                failures.append(f"{item.get('id')}: cannot resolve {dotted_path}: {error}")

    if failures:
        print("\n".join(failures))
        return 1
    print("scaffold valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
