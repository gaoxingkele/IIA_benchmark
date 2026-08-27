"""Validate config references, extracted-book boundaries, and source registry shape."""

from __future__ import annotations

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

    if failures:
        print("\n".join(failures))
        return 1
    print("scaffold valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
