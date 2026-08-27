"""Audit dataset presence, byte sizes, checksums, and git revisions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from download_public_datasets import ROOT, REGISTRY, checksum_ok


def main() -> int:
    with REGISTRY.open("r", encoding="utf-8") as stream:
        sources = json.load(stream)["sources"]
    rows = []
    failures = 0
    for source in sources:
        path = ROOT / source["path"]
        if source["kind"] == "git":
            present = (path / ".git").exists()
            revision = ""
            if present:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=path, text=True
                ).strip()
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if present else 0
            valid = present
        else:
            present = path.exists()
            size = path.stat().st_size if present else 0
            valid = checksum_ok(path, source.get("checksum"))
            revision = ""
        if source.get("default") and not valid:
            failures += 1
        rows.append(
            {
                "id": source["id"],
                "round": source["round"],
                "present": present,
                "valid": valid,
                "bytes": size,
                "revision": revision,
                "path": source["path"],
            }
        )
    output = ROOT / "data" / "public_datasets" / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['id']}: present={row['present']} valid={row['valid']} bytes={row['bytes']}")
    print(f"audit written: {output.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
