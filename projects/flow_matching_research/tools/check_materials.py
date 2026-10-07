"""Check registered project materials against a local benchmark asset root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def inspect(root, artifact, verify_hash=False):
    relative = artifact.get("path")
    if not relative:
        return "no_local_path"
    storage_source = root / 'src/iia_benchmark/config/storage.py'
    if storage_source.is_file():
        sys.path.insert(0, str(root / 'src'))
        from iia_benchmark.config.storage import resolve_project_path
        target = resolve_project_path(root, relative)
    else:
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Material path escapes asset root")
    if not target.exists():
        return "missing"
    if target.is_dir():
        return "directory_present_not_fully_verified"
    size = target.stat().st_size
    if not size or (artifact.get("bytes") is not None and size != artifact["bytes"]):
        return "invalid_size"
    with target.open("rb") as stream:
        prefix = stream.read(512).lstrip().lower()
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return "lfs_pointer"
    if artifact.get("format") != "html" and prefix.startswith((b"<!doctype html", b"<html")):
        return "html_instead_of_payload"
    if verify_hash and artifact.get("sha256"):
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            return "checksum_mismatch"
        return "sha256_verified"
    return "present_size_and_signature_checked"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=os.environ.get("IIA_BENCHMARK_ROOT"))
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "materials/catalog.json")
    parser.add_argument("--verify-hashes", action="store_true", help="Read every file, including large datasets")
    args = parser.parse_args()
    if args.asset_root is None:
        parser.error("Set IIA_BENCHMARK_ROOT or --asset-root")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    counts, unresolved = {}, []
    for artifact in catalog["artifacts"]:
        status = inspect(args.asset_root, artifact, args.verify_hashes)
        counts[status] = counts.get(status, 0) + 1
        if status not in {"present_size_and_signature_checked", "sha256_verified", "directory_present_not_fully_verified"}:
            unresolved.append({"id": artifact["id"], "status": status, "path": artifact.get("path")})
    known_gaps = catalog.get("unresolved_assets", [])
    print(json.dumps({"counts": counts, "unresolved": unresolved, "registered_gaps": known_gaps,
                      "boundary": "Presence check alone does not verify all publisher checksums, protocol splits or reproduction results"}, ensure_ascii=False, indent=2))
    return int(bool(unresolved or known_gaps))


if __name__ == "__main__":
    raise SystemExit(main())
