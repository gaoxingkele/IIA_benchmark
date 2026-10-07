"""Audit immutable registered downloads without modifying downloaded files.

Git-hosted regular files use Git blob SHA1, not the SHA1 of file bytes alone.
LFS files use the publisher's SHA256. Successful audits always record SHA256.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.config.storage import resolve_project_path
DEFAULT_REGISTRY = "configs/acquisition/fm_project_moment_pile.v1.json"
DEFAULT_REPORT = "papers/literature/flow_matching/project_moment_pile_integrity.json"


def contained_path(root: Path, value: str) -> Path:
    """Reject traversal and unauthorized links; allow configured data mounts."""
    return resolve_project_path(root, value, relative_only=True)


def source_signature(source: dict) -> str:
    fields = {key: source.get(key) for key in (
        "id", "path", "size_bytes", "checksum", "git_blob_sha1", "author_revision"
    )}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def verify_source(source: dict, root: Path, cached: dict | None = None) -> dict:
    record = {"id": source.get("id"), "path": source.get("path"),
              "author_revision": source.get("author_revision"),
              "source_signature": source_signature(source)}
    try:
        target = contained_path(root, source["path"])
        partials = [str(p.relative_to(root.resolve())) for p in (
            Path(str(target) + ".part"), Path(str(target) + ".part.aria2"),
            Path(str(target) + ".requests.part"), Path(str(target) + ".aria2")
        ) if p.exists()]
        record["partial_files"] = partials
        if not target.is_file():
            record["status"] = "partial" if partials else "missing"
            return record
        if Path(str(target) + ".aria2").exists():
            record["status"] = "partial"
            return record
        before = target.stat()
        record.update(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                      expected_size_bytes=source.get("size_bytes"))
        if source.get("size_bytes") is not None and before.st_size != source["size_bytes"]:
            record["status"] = "size_mismatch"
            return record
        expected = source.get("checksum") or {}
        expected_sha256 = expected.get("value") if expected.get("algorithm", "").lower() == "sha256" else None
        expected_blob = source.get("git_blob_sha1")
        record.update(expected_sha256=expected_sha256, expected_git_blob_sha1=expected_blob)
        if not expected_sha256 and not expected_blob:
            record["status"] = "unverifiable"
            return record
        if cached and cached.get("status") == "verified" and cached.get("sha256") and all(
            cached.get(k) == record.get(k) for k in ("source_signature", "size_bytes", "mtime_ns")
        ):
            record.update({k: cached[k] for k in ("sha256", "git_blob_sha1", "verified_at") if k in cached})
            record.update(status="verified", reused_previous_verification=True)
            return record
        sha = hashlib.sha256()
        blob = hashlib.sha1(b"blob " + str(before.st_size).encode("ascii") + b"\0")
        with target.open("rb") as stream:
            initial = stream.read(1024 * 1024)
            if initial.split(b"\n", 1)[0].rstrip(b"\r") == b"version https://git-lfs.github.com/spec/v1":
                record["status"] = "lfs_pointer"
                return record
            sha.update(initial)
            blob.update(initial)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                blob.update(block)
        after = target.stat()
        record.update(sha256=sha.hexdigest(), git_blob_sha1=blob.hexdigest(),
                      reused_previous_verification=False)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            record["status"] = "changing"
        elif ((expected_sha256 and record["sha256"] != expected_sha256.lower()) or
              (expected_blob and record["git_blob_sha1"] != expected_blob.lower())):
            record["status"] = "checksum_mismatch"
        else:
            record.update(status="verified", verified_at=datetime.now(timezone.utc).isoformat())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record.update(status="invalid", error=str(exc))
    return record


def audit_registry(config: dict, root: Path, previous: dict | None = None) -> dict:
    sources = config.get("sources", [])
    cached = {r.get("id"): r for r in (previous or {}).get("records", [])}
    records = [verify_source(source, root, cached.get(source.get("id"))) for source in sources]
    counts = Counter(r["status"] for r in records)
    declared = config.get("source_count", len(sources))
    identifiers = [s.get("id") for s in sources]
    paths = []
    for item in sources:
        try:
            paths.append(contained_path(root, item["path"]))
        except (ValueError, KeyError, TypeError):
            paths.append(item.get("path"))
    registered_bytes = sum(s.get("size_bytes") or 0 for s in sources)
    registry_valid = (declared == len(sources) and len(sources) > 0 and
                      config.get("total_bytes", registered_bytes) == registered_bytes and
                      len(set(identifiers)) == len(identifiers) and
                      len(set(paths)) == len(paths))
    complete = registry_valid and counts["verified"] == declared
    failures = sum(counts[k] for k in ("invalid", "checksum_mismatch", "size_mismatch", "lfs_pointer", "unverifiable"))
    return {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "scope": config.get("scope"), "upstream_revision": config.get("upstream_revision"),
            "verification_policy": "Publisher LFS SHA256 or Git blob SHA1; local SHA256 recorded; unchanged files reuse prior audit",
            "summary": {"complete": complete,
                        "status": "complete" if complete else "integrity_failed" if failures or not registry_valid else "running",
                        "registry_valid": registry_valid, "declared_source_count": declared,
                        "registered_source_count": len(sources), "status_counts": dict(counts),
                        "verified_count": counts["verified"],
                        "verified_lfs_sha256_count": sum(r["status"] == "verified" and bool(r.get("expected_sha256")) for r in records),
                        "verified_git_blob_sha1_count": sum(r["status"] == "verified" and bool(r.get("expected_git_blob_sha1")) for r in records),
                        "local_sha256_count": sum(r["status"] == "verified" and bool(r.get("sha256")) for r in records),
                        "registered_bytes": registered_bytes,
                        "verified_bytes": sum(r.get("size_bytes", 0) for r in records if r["status"] == "verified"),
                        "reused_count": sum(bool(r.get("reused_previous_verification")) for r in records)},
            "records": records}


def write_report(report: dict, output: Path, root: Path, sources: list[dict]) -> None:
    """Atomically replace verifier metadata, never a registered raw file."""
    output = output.resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError("Report must remain within the workspace")
    protected = {contained_path(root, s["path"]) for s in sources}
    temporary = output.with_name(output.name + ".tmp").resolve()
    if not temporary.is_relative_to(root.resolve()):
        raise ValueError("Temporary report must remain within the workspace")
    if output in protected or temporary in protected:
        raise ValueError("Report cannot overwrite registered raw data")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--watch", action="store_true", help="Audit until complete, with incremental verified-file reuse")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--force", action="store_true", help="Rehash every existing file in the initial scan")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("Watch interval must be at least 10 seconds")
    root = args.root.resolve()
    registry_path = contained_path(root, args.registry)
    output = contained_path(root, args.report)
    raw_registry = registry_path.read_bytes()
    config = json.loads(raw_registry)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() and not args.force else None
    registry_sha256 = hashlib.sha256(raw_registry).hexdigest()
    if previous and previous.get("registry_sha256") != registry_sha256:
        previous = None
    while True:
        report = audit_registry(config, root, previous)
        report.update(registry_path=args.registry, registry_sha256=registry_sha256)
        write_report(report, output, root, config["sources"])
        print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
        if report["summary"]["complete"]:
            return 0
        if not args.watch or report["summary"]["status"] == "integrity_failed":
            return 1
        previous = report
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
