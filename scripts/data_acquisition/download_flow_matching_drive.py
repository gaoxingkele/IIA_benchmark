"""Enumerate public author Drive folders, preserve existing files, and audit downloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import gdown

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.config.storage import resolve_project_path, project_relative_path
REGISTRY = ROOT / "configs/acquisition/flow_matching_drive_sources.json"
MANIFEST = ROOT / "papers/literature/flow_matching/drive_download_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def inspect(path, fmt=None):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("missing or empty payload")
    with path.open("rb") as stream:
        prefix = stream.read(512)
    if prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("HTML returned instead of data")
    signatures = {".npy": b"\x93NUMPY", ".h5": b"\x89HDF\r\n\x1a\n", ".zip": b"PK"}
    suffix = "." + fmt if fmt else path.suffix.lower()
    expected = signatures.get(suffix)
    if expected and not prefix.startswith(expected):
        raise ValueError("payload signature mismatch")
    if suffix == ".npy":
        import numpy as np
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran_order, dtype = reader(stream)
            if not dtype.hasobject:
                required = stream.tell() + int(np.prod(shape)) * dtype.itemsize
                if path.stat().st_size != required:
                    raise ValueError(f"NPY size mismatch: expected {required}, got {path.stat().st_size}")
    if suffix == ".h5":
        import h5py
        with h5py.File(path, "r") as archive:
            if not list(archive.keys()):
                raise ValueError("empty HDF5 archive")
    return {"bytes": path.stat().st_size, "sha256": sha256(path),
            "integrity_basis": "local SHA256; publisher checksum not supplied"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:17890")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reuse-inventory", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed file records once; preserve completed audit records")
    parser.add_argument("--retry-after-pid", type=int, help="Persistently wait for this Drive queue process, then retry failed files")
    parser.add_argument("--retry-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.retry_after_pid:
        import psutil
        print(f"Waiting for Drive queue PID {args.retry_after_pid}", flush=True)
        while psutil.pid_exists(args.retry_after_pid):
            time.sleep(10)
        for retry_round in range(1, args.retry_rounds + 1):
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
            failures = [r for r in manifest.get("records", []) if r.get("status") == "failed" and r.get("phase") != "enumerate"]
            if not failures:
                print("No failed Drive file records remain; retry supervisor complete", flush=True)
                return 0
            print(f"Drive retry round {retry_round}/{args.retry_rounds}: {len(failures)} failed files", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reuse-inventory", "--retry-failed",
                            "--workers", str(args.workers), "--proxy", args.proxy], cwd=ROOT, check=False)
        print("Retry limit reached; inspect drive_download_manifest.json for remaining failures", flush=True)
        return 0
    previous_records = json.loads(MANIFEST.read_text(encoding="utf-8")).get("records", []) if args.retry_failed and MANIFEST.exists() else []
    comparison = json.loads((ROOT / "configs/acquisition/flow_matching_comparison_data_sources.json").read_text(encoding="utf-8"))
    folders = [s for s in comparison["sources"] if s["id"] in ("dcrnn_traffic_bundle", "dcdetector_bundle")]
    sources, records, excluded = [], [], []

    def save():
        write_json(REGISTRY, {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Public official-author Google Drive folder file inventory", "sources": sources,
            "out_of_scope_sources": excluded,
            "download_scope": "DCRNN METR-LA/PeMS-BAY; DCdetector MSL, SMAP, SMD, PSM, SWAT, UCR, NIPS_TS_Creditcard, NIPS_TS_GECCO, NIPS_TS_Swan. UCR_AUG and unrelated benchmark archives excluded, not missing.",
            "enumeration_records": [r for r in records if r.get("phase") == "enumerate"]})
        write_json(MANIFEST, {"generated_at": datetime.now(timezone.utc).isoformat(),
            "registry": REGISTRY.relative_to(ROOT).as_posix(), "selected": len(sources),
            "completed": sum(r.get("phase") != "enumerate" for r in records), "records": records,
            "summary": {status: sum(r["status"] == status for r in records)
                        for status in sorted({r["status"] for r in records})}})

    folder_work = folders
    if args.reuse_inventory and REGISTRY.exists():
        previous = json.loads(REGISTRY.read_text(encoding="utf-8"))
        sources = previous.get("sources", []) + previous.get("out_of_scope_sources", [])
        records = previous.get("enumeration_records", [])
        folder_work = []
    for folder in folder_work:
        print("Enumerating", folder["id"], flush=True)
        try:
            root = resolve_project_path(ROOT, folder["path"])
            if not root.is_relative_to(ROOT / "data/public_datasets/flow_matching"):
                raise ValueError("folder target outside authorized data tree")
            entries = gdown.download_folder(url=folder["url"], output=str(root),
                proxy=args.proxy, skip_download=True, quiet=True, use_cookies=False)
            for entry in entries:
                target = resolve_project_path(ROOT, entry.local_path)
                if not target.is_relative_to(root):
                    raise ValueError("unsafe enumerated local path")
                sources.append({"id": "drive_" + entry.id, "drive_file_id": entry.id,
                    "url": "https://drive.google.com/uc?export=download&id=" + entry.id,
                    "path": project_relative_path(ROOT, target), "format": target.suffix.lstrip(".").lower(),
                    "checksum": None, "evidence_url": folder["evidence_url"],
                    "paper_ids": folder["paper_ids"], "access": "public", "parent_source_id": folder["id"],
                    "reproduction_boundary": "Official author distribution; exact experiment subset and split agreement still require audit."})
            records.append({"id": folder["id"], "phase": "enumerate", "status": "listed",
                            "files": len(entries), "evidence_url": folder["evidence_url"]})
        except Exception as exc:
            records.append({"id": folder["id"], "phase": "enumerate", "status": "failed",
                            "reason": str(exc), "url": folder["url"]})
        save()
    allowed = {"MSL", "SMAP", "SMD", "PSM", "SWAT", "UCR", "NIPS_TS_Creditcard", "NIPS_TS_GECCO", "NIPS_TS_Swan"}
    selected = []
    for source in sources:
        family = source["path"].split("author_bundle/", 1)[-1].split("/", 1)[0]
        if source.get("parent_source_id") == "dcrnn_traffic_bundle" or family in allowed:
            source["download_scope"] = "requested_paper_data"
            selected.append(source)
        else:
            source["download_scope"] = "excluded_unrelated_or_augmented"
            source["scope_reason"] = "Author folder inventory only; outside requested paper experiments. Not a missing acquisition."
            excluded.append(source)
    # Start direct benchmark numerical arrays before larger traffic h5 files.
    sources = sorted(selected, key=lambda source: ("author_bundle/UCR/" in source["path"], source.get("format") != "npy", source["path"]))
    save()
    if args.list_only:
        return 0
    def acquire(source):
        target = ROOT / source["path"]
        record = dict(source)
        try:
            if target.exists():
                details = inspect(target, source.get("format"))
                backend = "existing_preserved"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Retain old failed/interrupted transfers for diagnosis. Start
                # fresh: this proxy can return a tiny HTTP500 body on resumed
                # Google transfers, which gdown otherwise treats as complete.
                partial = target.with_name(target.name + ".transfer-" + uuid.uuid4().hex + ".part")
                result = gdown.download(id=source["drive_file_id"], output=str(partial),
                    proxy=args.proxy, quiet=True, use_cookies=False, resume=False)
                if not result:
                    raise RuntimeError("gdown returned no payload; public download may be unavailable or quota-limited")
                # Preserve registered suffix for signature validation of partial files.
                if target.suffix.lower() in (".npy", ".h5", ".zip"):
                    with partial.open("rb") as stream:
                        prefix = stream.read(8)
                    expected = {".npy": b"\x93NUMPY", ".h5": b"\x89HDF\r\n\x1a\n", ".zip": b"PK"}[target.suffix.lower()]
                    if not prefix.startswith(expected):
                        raise ValueError("downloaded partial signature mismatch")
                details = inspect(partial, source.get("format"))
                if target.exists():
                    raise FileExistsError("target appeared during download; existing file preserved")
                partial.rename(target)
                backend = "gdown_public_drive"
            record.update(status="available", backend=backend, **details)
        except Exception as exc:
            record.update(status="failed", reason=str(exc))
        return record
    pending_sources = sources
    if args.retry_failed:
        failed_ids = {r["id"] for r in previous_records if r.get("status") == "failed" and r.get("phase") != "enumerate"}
        pending_sources = [source for source in sources if source["id"] in failed_ids]
        records = [r for r in previous_records if r["id"] not in failed_ids]
        save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(acquire, source) for source in pending_sources]
        for task in as_completed(tasks):
            record = task.result()
            records.append(record)
            save()
            print(record["status"], record["id"], record["path"], flush=True)
    return int(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
