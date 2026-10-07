"""Acquire the registered FM literature/data bundle without replacing raw files."""
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
import threading
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import zipfile
import shutil

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.config.storage import resolve_project_path
sys.path.insert(0, str(Path(__file__).parent))
from download_public_datasets import find_aria2
HTTP_STATE = threading.local()


def http_session():
    if not hasattr(HTTP_STATE, "session"):
        HTTP_STATE.session = requests.Session()
        HTTP_STATE.session.trust_env = False
        HTTP_STATE.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, connect=3, read=3, backoff_factor=1,
            status_forcelist=(502, 503, 504), allowed_methods=("GET",))))
    return HTTP_STATE.session


def resolved_path(path, root=None):
    # Keep registered relative paths stable while a configured junction routes
    # the actual file access to external storage.
    try:
        return resolve_project_path(ROOT if root is None else root, path)
    except ValueError as error:
        raise ValueError(f'target outside repository or configured storage: {path}') from error


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(path, item):
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Git LFS pointer returned instead of actual dataset")
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("missing or empty payload")
    expected = item.get("checksum")
    if expected:
        algorithm, value = ((expected["algorithm"], expected["value"])
                            if isinstance(expected, dict) else expected.split(":", 1))
        if digest(path, algorithm) != value:
            raise ValueError("publisher checksum mismatch")
    if item.get("size_bytes") and path.stat().st_size != item["size_bytes"]:
        raise ValueError("publisher byte size mismatch")
    if item.get("format") == "pdf":
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("response is not a PDF")
        return {"pages": len(PdfReader(path).pages)}
    if item.get("format") in ("zip", "npz"):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"archive CRC failure: {bad}")
            return {"archive_members": len(archive.infolist())}
    signatures = {"npy": b"\x93NUMPY", "h5": b"\x89HDF\r\n\x1a\n",
                  "h5ad": b"\x89HDF\r\n\x1a\n", "png": b"\x89PNG\r\n\x1a\n"}
    if item.get("format") in signatures:
        with path.open("rb") as stream:
            magic = signatures[item["format"]]
            if stream.read(len(magic)) != magic:
                raise ValueError("payload signature does not match registered format")
    with path.open("rb") as stream:
        if item.get("format") != "html" and stream.read(512).lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError("HTML returned instead of dataset")
    return {}


def ranged_download(item, target, proxy):
    """Resume independent byte ranges for large public, short-lived signed URLs."""
    size = item["size_bytes"]
    parts = item.get("range_parts", 32)
    width = (size + parts - 1) // parts
    folder = target.with_name(target.name + ".chunks")
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(index):
        lo, hi = index * width, min(size, (index + 1) * width) - 1
        completed = folder / f"{index:04d}.bin"
        length = hi - lo + 1
        if completed.exists():
            if completed.stat().st_size != length:
                raise ValueError("preserved chunk has unexpected length")
            return completed
        partial = completed.with_suffix(".part")
        session = http_session()
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        last_error = None
        for attempt in range(5):
            start = lo + (partial.stat().st_size if partial.exists() else 0)
            if start > hi:
                if start != hi + 1:
                    raise ValueError("partial chunk exceeds range")
                partial.rename(completed)
                return completed
            parsed = urlsplit(item["url"])
            query = dict(parse_qsl(parsed.query))
            query[item.get("refresh_query", "cachebust")] = str(time.time_ns())
            url = urlunsplit(parsed._replace(query=urlencode(query)))
            try:
                with session.get(url, stream=True, timeout=(25, 90), headers={
                    "Range": f"bytes={start}-{hi}", "Cache-Control": "no-cache"}) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range", "") != f"bytes {start}-{hi}/{size}":
                        raise ValueError("server did not return the exact requested byte range")
                    with partial.open("ab") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if partial.stat().st_size != length:
                    raise ValueError("incomplete byte range")
                partial.rename(completed)
                print(f'range {item["id"]} {index + 1}/{parts} complete', flush=True)
                return completed
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 * (attempt + 1), 10))
        raise RuntimeError(f"range {index}: {last_error}")
    with ThreadPoolExecutor(max_workers=item.get("range_workers", 4)) as pool:
        chunks = list(pool.map(fetch, range(parts)))
    assembly = target.with_name(target.name + ".assembling")
    with assembly.open("wb") as output:
        for chunk in chunks:
            with chunk.open("rb") as stream:
                shutil.copyfileobj(stream, output, 1024 * 1024)
    details = validate(assembly, item)
    assembly.rename(target)
    return details


def acquire(item, proxy, audit_only=False):
    record = dict(item)
    target = resolved_path(ROOT / item["path"])
    if not target.is_relative_to(ROOT):
        raise ValueError(f"target outside repository: {target}; root={ROOT}")
    if item.get("format") == "directory" and target.is_dir():
        files = [{"path": p.relative_to(target).as_posix(), "bytes": p.stat().st_size,
                  "sha256": digest(p)} for p in sorted(target.rglob("*")) if p.is_file() and ".git" not in p.parts]
        return dict(record, status="available" if files else "missing", backend="existing_preserved",
                    bytes=sum(f["bytes"] for f in files), files=files,
                    integrity_basis="local SHA256 inventory of preserved files")
    if item.get("format") in {"gated", "manual", "synthetic_recipe", "drive_folder", "mega_folder", "share_link"}:
        return dict(record, status=item.get("access", "manual_download_required"))
    if not item.get("url") and not target.exists():
        return dict(record, status=item.get("access", "unavailable"))
    try:
        if target.exists():
            details = validate(target, item)
            backend = "existing_preserved"
        elif audit_only:
            return dict(record, status="missing")
        elif item.get("backend") == "range_requests":
            target.parent.mkdir(parents=True, exist_ok=True)
            details = ranged_download(item, target, proxy)
            backend = "requests_resumable_ranges"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            errors = []
            aria = find_aria2()
            use_aria = bool(aria and item.get("backend") != "requests")
            attempts = [(url, backend) for url in [item["url"], *item.get("fallback_urls", [])]
                        for backend in (["aria2c", "requests"] if use_aria else ["requests"])]
            for url, requested_backend in attempts:
                try:
                    partial = target.with_name(target.name + (".part" if requested_backend == "aria2c" else ".requests.part"))
                    if item.get("refresh_query"):
                        parsed = urlsplit(url)
                        query = dict(parse_qsl(parsed.query))
                        query[item["refresh_query"]] = str(time.time_ns())
                        url = urlunsplit(parsed._replace(query=urlencode(query)))
                    if requested_backend == "aria2c":
                        command = [aria, "--continue=true", "--allow-overwrite=false",
                                   "--auto-file-renaming=false", "--file-allocation=none",
                                   "--max-tries=2", "--retry-wait=3", "--timeout=45",
                                   "--connect-timeout=20", f'--split={item.get("connections", 4)}',
                                   f'--max-connection-per-server={item.get("connections", 4)}', "--console-log-level=warn",
                                   "--summary-interval=0", "--dir", str(partial.parent),
                                   "--out", partial.name]
                        if proxy:
                            command.append(f"--all-proxy={proxy}")
                        proc = subprocess.run(command + [url], capture_output=True)
                        if proc.returncode:
                            raise RuntimeError((proc.stdout + proc.stderr).decode("utf-8", errors="replace")[-2500:])
                        backend = "aria2c"
                    else:
                        proxies = {"http": proxy, "https": proxy} if proxy else {}
                        headers = dict(item.get("headers", {}))
                        start = partial.stat().st_size if partial.exists() else 0
                        if start:
                            headers["Range"] = f"bytes={start}-"
                        session = http_session()
                        with session.get(url, stream=True, timeout=(20, 90), proxies=proxies,
                                          headers=headers) as response:
                            response.raise_for_status()
                            if start and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                                raise ValueError("server refused safe resume of existing partial payload")
                            with partial.open("ab" if start else "wb") as stream:
                                for block in response.iter_content(1024 * 1024):
                                    stream.write(block)
                        backend = "requests"
                    details = validate(partial, item)
                    partial.rename(target)
                    record["resolved_url"] = url
                    break
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
            else:
                raise RuntimeError("; ".join(errors))
        return dict(record, status="available", backend=backend,
                    bytes=target.stat().st_size, sha256=digest(target),
                    integrity_basis="publisher checksum + local SHA256" if item.get("checksum") else "local SHA256; no publisher checksum registered",
                    **details)
    except Exception as exc:
        return dict(record, status="failed", reason=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:17890")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-bytes", type=int, default=0)
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    selected = [s for s in registry["sources"] if s.get("size_bytes", 0) >= args.min_bytes
                and (args.max_bytes is None or s.get("size_bytes", 0) <= args.max_bytes)]
    records = []
    def save():
        summary = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
        partial = args.manifest.with_suffix(".json.tmp")
        partial.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
            "registry": args.registry.as_posix(), "selected": len(selected), "completed": len(records),
            "records": sorted(records, key=lambda r: r["id"]), "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial.replace(args.manifest)
        return summary
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, item, args.proxy, args.audit_only): item for item in selected}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = dict(futures[future], status="failed", reason=str(exc))
            records.append(record)
            print(f'{len(records)}/{len(selected)} {record["status"]} {record["id"]}', flush=True)
            summary = save()
    summary = save()
    print(json.dumps(summary))
    return int(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
