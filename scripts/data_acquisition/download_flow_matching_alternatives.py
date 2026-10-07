"""Retry registered public artifacts using native curl and curl-cffi transports."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from curl_cffi import requests as cffi

sys.path.insert(0, str(Path(__file__).parent))
from download_flow_matching_bundle import ROOT, digest, resolved_path, validate
from download_flow_matching_drive import inspect


class PublicAccessGate(RuntimeError):
    pass


def fresh_url(url, item):
    if item.get("refresh_query"):
        parsed = urlsplit(url)
        query = dict(parse_qsl(parsed.query))
        query[item["refresh_query"]] = str(time.time_ns())
        return urlunsplit(parsed._replace(query=urlencode(query)))
    return url


def check_range(headers, start, end, total):
    if headers.get("content-range", "").lower() != f"bytes {start}-{end}/{total}":
        raise ValueError("server did not return the exact requested range")


def curl_transfer(url, output, proxy, byte_range=None):
    headers = output.with_name(output.name + ".headers")
    command = [shutil.which("curl.exe") or shutil.which("curl"), "--location", "--fail",
               "--silent", "--show-error", "--connect-timeout", "25", "--speed-time", "60",
               "--speed-limit", "1024", "--dump-header", str(headers), "--output", str(output)]
    if not command[0]:
        raise RuntimeError("native curl is not installed")
    if proxy:
        command += ["--proxy", proxy]
    else:
        command += ["--noproxy", "*"]
    if byte_range:
        command += ["--range", f"{byte_range[0]}-{byte_range[1]}", "--max-time", "90"]
    elif output.exists() and output.stat().st_size:
        command += ["--continue-at", "-"]
    proc = subprocess.run(command + [url], capture_output=True)
    text = headers.read_text(encoding="latin1") if headers.exists() else ""
    blocks = [b for b in re.split(r"\r?\n\r?\n", text) if b.startswith("HTTP/")]
    final = blocks[-1] if blocks else ""
    status = int(final.splitlines()[0].split()[1]) if final else 0
    parsed = {k.lower().strip(): v.strip() for k, v in
              (line.split(":", 1) for line in final.splitlines()[1:] if ":" in line)}
    return proc.returncode, status, parsed, proc.stderr.decode("utf-8", errors="replace")[-600:]


def cffi_transfer(url, output, proxy, byte_range=None):
    offset = output.stat().st_size if output.exists() and not byte_range else 0
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else (
        {"Range": f"bytes={offset}-"} if offset else {})
    with cffi.Session(impersonate="chrome", proxy=proxy or None, trust_env=False) as session:
        response = session.get(url, stream=True, timeout=90, headers=headers)
        response.raise_for_status()
        if (offset or byte_range) and response.status_code != 206 and "text/html" in response.headers.get("content-type", ""):
            probe = next(response.iter_content(4096), b"").lower()
            response.close()
            if b"quota exceeded" in probe:
                raise PublicAccessGate("Google Drive explicitly reports quota exceeded; preserved ranges remain intact")
        if offset and (response.status_code != 206 or not response.headers.get("content-range", "").startswith(f"bytes {offset}-")):
            raise ValueError("resume response does not start at preserved offset")
        if byte_range and response.status_code != 206:
            raise ValueError("range ignored")
        try:
            with output.open("ab" if offset else "wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        except cffi.RequestsError:
            # A validated 206 response may end early. Its received prefix can
            # still be checked against Content-Range and appended by segmented().
            if not byte_range or not output.exists() or not output.stat().st_size:
                raise
        finally:
            response.close()
        return {k.lower(): v for k, v in response.headers.items()}


def finish_details(path, item):
    with path.open("rb") as stream:
        probe = stream.read(4096).lower()
    if b"<html" in probe and b"quota exceeded" in probe:
        raise PublicAccessGate("Google Drive explicitly reports quota exceeded; retry after quota recovers")
    details = validate(path, item)
    if item.get("format") in ("npy", "h5"):
        details.update(inspect(path, item["format"]))
    if item.get("format") in ("gz", "tar.gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
    with path.open("rb") as stream:
        prefix = stream.read(128).strip().lower()
    if prefix in (b"internal server error", b"bad gateway", b"service unavailable"):
        raise ValueError("server error payload")
    details["integrity_basis"] = ("publisher checksum + local SHA256" if item.get("checksum")
                                  else "format validation + local SHA256; no publisher checksum registered")
    return details


def segmented(item, target, proxy):
    """Reuse original TEP chunks; bounded transfers limit lost work on TLS EOF."""
    total, parts = item["size_bytes"], item.get("range_parts", 32)
    width = (total + parts - 1) // parts
    folder = target.with_name(target.name + ".chunks")
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(index):
        lo, hi = index * width, min(total, (index + 1) * width) - 1
        completed = folder / f"{index:04d}.bin"
        partial = completed.with_suffix(".part")
        length = hi - lo + 1
        if completed.exists():
            if completed.stat().st_size != length:
                raise ValueError("existing chunk has incorrect length; preserved")
            return completed
        failures = 0
        while True:
            received = partial.stat().st_size if partial.exists() else 0
            if received > length:
                raise ValueError("preserved partial exceeds registered range")
            if received == length:
                partial.rename(completed)
                return completed
            start = lo + received
            end = min(hi, start + item.get("segment_bytes", 16 * 1024 * 1024) - 1)
            segment = folder / f"{index:04d}.{start}.{uuid.uuid4().hex}.segment"
            try:
                url = fresh_url(item["url"], item)
                if item.get("range_transport") == "curl_cffi":
                    headers = cffi_transfer(url, segment, proxy, (start, end))
                else:
                    code, status, headers, error = curl_transfer(url, segment, proxy, (start, end))
                    if status != 206:
                        raise RuntimeError(f"curl HTTP {status}: {error}")
                check_range(headers, start, end, total)
                count = segment.stat().st_size if segment.exists() else 0
                if not 0 < count <= end - start + 1:
                    raise ValueError("empty or excessive segment")
                with partial.open("ab") as output, segment.open("rb") as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                segment.unlink()  # This verified transport temporary has been appended.
                failures = 0
            except PublicAccessGate:
                raise
            except Exception:
                failures += 1
                if failures >= 4:
                    raise
                time.sleep(2 * failures)
    with ThreadPoolExecutor(max_workers=item.get("range_workers", 3)) as pool:
        chunks = list(pool.map(fetch, range(parts)))
    assembly = target.with_name(target.name + ".assembling")
    with assembly.open("wb") as output:
        for chunk in chunks:
            with chunk.open("rb") as stream:
                shutil.copyfileobj(stream, output, 1024 * 1024)
    details = finish_details(assembly, item)
    if target.exists():
        raise FileExistsError("target appeared; preserved")
    assembly.rename(target)
    return details


def acquire(item, proxy):
    record = dict(item)
    target = resolved_path(ROOT / item["path"], root=ROOT)
    if not target.is_relative_to(ROOT):
        return dict(record, status="failed", reason="outside workspace")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            details, backend = finish_details(target, item), "existing_preserved"
        elif item.get("backend") == "range_curl":
            details, backend = segmented(item, target, proxy), (
                "curl_cffi_segmented" if item.get("range_transport") == "curl_cffi" else "native_curl_segmented")
        else:
            errors = []
            for url in [item["url"], *item.get("fallback_urls", [])]:
                transports = ("curl_cffi", "native_curl") if item.get("preferred_transport") == "curl_cffi" else ("native_curl", "curl_cffi")
                for backend in transports:
                    tag = "." + item["partial_tag"] if item.get("partial_tag") else ""
                    partial = target.with_name(target.name + f".{backend}{tag}.part")
                    for attempt in range(2):
                        try:
                            if backend == "native_curl":
                                code, status, headers, error = curl_transfer(fresh_url(url, item), partial, proxy)
                                if code:
                                    raise RuntimeError(f"curl exit {code}, HTTP {status}: {error}")
                            else:
                                headers = cffi_transfer(fresh_url(url, item), partial, proxy)
                            # Derive full byte count from public response when no publisher size exists.
                            content_range = headers.get("content-range", "")
                            total = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else None
                            if total and partial.stat().st_size != total:
                                raise ValueError("incomplete response size")
                            details = finish_details(partial, item)
                            if target.exists():
                                raise FileExistsError("existing target preserved")
                            partial.rename(target)
                            return dict(record, status="available", backend=backend,
                                        bytes=target.stat().st_size, sha256=details.get("sha256") or digest(target), **{k:v for k,v in details.items() if k not in ("bytes", "sha256")})
                        except Exception as exc:
                            if isinstance(exc, PublicAccessGate):
                                raise
                            errors.append(f"{backend}: {exc}")
                            time.sleep(1)
            raise RuntimeError("; ".join(errors))
        return dict(record, status="available", backend=backend,
                    bytes=target.stat().st_size, sha256=details.get("sha256") or digest(target), **{k:v for k,v in details.items() if k not in ("bytes", "sha256")})
    except PublicAccessGate as exc:
        return dict(record, status="quota_exceeded", reason=str(exc))
    except Exception as exc:
        return dict(record, status="failed", reason=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:17890")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-large", action="store_true")
    args = parser.parse_args()
    items = json.loads(args.registry.read_text(encoding="utf-8"))["sources"]
    items = [s for s in items if (s.get("backend") == "range_curl") == args.only_large]
    records = []
    def save():
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        temp = args.manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),
            registry=args.registry.as_posix(), selected=len(items), completed=len(records), records=records,
            summary={s:sum(r["status"] == s for r in records) for s in {r["status"] for r in records}}), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.manifest)
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(acquire, item, args.proxy) for item in items]):
            record = future.result()
            records.append(record)
            save()
            print(len(records), len(items), record["status"], record["id"], flush=True)
    return int(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
