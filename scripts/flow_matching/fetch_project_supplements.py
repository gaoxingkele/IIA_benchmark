"""Fetch registered FM project supplements and verify identity and integrity."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile

from curl_cffi import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_acquisition.download_public_datasets import find_aria2


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(path: Path, item: dict) -> dict:
    if item["format"] == "pdf":
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("Response is not a PDF")
        reader = PdfReader(path, strict=True)
        if not reader.pages:
            raise ValueError("Empty PDF")
        first = normalize(" ".join(page.extract_text() or "" for page in reader.pages[:2]))
        compact = first.replace(" ", "")
        absent = [token for token in item["title_tokens"]
                  if normalize(token) not in first and normalize(token).replace(" ", "") not in compact]
        if absent:
            raise ValueError(f"PDF title verification failed: {absent}")
        details = {"pages": len(reader.pages), "title_verified": True}
    elif item["format"] == "zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {bad}")
            details = {"archive_members": len(archive.infolist()), "crc_verified": True,
                       "code_members": [n for n in archive.namelist() if n.endswith((".py", ".ipynb", ".sh", ".yaml", ".yml"))]}
    else:
        raise ValueError("Unsupported registered format")
    actual = sha256(path)
    if item.get("sha256") and item["sha256"] != actual:
        raise ValueError("Registered SHA256 mismatch")
    return {**details, "sha256": actual, "bytes": path.stat().st_size,
            "checksum_boundary": "Locally computed SHA256; no independent publisher checksum unless registered."}


def download(item: dict, proxy: str) -> dict:
    target = ROOT / item["path"]
    attempts = []
    if target.exists():
        return {"id": item["id"], "status": "existing_verified", "path": item["path"], **validate(target, item)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fm-supplement-", dir=ROOT / "tmp") as folder:
        temporary = Path(folder) / target.name
        aria2 = find_aria2()
        if aria2:
            command = [aria2, "--all-proxy=" + proxy, "--max-tries=1", "--connect-timeout=20",
                       "--timeout=40", "--max-connection-per-server=" + str(item.get("connections", 1)),
                       "--split=" + str(item.get("connections", 1)), "--min-split-size=1M",
                       "--file-allocation=none", "--allow-overwrite=false", "--auto-file-renaming=false",
                       "--console-log-level=error", "--summary-interval=0", "--dir=" + folder,
                       "--out=" + target.name, item["url"]]
            try:
                process = subprocess.run(command, capture_output=True, timeout=600)
                returncode = process.returncode
                attempts.append({"backend": "aria2c", "returncode": returncode})
            except subprocess.TimeoutExpired:
                returncode = None
                attempts.append({"backend": "aria2c", "error": "timeout"})
            if returncode == 0 and temporary.exists():
                try:
                    details = validate(temporary, item)
                    with target.open("xb") as output:
                        output.write(temporary.read_bytes())
                    return {"id": item["id"], "status": "downloaded", "path": item["path"],
                            "url": item["url"], "attempts": attempts, **details}
                except ValueError as exc:
                    attempts[-1]["validation_error"] = str(exc)
        with requests.Session(impersonate="chrome", proxy=proxy, trust_env=False) as session:
            response = session.get(item["url"], timeout=120)
            attempts.append({"backend": "curl_cffi", "http_status": response.status_code})
            response.raise_for_status()
            fallback = Path(folder) / (target.name + ".curl")
            with fallback.open("xb") as output:
                output.write(response.content)
            details = validate(fallback, item)
            with target.open("xb") as output:
                output.write(fallback.read_bytes())
            return {"id": item["id"], "status": "downloaded", "path": item["path"],
                    "url": item["url"], "attempts": attempts, **details}


def extract(path: Path, destination: Path, strip_prefix: bool = False) -> int:
    count = 0
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0] + "/" if strip_prefix else ""
        for entry in archive.infolist():
            name = entry.filename.removeprefix(prefix)
            if entry.is_dir() or not name:
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(entry)
            if target.exists():
                if target.read_bytes() != payload:
                    raise FileExistsError(f"Existing extracted source differs: {target}")
            else:
                with target.open("xb") as output:
                    output.write(payload)
            count += 1
    return count


def fetch_repository(item: dict, config: dict) -> dict:
    source = {"id": item["id"] + "_source", "format": "zip",
              "path": config["output_root"] + "/archives/" + item["id"] + "_" + item["commit"] + ".zip",
              "url": "https://codeload.github.com/" + item["repository"] + "/zip/" + item["commit"]}
    if item.get("archive_sha256"):
        source["sha256"] = item["archive_sha256"]
    record = download(source, config["proxy"])
    directory = ROOT / config["output_root"] / "sources" / item["id"]
    record.update({"repository": item["repository"], "commit": item["commit"],
                   "extracted_files": extract(ROOT / source["path"], directory / "original", True),
                   "original_path": (directory / "original").relative_to(ROOT).as_posix(),
                   "boundary": item["boundary"], "execution_status": "acquired_not_executed"})
    marker = directory / "snapshot.json"
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/acquisition/fm_project_fm_supplements.v1.json")
    parser.add_argument("--ids", nargs="*", help="Only fetch selected registered source/repository IDs")
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def run(item: dict) -> dict:
        if item.get("full_text_status") not in ("open_pdf", "open_archive"):
            record = {"id": item["id"], "status": "skipped", "url": item["url"],
                      "reason": item.get("access_note", item.get("full_text_status", "unknown"))}
            print(json.dumps(record, ensure_ascii=True), flush=True)
            return record
        try:
            record = download(item, config["proxy"])
            if item.get("extract_path"):
                record["extracted_files"] = extract(ROOT / item["path"], ROOT / item["extract_path"])
                record["original_path"] = item["extract_path"]
        except Exception as exc:
            record = {"id": item["id"], "status": "failed", "error": str(exc), "url": item["url"]}
        print(json.dumps({key: value for key, value in record.items() if key != "code_members"}, ensure_ascii=True), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, [item for item in config["sources"] if not args.ids or item["id"] in args.ids]))
    for item in config["repositories"]:
        if args.ids and item["id"] not in args.ids:
            continue
        try:
            record = fetch_repository(item, config)
        except Exception as exc:
            record = {"id": item["id"] + "_source", "status": "failed", "error": str(exc)}
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "code_members"}, ensure_ascii=True), flush=True)
    manifest = {"checked_at": datetime.now(timezone.utc).isoformat(), "config": str(args.config.relative_to(ROOT)),
                "records": records, "unavailable": config["unavailable"]}
    path = ROOT / config["output_root"] / "fm_supplement_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.ids and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        updated = {record["id"]: record for record in previous["records"]}
        updated.update({record["id"]: record for record in records})
        manifest["records"] = list(updated.values())
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
