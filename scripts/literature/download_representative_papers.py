"""Download registered open-access papers and emit a checksum/page manifest.

The PDFs themselves are intentionally git-ignored. The generated manifest is the
tracked evidence that identifies exactly what was available to the local run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "papers" / "literature" / "registry.json"
DEFAULT_OUTPUT = ROOT / "papers" / "literature" / "pdfs"
DEFAULT_MANIFEST = ROOT / "papers" / "literature" / "download_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    proc = subprocess.run(
        [pdfinfo, str(path)], capture_output=True, text=True, check=False
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download(url: str, destination: Path, proxy: str | None) -> str:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2:
        command = [
            aria2,
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--check-certificate=true",
            "--max-tries=3",
            "--retry-wait=2",
            "--dir",
            str(partial.parent),
            "--out",
            partial.name,
        ]
        if proxy:
            command.append(f"--all-proxy={proxy}")
        command.append(url)
        subprocess.run(command, check=True)
        backend = "aria2c"
    else:
        handlers: list[urllib.request.BaseHandler] = []
        if proxy:
            handlers.append(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers={"User-Agent": "IIA-benchmark/0.1"})
        with opener.open(request, timeout=90) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        backend = "urllib"
    if not valid_pdf(partial):
        partial.unlink(missing_ok=True)
        raise ValueError("response is not a valid PDF")
    partial.replace(destination)
    return backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTP(S) proxy, for example http://127.0.0.1:17890",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    previous_manifest = None
    previous_by_id: dict[str, dict[str, object]] = {}
    if args.manifest.is_file():
        previous_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        previous_by_id = {
            record["id"]: record for record in previous_manifest.get("records", [])
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures = 0
    for paper in registry["papers"]:
        destination = args.output_dir / f"{paper['id']}.pdf"
        record: dict[str, object] = {
            "id": paper["id"],
            "doi": paper["doi"],
            "path": destination.relative_to(ROOT).as_posix(),
            "source_url": paper.get("pdf_url"),
        }
        if not paper.get("pdf_url"):
            record.update(status="not_openly_downloadable", reason=paper["access"])
            records.append(record)
            continue
        try:
            if args.force or not valid_pdf(destination):
                record["backend"] = download(paper["pdf_url"], destination, args.proxy)
            else:
                record["backend"] = previous_by_id.get(paper["id"], {}).get(
                    "backend", "cache"
                )
            record.update(
                status="downloaded",
                bytes=destination.stat().st_size,
                sha256=sha256(destination),
                pages=pdf_pages(destination),
            )
        except Exception as exc:  # retain a complete audit instead of aborting the batch
            failures += 1
            record.update(status="download_failed", reason=str(exc))
        records.append(record)

    same_records = bool(previous_manifest) and previous_manifest.get("records") == records
    generated_at = (
        previous_manifest["generated_at"]
        if same_records
        else datetime.now(timezone.utc).isoformat()
    )
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "registry": args.registry.relative_to(ROOT).as_posix(),
        "records": records,
        "summary": {
            "registered": len(records),
            "downloaded": sum(r["status"] == "downloaded" for r in records),
            "failed": failures,
            "not_openly_downloadable": sum(
                r["status"] == "not_openly_downloadable" for r in records
            ),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
