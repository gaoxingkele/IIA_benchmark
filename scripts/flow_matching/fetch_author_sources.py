"""Acquire immutable author snapshots; preserve existing snapshots and raw data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/runs/flow_matching_campaign"


def fetch(paper_id: str, repository: str, proxy: str, commit: str | None = None):
    directory = BASE / "sources" / paper_id
    marker = directory / "snapshot.json"
    if marker.exists():
        result = json.loads(marker.read_text(encoding="utf-8"))
        if commit and result['commit'] != commit:
            raise ValueError('Existing source commit differs from preregistered commit')
        return result
    with requests.Session(impersonate="chrome", proxy=proxy, trust_env=False) as session:
        if commit is None:
            response = session.get(f"https://api.github.com/repos/{repository}/commits?per_page=1", timeout=60)
            response.raise_for_status()
            commit = response.json()[0]["sha"]
        url = f"https://codeload.github.com/{repository}/zip/{commit}"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
    archive = zipfile.ZipFile(io.BytesIO(payload))
    prefix = archive.namelist()[0].split("/")[0] + "/"
    directory.mkdir(parents=True, exist_ok=True)
    for entry in archive.infolist():
        relative = entry.filename.removeprefix(prefix)
        if not relative or entry.is_dir():
            continue
        target = (directory / "original" / relative).resolve()
        if not target.is_relative_to((directory / "original").resolve()):
            raise ValueError("Unsafe author archive path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(archive.read(entry))
    record = {"paper_id": paper_id, "repository": repository, "commit": commit,
              "archive_sha256": hashlib.sha256(payload).hexdigest(), "url": url,
              "original_path": str((directory / "original").relative_to(ROOT)),
              "status": "acquired_not_executed", "version_boundary": "Current author commit; paper-era commit not yet established"}
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproducibility/flow_matching_campaign.v1.json")
    parser.add_argument("--papers", nargs="*")
    parser.add_argument("--proxy", default="http://127.0.0.1:17890")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for paper in config["papers"]:
        if args.papers and paper["id"] not in args.papers:
            continue
        if not paper.get("author_repository"):
            continue
        try:
            record = fetch(paper["id"], paper["author_repository"], args.proxy, paper.get('author_commit'))
            print(json.dumps(record), flush=True)
        except Exception as exc:
            print(json.dumps({"paper_id": paper["id"], "status": "acquisition_failed", "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
