"""Audit DOI-level open-access and publisher links without bypassing access controls."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "papers/literature/registry.json"
DOWNLOAD_MANIFEST = ROOT / "papers/literature/download_manifest.json"
OUTPUT = ROOT / "papers/literature/access_audit.json"


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "IIA-benchmark/0.1 (open-access metadata audit)",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def query(url: str, attempts: int = 3) -> tuple[dict[str, Any] | None, str | None]:
    for attempt in range(attempts):
        try:
            return request_json(url), None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == attempts:
                return None, f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return None, "unreachable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Contact email required by Unpaywall")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DOWNLOAD_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    papers = json.loads(args.registry.read_text(encoding="utf-8"))["papers"]
    downloads = {
        record["id"]: record
        for record in json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    }
    records = []
    for paper in papers:
        doi = paper["doi"]
        encoded = urllib.parse.quote(doi, safe="")
        unpaywall, unpaywall_error = query(
            f"https://api.unpaywall.org/v2/{encoded}?email={urllib.parse.quote(args.email)}"
        )
        crossref, crossref_error = query(f"https://api.crossref.org/works/{encoded}")
        best = (unpaywall or {}).get("best_oa_location") or {}
        crossref_links = (crossref or {}).get("message", {}).get("link", [])
        publisher_pdf_candidates = [
            link["URL"]
            for link in crossref_links
            if "pdf" in str(link.get("content-type", "")).lower()
        ]
        download = downloads.get(paper["id"], {})
        oa_pdf = best.get("url_for_pdf")
        action = (
            "complete"
            if download.get("status") == "downloaded"
            else "retry_confirmed_open_pdf"
            if paper.get("pdf_url")
            else "review_oa_candidate"
            if oa_pdf
            else "manual_authenticated_author_copy"
            if paper.get("manual_author_copy_url")
            else "institutional_access_or_author_request"
        )
        records.append(
            {
                "id": paper["id"],
                "doi": doi,
                "download_status": download.get("status", "not_registered"),
                "access_class": paper["access"],
                "manual_author_copy": {
                    "url": paper.get("manual_author_copy_url"),
                    "note": paper.get("manual_copy_note"),
                },
                "unpaywall": {
                    "query_ok": unpaywall is not None,
                    "error": unpaywall_error,
                    "is_oa": (unpaywall or {}).get("is_oa"),
                    "oa_status": (unpaywall or {}).get("oa_status"),
                    "best_landing_url": best.get("url"),
                    "best_pdf_url": oa_pdf,
                    "host_type": best.get("host_type"),
                    "license": best.get("license"),
                },
                "crossref": {
                    "query_ok": crossref is not None,
                    "error": crossref_error,
                    "publisher_pdf_candidates": publisher_pdf_candidates,
                },
                "next_action": action,
            }
        )
    output = {
        "schema_version": 1,
        "audit_date": date.today().isoformat(),
        "policy": (
            "Only confirmed open or author-hosted PDFs are auto-downloaded; publisher "
            "candidates may still require institutional access and are never bypassed."
        ),
        "summary": {
            "registered": len(records),
            "downloaded": sum(record["download_status"] == "downloaded" for record in records),
            "unpaywall_oa": sum(record["unpaywall"]["is_oa"] is True for record in records),
            "oa_pdf_candidates": sum(bool(record["unpaywall"]["best_pdf_url"]) for record in records),
            "publisher_pdf_candidates": sum(bool(record["crossref"]["publisher_pdf_candidates"]) for record in records),
            "manual_author_copy_candidates": sum(
                bool(record["manual_author_copy"]["url"]) for record in records
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
