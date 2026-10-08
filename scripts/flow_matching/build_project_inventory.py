"""Consolidate registered material evidence for the independent FM project.

This is a local presence/manifest inventory, not a full new checksum pass over
hundreds of GB. Acquisition manifests carry earlier validation provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.config.storage import project_relative_path
SUCCESS = {"available", "downloaded", "verified", "verified_pdf", "open_pdf", "existing_verified"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(value):
    return project_relative_path(ROOT, value)


def nested_file_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield value
        for item in value.values():
            yield from nested_file_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_file_records(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/projects/flow_matching_research.v1.json")
    args = parser.parse_args()
    project = read(args.config)
    alias_configuration = read(ROOT / project["material_aliases"]) if project.get("material_aliases") else {"bindings": []}
    aliases = {binding["id"]: binding for binding in alias_configuration["bindings"]}
    retained_non_payloads = {relative(binding["legacy_path"]) for binding in aliases.values()}
    evidence = {p: read(ROOT / p) for p in project["evidence_reports"] if (ROOT / p).exists()}
    manifest_paths = list((ROOT / "papers/literature/flow_matching").glob("*manifest.json"))
    manifest_paths += list((ROOT / "experiments/runs/fm_project_acquisition").glob("*manifest.json"))
    manifest_paths += [ROOT / "experiments/runs/mtsad_protocol_audit/pdf_download_manifest.json"]
    manifest_paths += [ROOT / "papers/literature/flow_matching/project_moment_pile_integrity.json"]
    manifests = {p.relative_to(ROOT).as_posix(): read(p) for p in manifest_paths if p.exists()}
    manifests.update(evidence)
    manifests[project["material_aliases"]] = alias_configuration
    manifests[project["audit_report"]] = read(ROOT / project["audit_report"])
    observed = {}
    for origin, value in manifests.items():
        for record in nested_file_records(value):
            try:
                path = relative(record["path"])
            except (ValueError, OSError):
                continue
            if record.get("sha256"):
                # Merge only metadata; never discard an older verified digest
                # because a later acquisition attempt on another endpoint failed.
                previous = observed.get(path, {})
                if not previous or record.get("status") in SUCCESS:
                    observed[path] = {**record, "evidence_manifest": origin}

    sources, registries = {}, {}
    for config_path in project["acquisition_registries"]:
        path = ROOT / config_path
        if not path.exists():
            continue
        config = read(path)
        registries[config_path] = len(config.get("sources", []))
        for source in config.get("sources", []):
            if source.get("id") and source.get("path"):
                sources[source["id"]] = {**source, "registry": config_path}

    artifacts, unresolved, recipes, composites = {}, [], [], []
    for source_id, source in sources.items():
        if source_id in aliases:
            source = {**source, **aliases[source_id], "size_bytes": aliases[source_id]["bytes"],
                      "checksum": aliases[source_id]["publisher_checksum"]}
        path = relative(source["path"])
        target = ROOT / path
        if source.get("access") == "generated_no_download" or source.get("format") == "generated":
            recipes.append({"id": source_id, "registry": source["registry"], "status": "generation_recipe_required_not_external_download"})
            continue
        if source.get("member_source_ids"):
            members = source["member_source_ids"]
            missing = [m for m in members if m not in sources or not (ROOT / relative(sources[m]["path"])).is_file()]
            composites.append({"id": source_id, "member_count": len(members), "missing_members": missing,
                               "status": "members_present" if not missing else "incomplete"})
            continue
        if target.is_file():
            record = observed.get(path, {})
            expected = source.get("size_bytes", source.get("expected_size", record.get("bytes")))
            actual = target.stat().st_size
            status = "present_with_acquisition_evidence" if record else "present_without_matching_checksum_manifest"
            if not actual or expected is not None and actual != expected:
                status = "invalid_size"
            with target.open("rb") as stream:
                prefix = stream.read(256).lstrip().lower()
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                status = "lfs_pointer"
            elif source.get("format") != "html" and prefix.startswith((b"<html", b"<!doctype html")):
                status = "html_instead_of_payload"
            artifacts[path] = {
                "id": source_id, "path": path, "bytes": actual,
                "format": source.get("format"), "status": status,
                "sha256": record.get("sha256"), "publisher_checksum": source.get("checksum"),
                "upstream_revision": source.get("author_revision", source.get("upstream_revision")),
                "source_url": source.get("url"), "citation": source.get("evidence_url"),
                "paper_ids": source.get("paper_ids", []), "registry": source["registry"],
                "evidence_manifest": record.get("evidence_manifest"),
            }
            if status in {"invalid_size", "lfs_pointer", "html_instead_of_payload"}:
                unresolved.append({"id": source_id, "path": path, "status": status})
        elif target.is_dir():
            artifacts[path] = {"id": source_id, "path": path, "format": "directory", "status": "directory_present_not_fully_verified", "registry": source["registry"]}
        else:
            unresolved.append({"id": source_id, "path": path, "status": source.get("access", "missing"),
                               "citation": source.get("evidence_url"), "registry": source["registry"],
                               "reason": source.get("verification", source.get("boundary", "No complete payload at registered path"))})

    # Include verified papers, classic-five arrays, and supplementary archives
    # even when their legacy registrations use another schema.
    for path, record in observed.items():
        target = ROOT / path
        if path in artifacts or path in retained_non_payloads or not target.is_file():
            continue
        if record.get("bytes") is not None and target.stat().st_size != record["bytes"]:
            continue
        artifacts[path] = {
            "id": record.get("id", record.get("paper_id", path)), "path": path,
            "bytes": target.stat().st_size, "sha256": record.get("sha256"),
            "format": target.suffix.lstrip("."), "status": "present_with_acquisition_evidence",
            "citation": record.get("evidence_url", record.get("citation", record.get("url"))),
            "evidence_manifest": record["evidence_manifest"],
        }

    code_sources = []
    for area in ("flow_matching_campaign", "mtsad_protocol_audit", "fm_project_acquisition", "fm_code_completion", "fm_baseline_code_completion", "fm_foundation_baseline_sources", "fm_tsad_baseline_sources", "fm_imputation_baseline_sources"):
        for path in (ROOT / "experiments/runs" / area / "sources").glob("*/snapshot.json"):
            snapshot = read(path)
            original = snapshot.get("original_path")
            if original and (ROOT / relative(original)).is_dir():
                code_sources.append({**snapshot, "original_path": relative(original),
                                     "evidence_manifest": path.relative_to(ROOT).as_posix()})
    for origin, value in evidence.items():
        for record in nested_file_records(value):
            original = record.get("original_path")
            if original and (ROOT / relative(original)).is_dir():
                code_sources.append({k: v for k, v in {**record, "evidence_manifest": origin}.items()
                                     if k in {"id", "repository", "commit", "sha256", "archive_sha256", "original_path", "boundary", "execution_status", "evidence_manifest"}})
    catalog = {
        "schema_version": 1, "project": project["id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_root_environment": project["asset_root_environment"],
        "artifacts": sorted(artifacts.values(), key=lambda x: x["path"]),
        "unresolved_assets": unresolved, "generated_recipes": recipes, "composite_sources": composites,
        "code_sources": code_sources,
        "material_bindings": alias_configuration["bindings"],
        "boundary": "Current paths, sizes and basic signatures checked; hashes are inherited from acquisition evidence, not freshly recomputed here. Access placeholders and version-specific preprocessing remain separate gaps.",
    }
    report = {
        "schema_version": 1, "project": project, "generated_at": catalog["generated_at"],
        "all_materials_complete": False,
        "artifact_counts": dict(Counter(a["status"] for a in catalog["artifacts"])),
        "registered_source_count": len(sources), "unique_present_artifacts": len(artifacts),
        "present_file_bytes": sum(a.get("bytes", 0) for a in artifacts.values()),
        "registry_counts": registries, "remaining_registered_gaps": unresolved,
        "composite_sources": composites, "generation_recipes": recipes,
        "new_material_evidence": evidence,
        "new_checkpoint_download": manifests.get("papers/literature/flow_matching/project_checkpoints_and_retries_manifest.json"),
        "new_moment_pile_download": {k: v for k, v in manifests.get("papers/literature/flow_matching/project_moment_pile_manifest.json", {}).items() if k != "records"},
        "moment_pile_integrity": {k: v for k, v in manifests.get("papers/literature/flow_matching/project_moment_pile_integrity.json", {}).items() if k != "records"},
        "code_sources": code_sources,
        "boundary": "Not all material exists: original paper/code access gaps and registered restricted or generation/preprocessing dependencies remain. No new leaderboard result is asserted.",
    }
    destination = ROOT / project["catalog"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target = ROOT / project["acquisition_report"]
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("registered_source_count", "unique_present_artifacts", "present_file_bytes", "artifact_counts")}, indent=2))


if __name__ == "__main__":
    main()
