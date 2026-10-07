"""Report registered Flow Matching acquisitions without downloading or preprocessing."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.config.storage import resolve_project_path
CONFIGS = [ROOT / "configs/acquisition" / f"flow_matching_{group}_data_sources.json"
           for group in ("fm", "comparison", "foundation")]
BASE = ROOT / "papers/literature/flow_matching"


def read(path: Path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "papers", "sources", "items", "metadata"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value or "—").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def link(path):
    return f"[{cell(path)}](../../{str(path).replace(chr(92), '/')})" if path else "—"


def live_available(record):
    """Require a successful manifest record AND a matching current local payload."""
    if not record or record.get("status") not in ("available", "downloaded", "verified"):
        return False, (record or {}).get("reason") or (record or {}).get("status", "无下载记录")
    if not record.get("path"):
        return False, "成功记录缺少本地路径"
    try:
        path = resolve_project_path(ROOT, record["path"])
    except ValueError as error:
        return False, str(error)
    if not path.is_relative_to(ROOT) or not path.exists():
        return False, "manifest 标记成功，但本地路径不存在或越出仓库"
    if path.is_dir():
        return True, "现有目录；不等同于逐文件完整性审计"
    size = path.stat().st_size
    if not size:
        return False, "空文件"
    if record.get("bytes") is not None and size != record["bytes"]:
        return False, f"本地尺寸 {size} 与 manifest {record['bytes']} 不一致"
    with path.open("rb") as stream:
        prefix = stream.read(256).lstrip().lower()
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False, "Git LFS pointer; actual data object has not been downloaded at this path"
    if record.get("format") != "html" and prefix.startswith((b"<!doctype html", b"<html")):
        return False, "本地文件为 HTML，而非注册数据格式"
    return True, "当前文件存在且尺寸一致；摘要校验依据下载 manifest"


def material_kind(source, paper=None):
    fmt = str(source.get("format", "")).lower()
    access = str(source.get("access", "")).lower()
    text = " ".join(str(source.get(k, "")) for k in ("notes", "note", "download_note", "variant_note"))
    if fmt in ("py", "recipe", "generated", "code") or any(k in access for k in ("generated", "recipe")):
        return "生成配方/源码"
    if any(k in access for k in ("required", "proprietary", "restricted", "unavailable", "not_released", "not_found", "needs_institution")):
        return "受限/未公开"
    if source.get("exact_processed") is True or source.get("reproduction_status") == "exact_processed":
        return "已明确标注的论文预处理数据"
    # GRASP explicitly identifies this release as its experimental source. Do not
    # transfer that assertion to other papers using the same dataset families.
    if paper and canonical(paper.get("id")) == "grasp" and "mtsbench" in source.get("path", ""):
        return "论文指定 mTSBench 发布数据（实验索引仍需核对）"
    if any(k in text.lower() for k in ("preprocessed", "pre-processed", "prepared splits", "processed bundle", "author-preprocessed")) or "mtsbench" in source.get("path", ""):
        return "公开预处理/作者包（精确实验版本未确认）"
    return "原始/基础公开数据或作者数据包"


def generate(output: Path, data_manifest: Path, paper_manifest: Path, metadata: Path | None):
    documents = [read(p, required=True) for p in CONFIGS]
    documents.extend(read(ROOT / "configs/acquisition" / f"flow_matching_{group}_sources.json")
                     for group in ("drive", "mega"))
    papers, sources = {}, {}
    for doc in documents:
        for p in doc.get("papers", []):
            papers.setdefault(canonical(p["id"]), {}).update(p)
        for source in doc.get("sources", []):
            key = source["id"]
            old = sources.get(key, {})
            bindings = set(old.get("paper_ids", [])) | set(source.get("paper_ids", []))
            sources[key] = {**old, **source, "paper_ids": sorted(bindings)}
    dm, pm = read(data_manifest), read(paper_manifest)
    data_records = {r["id"]: r for r in rows(dm) if r.get("id")}
    by_path = {r.get("path"): r for r in data_records.values() if r.get("path")}
    # Independent queues publish completed records while larger transfers run.
    for manifest in sorted(BASE.glob("*manifest.json")):
        if manifest == paper_manifest:
            continue
        manifest_doc = read(manifest)
        completed_records = rows(manifest_doc)
        # MEGA lists enumeration results separately from authenticated payloads.
        for payload in manifest_doc.get("files", []):
            if payload.get("mega_mac_verified") and payload.get("status") in ("downloaded", "existing_not_overwritten"):
                record = dict(payload, id=payload["source_id"] + ":" + payload["relative_path"],
                              path=payload["relative_path"], status="verified")
                completed_records = completed_records + [record]
                sources[record["id"]] = dict(record, format="npy" if record["path"].endswith(".npy") else "npz")
        for record in completed_records:
            if not record.get("id"):
                continue
            if record.get("status") in ("available", "downloaded", "verified"):
                data_records[record["id"]] = record
                if record.get("path"):
                    by_path[record["path"]] = record
    for source in sources.values():
        if not live_available(data_records.get(source["id"]))[0] and source.get("path") in by_path:
            data_records[source["id"]] = by_path[source["path"]]
    pdf_records = {canonical(r["id"]): r for r in rows(pm) if r.get("id")}
    paper_registry = read(ROOT / "configs/acquisition/flow_matching_papers.json")
    for r in rows(paper_registry):
        papers.setdefault(canonical(r["id"]), {"id": r["id"]})
    meta = {}
    metadata_paths = [metadata] if metadata else sorted(BASE.glob("*metadata*.json"))
    for path in metadata_paths:
        for r in rows(read(path)):
            if r.get("id"):
                meta[canonical(r["id"])] = r
    assessments = {sid: live_available(data_records.get(sid)) for sid in sources}
    good = {sid for sid, (ok, _) in assessments.items() if ok}
    unique_files = {}
    for sid in good:
        record = data_records[sid]
        path = resolve_project_path(ROOT, record["path"])
        if path.is_file():
            unique_files[str(path)] = path.stat().st_size
    pdf_good = sum(live_available(r)[0] for r in pdf_records.values())
    lines = ["# Flow Matching 论文与实验数据获取审计（2026-10-01）", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}。", "",
             f"注册论文 {len(papers)} 篇；下载 manifest 中当前本地可用 PDF {pdf_good} 篇。"
             f"注册数据/配方来源 {len(sources)} 项；成功记录且现场可用 {len(good)} 项，"
             f"其中去重文件 {len(unique_files)} 个、合计 {sum(unique_files.values()) / 1024**3:.3f} GiB。", "",
             "本报告只审计资料获取，不报告模型性能，也不声明论文实验已经复现。原始数据、作者预处理包、生成配方与访问受限项分别标注；"
             "完整实验列表来自三个配置的论文条目。数据存在不代表作者划分、缺失掩码、超参数、随机种子或实验子集一致。", "",
             "现场检查包括 manifest 成功状态、本地存在、非空、与 manifest 尺寸一致及排除 HTML 占位。"
             "SHA256/发布方校验结果沿用下载 manifest，不在本报告脚本中重新读取全部大型数据。目录存在只表明可复用目录，不能代替逐文件审计。", "",
             "## 输入与完成状态", "", "| 输入 | 情况 |", "|---|---|"]
    for path, doc in [(data_manifest, dm), (paper_manifest, pm)]:
        lines.append(f"| {cell(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)} | "
                     f"{cell('缺失；不依据下载进行中子清单声称成功' if not doc else json.dumps({k:doc[k] for k in ('selected','completed','summary') if k in doc}, ensure_ascii=False))} |")
    lines.extend(["", "## 逐篇论文、完整实验列表与资料边界", ""])
    for pid, paper in papers.items():
        title = meta.get(pid, {}).get("title") or paper["id"]
        pdf = pdf_records.get(pid)
        ok, reason = live_available(pdf)
        lines.extend([f"### {cell(title)}", "", f"PDF：{'本地可用' if ok else '未取得可用全文'}；{cell(reason)}。"
                      + (f" {link(pdf.get('path'))}" if pdf else ""), ""])
        evidence = paper.get("evidence_url")
        if evidence:
            lines.extend([f"实验来源：[原论文/官方证据]({evidence})。", ""])
        datasets = paper.get("datasets", paper.get("experimental_datasets", paper.get("dataset_names", [])))
        if isinstance(datasets, dict):
            datasets = [f"{k}: {cell(v)}" for k, v in datasets.items()]
        if isinstance(datasets, str):
            datasets = [datasets]
        lines.append("论文列明实验数据集：" + ("；".join(cell(d) for d in datasets) if datasets else "配置尚未记录，不能由下载列表推断完整实验范围") + "。")
        lines.append("")
        related = [s for s in sources.values() if pid in {canonical(v) for v in s.get("paper_ids", [])}]
        counts = Counter(material_kind(s, paper) for s in related if s["id"] in good)
        lines.extend(["当前可用资料分类：" + ("；".join(f"{k} {v} 项" for k,v in counts.items()) if counts else "没有已完成且现场可用的注册资料") + "。", ""])
        for key in ("boundary", "reproduction_boundary", "notes", "download_note", "preprocessing_note", "synthetic_note", "restricted_items", "synthetic", "synthetic_generation"):
            if paper.get(key):
                lines.extend([f"- **{key}**：{cell(paper[key])}", ""])
        lines.extend(["| 来源 ID | 材料类型 | 现场状态 | 路径 |", "|---|---|---|---|"])
        for source in related:
            sid = source["id"]
            available, why = assessments[sid]
            lines.append(f"| {cell(sid)} | {cell(material_kind(source,paper))} | {cell('可用' if available else why)} | {link(source.get('path'))} |")
        if not related:
            lines.append("| — | — | 尚无绑定的来源项 | — |")
        lines.append("")
    lines.extend(["## 所有尚未可用来源与原因", "", "| 来源 ID | manifest 状态 | 原因/门禁 | 官方证据 |", "|---|---|---|---|"])
    for sid, source in sources.items():
        if sid in good:
            continue
        record = data_records.get(sid, {})
        reason = assessments[sid][1]
        if not record:
            reason += "; access=" + str(source.get("access", "unknown"))
        note = source.get("access_note") or source.get("reason") or source.get("notes") or source.get("download_note")
        if note:
            reason += "; " + str(note)
        url = source.get("evidence_url") or source.get("official_url")
        citation = f"[证据]({url})" if url and str(url).startswith("http") else cell(url)
        lines.append(f"| {cell(sid)} | {cell(record.get('status','无记录'))} | {cell(reason)} | {citation} |")
    if len(good) == len(sources):
        lines.append("| — | — | 全部注册来源现场可用；仍须遵守逐篇复现边界 | — |")
    for doc in documents:
        for key in ("related_release_boundary", "notes"):
            if doc.get(key):
                lines.extend(["", f"配置附注：{cell(doc[key])}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"papers": len(papers), "available_pdfs": pdf_good, "registered_sources": len(sources),
            "available_sources": len(good), "unique_files": len(unique_files), "bytes": sum(unique_files.values()),
            "report": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, default=BASE / "data_download_manifest.json")
    parser.add_argument("--paper-manifest", type=Path, default=BASE / "paper_download_manifest.json")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/flow_matching_acquisition_2026-10-01.md")
    args = parser.parse_args()
    print(json.dumps(generate(args.output.resolve(), args.data_manifest.resolve(),
                              args.paper_manifest.resolve(), args.metadata), ensure_ascii=False))


if __name__ == "__main__":
    main()
