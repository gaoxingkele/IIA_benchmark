"""Build the current paper/algorithm/dataset/task inventory report."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


DATASET_DESCRIPTIONS = {
    "tep_classic": "TEP 经典过程仿真；44 个 run、52 个变量",
    "pronto": "1.72 GB 多相流实验设施数据；过程、报警和故障标签",
    "piade": "5 台包装设备；429,394 行原始记录及 23,376 行小时序列",
    "skab": "35 个水循环异常实验 CSV",
    "tep_alarm_dataport": "16.98 GB；100 个 Tests run、1,000 条五类报警序列及异常场景变体",
    "npp_alarm_dataport": "101 个阈值层；每层 1,212 个 run、12 类事故/扰动加 Normal、192 个二值报警位",
    "fcc_alarm": "1,600 个 FCC 仿真 run、16 类异常、57 个报警位及 4,800 个配套时序 CSV",
    "comopi": "8 台包装设备、150,650 个十分钟 bin、123 类报警",
    "smd10towfgr": "10 台风机 SCADA；230,618 条日志、167 个事件代码",
    "enas": "219,893 条离散传感器、执行器和人工错误状态记录",
    "imaks": "211,200 条带异常和因果真值的合成 MQTT/传感器记录",
}

TASK_NAMES = {
    "T1": "报警生成与参数设计",
    "T2": "多变量动态报警限",
    "T3": "因果图与根因排序",
    "T4": "报警洪泛检测、聚类与分类",
    "T5": "next-alarm 与洪泛预测",
    "T6": "运维可视分析",
}

PAPER_STATUS = {
    "downloaded": "全文已归档",
    "not_openly_downloadable": "全文未归档（访问受限）",
    "download_failed": "全文未归档（下载失败）",
}


def read_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def implementation_text(value: str | list[str] | None) -> str:
    if isinstance(value, list):
        return "<br>".join(f"`{escape(item)}`" for item in value)
    if value:
        return f"`{escape(value)}`"
    return "—"


def current_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_report(cutoff: str) -> str:
    status = read_json("docs/status_audit.json")
    papers = read_json("papers/literature/registry.json")["papers"]
    paper_downloads = read_json("papers/literature/download_manifest.json")
    downloads_by_id = {item["id"]: item for item in paper_downloads["records"]}
    book_algorithms = read_json("configs/algorithms/book_algorithms.json")["algorithms"]
    sota_algorithms = read_json("configs/algorithms/sota_algorithms.json")["algorithms"]
    sources = read_json("configs/datasets/public_sources.json")["sources"]
    data_audit = read_json("data/public_datasets/audit.json")
    data_by_id = {item["id"]: item for item in data_audit}
    tasks = read_json("configs/tasks/downstream_tasks.json")["tasks"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        families[source["dataset_family"]].append(source)

    lines = [
        f"# IIA Benchmark 当前清单与完备性说明（{cutoff}）",
        "",
        f"仓库：<https://github.com/gaoxingkele/IIA_benchmark>；生成基线 revision：`{current_revision()}`。",
        "",
        "## 1. 总体结论",
        "",
        "| 范围 | 当前数量 | 完备性判断 |",
        "|---|---:|---|",
        f"| 登记参考论文 | {status['evidence']['registered_papers']} | 本地全文 {status['evidence']['downloaded_papers']}，缺 {status['evidence']['registered_papers'] - status['evidence']['downloaded_papers']} |",
        f"| 书籍算法交付项 | {status['algorithms']['book_deliverables']} | 可调用 {status['algorithms']['book_callable']}；verified {status['algorithms']['book_by_status']['verified']}，partial {status['algorithms']['book_by_status']['partial']} |",
        f"| SOTA 算法交付项 | {status['algorithms']['selected_sota_deliverables']} | 可调用 {status['algorithms']['sota_callable']}；verified {status['algorithms']['sota_by_status']['verified']}，partial {status['algorithms']['sota_by_status']['partial']} |",
        f"| 可调用方法族 / 模型配置 | {status['algorithms']['callable_method_families']} / {status['algorithms']['model_configs']} | 机制与单元测试可运行，不等于论文分数复现 |",
        f"| 逻辑数据集族 | {status['datasets']['logical_public_families']} | 有效主载荷 {status['datasets']['main_payload_available_families']}/{status['datasets']['logical_public_families']} |",
        f"| 下游任务 | {status['tasks']['defined']} | 6/6 有真实或已取得数据入口；T4 正式专用数据实验仍待 adapter |",
        f"| 正式排行榜切分 | {status['evidence']['leaderboard_eligible_splits']} | 尚无 leaderboard-eligible split |",
        f"| 真实数据验证报告 | {status['evidence']['real_data_validation_reports']} | 覆盖 {status['evidence']['registered_algorithms_with_real_data_execution']} 个登记算法；严格分数闭环仍为 {status['algorithms']['strict_score_closed']} |",
        "",
        "这里的‘可调用’表示本地实现有明确入口并通过机制/不变量测试；只有在论文原始数据、预处理、grouped split、指标、随机种子和参考分数均闭合后，才能升级为 `verified`。",
        "",
        "## 2. 参考论文清单",
        "",
        f"当前登记 {paper_downloads['summary']['registered']} 篇：全文已归档 {paper_downloads['summary']['downloaded']} 篇，访问受限 {paper_downloads['summary']['not_openly_downloadable']} 篇，自动下载失败 {paper_downloads['summary']['failed']} 篇。",
        "",
        "| ID | 年份 | 题目 | 期刊/会议 | DOI | 全文状态 | 对应角色 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for paper in papers:
        download = downloads_by_id.get(paper["id"], {})
        status_text = PAPER_STATUS.get(download.get("status", ""), "未登记下载状态")
        doi = paper.get("doi") or "—"
        lines.append(
            "| {id} | {year} | {title} | {venue} | {doi} | {status} | {role} |".format(
                id=f"`{escape(paper['id'])}`",
                year=escape(paper.get("year", "")),
                title=escape(paper.get("title", "")),
                venue=escape(paper.get("venue", "")),
                doi=f"`{escape(doi)}`" if doi != "—" else doi,
                status=status_text,
                role=escape(paper.get("role", "")),
            )
        )

    lines.extend(
        [
            "",
            "## 3. 算法集",
            "",
            "核心闭环账本为 20 项书籍算法与 10 项 SOTA。`docs/algorithm_matrix.md` 的 34 个可调用方法族还包含支撑基线和一个交付项拆出的多个机制，因此数量不与 30 项闭环账本一一相等。",
            "",
            "### 3.1 书籍算法（20 项）",
            "",
            "| ID | 章节 | 算法/方法 | 页码 | 本地入口 | 状态 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in book_algorithms:
        lines.append(
            f"| `{escape(item['id'])}` | {escape(item['chapter'])} | {escape(item['name'])} | 印刷页 {escape(item['printed_pages'])} / PDF {escape(item['physical_pages'])} | {implementation_text(item.get('implementation'))} | `{escape(item['status'])}` |"
        )

    lines.extend(
        [
            "",
            "### 3.2 SOTA 算法（10 项）",
            "",
            "| ID | 方法 | DOI | 本地入口 | 状态 | 完整闭环仍需 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in sota_algorithms:
        lines.append(
            f"| `{escape(item['id'])}` | {escape(item['name'])} | `{escape(item.get('doi', '—'))}` | {implementation_text(item.get('local_implementation'))} | `{escape(item['status'])}` | {escape(item.get('closure_gate', ''))} |"
        )

    lines.extend(
        [
            "",
            "## 4. 数据集清单",
            "",
            f"机器登记包含 {status['datasets']['registry_records']} 条来源记录，归并为 {status['datasets']['logical_public_families']} 个逻辑数据集族；当前 11/11 均至少有一个有效主载荷。原始数据文件受 `.gitignore` 保护，仓库只提交来源、路径、哈希、profile 和审计状态。",
            "",
            "| 数据集族 | 内容 | 有效主载荷 | 类型 | 可支持任务 | 当前边界 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for family in sorted(families):
        items = families[family]
        mains = [item for item in items if item.get("payload_role") == "main"]
        valid_mains = [
            item["id"] for item in mains if data_by_id.get(item["id"], {}).get("valid")
        ]
        task_names = sorted({task for item in mains for task in item.get("tasks", [])})
        synthetic = any(item.get("synthetic", False) for item in mains)
        if family == "tep_alarm_dataport":
            boundary = (
                "五类 ZIP adapter、G0、seeded split 与首批 6 个实验已完成；"
                "100-run/异常变体及论文 exact protocol 待补"
            )
        elif family == "fcc_alarm":
            boundary = "alarm/process adapter、G0、grouped split 与首批 9 个实验已完成；多 seed/论文协议待补"
        elif family == "npp_alarm_dataport":
            boundary = "原始载荷已取得；专用 adapter、grouped split 和正式实验待完成"
        elif family == "imaks":
            boundary = "仅用于合成因果/鲁棒性验证，不得作为真实工业性能"
        elif family == "pronto":
            boundary = "T4 使用故障窗代理，不是专家洪泛类别"
        else:
            boundary = "已取得；正式榜单仍需冻结 split 与参考分数"
        lines.append(
            f"| `{escape(family)}` | {escape(DATASET_DESCRIPTIONS.get(family, ''))} | {', '.join(f'`{escape(item)}`' for item in valid_mains) or '—'} | {'synthetic' if synthetic else 'public/acquired'} | {', '.join(f'`{escape(item)}`' for item in task_names)} | {boundary} |"
        )

    lines.extend(
        [
            "",
            "另有 4 个不可报告成绩的 smoke 生成器：`synthetic_step_fault`、`synthetic_multivariate`、`synthetic_root_cause`、`synthetic_alarm_floods`。",
            "",
            "## 5. 下游任务清单",
            "",
            "| ID | 任务 | 当前状态 | 数据集族 | 输出 | 说明 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for task in tasks:
        if task["id"] == "T4":
            note = (
                "TEP 五类与 FCC 专用 adapter、G0、grouped split 和首批实验已完成；"
                "NPP adapter、多 seed 与论文 exact protocol 待完成；PRONTO 仅保留为错配哨兵。"
            )
        else:
            note = "已有真实或已取得数据入口；正式榜单仍需统一 split。"
        lines.append(
            f"| `{task['id']}` | {TASK_NAMES[task['id']]} | `{escape(task['status'])}` | {', '.join(f'`{escape(item)}`' for item in task['dataset_families'])} | {', '.join(f'`{escape(item)}`' for item in task['outputs'])} | {escape(note)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 当前主要缺口与优先顺序",
            "",
            "1. 补齐 23 篇论文全文，更新 PDF SHA-256、页码证据和 ARA evidence；其中 22 篇访问受限、1 篇自动下载遭遇 HTTP 403。",
            "2. 为 NPP Alarm 建立只读 adapter；补齐 TEP 100-run/异常变体入口，并按 run/事故族/异常族生成稳定样本 ID。",
            "3. 建立首个 leaderboard-eligible grouped split；训练期确定全部超参，测试期冻结，报告多 seed 与 95% CI。",
            "4. TEP/FCC 已重跑 CASIM、CTFH、HDAM、ConE-AFC、Cross-Conformal；下一步补书籍序列方法、NPP、时间直方图与多 seed，保留 PRONTO 退化证据。",
            "5. 合法取得 CASIM、ConE-AFC 等官方 Code Ocean 工件，并复跑论文代表表格；在此之前 30 项算法均保持 `partial`。",
            "6. 完成 open-set 类别留一、prefix 早期分类、missing/spurious/jitter/delay 鲁棒性矩阵和跨数据集迁移实验。",
            "",
            "## 7. 复现入口",
            "",
            "```powershell",
            "python scripts/data_acquisition/audit_public_datasets.py",
            "python scripts/data_acquisition/profile_public_datasets.py",
            "python scripts/literature/verify_ara_collection.py",
            "python scripts/validate_scaffold.py",
            "python scripts/audit_benchmark_coverage.py",
            "python -m pytest -q",
            "```",
            "",
            "详细机器状态见 `docs/status_audit.json`，论文状态见 `papers/literature/download_manifest.json`，算法闭环账本见 `configs/algorithms/`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "docs" / "reports" / f"current_inventory_{args.date}.md"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(args.date), encoding="utf-8")
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
