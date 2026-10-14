# -*- coding: utf-8 -*-
"""生成 2026-10-14 周例会学术汇报 PPT：时间序列异常检测方法与数据集盘点。

用法: python scripts/reporting/build_weekly_ppt_20261014.py
输出: docs/reports/时间序列异常检测研究盘点_周例会汇报_20261014.pptx
      docs/reports/weekly_2026-10-14_assets/*.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(ROOT, "docs", "reports", "weekly_2026-10-14_assets")
OUT_PPTX = os.path.join(ROOT, "docs", "reports", "时间序列异常检测研究盘点_周例会汇报_20261014.pptx")
os.makedirs(ASSET_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DARK = RGBColor(0x1F, 0x4E, 0x79)
MID = RGBColor(0x2E, 0x75, 0xB6)
LIGHT = RGBColor(0xDE, 0xEB, 0xF7)
GRAY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
ORANGE = RGBColor(0xC5, 0x50, 0x0B)
RED = RGBColor(0xB0, 0x30, 0x30)

# ---------------------------------------------------------------- charts

def chart_mtsad_verdict():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    vals = [5, 9, 18]
    labels = ["matches\n(吻合)", "near\n(接近)", "off\n(偏离)"]
    colors = ["#2E7D32", "#C5500B", "#B03030"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=13, fontweight="bold")
    ax.set_title("MTSAD 复现总账：32 个对比对（124 runs）", fontsize=13)
    ax.set_ylabel("对比对数量")
    ax.set_ylim(0, 21)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "mtsad_verdict.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_timesnet():
    ds = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
    rerun = [0.8542, 0.8321, 0.7065, 0.9599, 0.9293]
    pub = [0.8512, 0.8295, 0.7051, 0.9548, 0.9210]
    x = range(len(ds))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([i - 0.18 for i in x], rerun, width=0.36, label="本仓库复现", color="#2E75B6")
    ax.bar([i + 0.18 for i in x], pub, width=0.36, label="论文报告", color="#9DC3E6")
    ax.set_xticks(list(x), ds)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("point-adjusted F1")
    ax.set_title("TimesNet：5/5 数据集复现与论文差 ≤ 0.03", fontsize=13)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "timesnet_repro.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_at_dc():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=False)
    ds = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
    at_r = [0.9034, 0.8557, 0.9021, 0.9528, 0.8699]
    at_p = [0.9033, 0.9393, 0.9033, 0.9502, 0.9322]
    dc_r = [0.8862, 0.8512, 0.8965, 0.9380, 0.9518]
    dc_p = [0.8914, 0.9660, 0.9037, 0.9447, 0.9633]
    for ax, r, p, name in [(axes[0], at_r, at_p, "Anomaly Transformer"),
                           (axes[1], dc_r, dc_p, "DCdetector")]:
        x = range(len(ds))
        ax.bar([i - 0.18 for i in x], r, width=0.36, label="复现", color="#2E75B6")
        ax.bar([i + 0.18 for i in x], p, width=0.36, label="论文/第三方", color="#C5500B")
        ax.set_xticks(list(x), ds, fontsize=9)
        ax.set_ylim(0.8, 1.0)
        ax.set_title(name, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("point-adjusted F1")
    axes[0].legend(fontsize=9)
    fig.suptitle("AT / DCdetector：部分数据集吻合，MSL 缺口为开放差异", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(ASSET_DIR, "at_dc.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_tep_f1():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = ["IID delay\n(单变量)", "自适应ECDF\n(单变量)", "Mahalanobis\n(多变量)", "凸包NOZ\n(多变量)", "搜索锥NOZ\n(多变量)"]
    vals = [0.8678, 0.9502, 0.8921, 0.7210, 0.6385]
    colors = ["#2E75B6", "#2E75B6", "#2E7D32", "#2E7D32", "#2E7D32"]
    bars = ax.bar(names, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("F1")
    ax.set_title("TEP 数据集上报警检测 F1（三种子均值）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "tep_f1.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_flood_ba():
    import numpy as np
    ds = ["TEP", "NPP", "FCC"]
    casim = [1.0000, 0.8182, 0.9922]
    hdam = [0.9967, 0.6932, 0.9375]
    jac = [0.9667, 0.6932, 0.9115]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.25, casim, width=0.24, label="CASIM", color="#1F4E79")
    ax.bar(x, hdam, width=0.24, label="HDAM", color="#2E75B6")
    ax.bar(x + 0.25, jac, width=0.24, label="Jaccard 基线", color="#9DC3E6")
    ax.set_xticks(x, ds)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("报警洪泛分类 BA：复杂方法未稳定超越基线", fontsize=13)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "flood_ba.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_dataset_scale():
    ds = ["SMD", "SWaT", "SMAP", "PSM", "MSL"]
    n = [708405, 495000, 135183, 132481, 58317]
    dim = [38, 51, 25, 25, 55]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.barh(ds[::-1], n[::-1], color="#2E75B6", height=0.55)
    for b, v, d in zip(bars, n[::-1], dim[::-1]):
        ax.text(v + 8000, b.get_y() + b.get_height() / 2, f"{v:,} × {d}维", va="center", fontsize=10)
    ax.set_xlim(0, 950000)
    ax.set_xlabel("样本点数")
    ax.set_title("MTSAD 五基准规模（样本数 × 维度）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "dataset_scale.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_fm_status():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    vals = [5341, 1776]
    labels = ["缺结果\n(5,341)", "有实测\n(1,776)"]
    ax.pie(vals, labels=labels, autopct="%1.0f%%", colors=["#B03030", "#2E75B6"],
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("flow_matching 子项目：7,117 条登记指标完成度", fontsize=13)
    fig.tight_layout()
    p = os.path.join(ASSET_DIR, "fm_status.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- pptx helpers

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _set_text(tf, items, base_size=16):
    """items: list of (text, level, bold, color, size_offset)"""
    tf.word_wrap = True
    first = True
    for text, level, bold, color, off in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = ("• " if level == 0 and not bold else ("– " if level > 0 else "")) + text
        f = run.font
        f.size = Pt(base_size + off)
        f.bold = bold
        f.color.rgb = color
        f.name = "Microsoft YaHei"


def add_rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    return sh


def title_bar(slide, title, subtitle=None):
    add_rect(slide, 0, 0, prs.slide_width, Inches(1.0), DARK)
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(12.4), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Microsoft YaHei"
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(13)
        r2.font.color.rgb = LIGHT
        r2.font.name = "Microsoft YaHei"


def content_slide(title, blocks, image=None, subtitle=None, img_w=Inches(5.8)):
    """blocks: list of bullet tuples; image: optional path placed right."""
    slide = prs.slides.add_slide(BLANK)
    title_bar(slide, title, subtitle)
    text_w = Inches(7.0) if image else Inches(12.3)
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.25), text_w, Inches(5.9))
    _set_text(tb.text_frame, blocks)
    if image:
        slide.shapes.add_picture(image, Inches(7.35), Inches(1.5), width=img_w)
    return slide


def section_slide(num, title, desc):
    slide = prs.slides.add_slide(BLANK)
    add_rect(slide, 0, 0, prs.slide_width, prs.slide_height, DARK)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.3), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = num
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x9D, 0xC3, 0xE6)
    r.font.name = "Microsoft YaHei"
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = title
    r2.font.size = Pt(36)
    r2.font.bold = True
    r2.font.color.rgb = WHITE
    r2.font.name = "Microsoft YaHei"
    p3 = tf.add_paragraph()
    r3 = p3.add_run()
    r3.text = desc
    r3.font.size = Pt(16)
    r3.font.color.rgb = LIGHT
    r3.font.name = "Microsoft YaHei"
    return slide


def info_card(slide, x, y, w, h, head, body, head_color=MID):
    add_rect(slide, x, y, w, h, LIGHT)
    add_rect(slide, x, y, w, Inches(0.42), head_color)
    tb = slide.shapes.add_textbox(x + Inches(0.08), y + Inches(0.04), w - Inches(0.16), Inches(0.36))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = head
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Microsoft YaHei"
    tb2 = slide.shapes.add_textbox(x + Inches(0.1), y + Inches(0.5), w - Inches(0.2), h - Inches(0.55))
    tf = tb2.text_frame
    tf.word_wrap = True
    first = True
    for line in body:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(3)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(11.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        r.font.name = "Microsoft YaHei"


def B(text, level=0, bold=False, color=RGBColor(0x33, 0x33, 0x33), off=0):
    return (text, level, bold, color, off)


def H(text):
    return B(text, bold=True, color=DARK, off=2)


def algo_slide(a):
    blocks = [H("出处 / 定位"), B(a["source"]), H("方法原理")]
    blocks += [B(x) for x in a["principle"]]
    blocks += [H("实现位置"), B(a["impl"])]
    if a.get("result"):
        blocks += [H("实验表现")]
        blocks += [B(x, color=a.get("rc", RGBColor(0x33, 0x33, 0x33))) for x in a["result"]]
    blocks += [H("适用场景"), B(a["usage"])]
    slide = content_slide(a["name"], blocks, subtitle=a.get("cat"))
    if a.get("badge"):
        txt, col = a["badge"]
        add_rect(slide, Inches(10.7), Inches(1.15), Inches(2.2), Inches(0.45), col)
        tb = slide.shapes.add_textbox(Inches(10.7), Inches(1.19), Inches(2.2), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = "Microsoft YaHei"
    return slide


def dataset_slide(d):
    blocks = [H("基本信息")]
    blocks += [B(f"类型：{d['type']}"), B(f"规模：{d['scale']}"), B(f"标签：{d['label']}"), B(f"获取状态：{d['status']}")]
    blocks += [H("内容说明")]
    blocks += [B(x) for x in d["desc"]]
    if d.get("note"):
        blocks += [H("使用备注")]
        blocks += [B(x) for x in d["note"]]
    slide = content_slide(d["name"], blocks, subtitle="数据集档案")
    # 状态徽标
    ok = "已下载" in d["status"] or "已取得" in d["status"]
    col = GREEN if ok else (ORANGE if "申请" in d["status"] or "未下载" in d["status"] else MID)
    add_rect(slide, Inches(11.1), Inches(1.15), Inches(1.8), Inches(0.45), col)
    tb = slide.shapes.add_textbox(Inches(11.1), Inches(1.19), Inches(1.8), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "可用" if ok else ("待申请" if not ok and "申请" in d["status"] or "未下载" in d["status"] else "受限")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Microsoft YaHei"
    return slide


# ---------------------------------------------------------------- data

ALG_UNIV = [
    dict(name="阈值 + on-delay + deadband 基线", cat="单变量报警 · 基础基线",
         source="工业报警系统标准做法（教科书式基线）",
         principle=["固定阈值触发，on/off delay 防抖，deadband 防抖动回切",
                    "实现为显式状态机，参数少、可解释、部署成本最低",
                    "是所有自适应/SOTA 方法的对照锚点"],
         impl="src/iia_benchmark/models/univariate.py · ThresholdDelayDeadband / design_alarm",
         result=["SKAB 上 F1 仅 0.1448 —— 暴露固定阈值在非平稳工况下的局限"],
         usage="任何新方法的必跑基线；也用于 FAR/MAR/AAD 联合搜索设计（书 Chapter 2）",
         badge=("基线", MID)),
    dict(name="IID on/off delay timer", cat="单变量报警 · 书 2.1",
         source="Xu et al., 2012, IEEE TASE（DOI 10.1109/TASE.2011.2176490）",
         principle=["把报警序列建模为马尔可夫链，推导 FAR/MAR/AAD 闭式解",
                    "给定性能指标约束可直接解析设计 on/off delay 参数",
                    "假设样本 IID，是 delay timer 理论的奠基工作"],
         impl="src/iia_benchmark/models/univariate_book.py · AlarmOnOffDelay / design_iid_delay_timer",
         result=["TEP 三种子：F1 0.8678 / FAR 0.0067 / MAR 0.1840",
                 "P0 论文级验证达 E4：Table VII 最大数值误差 5.94e-05（全仓库唯一 E4）"],
         usage="单变量报警参数设计的理论基准；论文复现的标杆案例",
         badge=("E4 复现", GREEN), rc=GREEN),
    dict(name="Non-IID delay timer", cat="单变量报警 · 书 2.2",
         source="Wang et al., 2022",
         principle=["放弃 IID 假设，用持续时间概率质量函数（PMF）刻画相关结构",
                    "Pettitt 变点分段 + 贝叶斯区间估计，适应非平稳过程",
                    "对自相关工业过程的 FAR/MAR 估计更诚实"],
         impl="src/iia_benchmark/models/univariate_book.py · design_non_iid_delay_timer",
         result=["书表迁移复现完成（E2/P2 证据级）"],
         usage="自相关强的过程变量（温度、液位缓变通道）的 delay 设计",
         badge=None),
    dict(name="Non-IID deadband 设计", cat="单变量报警 · 书 2.3",
         source="Wang et al., 2023",
         principle=["以最大幅值偏差为准则设计 deadband 宽度",
                    "Beta 后验更新，把历史报警质量反馈进宽度调整",
                    "与 delay timer 正交，可联合使用"],
         impl="src/iia_benchmark/models/univariate_book.py · deadband_index / design_deadband_width",
         result=["书表迁移复现完成（E2/P2 证据级）"],
         usage="报警抖动严重（chattering）通道的 deadband 定量设计",
         badge=None),
    dict(name="Alarm Probability Plot (APP)", cat="单变量报警 · 书 2.4",
         source="Yu et al., 2017",
         principle=["把'报警概率'作为连续量可视化，替代硬阈值二值化",
                    "支持多工况下阈值设计的图形化权衡分析",
                    "为操作员提供可解释的报警质量图谱"],
         impl="src/iia_benchmark/models/univariate_book.py · build_alarm_probability_plot",
         result=["书图复现完成"],
         usage="阈值/性能权衡的展示与沟通工具，适合配合设计类方法",
         badge=None),
    dict(name="自适应单变量套件（B0–B7 消融）", cat="单变量报警 · 本仓库适配层",
         source="本仓库原创适配（面向分布漂移与工况切换）",
         principle=["ECDF 归一化报警：经验分布自适应阈值，免人工调参",
                    "Page-Hinkley 变点检测：在线识别分布漂移并触发重校准",
                    "block 校准 + 工况条件 ECDF + 安全滚动 ECDF",
                    "自动路由 + 显式拒绝（OOD 时拒绝预测，输出不确定标记）"],
         impl="src/iia_benchmark/models/adaptive_univariate.py（EmpiricalCDFAlarm / PageHinkleyChangeAlarm / AdaptiveUnivariateAlarmRouter）；管线 adaptation/univariate_pipeline.py",
         result=["TEP 双侧 ECDF：F1 0.9502（基线 0.8678 → +0.082）",
                 "SKAB：B0 基线 0.1448 → B4 0.4124（但 FAR 仍 55 次/小时，负结果保留）"],
         usage="分布漂移、工况切换场景下的单变量检测主力；消融链完整，适合写论文",
         badge=("本周亮点", GREEN), rc=GREEN),
]

ALG_MULT = [
    dict(name="Mahalanobis MSPC", cat="多变量报警 · 经典基线",
         source="多元统计过程控制（MSPC）经典方法",
         principle=["用正常工况协方差构造 Mahalanobis 距离作为异常分数",
                    "假设线性高斯，计算高效、理论完备",
                    "是多变量报警的默认对照"],
         impl="src/iia_benchmark/models/multivariate.py · MahalanobisAlarm",
         result=["TEP F1 0.8921 —— 目前多变量报警主线最强", "PRONTO/SKAB 上明显偏弱（负结果保留）"],
         usage="多变量基线首选；线性相关性强的过程变量效果最佳",
         badge=("最强基线", GREEN), rc=GREEN),
    dict(name="凸包正常运行区（Convex NOZ）", cat="多变量报警 · 书 3.1.2",
         source="Yu et al., 2020",
         principle=["用正常运行区的凸包界定安全域，包外即报警",
                    "几何直观、无需分布假设",
                    "凸假设在非凸工况下偏保守"],
         impl="src/iia_benchmark/models/multivariate.py · ConvexHullNOZAlarm",
         result=["TEP F1 0.7210（低于 Mahalanobis 0.8921）"],
         usage="正常运行区边界清晰、样本覆盖充分的过程",
         badge=None),
    dict(name="搜索锥 NOZ（非凸）", cat="多变量报警 · 书 3.1.3",
         source="Wang et al., 2024",
         principle=["以搜索锥刻画非凸正常运行区，突破凸包保守性",
                    "对多工况簇状分布的边界刻画更贴合",
                    "实现含几何搜索，计算量高于凸包"],
         impl="src/iia_benchmark/models/multivariate.py · SearchConeNOZAlarm",
         result=["TEP F1 0.6385（迁移成绩低于凸包，负结果保留）"],
         usage="正常区明显非凸（多工况分离）的过程",
         badge=None),
    dict(name="变化方向报警（ATG + 规则矩阵）", cat="多变量报警 · 书 3.2",
         source="Chen et al., 2017",
         principle=["提取变量变化方向（升/降/不变）构成方向签名",
                    "规则矩阵匹配故障模式，天然带根因指向",
                    "对缓慢漂移型故障敏感"],
         impl="src/iia_benchmark/models/multivariate_book.py · VariationDirectionAlarm",
         result=["书表迁移完成（E2 级）"],
         usage="需要报警同时给出'哪个变量往哪变'的解释场景",
         badge=None),
    dict(name="贝叶斯时变回归（泵类设备）", cat="多变量报警 · 书 3.3",
         source="Xiong et al., 2018",
         principle=["滑动窗口贝叶斯回归刻画变量间时变关系",
                    "残差偏离即异常，系数轨迹可解释",
                    "适合泵、压缩机等旋转设备监测"],
         impl="src/iia_benchmark/models/multivariate_book.py · BayesianWindowRegressionAlarm",
         result=["书例迁移完成（E2 级）"],
         usage="变量间有明确物理回归关系的设备级监测",
         badge=None),
    dict(name="自适应多变量套件（M0–M3）", cat="多变量报警 · 本仓库适配层",
         source="本仓库原创适配",
         principle=["稳健收缩 Mahalanobis：小样本/病态协方差下更稳",
                    "block 校准 + 自动路由 + 显式拒绝",
                    "与单变量套件同构的消融链设计"],
         impl="src/iia_benchmark/models/adaptive_multivariate.py",
         result=["TEP/PRONTO/SKAB 迁移已跑通，PRONTO 上整体偏弱（负结果保留）"],
         usage="多变量方向的自适应扩展基座，后续消融与论文实验平台",
         badge=None),
    dict(name="因果 / 根因分析族", cat="多变量 · 书 Chapter 4",
         source="TE ranker、NTE/NDTE、IGTE/IGDTE、递归 BN、PLR 贡献度",
         principle=["传递熵（TE）及其归一化/动态变体做因果排序",
                    "递归贝叶斯网络做故障传播路径推断",
                    "PLR 贡献度定位异常变量"],
         impl="src/iia_benchmark/models/root_cause.py / root_cause_book.py",
         result=["iMAKS 合成因果真值上完成 smoke 验证（合成结果不进榜单）"],
         usage="报警之后归因：从'检测到异常'走向'解释异常来源'",
         badge=None),
]

ALG_MTSAD = [
    dict(name="TimesNet", cat="MTSAD 深度复现 · ICLR 2023",
         source="Wu et al., 2023（官方实现逐篇转录，带 # Grounding 溯源标签）",
         principle=["把一维时序折叠到二维（TimesBlock），捕获多周期结构",
                    "重建误差作为异常分数",
                    "通用时序骨干，异常检测是其任务头之一"],
         impl="src/iia_benchmark/models/mtsad_detectors.py（统一 Detector 接口）",
         result=["5/5 数据集复现与论文差 ≤ 0.03：SMD 0.8542 vs 0.8512；SWaT 0.9293 vs 0.9210",
                 "目前复现质量最好的深度方法"],
         usage="MTSAD 研究的可靠骨干；新方法的强基线",
         badge=("复现最佳", GREEN), rc=GREEN),
    dict(name="Anomaly Transformer", cat="MTSAD 深度复现 · ICLR 2022 Spotlight",
         source="Xu et al., 2022",
         principle=["Anomaly-Attention：关联差异（association discrepancy）建模",
                    "极小极大对抗训练放大异常的可区分性",
                    "逐点异常分数 + 可学习阈值"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["SMD 0.9034 vs 第三方重跑 0.9033（Δ 0.0001），PSM matches",
                 "MSL / SWaT off（0.8557 vs 0.9393）——协议差异待归因"],
         usage="注意力类异常检测的代表；部分数据集可放心引用",
         badge=("部分吻合", ORANGE), rc=ORANGE),
    dict(name="DCdetector", cat="MTSAD 深度复现 · KDD 2023",
         source="Yang et al., 2023",
         principle=["双注意力对比结构，无监督对比正常/异常表示",
                    "不依赖重建，直接学习判别性表示"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["4/5 数据集 near：SWaT 0.9518 vs 0.9633",
                 "MSL 缺口经 4 种子 + 200 分位阈值扫描排除随机性，oracle 上限 0.8646 < 0.9359，记录为开放差异"],
         usage="对比学习路线代表；MSL 结果引用需谨慎",
         badge=("4/5 接近", ORANGE), rc=ORANGE),
    dict(name="USAD", cat="MTSAD 深度复现 · KDD 2020",
         source="Audibert et al., 2020",
         principle=["双解码器自编码器 + 对抗式重建训练",
                    "训练快、推理轻"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["复现失败（off）——载荷与实现差异排查中"],
         usage="暂缓作为基线引用，待复现闭环",
         badge=("复现失败", RED), rc=RED),
    dict(name="TranAD", cat="MTSAD 深度复现 · VLDB 2022",
         source="Tuli et al., 2022",
         principle=["Transformer + 对抗训练 + 自条件化（self-conditioning）",
                    "论文报告了 NAB/UCR/MBA 单变量扩展"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["复现失败（off）；SWaT 基准载荷与论文表差 1,800 行，随结果如实报告、不静默裁剪"],
         usage="暂缓引用；其单变量扩展数据集已在库（NAB/UCR/MBA）",
         badge=("复现失败", RED), rc=RED),
    dict(name="GCAD", cat="MTSAD 深度复现",
         source="图神经 + 因果异常检测路线",
         principle=["变量建图 + 因果结构约束的异常评分"],
         impl="scripts/mtsad/ 侧评估管线（AUROC/AUPRC）",
         result=["AUROC 0.39–0.70 vs 论文 0.73–0.95，差距大，verdict withheld（暂不下结论）"],
         usage="暂缓引用；差距大到疑似协议/载荷问题，优先排查",
         badge=("存疑", RED), rc=RED),
    dict(name="iTransformer / CATCH", cat="MTSAD 深度复现 · 已转录",
         source="iTransformer（ICLR 2024）；CATCH",
         principle=["iTransformer：倒置注意力（变量为 token）",
                    "CATCH：通道感知的频域/ patch 建模",
                    "两者均按官方实现逐篇转录并接入统一接口"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["转录完成，系统性五基准复现排期中"],
         usage="下一代骨干候选，等待复现审计",
         badge=("待审计", MID)),
    dict(name="经典检测器族", cat="MTSAD · 经典基线组",
         source="PCA / kNN / Isolation Forest / OCSVM / Mahalanobis（逐点打分）",
         principle=["逐点打分、无训练或轻训练，作为深度方法的对照组",
                    "统一 Detector 接口，与深度方法共用评估协议"],
         impl="src/iia_benchmark/models/mtsad_detectors.py",
         result=["OCSVM-MSL matches（0.7015 vs 0.7082），其余多为 off —— 经典方法弱于深度方法的论文结论在本仓库得到支持"],
         usage="低成本基线组；协议验证的试金石",
         badge=None, rc=GREEN),
]

ALG_FLOOD = [
    dict(name="CASIM", cat="报警洪泛分类 · SOTA",
         source="报警序列相似性 + 模式挖掘路线",
         principle=["把报警洪泛片段编码为序列模式做开集分类",
                    "对已知故障类区分度极高"],
         impl="configs/algorithms/sota_algorithms.json 注册；实现见 src/iia_benchmark 报警族",
         result=["BA：TEP 1.0000 / NPP 0.8182 / FCC 0.9922",
                 "700/700 open-set fit 完成，但阈值/均值有差（P2 级）"],
         usage="报警洪泛分类当前最强；TEP/FCC 上接近满分",
         badge=("当前最强", GREEN), rc=GREEN),
    dict(name="ConE-AFC", cat="报警洪泛分类 · SOTA + conformal",
         source="作者工件级复现对象",
         principle=["对比嵌入 + 报警洪泛分类头",
                    "独立 conformal 层给出分布外校准"],
         impl="tests/test_cone_afc.py / test_cross_conformal_afc.py 对应实现",
         result=["250/250 作者任务通过；95/95 表均值在 0.02 内",
                 "独立 conformal 层 1,800/1,800 完全一致（仍非 P3 级）"],
         usage="复现纪律的标杆案例；conformal 不确定度可迁移到其他方法",
         badge=("复现标杆", GREEN), rc=GREEN),
    dict(name="HDAM / CTFH / TF-IDF-LSTM 等", cat="报警洪泛分类 · SOTA 族",
         source="sota_algorithms.json 注册 10 个方法",
         principle=["HDAM：层次化深度报警建模；CTFH：跨域迁移哈希",
                    "conformal 系列：分布外校准包装器"],
         impl="configs/algorithms/sota_algorithms.json",
         result=["HDAM BA：TEP 0.9967 / NPP 0.6932 / FCC 0.9375",
                 "PRONTO 表示错配导致 CTFH / HDAM 坍缩（BA 0.25 / 0.10），负结果保留",
                 "多种复杂方法未稳定超越 Jaccard 基线（0.9667 / 0.6932 / 0.9115）"],
         usage="论文对比组；PRONTO 迁移失败案例适合写'表示错配'分析",
         badge=("含负结果", ORANGE), rc=ORANGE),
    dict(name="flow_matching 生成式异常检测族", cat="生成式 MTSAD · 子项目（进行中）",
         source="projects/flow_matching_research/ 独立子项目",
         principle=["流匹配目标变体：independent / OT / rectified / SB / SF2M / VP / reflow",
                    "GRASP、CFM-TS、MOMENT 预训练零样本、MaelNet、CrossAD、TimeGAN/GT-GAN、BRITS/SAITS",
                    "严格无 point-adjustment 协议：验证集 99 分位定阈值"],
         impl="src/iia_benchmark/models/fm_*.py；结果 projects/flow_matching_research/results/2026-10-11/",
         result=["严格协议下 F1 百分数仅 3–41（rectified：SMD 41.07 / SWaT 28.96 / PSM 3.23）；MOMENT 零样本 2.6–42.6",
                 "7,117 条登记指标中 5,341 条缺结果，730 个严格组合仅 210 个完成 —— 现阶段不宜引用"],
         usage="探索性研究方向；与 PA 协议分数不可直接比较",
         badge=("进行中", ORANGE), rc=ORANGE),
]


DATASETS = [
    dict(name="SMD（Server Machine Dataset）", type="多变量（38 维）", scale="708,405 样本点；28 台机器",
         label="逐点异常标签，异常率 4.16%",
         status="已下载并逐文件 SHA-256 固定（TSLib 镜像）",
         desc=["服务器集群监控指标（CPU、内存、网络等）",
               "MTSAD 论文标配基准，训练/测试划分沿用 TSLib 约定",
               "本仓库复现锚点：TimesNet 0.8542、AT 0.9034（均 matches）"]),
    dict(name="MSL（Mars Science Laboratory）", type="多变量（55 维）", scale="58,317 样本点",
         label="逐点异常标签，异常率约 10.5%",
         status="已下载并逐文件 SHA-256 固定",
         desc=["NASA 火星好奇号 rover 遥测数据",
               "维度最高、通道间相关性复杂",
               "复现难点数据集：AT/DCdetector 在 MSL 上均为 off，DCdetector 缺口记录为开放差异"]),
    dict(name="SMAP（Soil Moisture Active Passive）", type="多变量（25 维）", scale="135,183 样本点",
         label="逐点异常标签，异常率约 12.8%",
         status="已下载并逐文件 SHA-256 固定",
         desc=["NASA 卫星遥测数据",
               "与 MSL 同属航天器遥测族，常联合报告"]),
    dict(name="PSM（Pooled Server Metrics）", type="多变量（25 维）", scale="132,481 样本点",
         label="逐点异常标签，异常率 27.76%（最高）",
         status="已下载并逐文件 SHA-256 固定",
         desc=["eBay 服务器池监控（TranAD 论文提出）",
               "异常率接近 28%，对阈值策略和训练污染鲁棒性是考验",
               "严格协议（无 PA）下生成式方法在 PSM 上仅 3.23 —— 高异常率放大了协议差异"]),
    dict(name="SWaT（Secure Water Treatment）", type="多变量（51 维）", scale="495,000 样本点",
         label="逐点异常标签，异常率约 12.2%",
         status="已下载并逐文件 SHA-256 固定",
         desc=["新加坡水处理厂工控安全测试床，攻击注入型异常",
               "工业控制安全领域旗舰基准",
               "复现锚点：TimesNet 0.9293 vs 论文 0.9210（matches）；载荷与 TranAD 论文表差 1,800 行，如实报告"]),
    dict(name="TEP（Tennessee Eastman Process）", type="多变量（52 过程变量）", scale="44 run（2 正常 + 42 故障）",
         label="故障起点已登记", status="已下载（Git 镜像）",
         desc=["化工过程仿真标杆，21 类预定义故障",
               "本仓库工业报警主线的核心测试床",
               "成绩锚点：单变量自适应 ECDF F1 0.9502；多变量 Mahalanobis 0.8921"],
         note=["衍生 tep_alarm_dataport 报警序列：50 报警位 × 1,000 run × 5 类，用于洪泛分类"]),
    dict(name="PRONTO", type="多变量（17 过程变量 + 报警列）", scale="3 个测试日，45,420 样本",
         label="故障阶段标签", status="已下载（1.72 GB，CRC 校验通过）",
         desc=["工业报警研究专用基准（含过程变量与报警日志双通道）",
               "报警洪泛与根因分析的标准测试床"],
         note=["洪泛方法迁移到此数据出现表示错配坍缩（CTFH BA 0.25 / HDAM 0.10），负结果保留——本身就是有价值的研究现象"]),
    dict(name="SKAB（Skoltech Anomaly Benchmark）", type="多变量（8 传感器）", scale="35 组实验 CSV",
         label="原生逐点异常标签（本仓库不做 point adjustment）", status="已下载",
         desc=["水泵试验台：阀开度、转速可控注入异常",
               "小而精，适合快速迭代与方法学实验"],
         note=["基线 F1 仅 0.1448，自适应 B4 升至 0.4124 但 FAR 仍 55 次/小时 —— 低信噪比场景的负结果同样保留"]),
    dict(name="NAB（Numenta Anomaly Benchmark）", type="单变量序列族", scale="7 组序列（本仓库镜像子集）",
         label="有标签", status="已下载（TranAD 仓库镜像）",
         desc=["真实流式单变量时序（AWS 指标、交通、温度等）",
               "自带 NAB 评分体系（早期检出奖励）",
               "单变量方向的公开可比基准"]),
    dict(name="UCR Archive 2021（+ MBA 心律）", type="单变量为主", scale="UCR 整包 zip 184 MB；MBA 心律序列",
         label="有标签", status="已下载，校验通过",
         desc=["UCR 时序档案是单变量分类/异常检测最大公开语料",
               "MBA 心律为 TranAD 单变量扩展数据",
               "下周计划：在 UCR/NAB 上建立自适应 ECDF 套件基线表"]),
    dict(name="GECCO 2018 / SWAN-SF", type="GECCO 多变量水质；SWAN 多变量太阳物理", scale="SWAN-SF 约 6.5 GB",
         label="有标签", status="已下载，校验通过",
         desc=["GECCO 2018：饮用水水质事件检测挑战赛数据",
               "SWAN-SF：太阳活动多变量流式基准（24 通道磁图特征）",
               "用于 flow_matching / CrossAD 等子项目扩展评估"]),
    dict(name="报警序列数据集（TEP / NPP / FCC）", type="二值报警位序列", scale="TEP 50 位×1,000 run×5 类；NPP 192 位×1,212 run；FCC 57 位×1,600 run×16 类",
         label="故障/事故类别标签", status="已取得并审计",
         desc=["报警洪泛分类任务的标准输入（事件序列而非连续时序）",
               "CASIM 在 TEP/FCC 上 BA ≈ 1.0 / 0.99，NPP 0.82"],
         note=["FCC 旧固定切分曾含重复轨迹泄漏，重做后 alignment BA 从 0.8875 降至 0.7736 —— 切分审计教训"]),
    dict(name="iMAKS（合成因果）", type="多变量（22 传感器）", scale="211,200 行",
         label="合成因果真值", status="已下载（登记为合成数据）",
         desc=["带显式因果图真值的合成过程数据",
               "仅用于因果/根因方法的 smoke 验证",
               "按纪律不得进入真实工业榜单"]),
    dict(name="WADI / MSDS", type="多变量", scale="—",
         label="仅登记元数据", status="未下载（需申请）",
         desc=["WADI：水分配网络工控安全测试床（SWaT 姊妹集）",
               "MSDS：工业监控系统数据集",
               "下周启动数据申请流程"]),
]

# ---------------------------------------------------------------- build

c_verdict = chart_mtsad_verdict()
c_timesnet = chart_timesnet()
c_atdc = chart_at_dc()
c_tep = chart_tep_f1()
c_flood = chart_flood_ba()
c_scale = chart_dataset_scale()
c_fm = chart_fm_status()

# 1 封面
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, prs.slide_width, prs.slide_height, DARK)
tb = slide.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(3.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "时间序列异常检测研究盘点"
r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = "Microsoft YaHei"
p = tf.add_paragraph()
r = p.add_run(); r.text = "—— 方法、数据集与实验进展（周例会汇报）"
r.font.size = Pt(24); r.font.color.rgb = LIGHT; r.font.name = "Microsoft YaHei"
p = tf.add_paragraph()
r = p.add_run(); r.text = "IIA_benchmark 仓库 · 2026-10-14"
r.font.size = Pt(16); r.font.color.rgb = LIGHT; r.font.name = "Microsoft YaHei"

# 2 目录
content_slide("汇报框架", [
    B("一、研究全景：三条研究线与方法总览", bold=True, color=DARK, off=2),
    B("二、单变量异常检测方法（6 个，逐页介绍）", bold=True, color=DARK, off=2),
    B("三、多变量工业报警方法（7 个，逐页介绍）", bold=True, color=DARK, off=2),
    B("四、MTSAD 深度复现族（8 组，逐页介绍）", bold=True, color=DARK, off=2),
    B("五、报警洪泛与生成式检测（4 组）", bold=True, color=DARK, off=2),
    B("六、数据集档案（14 个，逐页介绍）", bold=True, color=DARK, off=2),
    B("七、实验结果与图表分析", bold=True, color=DARK, off=2),
    B("八、问题风险与下周计划", bold=True, color=DARK, off=2),
])

# 3 全景
content_slide("研究全景：三条研究线", [
    H("线 1 · 工业报警主线（书籍 20 算法 + 10 个 SOTA）"),
    B("单通道变量→二值报警（T1）、多变量报警（T2）、报警洪泛分类（T3）、根因分析"),
    B("测试床：TEP / PRONTO / SKAB + 报警序列数据"),
    H("线 2 · MTSAD 深度复现（2026-09 起）"),
    B("统一 Detector 接口转录官方实现；SMD/MSL/SMAP/PSM/SWaT 五基准 SHA-256 固定"),
    B("自建 E1–E4 / P2–P3 证据分级，复现结论可审计"),
    H("线 3 · flow_matching 生成式检测子项目（进行中）"),
    B("流匹配目标变体 + GRASP / CFM-TS / MOMENT 零样本等，严格无 PA 协议"),
    H("关键纪律"),
    B("合成 smoke 结果只验证管线，不进排行榜；负结果保留；协议差异如实报告", color=RED),
])

# 4 方法总览表
slide = content_slide("方法总览（30 个正式注册算法）", [
    H("单变量（6）"),
    B("阈值基线 / IID delay timer / Non-IID delay / Non-IID deadband / APP / 自适应 ECDF 套件"),
    H("多变量工业报警（7）"),
    B("Mahalanobis MSPC / 凸包 NOZ / 搜索锥 NOZ / 变化方向 / 贝叶斯时变回归 / 自适应多变量 / 因果根因族"),
    H("MTSAD 深度复现（8 组）"),
    B("TimesNet / Anomaly Transformer / DCdetector / USAD / TranAD / GCAD / iTransformer·CATCH / 经典基线组"),
    H("报警洪泛 + 生成式（4 组）"),
    B("CASIM / ConE-AFC / HDAM·CTFH 族 / flow_matching 变体族"),
    H("证据等级现状"),
    B("仅 1 个 E4（Xu 2012）、0 个 P3；多数 E2/P2 —— '可调用+单测过' ≠ 论文复现", color=ORANGE),
], image=c_verdict, img_w=Inches(5.4))

# 5-10 单变量
section_slide("02", "单变量异常检测方法", "书 Chapter 2 系列 + 本仓库自适应适配层（B0–B7 消融）")
for a in ALG_UNIV:
    algo_slide(a)

# 11-17 多变量工业
section_slide("03", "多变量工业报警方法", "书 Chapter 3–4：正常运行区 / 方向规则 / 时变回归 / 因果根因")
for a in ALG_MULT:
    algo_slide(a)

# 18-25 MTSAD
section_slide("04", "MTSAD 深度复现族", "统一 Detector 接口 · 官方实现逐篇转录 · 五基准复现审计（124 runs / 32 对比对）")
for a in ALG_MTSAD:
    algo_slide(a)

# 26-29 洪泛与生成式
section_slide("05", "报警洪泛与生成式检测", "报警序列分类 SOTA + flow_matching 生成式子项目")
for a in ALG_FLOOD:
    algo_slide(a)

# 30 数据集总览
section_slide("06", "数据集档案", "14 个数据集：多变量 MTSAD 五基准 / 工业过程 / 单变量族 / 报警序列 / 待申请")
content_slide("数据集全景", [
    H("多变量 · 逐点标签（核心基准）"),
    B("SMD / MSL / SMAP / PSM / SWaT —— 已下载，逐文件 SHA-256 固定（TSLib 镜像）"),
    H("工业过程"),
    B("TEP（52 变量 44 run）/ PRONTO（1.72 GB，CRC 通过）/ SKAB（原生逐点标签）"),
    H("单变量"),
    B("NAB / UCR Archive 2021（184 MB）/ MBA 心律 / GECCO 2018 / SWAN-SF（6.5 GB）"),
    H("报警序列（洪泛分类）"),
    B("TEP 50 位×1,000 run / NPP 192 位×1,212 run / FCC 57 位×1,600 run"),
    H("受限与合成"),
    B("iMAKS（合成因果，仅 smoke）；WADI / MSDS 未下载（待申请）", color=ORANGE),
], image=c_scale, img_w=Inches(5.6))

for d in DATASETS:
    dataset_slide(d)

# 结果章节
section_slide("07", "实验结果与图表分析", "所有数值可溯源至 ara_mtsad/evidence 与 docs/reports 审计表")

content_slide("MTSAD 复现总账：5 matches / 9 near / 18 off", [
    H("结论"),
    B("TimesNet 是复现质量最好的深度方法（5/5 数据集 Δ≤0.03）"),
    B("Anomaly Transformer / DCdetector 部分数据集吻合；TranAD / USAD / GCAD 复现失败"),
    B("经典基线组整体弱于深度方法，支持论文结论"),
    H("方法学发现（重要）"),
    B("同一预测下 point-wise 与 point-adjusted F1 差 2.8–116 倍", color=RED),
    B("affiliation 与 PA 两族指标差达 0.53 —— 跨论文比较前必须对齐指标族、阈值规则、载荷版本", color=RED),
], image=c_verdict, img_w=Inches(5.2))

content_slide("TimesNet 复现：5/5 数据集吻合", [
    H("数值（复现 vs 论文，point-adjusted F1）"),
    B("SMD 0.8542 vs 0.8512（Δ 0.003）"),
    B("SWaT 0.9293 vs 0.9210（判 matches）"),
    B("其余三数据集 Δ 均 ≤ 0.03"),
    H("意义"),
    B("TimesNet 可作为本仓库 MTSAD 研究的可靠骨干与强基线"),
], image=c_timesnet, img_w=Inches(5.9))

content_slide("AT / DCdetector：部分吻合与开放差异", [
    H("Anomaly Transformer"),
    B("SMD 0.9034 vs 第三方重跑 0.9033（Δ 0.0001）；PSM matches"),
    B("MSL / SWaT off（0.8557 vs 0.9393）"),
    H("DCdetector"),
    B("4/5 near：SWaT 0.9518 vs 0.9633"),
    B("MSL：4 种子 + 200 分位阈值扫描后 oracle 上限 0.8646 < 0.9359，记为开放差异"),
    H("复现失败组"),
    B("TranAD / USAD off；GCAD AUROC 0.39–0.70 vs 论文 0.73–0.95（verdict withheld）", color=RED),
], image=c_atdc, img_w=Inches(6.6))

content_slide("工业报警主线：TEP 成绩", [
    H("单变量（三种子均值）"),
    B("IID delay：F1 0.8678 / FAR 0.0067 / MAR 0.1840"),
    B("自适应双侧 ECDF：F1 0.9502（+0.082）", color=GREEN),
    H("多变量"),
    B("Mahalanobis 0.8921 > 凸包 NOZ 0.7210 > 搜索锥 0.6385"),
    H("迁移负结果（保留）"),
    B("SKAB 基线 0.1448（B4 升至 0.4124 但 FAR 55 次/小时）；PRONTO/SKAB 多变量全面偏弱", color=ORANGE),
    H("论文级验证"),
    B("Xu 2012 达 E4：Table VII 最大误差 5.94e-05；ConE-AFC conformal 层 1,800/1,800 一致", color=GREEN),
], image=c_tep, img_w=Inches(5.9))

content_slide("报警洪泛分类：复杂方法未稳定超基线", [
    H("Balanced Accuracy（TEP / NPP / FCC）"),
    B("CASIM 1.0000 / 0.8182 / 0.9922 —— 当前最强"),
    B("HDAM 0.9967 / 0.6932 / 0.9375"),
    B("Jaccard 基线 0.9667 / 0.6932 / 0.9115 —— 多种复杂方法未能超越", color=ORANGE),
    H("负结果与教训"),
    B("PRONTO 表示错配：CTFH / HDAM 坍缩至 BA 0.25 / 0.10"),
    B("FCC 旧切分含重复轨迹泄漏：重做后 alignment BA 0.8875 → 0.7736（切分审计的必要性实证）", color=RED),
], image=c_flood, img_w=Inches(5.9))

content_slide("flow_matching 子项目：进行中，不宜引用", [
    H("进展"),
    B("7,117 条登记指标，1,776 条有实测，5,341 条缺结果"),
    B("730 个严格异常检测组合仅 210 个完成预定重复"),
    H("已有信号（严格无 PA 协议，验证集 99 分位阈值）"),
    B("流匹配变体 F1 百分数 3–41：rectified SMD 41.07 / SWaT 28.96 / PSM 3.23"),
    B("MOMENT 零样本 2.6–42.6"),
    H("引用纪律"),
    B("严格协议分数与论文 PA 分数不可直接比较；现阶段不对外引用", color=RED),
], image=c_fm, img_w=Inches(4.9))

# 风险与计划
section_slide("08", "问题、风险与下周计划", "")

content_slide("问题与风险", [
    H("1 · 排行榜为空"),
    B("leaderboard-eligible split 数量 = 0；目前只有复现审计表，无对外榜单"),
    H("2 · 证据等级偏低"),
    B("30 个正式算法仅 1 个 E4、0 个 P3；多数停留在 E2/P2"),
    H("3 · 复现失败未闭环"),
    B("TranAD / USAD / GCAD 无法复现；DCdetector-MSL 缺口为开放差异"),
    H("4 · 协议敏感性强"),
    B("point adjustment 可放大 F1 达 116 倍；结果表述必须绑定协议说明"),
    H("5 · 合成结果纪律"),
    B("synthetic smoke 高分（如 F1 0.9950）只验证管线，严禁当作 benchmark 性能"),
])

content_slide("下周计划", [
    B("1. 排查 TranAD / USAD / GCAD 复现失败原因（载荷 vs 实现）；DCdetector-MSL 归因实验", bold=True, color=DARK),
    B("2. 补齐 leaderboard-eligible split：分组切分 + 冻结超参 + 数据审计 + 多种子不确定性", bold=True, color=DARK),
    B("3. flow_matching：优先完成 730 个严格组合中已预定重复（当前 210/730）", bold=True, color=DARK),
    B("4. 单变量方向：自适应 ECDF 套件在 UCR / NAB 上建立基线表", bold=True, color=DARK),
    B("5. 启动 WADI / MSDS 数据申请流程", bold=True, color=DARK),
])

# 总结
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, prs.slide_width, prs.slide_height, DARK)
tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(11.3), Inches(3.2))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "一句话总结"
r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = RGBColor(0x9D, 0xC3, 0xE6); r.font.name = "Microsoft YaHei"
for line in ["MTSAD 五基准 + TimesNet / AT / DCdetector 复现管线是当前最可靠的研究起点；",
             "单变量方向有自适应 ECDF 套件 + UCR/NAB 数据可直接开工；",
             "flow_matching 子项目尚需时间，现阶段不引用其结果。"]:
    p = tf.add_paragraph()
    r = p.add_run(); r.text = line
    r.font.size = Pt(20); r.font.color.rgb = WHITE; r.font.name = "Microsoft YaHei"
    p.space_after = Pt(10)

prs.save(OUT_PPTX)
print(f"OK: {OUT_PPTX}  slides={len(prs.slides.__iter__.__self__._sldIdLst)}")
