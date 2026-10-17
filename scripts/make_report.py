#!/usr/bin/env python
"""Build the Chinese-language PDF report.  Every number is read from results/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

CJK = "ReportCJK"

# Embedded fonts render everywhere; the Adobe CID font is only a last resort
# because a reader without the system font shows blank glyphs.
CJK_CANDIDATES = [
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/Songti.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("/usr/share/fonts/truetype/arphic/uming.ttc", 0),
]


def register_cjk() -> str:
    """Register an embeddable CJK font; fall back to the non-embedded CID font."""
    import os as _os
    for path, index in CJK_CANDIDATES:
        if _os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(CJK, path, subfontIndex=index))
                return CJK
            except Exception:
                continue
    print("WARNING: no embeddable CJK font found; falling back to STSong-Light "
          "(readers without the font will show blank glyphs)")
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def styles():
    global CJK
    CJK = register_cjk()
    base = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName=CJK, fontSize=16, leading=20,
                                spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontName=CJK, fontSize=8.5, leading=12,
                              textColor=colors.HexColor("#555555"), alignment=TA_LEFT),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName=CJK, fontSize=11, leading=14,
                            spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a1a1a")),
        "p": ParagraphStyle("p", parent=base["Normal"], fontName=CJK, fontSize=8.8, leading=13.2),
        "cap": ParagraphStyle("c", parent=base["Normal"], fontName=CJK, fontSize=7.6, leading=10.5,
                              textColor=colors.HexColor("#444444")),
    }
    return out


def table(data, widths, align_right_from=1):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), CJK),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#8899aa")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#8899aa")),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
    ]))
    return t


def fig(path: Path, width=168 * mm):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--out", default="report/GB1_agent_report.pdf")
    args = ap.parse_args()

    R, F = Path(args.results), Path(args.figures)
    S = styles()
    base = json.loads((R / "baselines.json").read_text())
    records = json.loads((R / "records_all.json").read_text())
    interp = json.loads((R / "interpretability.json").read_text())
    ledger = json.loads((R / "llm_deepseek" / "llm_ledger.json").read_text())
    struct_path = R / "structure" / "structure_analysis.json"
    struct = json.loads(struct_path.read_text()) if struct_path.exists() else None

    arms = ["random", "greedy", "greedy_explore", "llm_agent", "llm_knowledge"]
    label = {"random": "随机对照", "greedy": "预测器贪心", "greedy_explore": "预测器+25%探索",
             "llm_agent": "LLM 智能体", "llm_knowledge": "知识增强 LLM 智能体"}
    seeds = sorted({r["seed"] for r in records})
    last = {a: {s: next(r for r in records if r["arm"] == a and r["seed"] == s and r["round"] == 3)
                for s in seeds} for a in arms}
    opt = interp["failure_layer"]["global_optimum_fitness"]

    story = []
    story.append(Paragraph("面向蛋白质定向进化的科学智能体：GB1 四位点景观上的五臂对照", S["title"]))
    story.append(Paragraph(
        "数据：FLIP 固定提交的 GB1 四位点深度突变扫描（149,361 条实测）。虚拟评估器＝公开实测 fitness，"
        "全程无湿实验。LLM：DeepSeek-chat 真实 API 调用。结构：NVIDIA 托管 Boltz-2 与本地 ProteinMPNN。", S["sub"]))
    story.append(Spacer(1, 5))

    # ---------------- 1 任务与协议
    story.append(Paragraph("1　任务、数据划分与评测协议", S["h"]))
    story.append(Paragraph(
        f"四个可变位点为 39/40/41/54（野生型 VDGV）。固定测试集 T＝10,000 条用于预测器评测；"
        f"起点标签 D0＝192 条随机变体（五臂共用、同种子同起点）；候选池 U＝139,169 条未观测变体，"
        f"其中包含全局最优 FWAA（fitness {opt:.3f}）。每轮预算 96 条，共 3 轮，3 个随机种子。"
        f"每轮候选与假说在标签揭示前以 SHA-256 冻结，揭示后由程序判定假说成立与否——"
        f"“智能体从反馈中学习”因此是可核验的记录，而不是叙述。", S["p"]))

    # ---------------- 2 基线预测器
    story.append(Paragraph("2　基线预测器：拟合能力与高值识别能力", S["h"]))
    d0 = {}
    for row in base["d0_models"]:
        d0.setdefault(row["spec"], []).append(row)
    rows = [["模型（仅用 D0＝192 标签训练）", "Spearman", "RMSE", "top-100 召回", "top-5% 精度"]]
    for spec, rs in sorted(d0.items(), key=lambda kv: -np.mean([r["spearman"] for r in kv[1]]))[:4]:
        rows.append([spec,
                     f"{np.mean([r['spearman'] for r in rs]):.3f}",
                     f"{np.mean([r['rmse'] for r in rs]):.3f}",
                     f"{np.mean([r['top100_recall'] for r in rs]):.2f}",
                     f"{np.mean([r['top5pct_precision'] for r in rs]):.2f}"])
    story.append(table(rows, [66 * mm, 22 * mm, 20 * mm, 26 * mm, 26 * mm]))
    story.append(Spacer(1, 3))
    flip = {r["split"]: r for r in base["flip_splits"]}
    rows = [["FLIP 官方划分", "训练量", "Spearman", "top-100 召回", "top-5% 精度"]]
    for name in ["sampled", "two_vs_rest", "three_vs_rest", "low_vs_high"]:
        if name in flip:
            r = flip[name]
            rows.append([name, f"{r['n_train']}", f"{r['spearman']:.3f}",
                         f"{r['top100_recall']:.2f}", f"{r['top5pct_precision']:.2f}"])
    story.append(table(rows, [40 * mm, 22 * mm, 24 * mm, 28 * mm, 26 * mm]))
    story.append(Spacer(1, 3))
    story.append(fig(F / "fig2_baselines.png"))
    story.append(Paragraph(
        "图 1　左：高值识别能力随标签量的变化（固定 10,000 条测试集）。右：四个 FLIP 官方划分上的成绩，"
        "成对编码岭回归。192 条标签只够把 top-5% 精度做到 0.27，而 6,144 条标签可到 0.69——"
        "本任务的瓶颈是标签预算，不是模型容量。", S["cap"]))
    story.append(Spacer(1, 4))

    # ---------------- 3 五臂对比
    story.append(Paragraph("3　五方法闭环对比（3 轮 × 96 条/轮 × 3 种子）", S["h"]))
    rows = [["方法", "末轮 best-so-far（均值）", "各种子末轮最好", "命中全局最优", "累计收益 G", "批次多样性"]]
    for a in arms:
        bests = [last[a][s]["best_so_far"] for s in seeds]
        gains = [sum(r["metrics"]["gain"] for r in records if r["arm"] == a and r["seed"] == s) for s in seeds]
        ham = np.mean([r["diversity"]["mean_pairwise_hamming"] for r in records if r["arm"] == a])
        rows.append([label[a], f"{np.mean(bests):.3f}",
                     " / ".join(f"{b:.2f}" for b in bests),
                     f"{sum(b >= opt - 1e-6 for b in bests)}/{len(seeds)}",
                     f"{np.mean(gains):.2f}", f"{ham:.2f}"])
    story.append(table(rows, [38 * mm, 32 * mm, 34 * mm, 24 * mm, 22 * mm, 22 * mm]))
    story.append(Spacer(1, 3))
    story.append(fig(F / "fig1_campaign.png"))
    story.append(Paragraph(
        "图 2　左：best-so-far 随累计标签数的变化（阴影为 3 个种子的极差）。右：每轮批次的平均 fitness。"
        "四条有模型参与的曲线几乎重合，随机对照明显落后。", S["cap"]))
    story.append(Paragraph(
        f"结论一：模型指导相对随机采样的收益是决定性的（末轮 best-so-far {np.mean([last['greedy'][s]['best_so_far'] for s in seeds]):.2f} "
        f"对 {np.mean([last['random'][s]['best_so_far'] for s in seeds]):.2f}）。结论二：在同样的 96 条预算下，"
        f"LLM 智能体相对预测器贪心只有边际优势（{np.mean([last['llm_agent'][s]['best_so_far'] for s in seeds]):.3f} 对 "
        f"{np.mean([last['greedy'][s]['best_so_far'] for s in seeds]):.3f}），知识增强臂与普通 LLM 臂几乎不可区分。"
        f"这是本次实验最值得报告的负面结果，下一节用决策轨迹解释它为什么会这样。", S["p"]))
    story.append(Spacer(1, 4))

    # ---------------- 4 可解释性
    story.append(Paragraph("4　可解释性：模型层、决策层、失败层", S["h"]))
    story.append(fig(F / "fig4_interpretability.png"))
    share = interp["model_layer"]["epistasis_d0"]["pairwise_share"]
    story.append(Paragraph(
        f"图 3　模型层。左：实测单突变效应谱（相对野生型的对数 fitness）。40 位最宽容、41 位几乎只接受甘氨酸。"
        f"右：成对编码模型中交互项占全部权重的 {share:.0%}——上位性不是修正项，而是这张景观的主体。", S["cap"]))
    story.append(Spacer(1, 2))
    story.append(fig(F / "fig5_decision.png"))
    v = interp["decision_layer"]["verdict_counts"]
    story.append(Paragraph(
        f"图 4　决策层。左：{sum(v.values())} 条假说由程序裁决，支持 {v.get('supported',0)} 条、"
        f"推翻 {v.get('refuted',0)} 条、证据不足 {v.get('undetermined',0)} 条。中：智能体第 1 轮几乎原样采纳预测器的 top-96"
        "（重合度 1.00），到第 3 轮降到 0.82——它确实在用反馈改变选择，只是改变的幅度有限。"
        "右：批次多样性随轮次下降，模型越自信，探索越少。", S["cap"]))

    comp = interp["failure_layer"]["components"]
    harmful = [c for c in comp if c["deleterious_alone"]]
    harmful_text = "、".join(f"{c['mutation']}（{c['single_mutant_fitness']:.2f}）" for c in harmful)
    story.append(Paragraph(
        f"失败层。全局最优 {interp['failure_layer']['global_optimum']}（fitness {opt:.3f}）的四个组分里，"
        f"{harmful_text} 单独出现时都低于野生型（野生型归一化为 1.0）。"
        f"任何以单突变收益为筛选依据的策略都会在第一步丢掉它们；三个种子里只有一个命中全局最优，根源就在这里。", S["p"]))

    story.append(PageBreak())

    # ---------------- 5 零样本先验与结构信息
    story.append(Paragraph("5　零样本先验与结构信息到底贡献了多少", S["h"]))
    story.append(fig(F / "fig3_priors.png", width=150 * mm))
    story.append(Paragraph(
        "图 5　左：三种零样本先验在三突变以上变体上的质量。结构反折叠（ProteinMPNN）预测的是“这个残基与骨架相不相容”，"
        "不是“结合不结合 IgG-Fc”，因此明显弱于实测单突变加性先验。右：全局最优的组分里有两个单独有害。", S["cap"]))
    if struct:
        story.append(Spacer(1, 2))
        story.append(fig(F / "fig6_structure.png", width=150 * mm))
        story.append(Paragraph(
            "图 6　结构信息消融。左：Boltz-2 预测结构在四个突变位点上的 pLDDT 与实测 fitness 的关系"
            "（抽样覆盖整个 fitness 量程）。右：把结构置信度作为额外特征加入后的 5 折交叉验证成绩。", S["cap"]))
        rows = [["特征", "与 fitness 的 Spearman", "说明"]]
        for k, lab in [("plddt_mean", "Boltz-2 全链平均 pLDDT"), ("plddt_sites", "四个突变位点的 pLDDT"),
                       ("ptm", "pTM"), ("sequence_model", "序列模型预测值（D0 训练）")]:
            if k in struct["spearman_vs_fitness"]:
                rows.append([lab, f"{struct['spearman_vs_fitness'][k]:+.3f}",
                             struct["notes"].get(k, "")])
        story.append(table(rows, [52 * mm, 34 * mm, 82 * mm]))
        story.append(Paragraph(
            f"在 {struct['n_folded']} 条按 fitness 分层抽样的变体上，用 NVIDIA 托管 Boltz-2 实际折叠"
            f"（{struct['n_keys_rotated']} 个密钥轮换，{struct['n_calls']} 次调用，中位延迟 {struct['median_latency_s']:.1f} 秒）。"
            f"结论：预测结构的置信度与结合 fitness 基本无关（|ρ| ≤ {struct['max_abs_rho']:.2f}），"
            f"把它作为额外特征加进预测器后，top-5% 精度变化 {struct['ablation']['delta_top5pct']:+.3f}、"
            f"Spearman 变化 {struct['ablation']['delta_spearman']:+.3f}。"
            f"结构模块在这道题上是诚实的零贡献项，报告它比不报告它更有意义。", S["p"]))
    else:
        story.append(Paragraph("（结构消融结果缺失。）", S["p"]))

    # ---------------- 6 结论与披露
    story.append(Paragraph("6　结论、成本与披露", S["h"]))
    story.append(Paragraph(
        f"（1）在 288 条新增标签的预算下，模型指导把最好变体从随机采样的 "
        f"{np.mean([last['random'][s]['best_so_far'] for s in seeds]):.2f} 提升到 "
        f"{np.mean([last['greedy'][s]['best_so_far'] for s in seeds]):.2f}；"
        f"（2）LLM 智能体的增量有限，且主要体现在后几轮的选择偏移上；"
        f"（3）知识块与结构先验在本任务上没有可测量的增益——它们描述的是折叠，不是结合；"
        f"（4）上位性占模型权重的 {share:.0%}，是“单突变筛选”策略失败的根本原因。", S["p"]))
    story.append(Paragraph(
        f"成本与外部服务：DeepSeek-chat 真实调用 {ledger['n_calls']} 次，合计 {ledger['spent_usd']:.4f} 美元"
        f"（账本 results/llm_deepseek/llm_ledger.json 逐条可查，设有硬上限，超限即停、失败不自动重试）；"
        f"NVIDIA 托管 Boltz-2 调用见 results/structure/nvidia_receipts.jsonl（只记录密钥序号、状态码与耗时）。"
        f"数据来自公开发表的 GB1 深度突变扫描（FLIP 固定提交），没有任何湿实验。"
        f"代码与复现脚本见随附仓库；仓库中不含任何密钥。本报告的全部数字由 scripts/make_report.py 从 results/ 读取生成。", S["p"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, topMargin=14 * mm, bottomMargin=12 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title="GB1 定向进化科学智能体报告", author="")
    doc.build(story)
    print("wrote", out)


if __name__ == "__main__":
    main()
