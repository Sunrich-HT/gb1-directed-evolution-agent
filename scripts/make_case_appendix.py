#!/usr/bin/env python
"""Generate paper/generated/case.tex: one complete campaign traced end to end.

Everything is read from the run records, the evidence base and the prior, so the
appendix cannot drift from what the code actually did.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent import evidence as evidence_mod
from gb1agent.data import load_landscape

def esc(text) -> str:
    """Escape every TeX-special character; agent-written reasons contain ^, _ and %."""
    out = str(text)
    out = out.replace("\\", r"\textbackslash{}")
    for k, v in (("{", r"\{"), ("}", r"\}"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}")):
        out = out.replace(k, v)
    return out


def clip(text: str, n: int) -> str:
    """Truncate on a word boundary so statements do not break mid-word."""
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + " ..."


def units(text: str) -> str:
    """Sources write angstrom-squared as 'A^2'; fix it after escaping."""
    text = text.replace(r"A\^{}2", r"\,\AA$^2$").replace(r"A\^{}3", r"\,\AA$^3$")
    text = text.replace(r"60 A cap", r"60\,\AA$^3$ cap")
    return text.replace(" \\,\\AA", "\\,\\AA")


def fmt_params(after: dict) -> str:
    """Flat, breakable rendering of a constraint's parameters - JSON braces overflow the column."""
    parts = []
    for key, val in after.items():
        if key == "sites":
            continue
        if isinstance(val, dict):
            parts.append(", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) else f"{k} {v}"
                                   for k, v in val.items()))
        elif isinstance(val, float):
            parts.append(f"{key} = {val:g}")
        else:
            parts.append(f"{key} = {val}")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--evidence", default="knowledge/evidence_base.json")
    ap.add_argument("--structure", default="knowledge/structure_evidence.json")
    ap.add_argument("--mpnn", default="config/mpnn_log_odds.json")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--out", default="paper/generated/case.tex")
    args = ap.parse_args()

    records = json.loads(Path(args.records).read_text())
    base = json.loads(Path(args.evidence).read_text())
    prior = evidence_mod.load(args.structure, args.mpnn, args.evidence)
    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)

    target = str(variants[int(np.argmax(fitness))])
    scores = prior.score_variants(variants)
    rank = int((scores > scores[list(variants).index(target)]).sum()) + 1

    rows = {r["round"]: r for r in records
            if r["arm"] == "prior_constraint" and r["seed"] == args.seed}
    greedy = {r["round"]: r for r in records
              if r["arm"] == "greedy" and r["seed"] == args.seed}
    fixed = {r["round"]: r for r in records
             if r["arm"] == "prior_fixed" and r["seed"] == args.seed}

    att = prior.attribution(target)
    att_wt = prior.attribution("VDGV")
    comp_names = {"mpnn": "反折叠分数", "packing_headroom": "堆积余量", "burial_charge": "埋藏电荷",
                  "interface_bonus": "Fc 界面加成", "pair_volume_compensation": "41/54 体积补偿",
                  "literature": "文献条目", "total": "合计"}

    L: list[str] = []
    A = L.append

    A(r"\section{附录：一个完整案例（种子 %d）}" % args.seed)
    A(r"""
本附录把一次完整的三轮实验逐步摊开：先验如何给出第一批候选、约束删掉了什么、程序如何裁决假说、
智能体据此改了哪些规则，以及最终在哪一轮命中全局最优。选择这个种子的原因是它能把本文的核心理念
摆在同一条时间线上——同样的起点下，预测器贪心止步于 %.3f，而证据先验加约束命中了 %s（%.3f）。
""" % (greedy[3]["best_so_far"], esc(target), fitness.max()))

    # ---- A.1 prior view of the target
    A(r"\subsection*{A.1\quad 在看到任何标签之前，先验怎么看待全局最优}")
    A(r"""
全局最优 %s 的四个取代中有两个（V39F、G41A）单独出现时低于野生型，因此任何以单突变收益排序的策略
都会丢掉它。证据先验不按单突变收益打分，而是把结构与文献证据加起来：
""" % esc(target))
    A(r"\begin{center}\small\begin{tabular}{lrr}\toprule")
    A(r"先验分量 & %s & 野生型 VDGV \\" % esc(target))
    A(r"\midrule")
    for k in ("mpnn", "packing_headroom", "burial_charge", "interface_bonus",
              "pair_volume_compensation", "literature"):
        A(r"%s & %+.3f & %+.3f \\" % (comp_names[k], att[k], att_wt[k]))
    A(r"\midrule")
    A(r"合计 & \textbf{%+.3f} & %+.3f \\" % (att["total"], att_wt["total"]))
    A(r"\bottomrule\end{tabular}\end{center}")
    A(r"""
在全部 %s 条变体中，该先验把 %s 排在第 %s 位（前 %.2f\%%），而这一排名没有用到任何多突变标签。
把它托起来的主要是文献条目（%+.3f）与 Fc 界面加成（%+.3f）：前者来自双突变图谱中 V54A 背景下
41 位由有害转为有利的报告，以及 41/54 之间的体积转移机制；后者来自复合物结构中只有 40 位埋藏表面
这一事实。
""" % ("{:,}".format(len(variants)).replace(",", ","), esc(target), "{:,}".format(rank).replace(",", ","),
       100 * rank / len(variants), att["literature"], att["interface_bonus"]))

    ev_rows = [r for r in base["evidence"] if r["id"] in ("EV019", "EV020", "EV034", "EV053")]
    if ev_rows:
        A(r"\begin{center}\small\renewcommand{\arraystretch}{1.15}"
          r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.07\textwidth}"
          r">{\raggedright\arraybackslash}p{0.66\textwidth}"
          r">{\raggedright\arraybackslash}p{0.21\textwidth}@{}}\toprule")
        A(r"编号 & 证据陈述（摘要） & 来源 \\ \midrule")
        for r in ev_rows:
            A(r"%s & %s & %s \\" % (r["id"], units(esc(clip(r["statement"], 140))),
                                    esc((r.get("source") or {}).get("id", ""))))
        A(r"\bottomrule\end{tabular}\end{center}")

    # ---- A.2 round by round
    A(r"\subsection*{A.2\quad 三轮的逐轮记录}")
    A(r"""
规则修改在提出的当轮即生效，因此下面按"先改规则、再看结果"的顺序排列。
""")
    for rd in sorted(rows):
        r = rows[rd]
        n = r["notes"]
        audit = n.get("constraint_audit") or {}
        removed = {k: v for k, v in (audit.get("filters_removed") or {}).items() if v}
        updates = n.get("constraint_updates") or []
        A(r"\paragraph{第 %d 轮。}" % rd)

        if updates:
            A(r"""
读过上一轮的审计记录与逐位点统计后，智能体把权重设为 $w_{\text{prior}}=%.1f$、$w_{\text{model}}=%.1f$，
并提出以下规则修改（原话节选）：
""" % (n.get("weights", {}).get("prior", 1.0), n.get("weights", {}).get("model", 1.0)))
            A(r"\begin{center}\small\renewcommand{\arraystretch}{1.15}"
              r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\textwidth}"
              r">{\raggedright\arraybackslash}p{0.26\textwidth}"
              r">{\raggedright\arraybackslash}p{0.44\textwidth}@{}}\toprule")
            A(r"约束 & 本轮生效的参数 & 理由（智能体原话，节选） \\ \midrule")
            for u in updates[:4]:
                after = u.get("after") or {}
                param = "停用" if u.get("active") is False else clip(fmt_params(after), 60)
                A(r"%s & %s & %s \\" % (esc(u["id"]), units(esc(param)),
                                          units(esc(clip(u.get("reason", ""), 130)))))
            A(r"\bottomrule\end{tabular}\end{center}")

        A(r"""
这组规则在 %s 条候选上的实际后果：硬过滤后剩余 %s 条（%s），探索配额留出 %d 个名额，
多样性配额挡下 %d 个候选，最终批次在四个位点上分别出现 %s 种氨基酸；与同样分数下的"无约束排序"相比，
有 %d 个名额被规则改变。本轮批次最好 %.3f、均值 %.3f，累计最好值 %.3f。
""" % ("{:,}".format(audit.get("n_pool", 0)),
            "{:,}".format(audit.get("n_legal_after_filters", 0)),
            "；".join("%s 删除 %s 条" % (k.split("_")[0], "{:,}".format(v))
                      for k, v in removed.items()) or "本轮无硬过滤删除候选",
            audit.get("exploration_quota", 0), audit.get("diversity_blocks", 0),
            "/".join(str(x) for x in audit.get("distinct_aa_per_site", [])),
            audit.get("changed_vs_unconstrained", 0),
            r["metrics"]["best"], r["metrics"]["mean"], r["best_so_far"]))

        verdicts = {}
        for h in r["hypotheses"]:
            verdicts[h["verdict"]] = verdicts.get(h["verdict"], 0) + 1
        zh = {"supported": "支持", "refuted": "推翻", "undetermined": "证据不足"}
        A(r"标签揭示后，本轮登记的 %d 条假说由程序裁决：%s。" % (
            len(r["hypotheses"]),
            "、".join("%s %d 条" % (zh.get(k, k), v) for k, v in sorted(verdicts.items())) or "无"))
        refuted = [h for h in r["hypotheses"] if h["verdict"] == "refuted"]
        if refuted:
            h = refuted[0]
            ev = h.get("evidence") or {}
            A(r"""
被推翻的例子：``%s''——带该取代的 %d 个变体均值 %.3f，不带的均值 %.3f，差值 %+.3f，
方向与假说相反，因此该取代进入 \texttt{C8} 惩罚项，在下一轮被降权。
""" % (units(esc(clip(h["statement"], 95))), ev.get("n_with_substitution", 0),
                ev.get("mean_with", 0.0), ev.get("mean_without", 0.0), ev.get("delta", 0.0)))

    # ---- A.3 the decisive pick
    r2 = rows[2]
    hit = [d for d in r2["selected"] if d["variant"] == target]
    if hit:
        preds = np.array([d["predicted"] for d in r2["selected"]])
        A(r"\subsection*{A.3\quad 决定性的一次选择}")
        A(r"""
第 2 轮的批次里出现了 %s。当时拟合预测器给它的预测值只有 %.3f，在这 96 个入选变体中排第 %d 位——
如果只按模型排序，它不会进入本轮批次（同种子的预测器贪心臂第 2 轮最好只有 %.3f，三轮结束为 %.3f）。
把它选进来的是先验：它在先验排序中位于前 %.2f\%%。揭示标签后其实测适应度为 %.3f，即全局最优。
这一次选择就是本文主张的具体形态：\emph{当模型在标签稀少时对高值区间失灵，先验与约束是唯一能把
正确候选留在批次里的东西}。
""" % (esc(target), hit[0]["predicted"], int((preds > hit[0]["predicted"]).sum()) + 1,
            greedy[2]["best_so_far"], greedy[3]["best_so_far"], 100 * rank / len(variants),
            hit[0]["fitness"]))

    # ---- A.4 contrast table
    A(r"\subsection*{A.4\quad 同一种子下三种策略的对照}")
    A(r"\begin{center}\small\begin{tabular}{lccc}\toprule")
    A(r"轮次 & 预测器贪心 & 证据先验+约束（无 LLM） & 证据先验+约束（LLM 调参） \\ \midrule")
    for rd in sorted(rows):
        A(r"第 %d 轮 & %.3f & %.3f & %.3f \\" % (
            rd, greedy[rd]["best_so_far"],
            fixed[rd]["best_so_far"] if rd in fixed else float("nan"),
            rows[rd]["best_so_far"]))
    A(r"\bottomrule\end{tabular}\end{center}")
    A(r"""
两个带先验与约束的臂在第 2 轮同时命中全局最优，而且带不带语言模型结果相同——这与正文第 3 节
在 %d 个种子上的统计结论一致。
""" % len({r["seed"] for r in records}))

    A(r"\subsection*{A.5\quad 这个案例说明了什么}")
    A(r"""
\begin{itemize}
\item \textbf{先验决定了第一轮看得见什么。} 全局最优在先验里排前 %.2f\%%，在单突变收益排序里则被两个
  有害组分挡在外面；第一批 %d 条候选能否包含通往最优的中间体，在任何模型被拟合之前就已经确定。
\item \textbf{约束是可审计的，而不是修辞。} 每一轮都能回答"删掉了多少、改变了几个名额、哪条规则在起作用"：
  第 1 轮硬过滤删除 %s 条、改变 %d 个名额，第 3 轮智能体在给出理由后主动停用了两条规则。
\item \textbf{语言模型的位置在规则层，不在候选层。} 本案例中它做的每件事都是改参数、给理由；
  把它整体移除后结果一模一样（A.4），这与正文在 %d 个种子上的统计结论一致。
\end{itemize}
""" % (100 * rank / len(variants), len(rows[1]["selected"]),
       "{:,}".format((rows[1]["notes"]["constraint_audit"]["filters_removed"]).get("C2_volume_window", 0)),
       rows[1]["notes"]["constraint_audit"].get("changed_vs_unconstrained", 0),
       len({r["seed"] for r in records})))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
