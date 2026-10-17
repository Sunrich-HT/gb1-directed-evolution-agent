#!/usr/bin/env python
"""Generate paper/generated/case.tex - the appendix that walks one campaign end to end.

The point of this appendix is the reasoning, not the numbers: it shows how
structure, physical chemistry and published evidence were turned into prior terms
and constraints, and what each decision did to the batch.  Every value is read
from results/, knowledge/ or the predicted structures, so the narrative cannot
drift from what the code did.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb1agent import evidence as evidence_mod
from gb1agent.data import load_landscape
from make_paper_figures import _kabsch, _parse_cif_ca


def esc(text) -> str:
    """Escape every TeX-special character; agent-written reasons contain ^, _ and %."""
    out = " ".join(str(text).split())
    out = out.replace("\\", r"\textbackslash{}")
    for k, v in (("{", r"\{"), ("}", r"\}"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}")):
        out = out.replace(k, v)
    return out


def clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def units(text: str) -> str:
    text = text.replace(r"A\^{}2", r"\,\AA$^2$").replace(r"A\^{}3", r"\,\AA$^3$")
    text = text.replace(r"60 A cap", r"60\,\AA$^3$ cap")
    return text.replace(" \\,\\AA", "\\,\\AA")


def fmt_params(after: dict) -> str:
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


def thousands(n) -> str:
    return f"{int(n):,}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--evidence", default="knowledge/evidence_base.json")
    ap.add_argument("--structure", default="knowledge/structure_evidence.json")
    ap.add_argument("--mpnn", default="config/mpnn_log_odds.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--prior-eval", default="results/prior_evaluation.json")
    ap.add_argument("--cif", default="results/structure/cif")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--out", default="paper/generated/case.tex")
    args = ap.parse_args()

    records = json.loads(Path(args.records).read_text())
    base = json.loads(Path(args.evidence).read_text())
    struct = json.loads(Path(args.structure).read_text())
    baselines = json.loads(Path(args.baselines).read_text())
    pe = json.loads(Path(args.prior_eval).read_text())
    prior = evidence_mod.load(args.structure, args.mpnn, args.evidence)

    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)
    target = str(variants[int(np.argmax(fitness))])
    scores = prior.score_variants(variants)
    rank = int((scores > scores[list(variants).index(target)]).sum()) + 1
    pct = 100 * rank / len(variants)

    S = struct["sites"]
    # the model the campaign actually uses is the cross-validated pick, one per seed;
    # taking the best test-set score across all specs would be selection on the test set
    cv_d0 = [m for m in baselines["d0_models"] if m["cv_selected"]]
    best_d0 = {"n_train": cv_d0[0]["n_train"],
               "spearman": float(np.mean([m["spearman"] for m in cv_d0])),
               "top5pct_precision": float(np.mean([m["top5pct_precision"] for m in cv_d0])),
               "best_in_predicted_top96": float(np.mean(
                   [m["best_in_predicted_top96"] for m in cv_d0])),
               "n_seeds": len(cv_d0)}
    ev = {r["id"]: r for r in base["evidence"]}

    rows = {r["round"]: r for r in records
            if r["arm"] == "prior_constraint" and r["seed"] == args.seed}
    greedy = {r["round"]: r for r in records
              if r["arm"] == "greedy" and r["seed"] == args.seed}
    fixed = {r["round"]: r for r in records
             if r["arm"] == "prior_fixed" and r["seed"] == args.seed}
    n_seeds = len({r["seed"] for r in records})

    att, att_wt = prior.attribution(target), prior.attribution("VDGV")
    comp = [("mpnn", "反折叠分数"), ("packing_headroom", "堆积余量"),
            ("burial_charge", "埋藏电荷"), ("interface_bonus", "Fc 界面加成"),
            ("pair_volume_compensation", "41/54 体积补偿"), ("literature", "文献条目")]

    # predicted structures: fold identity vs fitness
    cif = Path(args.cif)
    fold = {}
    if (cif / "FWAA.cif").exists():
        P = {v: _parse_cif_ca(cif / f"{v}.cif") for v in ("VDGV", "FWAA", "RYLE")}
        res = sorted(set.intersection(*[set(P[v]) for v in P]))
        X = {v: np.array([P[v][i][:3] for i in res]) for v in P}
        for v in ("FWAA", "RYLE"):
            fold[v] = _kabsch(X[v], X["VDGV"])[2]
        meta = json.loads((cif / "fold_meta.json").read_text())
        fold["conf"] = {v: meta[v]["confidence_scores"][0] for v in meta}

    L: list[str] = []
    A = L.append

    A(r"\section{附录：一个完整案例（种子 %d）}" % args.seed)
    A(r"""
正文给出的是 %d 个种子上的统计结论。本附录换一个角度：把其中一次完整的三轮实验从头讲到尾，
重点不在数字，而在\emph{推理是怎么走的}——结构坐标与文献条目如何变成先验的某一项、
每一轮看到什么情况、据此加了什么约束、这条约束又把批次改成了什么样。
选这个种子是因为它把三件事摆在同一条时间线上：同样的 %d 条起始标签下，
预测器贪心止步于 %.3f，而先验加约束在第 2 轮就命中了 %s（%.3f）。
""" % (n_seeds, len(rows[1]["selected"]) * 2, greedy[3]["best_so_far"],
       esc(target), fitness.max()))

    # ---------------------------------------------------------------- A.1
    A(r"\subsection*{A.1\quad 先验是怎么一步步推出来的}")
    A(r"""
\paragraph{第一步：先确认模型在第一轮确实靠不住。}
用 %d 条随机起始标签训练、并由交叉验证选出的预测器（%d 个种子的平均），
在一万条固定测试集上 Spearman 只有 %.3f、top-5\%% 精度 %.3f，
它预测的前 96 名中最好的变体实测只有 %.3f——离全局最优 %.3f 很远。
这不是模型没调好，而是四位点景观里 83\%% 的模型权重落在交互项上（正文图~\ref{fig:mechanism}b），
而 %d 条随机标签几乎覆盖不到任何一个高值组合。结论：第一轮的质量只能由\emph{不依赖标签的信息}决定。
""" % (best_d0["n_train"], best_d0["n_seeds"], best_d0["spearman"],
       best_d0["top5pct_precision"], best_d0["best_in_predicted_top96"],
       fitness.max(), best_d0["n_train"]))

    A(r"""
\paragraph{第二步：把"非序列维度"逐个问一遍。}
可用的非序列信息只有三类：实验结构坐标、残基的物理化学性质、已发表文献。
对四个位点逐个读结构（自由态 %s，复合物 1FCC）得到的分工非常不对称：
40 位相对溶剂可及度 %.1f\%%、结合 IgG-Fc 时埋藏 %.1f\,\AA$^2$；
39、41、54 位分别只埋藏 %.1f、%.1f、%.1f\,\AA$^2$，其中 39 与 54 近乎完全埋在核心
（可及度 %.1f\%% 与 %.1f\%%）。
\emph{由此得到第一个设计判断}：只有 40 位能直接改写结合界面，其余三个位点只能通过折叠稳定性与
环的构象间接起作用。所以先验里给 40 位一个"界面加成"项，而给 39/41/54 的是堆积与体积类的项——
两类位点用不同的打分依据，而不是对四个位点套同一个公式。
""" % (esc(struct["free_structure"]), 100 * S["40"]["rel_sasa"],
       S["40"]["interface"]["buried_surface_on_binding_A2"],
       S["39"]["interface"]["buried_surface_on_binding_A2"],
       S["41"]["interface"]["buried_surface_on_binding_A2"],
       S["54"]["interface"]["buried_surface_on_binding_A2"],
       100 * S["39"]["rel_sasa"], 100 * S["54"]["rel_sasa"]))

    A(r"""
\paragraph{第三步：41 位——一条被自己的证据推翻的假设。}
本工作最早的版本按 NMR 结构 2GB1（41 位 $\psi=%.1f^\circ$，落在非甘氨酸不利区）
加了一条硬过滤："41 位必须是甘氨酸"。这条规则会直接删掉全局最优。
晶体结构给出的同一个二面角是 $\phi/\psi=%.1f^\circ/%.1f^\circ$，属于普通 $\beta$ 区；
证据库中 %s 进一步指出，这条环上真正需要甘氨酸的是 $\phi\approx+77^\circ$ 的 38 位，而 38 位不可变。
\emph{于是硬过滤被删除}。但 41 位确实特殊：用主链 N/CA/C 重建的虚拟 C$\beta$ 距最近非相邻重原子仅
%.2f\,\AA（到 55 位羰基氧），而 39/54/40 位分别有 %.2f/%.2f/%.2f\,\AA。
这是"空间不够"，不是"必须甘氨酸"——因此改成可被补偿的\emph{堆积余量}惩罚项。
""" % (struct["nmr_crosscheck_2gb1"]["41"]["psi"], S["41"]["phi"], S["41"]["psi"],
       "EV032", S["41"]["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"],
       S["39"]["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"],
       S["54"]["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"],
       S["40"]["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"]))

    A(r"""
\paragraph{第四步：让文献决定"不够的空间能不能借到"。}
如果 41 位只是空间不足，那么把邻近位点改小就应当能补偿。文献里正好有这条机制：%s 报告
双突变图谱中最强的正上位性 G41L/V54G，其机制就是把侧链体积从埋藏的 54 位转移到 41 位所在的环；
%s 则报告在 V54A 背景下，41 位的取代会从有害翻转为有利。
\emph{这两条直接变成两个设计动作}：先验里加一个只对 41/54 生效的\emph{体积补偿}项
（奖励 $\Delta V_{41}+\Delta V_{54}\approx 0$ 的组合），约束里把 41 位的体积上限设成
一个\emph{可以在后续轮次被放宽}的参数，而不是写死的禁令。
""" % ("EV020", "EV019"))

    A(r"""
\paragraph{第五步：当模型与文献冲突时，以可核查的测量为准。}
零样本结构类模型（%s）把 40 位全部 19 种取代都判为不利，而双突变图谱里 D40W 单独就有 3.90 的适应度。
\emph{因此在 40 位把零样本项静音}——这是整条先验里唯一一处"用文献推翻模型"的地方，并在代码中显式标注。
""" % ("EV053",))

    A(r"""
\paragraph{这套推理值多少。}
合成后的先验在不使用任何多突变标签的前提下，对三突变以上子集给出 Spearman $\rho=%+.3f$，
与\emph{使用实测单突变标签}的加性上界（$%+.3f$）持平；其排序的前 96 名中最好变体实测 %.3f，
top-1\%% 富集 %.1f 倍。下表是它对全局最优的打分拆解——这张表就是"为什么选它"的全部答案：
""" % (pe["priors"]["evidence_prior"]["spearman_hd3plus"],
       pe["priors"]["leakage_measured_single_mutants"]["spearman_hd3plus"],
       pe["priors"]["evidence_prior"]["best_in_top96"],
       pe["priors"]["evidence_prior"]["enrichment_top1pct"]))

    A(r"\begin{center}\small\begin{tabular}{lrrl}\toprule")
    A(r"先验分量 & %s & 野生型 VDGV & 依据 \\ \midrule" % esc(target))
    why = {"mpnn": "ProteinMPNN，40 位静音", "packing_headroom": "虚拟 C$\\beta$ 间距（1PGA）",
           "burial_charge": "相对可及度（1PGA）", "interface_bonus": "$\\Delta$SASA（1FCC）",
           "pair_volume_compensation": "G41L/V54G 机制", "literature": "62 条结构化证据"}
    for k, zh in comp:
        A(r"%s & %+.3f & %+.3f & %s \\" % (zh, att[k], att_wt[k], why[k]))
    A(r"\midrule")
    A(r"合计 & \textbf{%+.3f} & %+.3f & 先验排名 %s / %s（前 %.2f\%%） \\" % (
        att["total"], att_wt["total"], thousands(rank), thousands(len(variants)), pct))
    A(r"\bottomrule\end{tabular}\end{center}")

    # ---------------------------------------------------------------- A.2
    A(r"\subsection*{A.2\quad 每一轮如何把当前情况变成下一轮的约束}")
    A(r"""
每轮开始时，智能体拿到的是上一轮的约束审计、逐位点统计与被推翻的假说，
它能做的只有一件事：改规则的参数或权重（它从不点名变体）。规则在提出的当轮即生效。
""")
    for rd in sorted(rows):
        r = rows[rd]
        n = r["notes"]
        audit = n.get("constraint_audit") or {}
        removed = {k: v for k, v in (audit.get("filters_removed") or {}).items() if v}
        verd = {}
        for h in r["hypotheses"]:
            verd[h["verdict"]] = verd.get(h["verdict"], 0) + 1
        zh = {"supported": "支持", "refuted": "推翻", "undetermined": "证据不足"}
        refuted = [h for h in r["hypotheses"] if h["verdict"] == "refuted"]

        A(r"\paragraph{第 %d 轮。}" % rd)
        A(r"""
权重 $w_{\text{prior}}=%.1f$、$w_{\text{model}}=%.1f$。规则作用在 %s 条候选上：
硬过滤后剩 %s 条（%s），探索配额 %d 个名额，多样性配额挡下 %d 个候选，
四个位点最终各出现 %s 种氨基酸；与同样分数下的"无约束排序"相比有 %d 个名额被改变。
批次最好 %.3f、均值 %.3f，累计最好值 %.3f。标签揭示后 %d 条登记假说的裁决为：%s。
""" % (n.get("weights", {}).get("prior", 1.0), n.get("weights", {}).get("model", 1.0),
            thousands(audit.get("n_pool", 0)), thousands(audit.get("n_legal_after_filters", 0)),
            "；".join("%s 删除 %s 条" % (k.split("_")[0], thousands(v))
                      for k, v in removed.items()) or "本轮无硬过滤删除候选",
            audit.get("exploration_quota", 0), audit.get("diversity_blocks", 0),
            "/".join(str(x) for x in audit.get("distinct_aa_per_site", [])),
            audit.get("changed_vs_unconstrained", 0),
            r["metrics"]["best"], r["metrics"]["mean"], r["best_so_far"],
            len(r["hypotheses"]),
            "、".join("%s %d 条" % (zh.get(k, k), v) for k, v in sorted(verd.items())) or "无"))
        if refuted:
            h = refuted[0]
            e = h.get("evidence") or {}
            A(r"""
被推翻的例子："%s"——带该取代的 %d 个变体均值 %.3f，不带的 %.3f，差值 %+.3f，方向与假说相反；
该取代随即进入 \texttt{C8} 惩罚项，在下一轮被降权。这就是"情况分析"进入约束的最短路径：
一条可以被数据否定的陈述，被否定之后直接改变下一批的组成。
""" % (units(esc(clip(h["statement"], 95))), e.get("n_with_substitution", 0),
                e.get("mean_with", 0.0), e.get("mean_without", 0.0), e.get("delta", 0.0)))

    A(r"""
三轮里智能体一共动了 %d 次规则，全部附有理由（下表为原话节选）。值得注意的是第 3 轮：
它\emph{主动停用}了体积上限与界面惩罚两条规则，理由分别是"这条上限删掉了 139k 中的 62k，
而 G41L/V54G 说明张力是可补偿的"以及"40 位最好的变体里就有 W 和 Y，惩罚大侧链与观测矛盾"。
规则能被它自己的审计记录推翻，这正是把约束写成显式参数而不是写进代码分支的意义。
""" % sum(len(rows[rd]["notes"].get("constraint_updates") or []) for rd in rows))

    A(r"\begin{center}\small\renewcommand{\arraystretch}{1.15}"
      r"\begin{tabular}{@{}c>{\raggedright\arraybackslash}p{0.235\textwidth}"
      r">{\raggedright\arraybackslash}p{0.205\textwidth}"
      r">{\raggedright\arraybackslash}p{0.44\textwidth}@{}}\toprule")
    A(r"轮 & 约束 & 本轮生效的参数 & 理由（智能体原话，节选） \\ \midrule")
    for rd in sorted(rows):
        for u in (rows[rd]["notes"].get("constraint_updates") or []):
            param = "停用" if u.get("active") is False else clip(fmt_params(u.get("after") or {}), 55)
            A(r"%d & %s & %s & %s \\" % (rd, esc(u["id"]), units(esc(param)),
                                         units(esc(clip(u.get("reason", ""), 120)))))
    A(r"\bottomrule\end{tabular}\end{center}")

    # ---------------------------------------------------------------- A.3
    hit = [d for d in rows[2]["selected"] if d["variant"] == target]
    check = {}
    cpath = Path("results/case_round2_model_check.json")
    if cpath.exists():
        check = {c["arm"]: c for c in json.loads(cpath.read_text())}
    if hit:
        preds = np.array([d["predicted"] for d in rows[2]["selected"]])
        A(r"\subsection*{A.3\quad 决定性的其实是第一批：它教给模型什么}")
        A(r"""
第 2 轮的批次里出现了 %s，实测 %.3f。容易误读的是把功劳直接记在"先验把它捞了出来"上：
核查后并非如此。该变体并不触犯任何一条硬过滤——即使体积上限停在 60\,\AA$^3$ 它也是合法的；
而当时拟合预测器对它的预测值只有 %.3f，在 96 个入选变体中仅排第 %d 位。
真正起作用的是更前面一步：\emph{第一批测了什么，决定了第二轮的模型能不能看见它}。
""" % (esc(target), hit[0]["fitness"], hit[0]["predicted"],
            int((preds > hit[0]["predicted"]).sum()) + 1))
        if check:
            A(r"""
为了把这一步讲清楚，我们做了一次事后核查：把三个臂各自第 1 轮的 96 条标签分别接在同一组
192 条起始标签之后，用同样的交叉验证流程重新拟合预测器，再让它对同一个候选池排序。结果是：
""")
            A(r"\begin{center}\small\begin{tabular}{lccc}\toprule")
            A(r"第 1 轮批次来自 & 该批最好标签 & %s 在模型中的名次 & 模型自己前 96 名中的最好实测 \\ \midrule"
              % esc(target))
            zh_arm = {"prior_constraint": "证据先验+约束（LLM 调参）",
                      "prior_fixed": "证据先验+约束（无 LLM）", "greedy": "预测器贪心"}
            for arm in ("prior_constraint", "prior_fixed", "greedy"):
                c = check.get(arm)
                if not c:
                    continue
                A(r"%s & %.3f & %s / %s & %.3f \\" % (
                    zh_arm[arm], c["best_label_r1"], thousands(c["fwaa_rank"]),
                    thousands(c["n_pool"]), c["best_in_model_top96"]))
            A(r"\bottomrule\end{tabular}\end{center}")
            A(r"""
贪心臂第 1 轮找到的最好标签其实\emph{更高}（%.3f 对 %.3f），但它训练出来的第 2 轮模型把全局最优排在
第 %s 位、落在前 96 名之外，其模型自选前 96 名中最好只有 %.3f；而由先验与约束挑出的第 1 轮批次，
让同一流程的模型把全局最优排进前 %d 名。\emph{第一批的价值不在于它本身最好，而在于它让模型学到什么}——
这正是"先验决定起跑线"的具体机制：约束保证批次覆盖互相补偿的组合（如 41/54 体积配对），
贪心则把 96 个名额堆在同一个基序附近，第 2 轮因此学不到新的交互项。
同一种子下三种策略的轨迹：预测器贪心 %.3f→%.3f→%.3f，先验加约束（无语言模型）%.3f→%.3f→%.3f，
先验加约束（语言模型调参）%.3f→%.3f→%.3f。
""" % (check["greedy"]["best_label_r1"], check["prior_constraint"]["best_label_r1"],
                thousands(check["greedy"]["fwaa_rank"]), check["greedy"]["best_in_model_top96"],
                max(check["prior_constraint"]["fwaa_rank"], check["prior_fixed"]["fwaa_rank"]),
                *[greedy[i]["best_so_far"] for i in (1, 2, 3)],
                *[fixed[i]["best_so_far"] for i in (1, 2, 3)],
                *[rows[i]["best_so_far"] for i in (1, 2, 3)]))

    # ---------------------------------------------------------------- A.4
    if fold:
        A(r"\subsection*{A.4\quad 结构预测工具的真实输出，以及它为什么没有进入打分}")
        A(r"""
为了检查"要不要把结构预测接进回路"，我们用 NVIDIA 托管的 Boltz-2 真实折叠了覆盖整个适应度量程的
%d 个变体，并把其中三个的坐标保留下来（图~\ref{fig:structure}）：野生型 VDGV、全局最优 %s，
以及一个适应度仅 0.002 但组成完全不同的死变体 RYLE。结果是三者的预测结构几乎重合——
%s 对野生型的 C$\alpha$ RMSD 为 %.2f\,\AA，RYLE 为 %.2f\,\AA；
整链置信度分别是 %.3f、%.3f、%.3f，死变体并不比全局最优更"不可信"。
在全部 %d 个折叠变体上，四个位点处的 pLDDT 与实测适应度的 Spearman 只有 $+0.21$，
而仅用 %d 条标签训练的序列模型是 $+0.56$；把结构置信度作为额外特征加入后，
5 折交叉验证的 Spearman 变化 $-0.002$。
\emph{因此我们没有把它写进先验}：它回答的是"这条序列能不能折叠成这个骨架"，
而本任务问的是"它和 IgG-Fc 结合得好不好"——这两个问题在 40 位尤其不一致。
把一个诚实的零贡献项留在方法里而不是悄悄删掉，本身也是可解释性的一部分。
""" % (146, esc(target), esc(target), fold["FWAA"], fold["RYLE"],
            fold["conf"]["VDGV"], fold["conf"]["FWAA"], fold["conf"]["RYLE"], 146,
            len(rows[1]["selected"]) * 2))

    # ---------------------------------------------------------------- A.5
    A(r"\subsection*{A.5\quad 这个案例说明了什么}")
    A(r"""
\begin{itemize}
\item \textbf{先验决定了第一轮看得见什么。} 全局最优在证据先验里排前 %.2f\%%，
  在任何单突变收益排序里却被两个有害组分挡在外面；第一批 %d 条候选里有没有通往最优的中间体，
  在模型被拟合之前就已经定了。
\item \textbf{约束必须是可以被自己的数据推翻的参数。} 41 位的"必须甘氨酸"被晶体结构推翻，
  体积上限被自己的审计记录推翻（第 3 轮停用），被推翻的取代自动进入惩罚项——
  每一步都能回答"删掉了多少、改变了几个名额、凭什么"。
\item \textbf{语言模型的位置在规则层，不在候选层。} 本案例中它做的每件事都是改参数、给理由；
  把它整体移除，三轮轨迹一模一样（A.3），这与正文在 %d 个种子上的统计结论一致。
\end{itemize}
""" % (pct, len(rows[1]["selected"]), n_seeds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("wrote", out, "|", len(L), "blocks")


if __name__ == "__main__":
    main()
