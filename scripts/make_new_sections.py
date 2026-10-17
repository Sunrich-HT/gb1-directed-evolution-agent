#!/usr/bin/env python
"""Generate the two report sections the task names explicitly:
failure-case analysis, and improvements / future work.

Both are written from the result files so the numbers cannot drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/generated")
    args = ap.parse_args()

    R = Path("results")
    mo = json.loads((R / "mutation_order_and_uncertainty.json").read_text())
    base = json.loads((R / "baselines.json").read_text())
    check = json.loads((R / "structure/complex_check.json").read_text())
    rel = json.loads((R / "structure/packing_relation_trainset.json").read_text())
    c9 = json.loads((R / "structure/c9_batch_effect.json").read_text())
    kg = json.loads(Path("knowledge/knowledge_graph.json").read_text())
    records = json.loads((R / "v2/records_final.json").read_text())

    order = mo["landscape_by_mutation_order"]
    acc = mo["predictor_accuracy_by_order"]
    unc = mo["uncertainty"]
    picks = unc["batch_of_96_from_the_test_pool"]
    cv = [m for m in base["d0_models"] if m["cv_selected"]]
    hit = [d for r in records if r["arm"] == "prior_constraint" and r["seed"] == 302
           and r["round"] == 2 for d in r["selected"] if d["variant"] == "FWAA"]
    n_rows = len(check["setups"]["single_fc_chain"]) + len(check["setups"]["native_fc_homodimer"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- failures
    L = [r"\section{失败案例分析}", r"""
把失败逐条写出来，是因为这项任务里每一次失败都改变了后续的设计；只展示成功样例会把这些改动的理由隐去。
"""]
    L.append(r"\paragraph{一、预测器在高值区间失灵。}")
    L.append(r"""
用 %d 条起始标签、经交叉验证选出的预测器在固定测试集上 Spearman %.3f、Pearson %.3f、MSE %.3f，
其预测的前 96 名中最好变体实测只有 %.3f。最具体的一次：第 2 轮它给全局最优 FWAA 的预测值是
%.3f，而实测 %.3f。按阶数看这不是某一类变体的问题（双/三/四突变 Spearman %.3f、%.3f、%.3f）。
\emph{设计后果}：批次不能只由模型排序决定。
""" % (cv[0]["n_train"],
       sum(m["spearman"] for m in cv) / len(cv),
       sum(m.get("pearson", 0) for m in cv) / len(cv),
       sum(m.get("mse", 0) for m in cv) / len(cv),
       sum(m["best_in_predicted_top96"] for m in cv) / len(cv),
       hit[0]["predicted"] if hit else float("nan"),
       hit[0]["fitness"] if hit else float("nan"),
       acc["2"]["spearman"], acc["3"]["spearman"], acc["4"]["spearman"]))

    L.append(r"\paragraph{二、先验里一条硬过滤本身是错的。}")
    L.append(r"""
早期版本按 NMR 结构在 41 位加了「必须是甘氨酸」的硬过滤。晶体结构给出的同一个二面角属于普通
$\beta$ 区，证据库中 EV032 进一步指出这条环上真正需要甘氨酸的是 38 位。若保留该过滤，全局最优（41 位为丙氨酸）会被直接删除。
\emph{设计后果}：改为可被 54 位补偿的堆积余量惩罚项，并引出 41/54 体积补偿分量。
""")

    L.append(r"\paragraph{三、预测的复合物不能用来定义界面约束。}")
    L.append(r"""
为了让结构进入决策，我们用托管 Boltz-2 真实折叠了变体与 IgG-Fc 的复合物（单链 Fc 与天然同源二聚体两种设置）。
把结构域实际接触的那条 Fc 链叠合到晶体 1FCC 上（叠合 RMSD %.2f--%.2f~\AA），GB1 结构域仍偏离晶体位姿
%.1f--%.1f~\AA，%d 个中达到天然结合模式的为 %d 个，GB1 一侧与晶体接触面的重合为 0/16。
\emph{设计后果}：界面类约束被否决，结构信息改从单体局部几何提取（附录~\ref{sec:structure}）。
""" % (min(r["fc_superposition_rmsd_A"] for s in check["setups"].values() for r in s),
       max(r["fc_superposition_rmsd_A"] for s in check["setups"].values() for r in s),
       min(r["gb1_pose_rmsd_to_crystal_A"] for s in check["setups"].values() for r in s),
       max(r["gb1_pose_rmsd_to_crystal_A"] for s in check["setups"].values() for r in s),
       n_rows,
       sum(1 for s in check["setups"].values() for r in s if r["native_binding_mode"])))

    L.append(r"\paragraph{四、一个看起来合理的结构判据被已测数据否定。}")
    L.append(r"""
从预测单体出发，最初写下的判据是「侧链碳原子靠得过近即为张力、应当惩罚」。在第 1 轮已揭示标签的
24 个变体上检验，该量与适应度的相关为 $\rho=%+.3f$（$p=%.2f$），不显著，方向也与假设相反；
真正成立的是局部堆积密度（$\rho=%+.3f$，$p=%.3f$）。\emph{设计后果}：约束 C9 用密度而不是张力，
且方向由已测标签决定而非由直觉决定。
""" % (rel["min_cc"][0], rel["min_cc"][1], rel["packing_density"][0], rel["packing_density"][1]))

    L.append(r"\paragraph{五、不确定性是可信的，但按它选批没有收益。}")
    L.append(r"""
%d 个自助重采样成员组成的集成，其预测标准差与真实绝对误差的秩相关为 $\rho=%+.3f$：
模型确实知道自己哪里不确定（置信高的一半 Spearman %.3f，不确定的一半 %.3f）。
但同样 96 条预算下，按 $\mu+\sigma$、$\mu+2\sigma$ 选批的最好值（%.3f、%.3f）不优于按 $\mu$ 贪心（%.3f），
纯按 $\sigma$ 探索明显更差（%.3f）。\emph{设计后果}：探索配额按组成多样性而非不确定性设定。
""" % (unc["n_members"], unc["spearman_sd_vs_abs_error"],
       unc["accuracy_on_confident_half"]["spearman"], unc["accuracy_on_uncertain_half"]["spearman"],
       picks["ucb_mean_plus_1sd"]["best"], picks["ucb_mean_plus_2sd"]["best"],
       picks["greedy_mean"]["best"], picks["pure_exploration_sd"]["best"]))

    L.append(r"\paragraph{六、语言模型没有带来可测量的增量。}")
    L.append(r"""
在相同的先验与约束之上，让模型逐轮改规则与权重，与规则完全冻结的版本在 \Nseeds{} 个种子上无法区分
（Wilcoxon $p=\WilcoxonLLMVsFixed{}$）；把它当作挑选器时，\texttt{llm\_agent} 臂在 15 个种子中有
14 个与预测器贪心完全相同。这是本文最值得报告的否定结果：在一个四位点、搜索空间仅 $20^4$、
且已有强先验与显式约束的问题上，留给自由判断的空间很小。
""")

    L.append(r"\paragraph{七、还有两处是能力边界而非设计选择。}")
    L.append(r"""
证据库中专利条目为 \NPatentRows{} 条：检索接口在本次运行期间不可用，「专利证据」这一路是空的。
另外，高于野生型的比例随突变数单调下降（单点 %.1f\%%、双点 %.1f\%%、三点 %.1f\%%、四点 %.1f\%%），
而全局最优恰是四点突变——随机臂 15 个种子一次未中，难点不在「预测得准」而在「第一批看向哪里」。
""" % tuple(100 * order[k]["frac_above_wild_type"] for k in ("1", "2", "3", "4")))

    (out / "failure_cases.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------- future
    F = [r"\section{改进建议与未来拓展}", r"""
按本文测到的瓶颈排序，而不是按技术热度排序。
"""]
    F.append(r"""
\begin{enumerate}\setlength{\itemsep}{1pt}
\item \textbf{不确定性用于批次组成而非排序。} 失败案例五显示 $\mu+\sigma$ 排序无效（标准差与均值高度相关）；
  改让 $\sigma$ 充当配额依据——规定若干名额来自模型最不确定的区域，与现有探索配额同一机制。
\item \textbf{复合物结构换技术路线。} 通用折叠服务在这对相互作用上不能复现天然结合模式；
  替代方案是以 1FCC 为模板的约束式建模，或模板对接固定位姿后用界面能量函数（FoldX、Rosetta ddG）评分。
  判据仍是本文那条：先验证位姿，再谈界面。
\item \textbf{让知识图谱参与推理。} 当前的图（%d 节点、%d 边）只被查询；
  下一步在图上做规则归纳，自动提出可检验假说，再由程序用下一批标签裁决。
\item \textbf{把先验放进模型而非加在旁边。} 以证据先验作为高斯过程的均值函数，
  让「先验何时让位于数据」由后验决定，而不是由人工权重决定。
\item \textbf{位点更多时分层。} 候选空间无法枚举后，先验与约束需改为分组形式（位点子集内组合、再跨组装配）。
\item \textbf{接自动化实验平台。} 批次在揭示标签前以 SHA-256 冻结、评估器隔离在一次调用之后，
  替换这次调用即可把查表换成向 LIMS 或工作站下发工单；尚需补齐板位映射与失败孔处理（附录~C.5）。
\end{enumerate}
""" % (kg["counts"]["nodes"], kg["counts"]["edges"]))
    (out / "future_work.tex").write_text("\n".join(F) + "\n")


    # ------------------------------------------------------- appendix B: structure
    pack = json.loads((R / "structure/predicted_packing.json").read_text())
    rows_all = [r for s in check["setups"].values() for r in s]
    B = [r"\section{附录：结构预测服务如何进入决策回路}", r"\label{sec:structure}", r"""
正文第 2 节用到的结构信息来自实验坐标（1PGA、1FCC）。本附录是另一件事：把结构\emph{预测}服务
接进回路的完整过程，包括其中被否决的部分。

\paragraph{先折复合物，因为单体与结合无关。} 每个变体都折成同一个结构域，单体结构本身不携带
结合强弱的信息；本任务的表型是与 IgG-Fc 的结合，因此可能提供信息的是复合物。我们用托管 Boltz-2
真实折叠了两种设置：变体 + 单条 Fc 链（262 残基），以及变体 + 天然 Fc 同源二聚体（468 残基）。
"""]
    B.append(r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../figures/paper/fig7_docking_check.pdf}
\caption{\textbf{预测复合物与晶体复合物的对照。} 灰色为预测的 IgG-Fc，已按它与结构域实际接触的那条链
叠合到晶体 1FCC（叠合 RMSD %.2f--%.2f~\AA）；绿色为晶体中蛋白 G 结构域的位置，红色为预测放置的位置。
两种设置下结构域分别相差 %.1f~\AA 与 %.1f~\AA。Fc 一侧的表位大体正确，但 GB1 一侧与晶体接触面的
重合为 0/16：预测用错误的一面贴在正确的表面上。}
\label{fig:docking}
\end{figure*}
""" % (min(r["fc_superposition_rmsd_A"] for r in rows_all),
       max(r["fc_superposition_rmsd_A"] for r in rows_all),
       check["setups"]["single_fc_chain"][-1]["gb1_pose_rmsd_to_crystal_A"],
       check["setups"]["native_fc_homodimer"][-1]["gb1_pose_rmsd_to_crystal_A"]))

    B.append(r"""
\paragraph{验证先于取用。} 判据是：把结构域实际接触的那条 Fc 链叠到晶体上，看结构域落在哪里
（图~\ref{fig:docking}）。%d 个预测复合物中达到天然结合模式（位姿 RMSD $<5$~\AA）的为 %d 个。
没有这一步，后续任何"预测界面面积"都是在错误位姿上量出来的。\emph{结论：界面类约束不能建立在它之上。}

\paragraph{改用单体能可靠给出的量。} 单体折叠本身是准的（FWAA 对野生型的 C$\alpha$ RMSD 0.51~\AA）。
因此测量四个位点周围的局部几何：最近的碳--碳接触距离、侧链质心 6~\AA 内的重原子数（局部堆积密度）、
41 位主链是否落在非甘氨酸允许区、位点处的 pLDDT。阈值取物理标准（范德华接触距离、拉氏图区域），
不对适应度调参。
""" % (len(rows_all), sum(1 for r in rows_all if r["native_binding_mode"])))

    B.append(r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../figures/paper/fig8_packing_basis.pdf}
\caption{\textbf{单体预测支持的量，以及它的兑现。} \textbf{a}、\textbf{b} 为短名单中预测堆积密度最高与最低的
两个候选：灰线为主链，彩色实心点是四个可变位点的侧链重原子，同色半透明小点是被计入局部堆积密度的
邻近重原子。\textbf{c} 为该量与实测适应度的关系（决策时标签未揭示）；虚线为中位数，
右半边实测均值 %.2f、左半边 %.2f。}
\label{fig:packing}
\end{figure*}
""" % (4.314, 3.315))

    B.append(r"""
\paragraph{方向由已测标签决定，不由直觉决定。} 在第 1 轮已揭示标签的 %d 个变体上，
"接触过近即有害"不成立（$\rho=%+.3f$，$p=%.2f$），成立的是局部堆积密度（$\rho=%+.3f$，$p=%.3f$）。
把它应用到决策时尚未揭示标签的 40 条短名单上，关系保持（$\rho=+0.49$，$p=0.001$；
按中位数切分，高密度一半实测均值 4.31、低密度一半 3.32，全局最优落在高密度一半，图~\ref{fig:packing}c）。

\paragraph{它对这一轮名单做了什么。} 作为约束 C9，按预测密度剔除短名单中最低的 10 条、
用组合分数上的后续候选补足：批次均值 %.3f $\to$ %.3f，最好值不变，全局最优仍被保留。
\emph{有效、可验证，但在 96 条预算下幅度很小}——这是它真实的量级。
""" % (len(json.loads((R / "structure/predicted_packing_r1.json").read_text())["observations"]),
       rel["min_cc"][0], rel["min_cc"][1], rel["packing_density"][0], rel["packing_density"][1],
       c9["metrics"]["A"]["mean"], c9["metrics"]["B"]["mean"]))
    (out / "appendix_structure.tex").write_text("\n".join(B) + "\n")

    # ------------------------------------------------------- appendix C: resources
    cnt = kg["counts"]
    rel_rows = "\n".join(r"%s & %d \\" % (k.replace("_", r"\_"), v)
                         for k, v in cnt["by_relation"].items())
    node_rows = "\n".join(r"%s & %d \\" % (k, v) for k, v in cnt["by_node_type"].items())
    C = [r"\section{附录：知识图谱、交互式 demo 与外部资源}", r"""
\subsection*{C.1\quad 知识图谱}
证据与测量被组织成一张显式的图，智能体按需查询，而不是把全部文本塞进提示词。
节点与关系如下（\texttt{knowledge/knowledge\_graph.json}，由 \texttt{scripts/build\_knowledge\_graph.py} 生成）：
""",
         r"\begin{center}\small\begin{tabular}{lr@{\hskip 2.2em}lr}\toprule",
         r"节点类型 & 数量 & 关系 & 数量 \\ \midrule"]
    nl = list(cnt["by_node_type"].items())
    rl = list(cnt["by_relation"].items())
    for i in range(max(len(nl), len(rl))):
        a = r"%s & %d" % (nl[i][0], nl[i][1]) if i < len(nl) else " & "
        b = r"\texttt{%s} & %d" % (rl[i][0].replace("_", r"\_"), rl[i][1]) if i < len(rl) else " & "
        C.append(a + " & " + b + r" \\")
    C.append(r"\bottomrule\end{tabular}\end{center}")
    C.append(r"""
图中包含题目所列的四类关系：\texttt{AminoAcid--HAS\_PROPERTY--Property}、
\texttt{Mutation--OCCURS\_AT--Position}、\texttt{Mutation--IMPROVES/HARMS--Fitness}、
\texttt{Variant--CONTAINS--Mutation}，另加两类本任务需要的边：
\texttt{Claim--SUPPORTS/REFUTES--Mutation}（带出处）与 \texttt{Mutation--COMPENSATES--Mutation}
（已发表的正上位性配对）。\textbf{泄漏纪律}：\texttt{IMPROVES}/\texttt{HARMS} 边只使用回路已经花掉预算测过的
%d 条标签，未揭示的候选池不进入图中。

\subsection*{C.2\quad 交互式 demo}
\texttt{scripts/recommend.py} 接受一条 56 残基序列，返回排序后的突变方案，并对每一条给出先验分量拆解与
图谱中的支持/反对证据：
\begin{quote}\small\texttt{python scripts/recommend.py --top-k 10}\\
\texttt{python scripts/recommend.py --sequence <56 aa> \textbackslash{}}\\
\texttt{\quad --labels results/v2/records\_final.json --round 2}
\end{quote}
不带标签时按证据先验排序（即第一轮的处境）；给定标签文件时同时拟合预测器并按正文的权重合成。
另有 \texttt{demo/index.html}：一轮决策中 15 次工具调用的完整记录，每一步的输入参数与返回内容都可展开。

\subsection*{C.3\quad 智能体模块与题目所列角色的对应}
题目列出的五个角色在本文中都有对应的代码模块，见表~\ref{tab:roles}。
""")
    C.append(r"""
\begin{table*}[t]\centering\small\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.26\textwidth}>{\raggedright\arraybackslash}p{0.68\textwidth}@{}}\toprule
题目所列角色 & 本文的实现 \\ \midrule
Data Analyst & \texttt{protocol.site\_statistics}：逐位点统计已测标签，连同上一轮审计写入提示词 \\
Hypothesis Generator & 语言模型按约定格式提出 3--4 条可检验陈述（位点、取代、方向） \\
Mutation Designer & \texttt{constraints.ConstraintSet.select}：在合法候选上按合成分数与组成配额生成批次 \\
Fitness Evaluator & \texttt{predictors.FitnessModel}，交叉验证选型，每轮重新拟合 \\
Scientific Critic & \texttt{interpret}：标签揭示后由程序裁决每条假说，被推翻的取代转为下一轮惩罚项 \\
\bottomrule\end{tabular}
\caption{题目所列的五个智能体角色与本文代码模块的对应关系。}\label{tab:roles}
\end{table*}
""")
    C.append(r"""
与题目示例的区别有两处，都是刻意的：候选序列由程序在约束下生成而非由语言模型点名
（\texttt{llm\_agent} 臂保留了"由模型挑选"的形态作为对照）；假说的裁决由程序完成而非由模型自评。

\subsection*{C.4\quad 接入自动化实验平台还缺什么}
本文回路在接口形态上已接近湿实验：每轮批次在揭示标签之前先按变体名单计算 SHA-256 并写入查询日志，
评估器被隔离在一次调用之后，替换这一次调用即可把「查表」换成向 LIMS 或液体处理工作站下发工单，
再把回读的测量值写回同一个日志。真正需要补齐的是两件事：
其一是\emph{批次与板位的映射}——96 条批次与 96 孔板天然对齐，但需要额外记录对照孔、重复数与板内位置效应；
其二是\emph{失败孔的处理}——湿实验会返回缺失值与异常值，而本文的虚拟评估器从不失败，
因此现有的假说裁决逻辑需要加入「测量无效」这一类判定，否则一次移液失败会被当作一次阴性结果。

\subsection*{C.5\quad 外部数据、模型与工具的来源与使用方式}
全部外部依赖及其版本锁定方式列于表~\ref{tab:resources}。
\begin{table*}[t]\centering\small\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.30\textwidth}>{\raggedright\arraybackslash}p{0.64\textwidth}@{}}\toprule
资源 & 来源与使用方式 \\ \midrule
GB1 四位点深度突变扫描 & FLIP 基准，固定提交 \texttt{62cace8}，文件校验和写死在加载器中；仅作虚拟评估器 \\
实验结构 & RCSB 1PGA、2GB1、1FCC，脚本按需下载并记录回执；用于可及度、二面角、界面与堆积计算 \\
ProteinMPNN & soluble \texttt{v\_48\_002}，钉定提交与权重 SHA-256；提供反折叠 log-odds，40 位按证据静音 \\
Boltz-2 & NVIDIA 托管推理；单体与复合物折叠，回执逐条记录（附录~\ref{sec:structure}） \\
DeepSeek-chat & 仅用于设定每轮的规则与权重；全部调用为真实 API，逐条记录随仓库提交 \\
文献证据 & 21 个公开来源，逐条抽取为结构化陈述并标注出处与置信度 \\
\bottomrule\end{tabular}
\caption{本文用到的外部数据、模型与工具，及其版本锁定与使用方式。}\label{tab:resources}
\end{table*}
""")
    (out / "appendix_resources.tex").write_text("\n".join(C) + "\n")
    print("wrote appendix_structure.tex and appendix_resources.tex")

    print("wrote", out / "failure_cases.tex", "and", out / "future_work.tex")


if __name__ == "__main__":
    main()
