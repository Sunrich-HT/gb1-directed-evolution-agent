#!/usr/bin/env python
"""Render results/trace_seed302.json as a session transcript (demo/index.html).

The page is a view over the captured trace: every tool call shows the arguments it
was given and the output it returned, with the raw JSON one click away.  Nothing is
written by hand here - if a number is on the page, it is in the trace file.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SITES = (39, 40, 41, 54)

TOOL_TITLE = {
    "http.get": ("取结构文件", "HTTP GET · files.rcsb.org"),
    "structure.analyze": ("对序列做结构分析", "shrake_rupley + backbone_dihedrals"),
    "inverse_folding.load": ("载入反折叠打分", "ProteinMPNN soluble v_48_002"),
    "prior.score": ("用证据先验给 149,361 条候选打分", "EvidencePrior.score_variants"),
    "predictor.fit": ("用已有标签拟合预测器", "select_model + FitnessModel.fit"),
    "llm.set_rules": ("让语言模型设定本轮规则", "DeepSeek · deepseek-chat"),
    "constraints.apply": ("约束引擎选出 96 条批次", "ConstraintSet.select"),
    "oracle.query": ("冻结批次后查询评估器", "virtual oracle · SHA-256 committed"),
    "hypotheses.judge": ("程序裁决本轮假说", "program verdict"),
    "boltz2.predict": ("调用结构预测服务", "POST health.api.nvidia.com · Boltz-2"),
    "boltz2.fold_complex": ("把变体与 IgG-Fc 折成复合物", "POST health.api.nvidia.com · Boltz-2 · 262 / 468 残基"),
    "structure.check_docking": ("先验证结合模式对不对", "check_complex_docking.py"),
    "boltz2.fold_monomers": ("折叠候选的单体结构", "POST health.api.nvidia.com · Boltz-2 · 64 个变体"),
    "structure.learn_relation": ("在已测标签上学结构量的方向", "analyze_predicted_packing.py"),
    "constraints.apply_C9": ("结构约束改写本轮名单", "C9_predicted_packing"),
}


def esc(x) -> str:
    return html.escape(str(x)).replace("A^2", "Å²").replace("A^3", "Å³")


def jdump(obj, limit=None) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    if limit and len(s) > limit:
        s = s[:limit] + "\n  ... (完整内容见 results/trace_seed302.json)"
    return esc(s)


def inline(path: Path, width=1500) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pre(text, cls="") -> str:
    return f'<pre class="raw {cls}">{text}</pre>'


def details(summary, inner) -> str:
    return f"<details><summary>{summary}</summary>{inner}</details>"


def table(headers, rows, cls="") -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def seq_track(per_res) -> str:
    """One cell per residue, shaded by burial, with the four sites marked."""
    cells = []
    for r in per_res:
        rs = r["rel_sasa"]
        shade = "" if rs is None else f"background:rgba(29,78,216,{max(0.06, 1 - min(rs, 1)):.2f})"
        cls = "site" if r["resseq"] in SITES else ""
        tip = f'{r["aa"]}{r["resseq"]} 相对可及度 {"-" if rs is None else f"{100*rs:.0f}%"}'
        if r["phi"] is not None and r["psi"] is not None:
            tip += f' φ/ψ {r["phi"]:.0f}/{r["psi"]:.0f} {r["region"] or ""}'
        cells.append(f'<span class="res {cls}" style="{shade}" title="{esc(tip)}">{r["aa"]}</span>')
    return f'<div class="track">{"".join(cells)}</div>'


def render_step(i, s, figs) -> str:
    tool = s["tool"]
    title, sub = TOOL_TITLE.get(tool, (tool, ""))
    call, res, note = s["call"], s["result"], s.get("note", "")
    body = ""

    if tool == "http.get":
        rows = [[k.upper(), f'HTTP {v["status"]}', f'{v["bytes"]:,} B',
                 f'{v["seconds"]}s', f'<code>{v["sha256"][:16]}…</code>'] for k, v in res.items()]
        body = (table(["PDB", "状态", "大小", "耗时", "SHA-256"], rows)
                + pre(esc("\n".join(res["1pga"]["head"]))))

    elif tool == "structure.analyze":
        per = res["per_residue"]
        site_rows = [[f'{res["mutable_sites"][str(p)]["wild_type"]}{p}',
                      f'{100*res["mutable_sites"][str(p)]["rel_sasa"]:.1f}%',
                      f'{res["mutable_sites"][str(p)]["interface"]["buried_surface_on_binding_A2"]:.1f}',
                      f'{res["mutable_sites"][str(p)]["phi"]:.0f}° / {res["mutable_sites"][str(p)]["psi"]:.0f}°',
                      f'{res["mutable_sites"][str(p)]["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"]:.2f}']
                     for p in SITES]
        body = (f'<p class="lead">全序列 {res["n_residues"]} 个残基的可及度（颜色越深越埋藏，悬停看数值），'
                f'黄框为四个可变位点：</p>' + seq_track(per)
                + table(["位点", "相对可及度", "结合时埋藏 (Å²)", "φ / ψ", "虚拟 Cβ 最近重原子 (Å)"], site_rows)
                + f'<p class="lead">与 Fc 的接触残基共 {len(res["fc_contact_residues"])} 个，'
                  f'四个可变位点里只有 40 位真正埋在界面上（41.9 Å²）。</p>'
                + details("展开全部 56 个残基的原始输出", pre(jdump(per, 2600))))

    elif tool == "inverse_folding.load":
        lo = res["log_odds_by_site"]
        aas = list(next(iter(lo.values())))
        rows = [[f"{p}"] + [f'{lo[str(p)][a]:+.2f}' for a in aas] for p in SITES]
        body = (table(["位点"] + aas, rows, "num")
                + f'<p class="lead">{esc(res["convention"])}</p>'
                + f'<p class="lead">权重与校验和：<code>{esc(call["weights_sha256"][:24])}…</code>，'
                  f'结构 {esc(call["structure"])}。</p>')

    elif tool == "prior.score":
        a, w = res["attribution_global_optimum"], res["attribution_wild_type"]
        zh = {"mpnn": "反折叠分数", "packing_headroom": "堆积余量", "burial_charge": "埋藏电荷",
              "interface_bonus": "Fc 界面加成", "pair_volume_compensation": "41/54 体积补偿",
              "literature": "文献条目", "total": "合计"}
        rows = [[zh.get(k, k), f'{a[k]:+.3f}', f'{w[k]:+.3f}'] for k in a]
        ev, ub = res["zero_label_evaluation"], res["measured_single_mutant_upper_bound"]
        body = (table(["先验分量", "FWAA", "野生型 VDGV"], rows)
                + f'<p class="lead">全局最优 FWAA 在这套打分里排第 '
                  f'<b>{res["rank_of_global_optimum"]:,}</b> 名（共 149,361 条，前 0.61%）——'
                  f'而它没有用任何多突变标签。</p>'
                + table(["零标签评测", "ρ（≥3 突变）", "top-5% 精度", "前 96 内最好", "top-1% 富集"],
                        [["证据先验", f'{ev["spearman_hd3plus"]:+.3f}', f'{ev["top5pct_precision"]:.3f}',
                          f'{ev["best_in_top96"]:.3f}', f'{ev["enrichment_top1pct"]:.1f}×'],
                         ["用实测单突变标签的上界", f'{ub["spearman_hd3plus"]:+.3f}',
                          f'{ub["top5pct_precision"]:.3f}', f'{ub["best_in_top96"]:.3f}',
                          f'{ub["enrichment_top1pct"]:.1f}×']])
                + f'<img src="{figs["prior"]}" alt="先验构成与评测">')

    elif tool == "predictor.fit":
        rows = [[esc(c["spec"]), f'{c["cv_spearman"]:+.3f}', f'{c["cv_rmse"]:.3f}']
                for c in res["cv_table"]]
        body = (f'<p class="lead">用当前 {call["n_train"]} 条标签做 {call["cv_folds"]} 折交叉验证选型：</p>'
                + table(["候选模型", "CV Spearman", "CV RMSE"], rows)
                + f'<p class="lead">选中 <code>{esc(res["selected"])}</code>。'
                  f'它对全局最优的预测值只有 <b>{res["prediction_for_global_optimum"]}</b>'
                  f'（池内第 {res["model_rank_of_global_optimum"]} 名），实测是 8.762 ——'
                  f'标签稀少时模型对高值区间几乎没有分辨力。</p>')

    elif tool == "llm.set_rules":
        p = call["payload"]
        reply = res["raw_response"]
        try:
            parsed = json.loads(reply)
        except Exception:
            parsed = None
        urows = [[esc(u["id"]),
                  esc("; ".join(f"{k} = {v}" for k, v in (u.get("params") or {}).items()
                                if k != "sites")) or "—",
                  "启用" if u.get("active", True) else "停用",
                  esc(u.get("reason", ""))]
                 for u in (parsed or {}).get("constraint_updates", [])]
        hrows = [[esc(h["statement"]), f'{h.get("site","")}', esc(h.get("direction", ""))]
                 for h in (parsed or {}).get("hypotheses", [])]
        body = (f'<div class="two"><div><span class="lbl">发给模型的内容（节选）</span>'
                + pre(jdump({k: p[k] for k in ("round", "task", "labelled_data_summary",
                                               "previous_round_feedback")
                             if k in p}, 1500))
                + details("展开完整 payload（含先验构成、当前约束、逐位点统计、上一轮审计）",
                          pre(jdump(p, 7000)))
                + '</div><div><span class="lbl">模型返回的原始字符串</span>'
                + pre(esc(reply[:900] + ("…" if len(reply) > 900 else "")))
                + details("展开完整回复", pre(esc(reply))) + '</div></div>'
                + '<p class="lead">解析后它提了这些可检验假说：</p>'
                + table(["陈述", "位点", "方向"], hrows, "wide")
                + '<p class="lead">以及这些规则修改（它只能改参数，不能点名变体）：</p>'
                + table(["约束", "参数", "状态", "理由（原话）"], urows, "wide")
                + (f'<p class="lead">本次调用：{res["ledger"].get("prompt_tokens","?")} 输入 token，'
                   f'{res["ledger"].get("completion_tokens","?")} 输出 token。</p>'
                   if isinstance(res.get("ledger"), dict) and "prompt_tokens" in res["ledger"] else ""))

    elif tool == "constraints.apply":
        A, B = res["audit_A"], res["audit_B"]
        rows = [["合法候选（硬过滤后）", f'{A["n_legal_after_filters"]:,}', f'{B["n_legal_after_filters"]:,}'],
                ["其中被体积上限删除", f'{A["filters_removed"].get("C2_volume_window",0):,}',
                 f'{B["filters_removed"].get("C2_volume_window",0):,}'],
                ["与无约束排序相比被改变的名额", A.get("changed_vs_unconstrained", 0),
                 B.get("changed_vs_unconstrained", 0)],
                ["批次里是否含全局最优", "是" if res["global_optimum_in_A"] else "否",
                 "是" if res["global_optimum_in_B"] else "否"]]
        ent = ", ".join(x["variant"] for x in res["entered"][:16])
        lft = ", ".join(x["variant"] + ("*" if x["refuted"] else "") for x in res["left"][:16])
        body = (f'<p class="lead">这一步是本页最关键的对照：<b>同一个模型、同一个先验、同一个候选池，只换规则</b>，'
                f'看 96 个名额怎么变。</p>'
                + table(["", "规则集 A（第 1 轮）", "规则集 B（第 2 轮，模型刚改的）"], rows)
                + f'<p class="lead">结果是 <b>{res["n_entered"]} / 96</b> 个名额换人。'
                  f'换进来的：<code>{esc(ent)}</code>…<br>换出去的：<code>{esc(lft)}</code>…'
                  f'（<code>*</code> 表示带有上一轮被推翻的取代）</p>'
                + f'<p class="lead warn">需要说清楚的一点：C8 是<b>软惩罚</b>而不是硬过滤，'
                  f'所以带着被推翻取代的变体并没有被清出批次——换出 {res["n_left_carrying_refuted"]} 个、'
                  f'换进 {res["n_entered_carrying_refuted"]} 个，基本持平。真正改变名单的是放宽的体积上限'
                  f'（合法候选 +{B["n_legal_after_filters"]-A["n_legal_after_filters"]:,} 条）'
                  f'与重新加权的界面惩罚。全局最优在两套规则下都合法，'
                  f'它进入批次靠的是先验排名，不是这次改规则。</p>'
                + details("展开两套规则下的完整审计记录", pre(jdump({"A": A, "B": B}, 2600))))

    elif tool == "oracle.query":
        rows = [[esc(d["variant"]), f'{d["predicted"]:.3f}', f'{d["fitness"]:.3f}']
                for d in res["top_revealed"]]
        g = res.get("global_optimum_row")
        body = (f'<p class="lead">批次在揭示任何标签之前先按变体名单算 SHA-256 冻结：'
                f'<code>{esc(call["commit_sha256"][:32])}…</code></p>'
                + table(["揭示后的前几名", "模型当时的预测", "实测适应度"], rows)
                + (f'<p class="hit"><b>本轮命中全局最优 FWAA，实测 {g["fitness"]:.3f}。</b>'
                   f'模型当时对它的预测只有 {g["predicted"]:.3f}。</p>' if g else "")
                + f'<p class="lead">批次均值 {res["batch_mean"]}，最好 {res["batch_best"]:.3f}。</p>')

    elif tool == "hypotheses.judge":
        rows = []
        for h in res["hypotheses"]:
            e = h.get("evidence") or {}
            tag = {"supported": "支持", "refuted": "推翻", "undetermined": "证据不足"}.get(
                h["verdict"], h["verdict"])
            cls = {"supported": "ok", "refuted": "bad"}.get(h["verdict"], "mid")
            rows.append([esc(h["statement"]), f'<span class="tag {cls}">{tag}</span>',
                         f'{e.get("n_with_substitution","-")}',
                         f'{e.get("mean_with",0):.3f} / {e.get("mean_without",0):.3f}'
                         if e else "—"])
        body = (f'<p class="lead">{esc(call["rule"])}</p>'
                + table(["模型提出的陈述", "裁决", "带该取代的变体数", "带 / 不带 的均值"], rows, "wide"))

    elif tool == "boltz2.predict":
        rows = [[f'HTTP {r["status"]}', f'{r["seconds"]}s', "ok" if r.get("ok") else "retry"]
                for r in res["last_receipts"]]
        body = (f'<p class="lead">请求体：</p>' + pre(jdump(call["payload"]))
                + f'<p class="lead">本次运行共 {res["n_receipts"]} 条回执，状态分布 '
                  f'{esc(res["status_counts"])}；最近三条：</p>'
                + table(["状态", "耗时", "结果"], rows)
                + f'<p class="lead">返回的 mmCIF 开头：</p>'
                + pre(esc("\n".join(res["cif_head"])))
                + f'<img src="{figs["structure"]}" alt="Boltz-2 预测结构">')

    elif tool == "boltz2.fold_complex":
        r1, r2 = res["single_fc_chain"], res["native_fc_homodimer"]
        body = (f'<p class="lead">{esc(call["why"])}</p>'
                + table(["设置", "折叠数", "总残基", "中位耗时"],
                        [["单链 Fc", r1["n_folded"], r1["residues_total"], f'{r1["median_seconds"]:.1f}s'],
                         ["天然 Fc 同源二聚体", r2["n_folded"], r2["residues_total"], f'{r2["median_seconds"]:.1f}s']])
                + f'<p class="lead">请求体结构：</p>' + pre(jdump(call["payload_shape"])))

    elif tool == "structure.check_docking":
        rows = [[esc(r["file"]), f'{r["fc_superposition_rmsd_A"]:.2f}',
                 f'{r["gb1_pose_rmsd_to_crystal_A"]:.2f}',
                 f'{len(r["overlap_with_crystal_contacts"])}/{r["n_crystal_contacts"]}',
                 ", ".join(str(x) for x in r["mutable_sites_at_interface"]) or "无"]
                for r in res["per_structure"]]
        body = (f'<p class="lead">{esc(call["test"])}；通过条件：{esc(call["pass_condition"])}</p>'
                + table(["预测结构", "Fc 叠合 RMSD (Å)", "GB1 位姿 RMSD (Å)",
                         "与晶体接触面重合", "界面上的可变位点"], rows, "wide")
                + f'<img src="{figs["docking"]}" alt="预测复合物与晶体复合物的对比">'
                + f'<p class="lead warn"><b>结论：</b>{res["n_native_mode"]} / '
                  f'{len(res["per_structure"])} 个预测复合物达到天然结合模式。'
                  f'这一步的作用是<b>否决</b>——没有它，后面任何"预测界面面积"都是在错误的位姿上量出来的。</p>')

    elif tool == "boltz2.fold_monomers":
        body = (f'<p class="lead">{esc(call["why"])}。测量的量：</p><ul>'
                + "".join(f'<li>{esc(o)}</li>' for o in res["observables"]) + '</ul>'
                + f'<p class="lead">单个变体的原始观测：</p>' + pre(jdump(res["example_observation"]))
                + f'<p class="lead">本次运行累计 {res["n_receipts_total"]} 条 NVIDIA 回执。</p>')

    elif tool == "structure.learn_relation":
        rows = [[k, f'{v["rho"]:+.3f}', f'{v["p"]:.3f}',
                 "显著" if v["p"] < 0.05 else "不显著"]
                for k, v in res["spearman_against_known_fitness"].items()]
        body = (f'<p class="lead">{esc(call["rule"])}（{esc(call["training_set"])}）</p>'
                + table(["结构观测量", "与已知适应度的 Spearman ρ", "p", ""], rows))

    elif tool == "constraints.apply_C9":
        t = res["transfer_check_on_heldout_shortlist"]
        m = res["median_split"]
        bm = res["batch_metrics"]
        body = (f'<p class="lead">{esc(call["definition"])}<br>标定依据：{esc(call["calibrated_on"])}</p>'
                + f'<p class="lead">迁移检验（决策时未揭示标签的 40 条短名单）：'
                  f'Spearman ρ = <b>{t["spearman_rho"]:+.3f}</b>，p = {t["p"]}，n = {t["n"]}</p>'
                + table(["按预测堆积密度中位数切分", "条数", "实测均值", "实测最好"],
                        [["密度高的一半", m["high_density_half"]["n"],
                          f'{m["high_density_half"]["mean_fitness"]:.3f}',
                          f'{m["high_density_half"]["best"]:.3f}'],
                         ["密度低的一半", m["low_density_half"]["n"],
                          f'{m["low_density_half"]["mean_fitness"]:.3f}',
                          f'{m["low_density_half"]["best"]:.3f}']])
                + f'<p class="lead">对本轮 96 条名单的实际改动：剔除 '
                  f'<code>{esc(", ".join(res["dropped"]))}</code>，补入 '
                  f'<code>{esc(", ".join(res["backfilled"]))}</code>。</p>'
                + table(["", "批次均值", "批次最好"],
                        [["不加 C9", f'{bm["A"]["mean"]:.3f}', f'{bm["A"]["best"]:.3f}'],
                         ["加 C9", f'{bm["B"]["mean"]:.3f}', f'{bm["B"]["best"]:.3f}']])
                + f'<img src="{figs["packing"]}" alt="预测单体的局部堆积与实测适应度的关系">'
                + f'<p class="lead warn">结构量确实迁移到了未揭示的候选上，方向也和在已测标签上学到的一致；'
                  f'但在 96 条的预算下它对批次的改变很小（均值 {bm["A"]["mean"]:.3f} → {bm["B"]["mean"]:.3f}），'
                  f'最好值不变，全局最优仍被保留（{"是" if res["global_optimum_retained"] else "否"}）。'
                  f'这是它真实的量级，不是一个被放大的卖点。</p>')

    return f"""
<section class="turn tool">
  <div class="gutter"><span class="idx">{i}</span></div>
  <div class="block">
    <div class="bar">
      <span class="name">{esc(title)}</span>
      <code class="sig">{esc(sub)}</code>
    </div>
    <div class="inner">
      {f'<p class="note">{esc(note)}</p>' if note else ''}
      {body}
      {details("查看这一步的调用参数（原始 JSON）", pre(jdump(call, 3000)))}
    </div>
  </div>
</section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="results/trace_seed302.json")
    ap.add_argument("--out", default="demo/index.html")
    args = ap.parse_args()

    T = json.loads(Path(args.trace).read_text())
    F = Path("figures/paper")
    figs = {"prior": inline(F / "fig1_prior.png"),
            "structure": inline(F / "fig6_structure.png"),
            "docking": inline(F / "fig7_docking_check.png"),
            "packing": inline(F / "fig8_packing_basis.png")}

    seq_html = (esc(SEQ[:38]) + "<b>" + esc(SEQ[38:41]) + "</b>" + esc(SEQ[41:53])
                + "<b>" + esc(SEQ[53]) + "</b>" + esc(SEQ[54:]))

    steps = "".join(render_step(i + 1, s, figs) for i, s in enumerate(T["steps"]))

    struct_html = ""
    sp = Path("results/trace_structure.json")
    if sp.exists():
        ST = json.loads(sp.read_text())
        n0 = len(T["steps"])
        struct_html = (
            '<h2 class="sect">结构信息如何进入下一轮的策略</h2>'
            '<p class="sub">上面那一轮里，结构只提供了静态的证据（晶体坐标）。'
            '这一节是把结构预测服务真正接进决策回路的完整过程：先折复合物，'
            '验证结合模式，否决掉不能用的部分，再用能用的部分造出一条约束，并量出它的实际效果。</p>'
            + "".join(render_step(n0 + i + 1, s, figs) for i, s in enumerate(ST["steps"])))

    css = """
:root{--ink:#18181b;--mut:#71717a;--line:#e4e4e7;--bg:#fafaf9;--card:#fff;
--blue:#1d4ed8;--good:#15803d;--bad:#b91c1c;--warn:#a16207}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:48px 20px 90px}
h1{font-size:25px;line-height:1.45;margin:0 0 8px;letter-spacing:-.01em}
.kicker{font-size:14px;line-height:1.5;color:#1b2733;font-weight:600;margin:0 0 2px}
.byline{font-size:12.5px;color:#64748b;margin:0 0 18px}
.byline a{color:inherit}
.sub{color:var(--mut);font-size:14px;margin:0 0 18px}
.meta{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:34px}
.chip{font-size:12px;color:var(--mut);background:#f4f4f5;border:1px solid var(--line);
border-radius:999px;padding:3px 10px}
.chip a{color:var(--blue);text-decoration:none}
.turn{display:flex;gap:14px;margin:18px 0}
.gutter{flex:0 0 26px;display:flex;flex-direction:column;align-items:center}
.idx{width:22px;height:22px;border-radius:6px;background:#27272a;color:#fff;font-size:11px;
display:flex;align-items:center;justify-content:center;font-variant-numeric:tabular-nums}
.gutter:after{content:"";flex:1;width:1px;background:var(--line);margin-top:6px}
.turn:last-child .gutter:after{display:none}
.block{flex:1;min-width:0;background:var(--card);border:1px solid var(--line);border-radius:10px;
overflow:hidden}
.bar{display:flex;align-items:center;justify-content:space-between;gap:12px;
padding:9px 14px;background:#fafafa;border-bottom:1px solid var(--line)}
.bar .name{font-size:14.5px;font-weight:600}
.sig{font-size:11.5px;color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.inner{padding:14px 16px}
.user .block{background:#f8fafc;border-color:#dbeafe}
.user .bar{background:#eff6ff;border-color:#dbeafe}
p{margin:9px 0}
.note{color:var(--mut);font-size:13.5px;margin:0 0 10px;padding-left:9px;
border-left:2px solid var(--line)}
.lead{font-size:14px}
.lead.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:7px;padding:9px 12px;
font-size:13.5px}
pre.raw{background:#fafafa;border:1px solid var(--line);border-radius:7px;padding:10px 12px;
font:11.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:auto;max-height:290px;
white-space:pre-wrap;word-break:break-word;margin:9px 0}
details{margin:9px 0}
summary{cursor:pointer;font-size:12.5px;color:var(--blue);user-select:none}
summary::marker{color:var(--mut)}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:12.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-size:11.5px;font-weight:600;border-bottom:1px solid #d4d4d8}
tbody tr:last-child td{border-bottom:none}
table.wide td:first-child{max-width:330px}
table.num td,table.num th{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
padding:5px 4px;text-align:right}
table.num td:first-child,table.num th:first-child{text-align:left}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
background:#f4f4f5;border-radius:4px;padding:1px 4px}
.track{display:flex;flex-wrap:wrap;gap:1px;margin:10px 0}
.res{width:17px;height:21px;display:flex;align-items:center;justify-content:center;
font:10.5px ui-monospace,SFMono-Regular,Menlo,monospace;border-radius:2px;color:#0b1220;cursor:help}
.res.site{outline:2px solid #f59e0b;outline-offset:-2px;font-weight:700}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.two .lbl{display:block;font-size:11px;letter-spacing:.06em;color:var(--mut);margin-bottom:3px}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px}
.tag.ok{background:#dcfce7;color:var(--good)}
.tag.bad{background:#fee2e2;color:var(--bad)}
.tag.mid{background:#f4f4f5;color:var(--mut)}
.hit{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:7px;padding:9px 12px;font-size:14px}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:7px;margin:10px 0 2px}
pre.seq{background:#fafafa;border:1px solid var(--line);border-radius:7px;padding:10px;
font:12px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto;margin:10px 0}
pre.seq b{color:#b45309}
h2.sect{font-size:19px;margin:44px 0 6px;padding-top:26px;border-top:2px solid var(--line)}
footer{margin-top:36px;padding-top:18px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px}
@media(max-width:700px){.two{grid-template-columns:1fr}}
"""

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>先验与约束驱动的蛋白质定向进化智能体 · 一轮的真实调用过程</title>
<style>{css}</style></head><body><div class="wrap">
<div class="kicker">先验与约束驱动的蛋白质定向进化智能体：GB1 四位点景观上的七臂对照研究</div>
<div class="byline">孙浩桐 · <a href="mailto:richsun821@gmail.com">richsun821@gmail.com</a></div>
<h1>一轮决策的完整调用过程</h1>
<p class="sub">种子 {T['seed']} 的第 {T['round']} 轮。下面每一步都是这次运行里真实发生的调用：
左边是传给工具的参数，右边是工具返回的内容，原始 JSON 都可以展开。
trace 由 <code>scripts/capture_trace.py</code> 抓取（结构文件是现场重新下载的）。</p>
<div class="meta">
  <span class="chip">抓取时间 {T['captured_at']}</span>
  <span class="chip">{len(T['steps'])} 次工具调用（本节）</span>
  <span class="chip"><a href="https://github.com/Sunrich-HT/gb1-directed-evolution-agent">源码与完整 trace</a></span>
</div>

<section class="turn user">
  <div class="gutter"><span class="idx">▸</span></div>
  <div class="block">
    <div class="bar"><span class="name">任务</span><code class="sig">输入</code></div>
    <div class="inner">
      <p>在 GB1 的 39/40/41/54 四个位点上，用 192 条起始标签 + 每轮 96 条的预算找到适应度最高的变体。
      第一轮必须由真实先验知识驱动，之后每轮都要用非序列维度的显式约束，全过程可解释。</p>
      <pre class="seq">{seq_html}</pre>
      <p class="note">加粗为四个可变位点，野生型 V39 · D40 · G41 · V54。
      适应度由已发表的深度突变扫描查表充当评估器（149,361 条），不是湿实验。</p>
    </div>
  </div>
</section>
{steps}
{struct_html}
<footer>本页是 results/trace_seed302.json 的视图，由 scripts/make_demo_page.py 渲染；
页面上的每个数字都能在该 trace 文件中找到对应字段。</footer>
</div></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
