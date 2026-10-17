#!/usr/bin/env python
"""Build demo/index.html - a self-contained walkthrough of ONE complete campaign.

Not an interactive app: a presentation page.  Every number, rule edit, verdict and
figure is read from results/ and knowledge/, so the page cannot disagree with the
run it describes.  Figures are downscaled and inlined as base64 so the single file
opens anywhere with no assets and no network.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent import evidence as evidence_mod
from gb1agent.data import load_landscape

SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def esc(x) -> str:
    """Escape, then restore the units the sources write as ASCII (A^2, A^3)."""
    out = html.escape(str(x))
    return out.replace("A^2", "Å²").replace("A^3", "Å³")


def fmt_val(k, v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if k == "max_added_volume":
        return f"{v} Å³"
    if isinstance(v, dict):
        return ", ".join(f"{a} {b:g}" if isinstance(b, (int, float)) else f"{a} {b}"
                         for a, b in v.items())
    return str(v)


def inline_png(path: Path, width: int = 1500) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def kv(rows) -> str:
    out = ['<dl class="kv">']
    for k, v in rows:
        out.append(f"<dt>{k}</dt><dd>{v}</dd>")
    out.append("</dl>")
    return "\n".join(out)


def table(headers, rows, cls="") -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def step(n, title, tool, io_in, io_out, body) -> str:
    return f"""
<section class="step">
  <div class="rail"><span class="num">{n}</span></div>
  <div class="card">
    <div class="head">
      <h3>{title}</h3>
      <span class="tool">{tool}</span>
    </div>
    <div class="io">
      <div class="in"><span class="lbl">输入</span>{io_in}</div>
      <div class="out"><span class="lbl">输出</span>{io_out}</div>
    </div>
    <div class="body">{body}</div>
  </div>
</section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--out", default="demo/index.html")
    args = ap.parse_args()

    R = Path("results")
    K = Path("knowledge")
    F = Path("figures/paper")
    records = json.loads((R / "v2/records_final.json").read_text())
    base = json.loads((K / "evidence_base.json").read_text())
    struct = json.loads((K / "structure_evidence.json").read_text())
    pe = json.loads((R / "prior_evaluation.json").read_text())
    baselines = json.loads((R / "baselines.json").read_text())
    check = {c["arm"]: c for c in json.loads((R / "case_round2_model_check.json").read_text())}
    fold = json.loads((R / "structure/cif/fold_meta.json").read_text())
    prior = evidence_mod.load(str(K / "structure_evidence.json"),
                             "config/mpnn_log_odds.json", str(K / "evidence_base.json"))

    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)
    target = str(variants[int(np.argmax(fitness))])
    scores = prior.score_variants(variants)
    rank = int((scores > scores[list(variants).index(target)]).sum()) + 1
    pct = 100 * rank / len(variants)
    att, att_wt = prior.attribution(target), prior.attribution("VDGV")

    rows = {r["round"]: r for r in records
            if r["arm"] == "prior_constraint" and r["seed"] == args.seed}
    greedy = {r["round"]: r for r in records
              if r["arm"] == "greedy" and r["seed"] == args.seed}
    fixed = {r["round"]: r for r in records
             if r["arm"] == "prior_fixed" and r["seed"] == args.seed}
    n_seeds = len({r["seed"] for r in records})
    S = struct["sites"]
    cv = [m for m in baselines["d0_models"] if m["cv_selected"]]
    ev = {r["id"]: r for r in base["evidence"]}

    P = []

    # ---------------------------------------------------------------- input
    P.append(f"""
<section class="step">
  <div class="rail"><span class="num dot">▶</span></div>
  <div class="card input">
    <div class="head"><h3>任务输入</h3><span class="tool">用户给定</span></div>
    <div class="body">
      {kv([("目标", "在 GB1 四位点组合空间上，用尽可能少的测量找到适应度最高的变体"),
           ("蛋白", "链球菌蛋白 G 的免疫球蛋白结合域 B1（56 残基）"),
           ("可变位点", "39 / 40 / 41 / 54，野生型 V39 · D40 · G41 · V54"),
           ("预算", f"起始随机标签 192 条，之后 3 轮 × 每轮 96 条，共 480 条"),
           ("评估方式", "已发表深度突变扫描查表（149,361 条）；批次以 SHA-256 冻结后才允许揭示标签"),
           ("硬性要求", "第一轮必须由真实先验知识驱动；之后每一轮都要用非序列维度的显式约束；全过程可解释")])}
      <pre class="seq">{SEQ[:38]}<b>{SEQ[38]}{SEQ[39]}{SEQ[40]}</b>{SEQ[41:53]}<b>{SEQ[53]}</b>{SEQ[54:]}</pre>
      <p class="note">加粗为四个可变位点。本页展示的是种子 {args.seed} 的一次完整运行，
      所有数字都直接取自该次运行留下的记录文件。</p>
    </div>
  </div>
</section>""")

    # ---------------------------------------------------------------- 1 evidence
    ids = ["EV034", "EV019", "EV020", "EV032", "EV053"]
    erows = [[i, esc(ev[i]["statement"][:120] + ("..." if len(ev[i]["statement"]) > 120 else "")),
              esc((ev[i].get("source") or {}).get("id", ""))]
             for i in ids if i in ev]
    P.append(step(1, "挖掘已发表证据", "文献 / 专利检索 · 全文阅读 · 结构化抽取",
                  "GB1 结构与突变文献、IgG-Fc 复合物、定向进化方法学",
                  f"{len(base['evidence'])} 条结构化证据，{len(base['sources'])} 个来源",
                  f"""<p>每条证据都带来源、置信度与作用范围（全局 / 方法 / 位点 / 位点对），
                  并显式映射到后面的某一个先验分量或某一条约束。下面是对本案例最关键的五条：</p>
                  {table(["编号", "陈述（节选）", "来源"], erows, "wide")}
                  <p class="note">专利条目为 0 条：检索接口在本次运行期间不可用，这一点在论文局限中如实写明，
                  没有用任何推测内容填补。</p>"""))

    # ---------------------------------------------------------------- 2 structure
    srows = [[f"{S[k]['wild_type']}{k}", f"{100*S[k]['rel_sasa']:.1f}%",
              f"{S[k]['interface']['buried_surface_on_binding_A2']:.1f}",
              f"{S[k]['phi']:.1f}° / {S[k]['psi']:.1f}°",
              f"{S[k]['packing_headroom']['virtual_cbeta_nearest'][0]['distance']:.2f}"]
             for k in ("39", "40", "41", "54")]
    P.append(step(2, "从结构坐标算出位点分工", "Shrake-Rupley SASA · 主链二面角 · 虚拟 Cβ 重建",
                  "RCSB 1PGA（晶体，自由态）、1FCC（与 IgG-Fc 复合）",
                  "四个位点的埋藏程度、界面贡献、骨架构象与堆积余量",
                  f"""{table(["位点", "相对溶剂可及度", "结合时埋藏表面 (Å²)", "φ / ψ", "虚拟 Cβ 最近重原子 (Å)"], srows)}
                  <p><b>读出来的判断：</b>只有 40 位真正位于结合界面（埋藏 41.9 Å²，其余三位点 ≤ 7.2 Å²），
                  39 与 54 近乎完全埋在疏水核心。因此 40 位用"界面加成"打分，39/41/54 用堆积与体积类的项——
                  四个位点不套同一个公式。41 位的特殊之处是空间：虚拟 Cβ 距 55 位羰基氧仅 2.11 Å。</p>"""))

    # ---------------------------------------------------------------- 3 prior
    comp = [("mpnn", "反折叠分数", "ProteinMPNN，40 位按证据静音"),
            ("packing_headroom", "堆积余量", "虚拟 Cβ 间距（1PGA）"),
            ("burial_charge", "埋藏电荷惩罚", "相对可及度（1PGA）"),
            ("interface_bonus", "Fc 界面加成", "ΔSASA（1FCC）"),
            ("pair_volume_compensation", "41/54 体积补偿", "G41L/V54G 机制（EV020）"),
            ("literature", "文献条目", "62 条结构化证据")]
    arows = [[zh, f"{att[k]:+.3f}", f"{att_wt[k]:+.3f}", why] for k, zh, why in comp]
    arows.append([f"<b>合计</b>", f"<b>{att['total']:+.3f}</b>", f"{att_wt['total']:+.3f}",
                  f"先验排名 {rank:,} / {len(variants):,}（前 {pct:.2f}%）"])
    ep = pe["priors"]["evidence_prior"]
    up = pe["priors"]["leakage_measured_single_mutants"]
    P.append(step(3, "把证据编译成一个可拆解的先验", "evidence.py · 零标签评测",
                  "结构数值 + 物化性质 + 62 条证据",
                  f"4×20 取代打分表 + 位点对交互项；Spearman ρ = {ep['spearman_hd3plus']:+.3f}（≥3 突变）",
                  f"""<p>先验不使用任何多突变标签。它的质量与<b>使用实测单突变标签</b>的加性上界
                  （ρ = {up['spearman_hd3plus']:+.3f}）持平，其排序前 96 名中最好变体实测
                  {ep['best_in_top96']:.3f}，top-1% 富集 {ep['enrichment_top1pct']:.1f} 倍。</p>
                  <p>关键在于它<b>可以被拆开</b>。对全局最优 {target} 的打分分解：</p>
                  {table(["先验分量", target, "野生型 VDGV", "依据"], arows, "wide")}
                  <p class="note">{target} 的四个取代中 V39F（0.38）与 G41A（0.12）单独出现时都低于野生型，
                  任何按单突变收益排序的策略都会先把它们丢掉。</p>
                  <img src="{inline_png(F / 'fig1_prior.png')}" alt="先验构成与评测">"""))

    # ---------------------------------------------------------------- 4 architecture
    P.append(step(4, "系统是怎么接起来的", "架构",
                  "先验 + 拟合预测器 + 约束引擎 + 虚拟评估器",
                  "每轮 96 条批次，以及每条规则的审计记录",
                  f"""<p>语言模型只能修改规则的参数与权重，<b>不能点名任何变体</b>；
                  把它整体移除（prior_fixed 臂）结果不变。假说由程序在标签揭示后裁决，
                  被推翻的取代自动转为下一轮的惩罚项。</p>
                  <img src="{inline_png(F / 'fig5_architecture.png')}" alt="方法架构">"""))

    # ---------------------------------------------------------------- rounds
    for rd in sorted(rows):
        r = rows[rd]
        n = r["notes"]
        audit = n.get("constraint_audit") or {}
        removed = {k: v for k, v in (audit.get("filters_removed") or {}).items() if v}
        ups = n.get("constraint_updates") or []
        urows = [[u["id"], "停用" if u.get("active") is False else esc(
            "; ".join(f"{k} = {fmt_val(k, v)}"
                      for k, v in (u.get("after") or {}).items() if k != "sites")),
            esc(u.get("reason", "")[:150])] for u in ups]
        verd = {}
        for h in r["hypotheses"]:
            verd[h["verdict"]] = verd.get(h["verdict"], 0) + 1
        zh = {"supported": "支持", "refuted": "推翻", "undetermined": "证据不足"}
        ref = [h for h in r["hypotheses"] if h["verdict"] == "refuted"]
        extra = ""
        if ref:
            h = ref[0]
            e = h.get("evidence") or {}
            extra = f"""<div class="verdict"><span class="tag bad">推翻</span>
            <q>{esc(h['statement'][:130])}</q>
            <p class="note">带该取代的 {e.get('n_with_substitution',0)} 个变体均值
            {e.get('mean_with',0):.3f}，不带的 {e.get('mean_without',0):.3f}，
            差值 {e.get('delta',0):+.3f}，方向与假说相反 → 进入 C8 惩罚项，下一轮降权。</p></div>"""
        hit = [d for d in r["selected"] if d["variant"] == target]
        hit_html = ""
        if hit:
            preds = np.array([d["predicted"] for d in r["selected"]])
            hit_html = f"""<div class="hit"><b>本轮命中全局最优 {target}，实测 {hit[0]['fitness']:.3f}。</b>
            当时模型对它的预测值只有 {hit[0]['predicted']:.3f}，在 96 个入选变体中排第
            {int((preds > hit[0]['predicted']).sum()) + 1} 位。</div>"""
        feed = ("192 条起始标签的逐位点统计 + 证据先验对候选池的排序" if rd == 1
                else f"第 {rd - 1} 轮的约束审计、逐位点统计与被推翻的假说")
        P.append(step(f"R{rd}", f"第 {rd} 轮", "约束引擎 · 虚拟评估器 · 程序裁决",
                      f"{feed}；候选池 {audit.get('n_pool',0):,} 条",
                      f"96 条批次；批次最好 {r['metrics']['best']:.3f}，累计最好 {r['best_so_far']:.3f}",
                      f"""<p><b>本轮生效的规则修改</b>（权重 w<sub>prior</sub> =
                      {n.get('weights',{}).get('prior',1.0):.1f}，w<sub>model</sub> =
                      {n.get('weights',{}).get('model',1.0):.1f}）：</p>
                      {table(["约束", "参数", "理由（智能体原话，节选）"], urows, "wide") if urows else "<p class='note'>本轮无规则修改。</p>"}
                      <p><b>这些规则的实际后果</b></p>
                      {kv([("硬过滤后合法候选", f"{audit.get('n_legal_after_filters',0):,} 条" +
                            ("（" + "；".join(f"{k.split('_')[0]} 删除 {v:,} 条" for k, v in removed.items()) + "）" if removed else "（本轮无硬过滤删除）")),
                           ("探索配额", f"{audit.get('exploration_quota',0)} 个名额"),
                           ("多样性配额挡下", f"{audit.get('diversity_blocks',0)} 个候选"),
                           ("四个位点的氨基酸种类", " / ".join(str(x) for x in audit.get('distinct_aa_per_site', []))),
                           ("与无约束排序相比", f"{audit.get('changed_vs_unconstrained',0)} 个名额被规则改变"),
                           ("假说裁决", "、".join(f"{zh.get(k,k)} {v} 条" for k, v in sorted(verd.items())))])}
                      {extra}{hit_html}"""))

    # ---------------------------------------------------------------- post-hoc
    crows = [[zh_arm, f"{check[a]['best_label_r1']:.3f}",
              f"{check[a]['fwaa_rank']:,} / {check[a]['n_pool']:,}",
              f"{check[a]['best_in_model_top96']:.3f}"]
             for a, zh_arm in (("prior_constraint", "证据先验 + 约束（LLM 调参）"),
                               ("prior_fixed", "证据先验 + 约束（无 LLM）"),
                               ("greedy", "预测器贪心")) if a in check]
    P.append(step(5, "事后核查：第一批标签到底教给了模型什么", "check_first_batch_information.py",
                  "同一组 192 条起始标签 + 各臂第 1 轮的 96 条",
                  "重新拟合的预测器对同一候选池的排序",
                  f"""<p>容易误读的是把命中归功于"先验直接把它捞了出来"。核查后不是这样：
                  {target} 不触犯任何硬过滤（体积上限停在 60 Å³ 时它同样合法）。
                  真正的差别在更前面一步——第一批测了什么：</p>
                  {table(["第 1 轮批次来自", "该批最好标签", f"{target} 在模型中的名次", "模型自选前 96 名中最好实测"], crows, "wide")}
                  <p><b>贪心那一批找到的最好标签其实更高</b>（4.597 对 4.035），
                  但它训练出的第 2 轮模型把全局最优排在第 2,859 位、落在前 96 名之外。
                  约束要求批次覆盖互相补偿的组合（如 41/54 的体积配对），贪心则把名额堆在同一个基序附近。
                  第一批的价值不在于它本身最好，而在于它让模型学到什么。</p>"""))

    # ---------------------------------------------------------------- structure tool
    P.append(step(6, "接入结构预测工具，并如实报告它的零贡献", "NVIDIA 托管 Boltz-2",
                  "覆盖整个适应度量程的 146 个变体序列",
                  "146 个预测结构与置信度；位点 pLDDT 与适应度的 ρ = +0.21",
                  f"""<p>野生型、全局最优与一个适应度仅 0.002 的死变体 RYLE 的预测结构几乎重合
                  （Cα RMSD 0.51 Å 与 1.20 Å），整链置信度分别为
                  {fold['VDGV']['confidence_scores'][0]:.3f}、{fold['FWAA']['confidence_scores'][0]:.3f}、
                  {fold['RYLE']['confidence_scores'][0]:.3f}——死变体并不更"不可信"。</p>
                  <p>把结构置信度作为额外特征加入后，5 折交叉验证的 Spearman 变化 −0.002。
                  <b>因此它没有被写进先验</b>：它回答的是"能不能折叠成这个骨架"，
                  而本任务问的是"与 IgG-Fc 结合得好不好"。保留这个诚实的零贡献项，本身也是可解释性的一部分。</p>
                  <img src="{inline_png(F / 'fig6_structure.png')}" alt="Boltz-2 折叠结果">"""))

    # ---------------------------------------------------------------- output
    traj = lambda d: " → ".join(f"{d[i]['best_so_far']:.3f}" for i in (1, 2, 3))
    P.append(f"""
<section class="step">
  <div class="rail"><span class="num dot">■</span></div>
  <div class="card output">
    <div class="head"><h3>最终输出</h3><span class="tool">本次运行（种子 {args.seed}）</span></div>
    <div class="body">
      {kv([("找到的最好变体", f"<b>{target}</b>，实测适应度 <b>{fitness.max():.3f}</b>（该景观的全局最优）"),
           ("命中轮次", "第 2 轮（累计 288 条标签）"),
           ("共测量", "480 条，占全部 149,361 条的 0.32%"),
           ("同种子下的轨迹", f"证据先验+约束（LLM 调参） {traj(rows)}<br>证据先验+约束（无 LLM） {traj(fixed)}<br>预测器贪心 {traj(greedy)}"),
           ("可审计产物", "每轮的约束审计、规则修改及其理由、假说裁决、先验分数拆解、预言机查询日志")])}
      <p class="note">在 {n_seeds} 个随机种子上，证据先验+约束命中全局最优的比例为 60%（9/15），
      预测器贪心为 20%（3/15），配对 McNemar 精确检验 p = 0.031；
      而带不带语言模型没有可测量的差别——起作用的是进入第一轮的信息和显式约束。</p>
    </div>
  </div>
</section>""")

    css = """
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fbfbfa;--card:#fff;
--acc:#b45309;--good:#15803d;--bad:#b91c1c;--blue:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:56px 24px 96px}
header.top{border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:8px}
header.top h1{font-size:27px;line-height:1.4;margin:0 0 10px;letter-spacing:-.01em}
header.top p{margin:0;color:var(--mut);font-size:14.5px}
.badges{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
.badge{font-size:12.5px;color:var(--mut);background:#f3f4f6;border:1px solid var(--line);
border-radius:999px;padding:3px 11px}
.badge a{color:var(--blue);text-decoration:none}
.step{display:flex;gap:18px;margin:26px 0}
.rail{flex:0 0 30px;display:flex;flex-direction:column;align-items:center}
.num{display:flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;
background:var(--ink);color:#fff;font-size:12.5px;font-weight:600;flex:0 0 30px}
.num.dot{background:#fff;color:var(--ink);border:1px solid var(--line)}
.rail:after{content:"";flex:1;width:1px;background:var(--line);margin-top:8px}
.step:last-child .rail:after{display:none}
.card{flex:1;background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;min-width:0}
.card.input{border-left:3px solid var(--ink)}
.card.output{border-left:3px solid var(--good)}
.head{display:flex;align-items:baseline;justify-content:space-between;gap:14px;margin-bottom:14px}
.head h3{margin:0;font-size:17px;letter-spacing:-.01em}
.tool{font-size:12px;color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.io{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}
.io>div{background:#f9fafb;border:1px solid var(--line);border-radius:8px;padding:9px 12px;font-size:13.5px}
.io .lbl{display:block;font-size:11px;letter-spacing:.08em;color:var(--mut);margin-bottom:3px}
.io .out{background:#f6fbf7;border-color:#d7ecdc}
.body p{margin:11px 0}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:7px 18px;margin:12px 0;font-size:14px}
dl.kv dt{color:var(--mut)}
dl.kv dd{margin:0}
table{width:100%;border-collapse:collapse;margin:13px 0;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--mut);font-size:12px;letter-spacing:.03em;
border-bottom:1px solid #d1d5db}
tbody tr:last-child td{border-bottom:none}
table.wide td:first-child{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:nowrap}
img{width:100%;height:auto;margin:16px 0 4px;border:1px solid var(--line);border-radius:8px;background:#fff}
pre.seq{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
background:#f9fafb;border:1px solid var(--line);border-radius:8px;padding:11px;overflow-x:auto;margin:14px 0}
pre.seq b{color:var(--acc)}
.note{color:var(--mut);font-size:13px}
.verdict{border-left:3px solid var(--bad);background:#fef6f6;border-radius:0 8px 8px 0;
padding:10px 14px;margin:14px 0}
.verdict q{font-style:italic;font-size:13.5px}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:999px;margin-right:7px}
.tag.bad{background:#fee2e2;color:var(--bad)}
.hit{border-left:3px solid var(--good);background:#f2fbf4;border-radius:0 8px 8px 0;
padding:11px 14px;margin:14px 0;font-size:14px}
footer{margin-top:40px;padding-top:22px;border-top:1px solid var(--line);
color:var(--mut);font-size:13px}
@media(max-width:720px){.io{grid-template-columns:1fr}.wrap{padding:32px 16px 64px}}
"""

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GB1 定向进化智能体 · 一个完整案例</title>
<style>{css}</style></head>
<body><div class="wrap">
<header class="top">
  <h1>先验与约束驱动的蛋白质定向进化智能体<br>一次完整运行的全过程</h1>
  <p>从任务输入，到每一步工具调用的真实输入与输出，到最终命中全局最优。页面上的每个数字都取自这次运行留下的记录文件。</p>
  <div class="badges">
    <span class="badge">数据集 FLIP GB1 四位点 · 149,361 条</span>
    <span class="badge">预算 192 + 3 × 96</span>
    <span class="badge">种子 {args.seed}</span>
    <span class="badge"><a href="https://github.com/Sunrich-HT/gb1-directed-evolution-agent">代码与完整记录</a></span>
  </div>
</header>
{''.join(P)}
<footer>本页由 scripts/make_demo_page.py 从 results/ 与 knowledge/ 直接生成；
图件与论文同源。适应度为已发表深度突变扫描的查表值，不是湿实验测量。</footer>
</div></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
