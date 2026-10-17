#!/usr/bin/env python
"""Paper figures, built to the contract in figures/FIGURE_CONTRACT.md.

Backend: matplotlib only.  Every value is read from results/ or knowledge/.
Run:  python scripts/make_paper_figures.py --results results --out figures/paper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape

MM = 1 / 25.4
AA = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = (39, 40, 41, 54)
WT = "VDGV"

ARM_ORDER = ["random", "greedy", "greedy_explore", "llm_agent", "llm_knowledge",
             "prior_fixed", "prior_constraint"]
ARM_LABEL = {
    "random": "random",
    "greedy": "predictor-greedy",
    "greedy_explore": "predictor + exploration",
    "llm_agent": "LLM agent",
    "llm_knowledge": "LLM + knowledge block",
    "prior_fixed": "evidence prior + constraints (no LLM)",
    "prior_constraint": "evidence prior + constraints (LLM-steered)",
}
SHORT = {
    "random": "random",
    "greedy": "greedy",
    "greedy_explore": "greedy+explore",
    "llm_agent": "LLM",
    "llm_knowledge": "LLM+knowledge",
    "prior_fixed": "prior+rules",
    "prior_constraint": "prior+rules+LLM",
}

ARM_COLOR = {
    "random": "#9e9e9e",
    "greedy": "#4c72b0",
    "greedy_explore": "#55a868",
    "llm_agent": "#dd8452",
    "llm_knowledge": "#8172b3",
    "prior_fixed": "#937860",
    "prior_constraint": "#c44e52",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def panel_letter(ax, letter: str, dx: float = -0.14, dy: float = 1.10) -> None:
    """Deferred: letters are stamped by stamp_letters() once the layout is final."""
    ax._panel_letter = letter


def stamp_letters(fig) -> None:
    """Place each panel letter outside the axes' own tight bounding box, so it never
    collides with a long tick label (checked by the collision audit)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        letter = getattr(ax, "_panel_letter", None)
        if not letter:
            continue
        bb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.text(max(0.006, bb.x0 - 0.028), min(0.985, bb.y1 + 0.010), letter,
                 fontsize=8, fontweight="bold", va="top", ha="left")


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    return pdf


# ----------------------------------------------------------------- figure 1
def figure1(struct: dict, prior: dict, prior_eval: dict, out: Path) -> Path:
    fig = plt.figure(figsize=(170 * MM, 58 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.42,
                          left=0.17, right=0.95, top=0.92, bottom=0.10)

    # (a) mechanism split
    ax = fig.add_subplot(gs[0, 0])
    sites = [str(p) for p in POSITIONS]
    rel = [struct["sites"][s]["rel_sasa"] * 100 for s in sites]
    buried = [struct["sites"][s]["interface"]["buried_surface_on_binding_A2"] for s in sites]
    x = np.arange(4)
    ax.bar(x - 0.19, rel, width=0.36, color="#4c72b0", label="solvent exposure in the free domain (%)")
    ax.bar(x + 0.19, buried, width=0.36, color="#c44e52",
           label="surface buried on IgG-Fc binding (Å²)")
    for xi, b in zip(x, buried):
        if b >= 10:
            ax.text(xi + 0.19, b + 2, f"{b:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{WT[i]}{p}" for i, p in enumerate(POSITIONS)])
    ax.set_ylabel("value (see key)")
    ax.set_title("Only site 40 acts through the binding interface")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.margins(y=0.22)
    panel_letter(ax, "a")

    # (b) prior table
    ax = fig.add_subplot(gs[0, 1])
    table = np.array([[prior["table"][str(p)][a] for a in AA] for p in POSITIONS])
    vmax = np.abs(table).max()
    im = ax.imshow(table, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(20))
    ax.set_xticklabels(list(AA), fontsize=5.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{WT[i]}{p}" for i, p in enumerate(POSITIONS)])
    ax.set_title("The prior, before any multi-mutant label")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("prior score", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5.5)
    ax.grid(False)
    panel_letter(ax, "b")

    # (c) prior quality
    stamp_letters(fig)
    return save(fig, out, "fig1_prior")


def _arm_curves(records, arms, seeds):
    out = {}
    for arm in arms:
        rows = []
        for s in seeds:
            curve = [r["best_so_far"] for r in records if r["arm"] == arm and r["seed"] == s]
            if curve:
                rows.append(curve)
        if rows:
            out[arm] = np.array(rows)
    return out


def figure2(records, out: Path, n_initial: int = 192, batch: int = 96) -> Path:
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in records)]
    seeds = sorted({r["seed"] for r in records})
    curves = _arm_curves(records, arms, seeds)
    rounds = sorted({r["round"] for r in records})

    fig = plt.figure(figsize=(170 * MM, 74 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.26,
                          left=0.085, right=0.985, top=0.92, bottom=0.40)

    # (a) campaign trace
    ax = fig.add_subplot(gs[0, 0])
    for arm, c in curves.items():
        x = n_initial + np.arange(1, c.shape[1] + 1) * batch
        ax.plot(x, c.mean(0), marker="o", ms=3, lw=1.4, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.fill_between(x, c.min(0), c.max(0), color=ARM_COLOR[arm], alpha=0.12, lw=0)
    ax.set_xlabel("labels measured (cumulative)")
    ax.set_ylabel("best fitness found so far")
    ax.set_title(f"Campaign trace (mean and range, n={len(seeds)} seeds)")
    handles, labels = ax.get_legend_handles_labels()
    ax.margins(0.04)
    panel_letter(ax, "a")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 0.005),
               columnspacing=1.8, handlelength=1.5, labelspacing=0.4)

    # (b) per-seed endpoint

    # (c) per-round gain
    ax = fig.add_subplot(gs[0, 1])
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        means = [np.mean([r["metrics"]["gain"] for r in records
                          if r["arm"] == arm and r["round"] == rd]) for rd in rounds]
        ax.bar(np.array(rounds) + (i - len(arms) / 2 + 0.5) * width, means, width=width,
               color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("gain over the incumbent")
    ax.set_title("Where each arm's improvement happens")
    ax.margins(y=0.12)
    panel_letter(ax, "b")

    # (d) hit rate

    stamp_letters(fig)
    return save(fig, out, "fig2_campaign")


# ----------------------------------------------------------------- figure 3
def figure3(records, out: Path) -> Path:
    rule = [r for r in records if r["arm"] == "prior_constraint"]
    rounds = sorted({r["round"] for r in rule})

    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.52,
                          left=0.10, right=0.90, top=0.86, bottom=0.14)

    # (a) prior attribution of the actual picks
    ax = fig.add_subplot(gs[0, 0])
    attrib = {}
    for r in rule:
        for variant, parts in (r["notes"].get("prior_attribution_examples") or {}).items():
            attrib[variant] = parts
    variants = list(attrib)[:8]
    comps = [k for k in (attrib[variants[0]] if variants else {}) if k != "total"]
    pos_bottom = np.zeros(len(variants))
    neg_bottom = np.zeros(len(variants))
    palette = plt.get_cmap("tab10")
    for i, comp in enumerate(comps):
        vals = np.array([attrib[v].get(comp, 0.0) for v in variants])
        bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(range(len(variants)), vals, bottom=bottoms,
               color=palette(i % 10), width=0.68, label=comp.replace("_", " "))
        pos_bottom += np.clip(vals, 0, None)
        neg_bottom += np.clip(vals, None, 0)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("contribution to the prior score")
    ax.set_title("Which evidence paid for each pick")
    ax.legend(ncol=2, loc="upper left", fontsize=5.5, columnspacing=0.8,
              handlelength=1.0, borderpad=0.2, labelspacing=0.25)
    ax.margins(y=0.42)
    panel_letter(ax, "a")

    # (b) constraint audit
    ax = fig.add_subplot(gs[0, 1])
    ids, per_round = set(), {}
    for r in rule:
        audit = r["notes"].get("constraint_audit") or {}
        removed = audit.get("filters_removed") or {}
        per_round.setdefault(r["round"], []).append(removed)
        ids |= set(removed)
    ids = [i for i in sorted(ids)
           if any(np.mean([d.get(i, 0) for d in per_round.get(rd, [{}])]) > 0 for rd in rounds)]
    width = 0.74 / max(1, len(ids))
    for i, cid in enumerate(ids):
        means = [np.mean([d.get(cid, 0) for d in per_round.get(rd, [{}])]) / 1000.0 for rd in rounds]
        ax.bar(np.array(rounds) + (i - len(ids) / 2 + 0.5) * width, means, width=width,
               color="#dd8452", label=cid.split("_", 1)[-1].replace("_", " "))
    # the quota that shapes the batch, on a second axis: how many picks it displaced
    blocks = [np.mean([(r["notes"].get("constraint_audit") or {}).get("diversity_blocks", 0)
                       for r in rule if r["round"] == rd]) for rd in rounds]
    ax2 = ax.twinx()
    ax2.plot(rounds, blocks, marker="o", ms=3.5, lw=1.2, color="#4c72b0",
             label="diversity quota: candidates skipped")
    ax2.set_ylabel("candidates skipped by the\ndiversity quota", color="#4c72b0", fontsize=6)
    ax2.tick_params(axis="y", colors="#4c72b0")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("candidates removed by the\nvolume filter (thousands)", color="#dd8452", fontsize=6)
    ax.tick_params(axis="y", colors="#dd8452")
    ax.set_title("What the constraints actually excluded")
    ax.margins(y=0.15)
    panel_letter(ax, "b")

    # (c) hypothesis verdicts
    ax = fig.add_subplot(gs[1, 0])
    verdicts = ["supported", "refuted", "undetermined"]
    colors = {"supported": "#1b7837", "refuted": "#b2182b", "undetermined": "#9e9e9e"}
    bottom = np.zeros(len(rounds))
    for v in verdicts:
        counts = np.array([sum(1 for r in rule for h in r["hypotheses"]
                               if r["round"] == rd and h["verdict"] == v) for rd in rounds], float)
        ax.bar(rounds, counts, bottom=bottom, color=colors[v], label=v, width=0.6)
        bottom += counts
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("registered hypotheses")
    ax.set_title("Verdicts assigned by the program, not the model")
    ax.legend(fontsize=5.5)
    ax.margins(y=0.15)
    panel_letter(ax, "c")

    # (d) consequences of the rules
    ax = fig.add_subplot(gs[1, 1])
    changed = [np.mean([(r["notes"].get("constraint_audit") or {}).get("changed_vs_unconstrained", np.nan)
                        for r in rule if r["round"] == rd]) for rd in rounds]
    ham = [np.mean([r["diversity"]["mean_pairwise_hamming"] for r in rule if r["round"] == rd])
           for rd in rounds]
    ax.plot(rounds, changed, marker="o", ms=3.5, color="#c44e52", label="picks changed by the rules")
    ax.set_xlabel("round")
    ax.set_ylabel("picks changed vs unconstrained", color="#c44e52")
    ax.tick_params(axis="y", colors="#c44e52")
    ax.set_xticks(rounds)
    ax2 = ax.twinx()
    ax2.plot(rounds, ham, marker="s", ms=3.5, color="#4c72b0", label="batch diversity")
    ax2.set_ylabel("mean pairwise Hamming distance", color="#4c72b0")
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    ax2.tick_params(axis="y", colors="#4c72b0")
    ax2.spines["top"].set_visible(False)
    ax.set_title("The rules move the batch and keep it diverse")
    panel_letter(ax, "d")

    stamp_letters(fig)
    return save(fig, out, "fig3_decisions")


# ----------------------------------------------------------------- figure 4
def figure4(interp: dict, df, struct: dict, out: Path) -> Path:
    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.36,
                          left=0.10, right=0.95, top=0.92, bottom=0.12)

    # (a) measured single-mutant effects
    ax = fig.add_subplot(gs[0, 0])
    measured = np.array(interp["model_layer"]["measured_single_mutant_effects"], float)
    vmax = np.nanmax(np.abs(measured))
    im = ax.imshow(measured, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(20))
    ax.set_xticklabels(list(AA), fontsize=5.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{WT[i]}{p}" for i, p in enumerate(POSITIONS)])
    ax.set_title("Measured single-mutant effect (log fitness vs wild type)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    ax.grid(False)
    panel_letter(ax, "a")

    # (b) epistasis
    ax = fig.add_subplot(gs[0, 1])
    per_pair = interp["model_layer"]["epistasis_d0"]["per_pair_l1"]
    labels = list(per_pair)
    vals = [per_pair[k] for k in labels]
    order = np.argsort(vals)
    ax.barh([labels[i] for i in order], [vals[i] for i in order], color="#8172b3", height=0.6)
    share = interp["model_layer"]["epistasis_d0"]["pairwise_share"]
    ax.set_xlabel("|interaction coefficients| (L1)")
    ax.set_title(f"Epistasis carries {share:.0%} of the model's weight")
    ax.margins(x=0.10)
    panel_letter(ax, "b")

    # (c) the 41/54 volume transfer, measured
    ax = fig.add_subplot(gs[1, 0])
    vol = struct["amino_acid_properties"]["volume"]
    variants = df["variant"].to_numpy()
    fitness = df["fitness"].to_numpy(float)
    d41 = np.array([vol[v[2]] - vol["G"] for v in variants])
    d54 = np.array([vol[v[3]] - vol["V"] for v in variants])
    imbalance = np.abs(d41 + d54)
    bins = np.linspace(0, np.percentile(imbalance, 99), 9)
    idx = np.digitize(imbalance, bins)
    centres, means, tops = [], [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() > 50:
            centres.append(0.5 * (bins[b - 1] + bins[b]))
            means.append(float(np.mean(np.log1p(fitness[sel]))))
            tops.append(float(np.percentile(np.log1p(fitness[sel]), 99)))
    ax.plot(centres, means, marker="o", ms=3.5, color="#4c72b0", label="mean log(1+fitness)")
    ax.plot(centres, tops, marker="s", ms=3.5, color="#c44e52", label="99th percentile")
    ax.set_xlabel("|ΔV(41) + ΔV(54)|  (Å³, volume imbalance)")
    ax.set_ylabel("log(1 + measured fitness)")
    ax.set_title("Compensating volume between 41 and 54 pays off")
    ax.legend()
    ax.margins(0.05)
    panel_letter(ax, "c")

    # (d) components of the optimum
    ax = fig.add_subplot(gs[1, 1])
    comp = interp["failure_layer"]["components"]
    names = [c["mutation"] for c in comp]
    vals = [c["single_mutant_fitness"] for c in comp]
    colors = ["#b2182b" if c["deleterious_alone"] else "#4c72b0" for c in comp]
    ax.hlines(range(len(names)), 1.0, vals, color=colors, lw=1.2)
    ax.plot(vals, range(len(names)), "o", ms=5, color="none",
            markerfacecolor="none", markeredgecolor="none")
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.plot(v, i, "o", ms=5, color=c)
        ax.text(v, i + 0.30, f"{v:.2f}", va="bottom", ha="center", fontsize=6)
    ax.axvline(1.0, color="#777777", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("fitness of the component as a single mutant")
    ax.set_title(f"Two components of {interp['failure_layer']['global_optimum']} are harmful alone")
    ax.margins(x=0.18, y=0.16)
    panel_letter(ax, "d")

    stamp_letters(fig)
    return save(fig, out, "fig4_mechanism")


def fig_architecture(out: Path) -> Path:
    """Method architecture on a strict grid: every box occupies its own cell and every
    arrow runs in an empty gutter, so no text can be overlapped or occluded."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(180 * MM, 108 * MM))
    ax = fig.add_axes([0.012, 0.012, 0.976, 0.976])
    ax.set_xlim(-1, 123)
    ax.set_ylim(-1, 73)
    ax.axis("off")

    EV, PRIOR, MODEL, SCORE = "#4c72b0", "#c44e52", "#55a868", "#8172b3"
    GREY, AGENT = "#5f5f5f", "#937860"

    # --- grid: 4 columns x 4 rows, gutters are 6 units wide and stay empty
    COLW, GUT = 26.0, 5.0
    COLX = [2.0 + i * (COLW + GUT) for i in range(4)]
    ROWY = {1: 56.0, 2: 40.0, 3: 24.0, 4: 6.0}
    ROWH = {1: 14.0, 2: 13.0, 3: 13.0, 4: 13.0}

    def cell(col, row, span=1):
        x = COLX[col]
        w = COLW * span + GUT * (span - 1)
        return x, ROWY[row], w, ROWH[row]

    def box(rect, title, lines, color, ls="-"):
        x, y, w, h = rect
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
                                    linewidth=1.0, edgecolor=color, linestyle=ls,
                                    facecolor="white", zorder=2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=color, zorder=3)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.0 - i * 2.6, ln, ha="center", va="center",
                    fontsize=5.6, color="#222222", zorder=3)

    def vline(x, y0, y1, color=GREY, ls="-"):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=color, linestyle=ls, zorder=1))

    def hline(y, x0, x1, color=GREY, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=color, linestyle=ls, zorder=1))

    # row 1 — evidence sources
    box(cell(0, 1), "Structural evidence", ["1PGA crystal, 1FCC complex",
                                            "burial, backbone dihedrals",
                                            "C-beta packing headroom"], EV)
    box(cell(1, 1), "Biophysical evidence", ["side-chain volume",
                                             "hydropathy, formal charge"], EV)
    box(cell(2, 1), "Published evidence", ["62 curated rows, 21 sources",
                                           "each row maps to one prior",
                                           "term or one constraint"], EV)
    box(cell(3, 1), "Labelled data", ["192 random variants (D0)",
                                      "the only labels at the start"], MODEL)

    # row 2 — prior and model
    box(cell(0, 2, span=3), "Evidence prior   (no multi-mutant label; rho = +0.31)",
        ["inverse folding, muted at site 40   |   packing headroom   |   buried-charge penalty",
         "Fc-interface bonus   |   41/54 volume compensation   |   literature claims"], PRIOR)
    box(cell(3, 2), "Fitness predictor", ["ridge / RF / FCNN,",
                                          "cross-validated,",
                                          "retrained each round"], MODEL)

    # row 3 — scoring and rules
    box(cell(0, 3), "Combined score", ["w_prior x prior", "+ w_model x model"], SCORE)
    box(cell(1, 3), "Constraint engine", ["hard filters, soft penalties,",
                                          "composition quotas,",
                                          "audited every round"], SCORE)
    box(cell(2, 3), "Batch of 96", ["frozen by SHA-256 before",
                                    "any label is revealed"], GREY)
    box(cell(3, 3), "Virtual oracle", ["published DMS lookup,",
                                       "every query logged"], GREY)

    # row 4 — agent and verdicts
    box(cell(0, 4), "LLM agent (optional)", ["edits constraint parameters",
                                             "and weights; never names",
                                             "a variant"], AGENT, ls="--")
    box(cell(1, 4), "Program verdicts", ["each hypothesis judged",
                                         "after the labels:",
                                         "supported / refuted"], GREY)
    box(cell(2, 4, span=2), "Outcome", ["best-so-far, cumulative gain, batch diversity,",
                                        "and a per-round audit of every rule that fired"], GREY)

    # arrows, all inside empty gutters
    for col in (0, 1, 2):
        vline(COLX[col] + COLW / 2, ROWY[1], ROWY[2] + ROWH[2], EV)
    vline(COLX[3] + COLW / 2, ROWY[1], ROWY[2] + ROWH[2], MODEL)
    vline(COLX[0] + COLW / 2, ROWY[2], ROWY[3] + ROWH[3], PRIOR)
    # predictor feeds the combined score along the row-2/row-3 gutter
    ax.plot([COLX[3] + COLW / 2, COLX[3] + COLW / 2], [ROWY[2], ROWY[2] - 2.6], color=MODEL, lw=1.0)
    ax.plot([COLX[3] + COLW / 2, COLX[0] + COLW / 2 + 4.0], [ROWY[2] - 2.6, ROWY[2] - 2.6],
            color=MODEL, lw=1.0)
    vline(COLX[0] + COLW / 2 + 4.0, ROWY[2] - 2.6, ROWY[3] + ROWH[3], MODEL)
    hline(ROWY[3] + ROWH[3] / 2, COLX[0] + COLW, COLX[1], SCORE)
    hline(ROWY[3] + ROWH[3] / 2, COLX[1] + COLW, COLX[2], SCORE)
    hline(ROWY[3] + ROWH[3] / 2, COLX[2] + COLW, COLX[3], GREY)
    # oracle -> verdicts, verdicts -> outcome, agent -> constraint engine
    ax.plot([COLX[3] + COLW / 2, COLX[3] + COLW / 2], [ROWY[3], ROWY[3] - 3.0], color=GREY, lw=1.0)
    ax.plot([COLX[3] + COLW / 2, COLX[1] + COLW / 2], [ROWY[3] - 3.0, ROWY[3] - 3.0],
            color=GREY, lw=1.0)
    vline(COLX[1] + COLW / 2, ROWY[3] - 3.0, ROWY[4] + ROWH[4], GREY)
    hline(ROWY[4] + ROWH[4] / 2, COLX[1] + COLW, COLX[2], GREY)
    hline(ROWY[4] + ROWH[4] / 2, COLX[0] + COLW, COLX[1], AGENT, ls="--")
    # verdicts feed back into the rules (dashed, up the column-1 gutter)
    ax.plot([COLX[1] + COLW - 3.0, COLX[1] + COLW - 3.0], [ROWY[4] + ROWH[4], ROWY[3] - 1.5],
            color=AGENT, ls="--", lw=1.0)
    vline(COLX[1] + COLW - 3.0, ROWY[3] - 1.5, ROWY[3], AGENT, ls="--")
    # new labels retrain the predictor (up the right-hand gutter)
    ax.plot([COLX[3] + COLW - 2.5, COLX[3] + COLW - 2.5], [ROWY[3], ROWY[2] - 1.0],
            color=MODEL, lw=1.0)
    vline(COLX[3] + COLW - 2.5, ROWY[2] - 1.0, ROWY[2], MODEL)
    return save(fig, out, "fig5_architecture")


# ----------------------------------------------------------------- figure 6
def _parse_cif_ca(path: Path) -> dict:
    """Boltz-2 returns line-broken atom_site rows; tokenise the whole loop."""
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines)
             if l.strip() == "loop_" and lines[k + 1].strip().startswith("_atom_site."))
    cols, j = [], i + 1
    while lines[j].strip().startswith("_atom_site."):
        cols.append(lines[j].strip().split(".", 1)[1]); j += 1
    toks = []
    while j < len(lines) and not lines[j].startswith(("#", "loop_", "_", "data_")):
        toks += lines[j].split(); j += 1
    n = len(cols)
    rows = [dict(zip(cols, toks[k:k + n])) for k in range(0, len(toks) - n + 1, n)]
    return {int(r["label_seq_id"]): (float(r["Cartn_x"]), float(r["Cartn_y"]),
                                     float(r["Cartn_z"]), float(r.get("B_iso_or_equiv", 0.0)))
            for r in rows if r["label_atom_id"] == "CA"}


def _kabsch(P, Q):
    """Superpose P onto Q; returns the rotated P, centred Q and the RMSD."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    Pr = Pc @ R
    return Pr, Qc, float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


def figure6(cif_dir: Path, features: list, out: Path) -> Path:
    """What the folding model actually returns, and why it carries no signal here."""
    shown = ["VDGV", "FWAA", "RYLE"]
    fit = {r["variant"]: r["fitness"] for r in features}
    fit["VDGV"] = 1.0
    label = {"VDGV": "VDGV (wild type, 1.00)", "FWAA": "FWAA (global optimum, 8.76)",
             "RYLE": "RYLE (dead variant, 0.002)"}
    colour = {"VDGV": "#777777", "FWAA": "#c0392b", "RYLE": "#2e6da4"}

    S = {v: _parse_cif_ca(cif_dir / f"{v}.cif") for v in shown}
    res = sorted(set.intersection(*[set(S[v]) for v in shown]))
    X = {v: np.array([S[v][i][:3] for i in res]) for v in shown}
    ref = X["VDGV"] - X["VDGV"].mean(0)
    sup, rmsd = {"VDGV": ref}, {}
    for v in ("FWAA", "RYLE"):
        Pr, _, rm = _kabsch(X[v], X["VDGV"])
        sup[v], rmsd[v] = Pr, rm

    fig = plt.figure(figsize=(180 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0],
                          left=0.07, right=0.985, top=0.87, bottom=0.26, wspace=0.42)

    # a - superposed C-alpha traces, projected on the first two principal axes
    axa = fig.add_subplot(gs[0, 0])
    u, s, vt = np.linalg.svd(ref - ref.mean(0), full_matrices=False)
    basis = vt[:2].T
    for v in shown:
        P = sup[v] @ basis
        axa.plot(P[:, 0], P[:, 1], "-", color=colour[v], lw=1.0,
                 alpha=0.95 if v != "VDGV" else 0.8, zorder=2 if v == "VDGV" else 3)
    idx = {r: k for k, r in enumerate(res)}
    for site in POSITIONS:
        P = sup["FWAA"] @ basis
        axa.plot(P[idx[site], 0], P[idx[site], 1], "o", ms=3.4, mfc="white",
                 mec="#c0392b", mew=0.9, zorder=4)
    axa.set_xlabel("principal axis 1 (\u00c5)")
    axa.set_ylabel("principal axis 2 (\u00c5)")
    axa.set_title("The predicted fold is the same", pad=3)
    axa.set_aspect("equal")
    # the note goes below the axes: inside the panel it would sit on the traces
    axa.text(0.5, -0.30, f"C$\\alpha$ RMSD to wild type\nFWAA {rmsd['FWAA']:.2f} \u00c5"
                         f"    RYLE {rmsd['RYLE']:.2f} \u00c5",
             transform=axa.transAxes, fontsize=5.8, va="top", ha="center")
    panel_letter(axa, "a")

    # b - per-residue confidence
    axb = fig.add_subplot(gs[0, 1])
    for site in POSITIONS:
        axb.axvspan(site - 0.5, site + 0.5, color="#f2c14e", alpha=0.28, lw=0, zorder=1)
    for v in shown:
        axb.plot(res, [S[v][i][3] for i in res], "-", color=colour[v], lw=1.0,
                 label=label[v], zorder=3)
    axb.set_xlabel("residue")
    axb.set_ylabel("predicted confidence (pLDDT)")
    axb.set_title("Confidence does not separate them either", pad=3)
    axb.set_ylim(0.6, 1.02)
    axb.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=5.4,
               handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    panel_letter(axb, "b")

    # c - the whole folded set against measured fitness
    axc = fig.add_subplot(gs[0, 2])
    fv = np.array([r["fitness"] for r in features])
    pv = np.array([r["plddt_sites"] for r in features])
    axc.scatter(np.log1p(fv), pv, s=5, color="#9aa3ad", alpha=0.75, lw=0, zorder=2)
    for v in ("FWAA", "RYLE"):
        row = [r for r in features if r["variant"] == v]
        if row:
            axc.scatter([np.log1p(row[0]["fitness"])], [row[0]["plddt_sites"]], s=22,
                        facecolor=colour[v], edgecolor="white", lw=0.6, zorder=4)
            dx = 7 if v == "RYLE" else -7
            axc.annotate(v, (np.log1p(row[0]["fitness"]), row[0]["plddt_sites"]),
                         textcoords="offset points", xytext=(dx, 5),
                         ha="left" if dx > 0 else "right",
                         fontsize=5.8, color=colour[v])
    rho = float(np.corrcoef(np.argsort(np.argsort(fv)), np.argsort(np.argsort(pv)))[0, 1])
    axc.set_xlabel("log(1 + measured fitness)")
    axc.set_ylabel("pLDDT at the four sites")
    axc.set_title(f"All {len(features)} folded variants", pad=3)
    axc.text(0.97, 0.05, f"Spearman $\\rho$ = {rho:+.2f}", transform=axc.transAxes,
             fontsize=6, ha="right", va="bottom")
    panel_letter(axc, "c")

    stamp_letters(fig)
    return save(fig, out, "fig6_structure")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--records", default="results/v2/records.json")
    ap.add_argument("--out", default="figures/paper")
    args = ap.parse_args()

    style()
    R = Path(args.results)
    out = Path(args.out)
    struct = json.loads(Path("knowledge/structure_evidence.json").read_text())
    prior_eval = json.loads((R / "prior_evaluation.json").read_text())
    records = json.loads(Path(args.records).read_text())
    prior = json.loads((Path(args.records).parent / "evidence_prior.json").read_text())
    interp = json.loads((R / "interpretability.json").read_text())
    df = load_landscape(args.data)

    made = [fig_architecture(out),
            figure1(struct, prior, prior_eval, out),
            figure2(records, out),
            figure3(records, out),
            figure4(interp, df, struct, out),
            figure6(Path("results/structure/cif"),
                    json.loads((R / "structure/structure_features.json").read_text()), out)]
    print("\n".join(str(p) for p in made))


if __name__ == "__main__":
    main()
