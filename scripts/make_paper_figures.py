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
    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.42,
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
    ax = fig.add_subplot(gs[1, 0])
    show = [("reference_random", "random"),
            ("only_mpnn", "inverse folding only"),
            ("only_literature", "literature only"),
            ("only_packing_headroom", "packing headroom only"),
            ("evidence_prior", "full evidence prior"),
            ("leakage_measured_single_mutants", "measured single mutants*")]
    vals = [prior_eval["priors"][k]["spearman_hd3plus"] for k, _ in show]
    colors = ["#9e9e9e", "#4c72b0", "#55a868", "#8172b3", "#c44e52", "#333333"]
    ax.barh(range(len(show)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([lab for _, lab in show])
    ax.invert_yaxis()
    ax.axvline(prior_eval["priors"]["leakage_measured_single_mutants"]["spearman_hd3plus"],
               color="#333333", lw=0.7, ls=":")
    ax.set_xlabel("Spearman ρ against measured fitness (≥3 mutations)")
    ax.set_title("Evidence alone matches the labelled upper bound")
    ax.margins(x=0.12)
    panel_letter(ax, "c")

    # (d) operational metric
    ax = fig.add_subplot(gs[1, 1])
    vals = [prior_eval["priors"][k]["best_in_top96"] for k, _ in show]
    ax.barh(range(len(show)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    opt = prior_eval["global_optimum"]["fitness"]
    ax.axvline(opt, color="#c44e52", lw=0.8, ls="--")
    ax.text(opt, -0.75, f"global optimum {opt:.2f}", color="#c44e52", fontsize=5.5, ha="right")
    ax.set_xlabel("best measured fitness inside the prior's top 96")
    ax.set_title("What one batch of 96 would have caught")
    ax.margins(x=0.12, y=0.12)
    panel_letter(ax, "d")

    stamp_letters(fig)
    return save(fig, out, "fig1_prior")


# ----------------------------------------------------------------- figure 2
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

    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = fig.add_gridspec(2, 2, hspace=1.30, wspace=0.26,
                          left=0.09, right=0.97, top=0.94, bottom=0.27)

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
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=5.5,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.4, handlelength=1.5)

    # (b) per-seed endpoint
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        finals = curves[arm][:, -1]
        ax.scatter(np.full(len(finals), i) + rng.uniform(-0.12, 0.12, len(finals)), finals,
                   s=14, color=ARM_COLOR[arm], alpha=0.85, edgecolor="none", zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.median(finals)] * 2, color="#333333", lw=1.2, zorder=4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], rotation=90, ha="center", fontsize=5.5)
    ax.set_ylabel("final best fitness")
    ax.set_title("Every seed, with the median marked")
    ax.margins(y=0.12)
    panel_letter(ax, "b")

    # (c) per-round gain
    ax = fig.add_subplot(gs[1, 0])
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
    panel_letter(ax, "c")

    # (d) hit rate
    ax = fig.add_subplot(gs[1, 1])
    opt = max(r["best_so_far"] for r in records)
    hits = [float(np.mean(curves[a][:, -1] >= opt - 1e-6)) * 100 for a in arms]
    ax.bar(range(len(arms)), hits, color=[ARM_COLOR[a] for a in arms], width=0.62)
    for i, h in enumerate(hits):
        ax.text(i, h + 1.5, f"{h:.0f}%", ha="center", fontsize=6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], rotation=90, ha="center", fontsize=5.5)
    ax.set_ylabel("seeds reaching it (%)")
    ax.set_title(f"Hit rate of the global optimum ({opt:.2f})")
    ax.margins(y=0.18)
    panel_letter(ax, "d")

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

    made = [figure1(struct, prior, prior_eval, out),
            figure2(records, out),
            figure3(records, out),
            figure4(interp, df, struct, out)]
    print("\n".join(str(p) for p in made))


if __name__ == "__main__":
    main()
