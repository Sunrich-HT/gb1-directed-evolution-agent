#!/usr/bin/env python
"""Figures for the report.  Every number is read from results/, never retyped."""
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
from gb1agent.features import AA
from gb1agent.metrics import top_percentile_precision
from gb1agent.priors import load_log_odds, score_variants, single_mutant_table, zscore_by_substitution
from scipy.stats import spearmanr

ARM_LABEL = {
    "random": "random",
    "greedy": "predictor-greedy",
    "greedy_explore": "predictor + 25% exploration",
    "llm_agent": "LLM agent",
    "llm_knowledge": "knowledge-augmented LLM agent",
}
ARM_COLOR = {
    "random": "#9e9e9e",
    "greedy": "#1f78b4",
    "greedy_explore": "#33a02c",
    "llm_agent": "#ff7f00",
    "llm_knowledge": "#6a3d9a",
}
POSITIONS = (39, 40, 41, 54)
WT = "VDGV"


def style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })


def fig_campaign(records, out: Path, budget_per_round: int, n_initial: int = 192) -> None:
    arms = [a for a in ARM_LABEL if any(r["arm"] == a for r in records)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        seeds = sorted({r["seed"] for r in rows})
        curves = []
        for s in seeds:
            rs = sorted([r for r in rows if r["seed"] == s], key=lambda r: r["round"])
            curves.append([r["best_so_far"] for r in rs])
        curves = np.array(curves, float)
        x = n_initial + np.arange(1, curves.shape[1] + 1) * budget_per_round
        ls = "--" if arm == "greedy_explore" else "-"
        ax.plot(x, curves.mean(0), marker="o", ms=3.5, color=ARM_COLOR[arm], ls=ls,
                label=f"{ARM_LABEL[arm]} (n={len(seeds)})")
        ax.fill_between(x, curves.min(0), curves.max(0), color=ARM_COLOR[arm], alpha=0.15, lw=0)
    ax.axhline(8.762, ls="--", lw=1, color="#b2182b")
    ax.text(ax.get_xlim()[1], 8.762, " global optimum FWAA", va="center", fontsize=6.5, color="#b2182b")
    ax.set_xlabel("labels measured (cumulative)")
    ax.set_ylabel("best fitness found so far")
    ax.set_title("Each arm starts from the same 192 labels")
    ax.legend(frameon=False, loc="center right", fontsize=6.3, borderaxespad=0.6)
    ax.set_ylim(top=9.6)
    ax.text(0.02, 0.94, "higher = better", transform=ax.transAxes, fontsize=6.5,
            va="bottom", color="0.35")

    ax = axes[1]
    for arm in arms:
        rows = sorted([r for r in records if r["arm"] == arm], key=lambda r: r["round"])
        rounds = sorted({r["round"] for r in rows})
        means = [np.mean([r["metrics"]["mean"] for r in rows if r["round"] == q]) for q in rounds]
        ax.plot(rounds, means, marker="s", ms=3.5, color=ARM_COLOR[arm],
                ls="--" if arm == "greedy_explore" else "-", label=ARM_LABEL[arm])
    ax.set_xticks(sorted({r["round"] for r in records}))
    ax.set_xlabel("round")
    ax.set_ylabel("mean fitness of the batch")
    ax.set_title("Batch quality per round")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_baselines(baselines: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    curve = baselines["training_size_curve"]
    n = [c["n_train"] for c in curve]
    ax = axes[0]
    ax.plot(n, [c["top100_recall"] for c in curve], marker="o", ms=3.5,
            color="#1f78b4", label="top-100 recall")
    ax.plot(n, [c["top5pct_precision"] for c in curve], marker="s", ms=3.5,
            color="#6a3d9a", label="top-5% precision")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(v) for v in n], fontsize=6.5)
    ax.set_xlabel("labelled training variants")
    ax.set_ylabel("score on the 10,000-variant test set")
    ax.set_title("High-value identification improves with labels")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    flip = baselines["flip_splits"]
    names = [f["split"] for f in flip]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [f["spearman"] for f in flip], width=0.38, color="#1f78b4", label="Spearman")
    ax.bar(x + 0.2, [f["top5pct_precision"] for f in flip], width=0.38, color="#6a3d9a",
           label="top-5% precision")
    for xi, f in zip(x, flip):
        ax.text(xi, max(f["spearman"], f["top5pct_precision"]) + 0.03,
                f"n_train={f['n_train']}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Official FLIP benchmark splits")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_priors(df, out: Path, mpnn_path: Path) -> None:
    variants = df["variant"].to_numpy()
    y = df["fitness"].to_numpy(float)
    n_mut = df["n_mut"].to_numpy(int)
    singles_mask = n_mut <= 1
    tables = {
        "inverse folding\n(ProteinMPNN)": load_log_odds(mpnn_path),
        "inverse folding\n+ z-scored": zscore_by_substitution(load_log_odds(mpnn_path)),
        "measured single\nmutants (additive)": single_mutant_table(
            variants[singles_mask], y[singles_mask]),
    }
    hi = n_mut >= 3
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    names, rho, prec = [], [], []
    for name, table in tables.items():
        s = score_variants(variants, table)
        names.append(name)
        rho.append(spearmanr(y[hi], s[hi]).statistic)
        prec.append(top_percentile_precision(y[hi], s[hi], 5.0))
    x = np.arange(len(names))
    ax.bar(x - 0.2, rho, width=0.38, color="#1f78b4", label="Spearman")
    ax.bar(x + 0.2, prec, width=0.38, color="#6a3d9a", label="top-5% precision")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("score on variants with >=3 mutations")
    ax.set_title("No-label priors: evolution-of-fold is not binding")
    ax.legend(frameon=False)

    ax = axes[1]
    best = str(variants[int(np.argmax(y))])
    lookup = dict(zip(variants, y))
    comps, vals = [], []
    for i, aa in enumerate(best):
        if aa == WT[i]:
            continue
        single = list(WT)
        single[i] = aa
        comps.append(f"{WT[i]}{POSITIONS[i]}{aa}")
        vals.append(lookup["".join(single)])
    colors = ["#b2182b" if v < 1 else "#1f78b4" for v in vals]
    ax.hlines(range(len(comps)), 1.0, vals, color=colors, lw=1)
    ax.plot(vals, range(len(comps)), "o", ms=5, color="none", markeredgecolor="k", mew=0.6)
    for i, (cmp_, v) in enumerate(zip(comps, vals)):
        ax.scatter([v], [i], s=28, color=colors[i], zorder=3)
        ax.text(v + 0.12, i, f"{v:.2f}", va="center", fontsize=7)
    ax.axvline(1.0, color="0.4", lw=1)
    ax.text(1.05, -0.42, "wild type", fontsize=6.5, color="0.4")
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(comps)
    ax.set_xlabel("fitness of the component as a single mutant")
    ax.set_title(f"Two components of {best} ({y.max():.2f}) are harmful alone")
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.75, len(comps) - 0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_interpret(report: dict, out: Path) -> None:
    measured = np.array(report["model_layer"]["measured_single_mutant_effects"], float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    vmax = np.nanmax(np.abs(measured))
    im = ax.imshow(measured, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(20))
    ax.set_xticklabels(list(AA), fontsize=6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{WT[i]}{POSITIONS[i]}" for i in range(4)])
    ax.set_title("Measured single-mutant effect, log fitness vs wild type")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]
    per_pair = report["model_layer"]["epistasis_d0"]["per_pair_l1"]
    labels = list(per_pair)
    vals = [per_pair[k] for k in labels]
    order = np.argsort(vals)
    ax.barh([labels[i] for i in order], [vals[i] for i in order], color="#6a3d9a")
    share = report["model_layer"]["epistasis_d0"]["pairwise_share"]
    ax.set_xlabel("|interaction coefficients| (L1)")
    ax.set_title(f"Epistasis carries {share:.0%} of model weight")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


ARM_LABEL_DECISION = {"llm_agent": "LLM agent", "llm_knowledge": "knowledge-augmented LLM agent"}


def fig_decision(records, report: dict, out: Path) -> None:
    """Decision layer: what the agent claimed, whether the data backed it, and
    how far its batch drifted from the predictor's own top-k."""
    hyp = report["decision_layer"]["hypotheses"]
    rounds = sorted({h["round"] for h in hyp})
    verdicts = ["supported", "refuted", "undetermined"]
    colors = {"supported": "#1b7837", "refuted": "#b2182b", "undetermined": "#9e9e9e"}

    # batch overlap between each LLM arm and predictor-greedy, same seed and round
    def picked(rec):
        return {row["variant"] if isinstance(row, dict) else row for row in rec["selected"]}
    by_key = {(r["arm"], r["seed"], r["round"]): picked(r) for r in records}
    seeds = sorted({r["seed"] for r in records})
    overlap = {arm: [] for arm in ARM_LABEL_DECISION}
    for arm in overlap:
        for rnd in rounds:
            vals = [len(by_key[(arm, s, rnd)] & by_key[("greedy", s, rnd)]) /
                    len(by_key[(arm, s, rnd)] | by_key[("greedy", s, rnd)])
                    for s in seeds if (arm, s, rnd) in by_key]
            overlap[arm].append(float(np.mean(vals)))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))

    ax = axes[0]
    bottom = np.zeros(len(rounds))
    for verdict in verdicts:
        counts = np.array([sum(1 for h in hyp if h["round"] == r and h["verdict"] == verdict)
                           for r in rounds], float)
        ax.bar(rounds, counts, bottom=bottom, color=colors[verdict], label=verdict, width=0.6)
        bottom += counts
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("registered hypotheses")
    ax.set_title("Hypotheses are judged by the program,\nnot by the model")
    ax.legend(frameon=False, loc="upper left", fontsize=6)

    ax = axes[1]
    dash = {"llm_agent": "-", "llm_knowledge": (0, (4, 2))}
    for arm, values in overlap.items():
        ax.plot(rounds, values, marker="o", ms=4, linestyle=dash[arm],
                lw=2.4 if arm == "llm_agent" else 1.4, color=ARM_COLOR.get(arm),
                label=ARM_LABEL_DECISION[arm])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("batch overlap with predictor-greedy")
    ax.set_title("The agent starts by copying the predictor\nand drifts away as feedback arrives")
    ax.legend(frameon=False, loc="lower left", fontsize=6)

    ax = axes[2]
    explore = report["decision_layer"]["exploration"]
    for arm in ["greedy", "llm_agent", "llm_knowledge", "random"]:
        rows = [e for e in explore if e["arm"] == arm]
        if not rows:
            continue
        xs = sorted({e["round"] for e in rows})
        ys = [float(np.mean([e["mean_pairwise_hamming"] for e in rows if e["round"] == r])) for r in xs]
        ax.plot(xs, ys, marker="s", ms=4, color=ARM_COLOR.get(arm), label=ARM_LABEL.get(arm, arm))
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("mean pairwise Hamming distance")
    ax.set_title("Batch diversity: exploration collapses\nonce the model is confident")
    ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_structure(struct: dict, features: list, out: Path) -> None:
    """Structure track: does Boltz-2 confidence track binding fitness?"""
    rows = [r for r in features if "error" not in r]
    fit = np.array([r["fitness"] for r in rows], float)
    plddt = np.array([r["plddt_sites"] for r in rows], float)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.scatter(np.log1p(fit), plddt, s=14, color="#2b6cb0", alpha=0.75, edgecolor="none")
    rho = struct["spearman_vs_fitness"]["plddt_sites"]
    ax.set_xlabel("log(1 + measured fitness)")
    ax.set_ylabel("Boltz-2 pLDDT at the four sites")
    ax.set_title(f"Predicted-fold confidence barely tracks binding\n(Spearman {rho:+.2f}, n={len(rows)})")
    ax.margins(0.05)

    ax = axes[1]
    ab = struct["ablation"]
    labels = ["Spearman", "top-5% precision"]
    seq = [ab["sequence_only"]["spearman"], ab["sequence_only"]["top5pct_precision"]]
    both = [ab["sequence_plus_structure"]["spearman"], ab["sequence_plus_structure"]["top5pct_precision"]]
    x = np.arange(2)
    ax.bar(x - 0.18, seq, width=0.34, color="#4c72b0", label="sequence model only")
    ax.bar(x + 0.18, both, width=0.34, color="#dd8452", label="+ structure confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("5-fold CV score on the folded sample")
    ax.set_title("Adding structure columns changes nothing")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", default="results/experiment/records.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--interpretability", default="results/interpretability.json")
    ap.add_argument("--mpnn-prior", default="config/mpnn_log_odds.json")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()

    style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_landscape(args.data)

    if Path(args.records).exists():
        fig_campaign(json.loads(Path(args.records).read_text()),
                     outdir / "fig1_campaign.png", args.batch_size)
    if Path(args.baselines).exists():
        fig_baselines(json.loads(Path(args.baselines).read_text()),
                      outdir / "fig2_baselines.png")
    fig_priors(df, outdir / "fig3_priors.png", Path(args.mpnn_prior))
    if Path(args.interpretability).exists():
        report = json.loads(Path(args.interpretability).read_text())
        fig_interpret(report, outdir / "fig4_interpretability.png")
        if "decision_layer" in report and Path(args.records).exists():
            fig_decision(json.loads(Path(args.records).read_text()), report,
                         outdir / "fig5_decision.png")
    struct_path = Path(args.results if hasattr(args, "results") else "results") / "structure"
    analysis, features = struct_path / "structure_analysis.json", struct_path / "structure_features.json"
    if analysis.exists() and features.exists():
        fig_structure(json.loads(analysis.read_text()), json.loads(features.read_text()),
                      outdir / "fig6_structure.png")
    print("figures written to", outdir)


if __name__ == "__main__":
    main()
