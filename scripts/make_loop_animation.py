#!/usr/bin/env python
"""Render the closed loop of one campaign as an animated GIF.

Every frame is drawn from results/v2/records_final.json: the candidate pool as
the constraints cut it down, the 96-variant batch, the labels that came back and
the best-so-far curve. Nothing in the animation is mocked up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SITES = ["39", "40", "41", "54"]
WILD = "VDGV"
INK = "#1b2733"
MUT = "#64748b"
POOL = "#b9c2cc"
LEGAL = "#5b7fa6"
BATCH = "#c44e52"
HIT = "#b2182b"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.9, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 110, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
    })


def thousands(n: int) -> str:
    return f"{n:,}"


def frame(rounds, r_idx, stage, start_labels, optimum):
    """One frame: r_idx indexes the round, stage says how much of it is revealed."""
    rec = rounds[r_idx]
    audit = rec["notes"]["constraint_audit"]
    fits = np.array([s["fitness"] for s in rec["selected"]])
    n_pool = audit["n_pool"]
    n_legal = audit["n_legal_after_filters"]

    fig = plt.figure(figsize=(10.0, 5.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.22], height_ratios=[1.0, 0.72],
                          left=0.175, right=0.975, top=0.80, bottom=0.095,
                          wspace=0.26, hspace=0.52)
    ax_funnel = fig.add_subplot(gs[0, 0])
    ax_batch = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[:, 1])

    fig.text(0.035, 0.945, "A prior-and-constraint round on the GB1 four-site landscape",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.035, 0.893, f"seed 302 · round {rec['round']} of {len(rounds)} · "
                           f"96 measurements per round · the evaluator stays behind one call",
             fontsize=9, color=MUT)

    # ---- the pool, as the constraints cut it -------------------------------
    removed = audit.get("filters_removed", {})
    bars = [("candidate pool", n_pool, POOL)]
    if stage >= 1:
        bars.append(("legal after hard filters", n_legal, LEGAL))
    if stage >= 2:
        bars.append(("batch sent to the evaluator", len(fits), BATCH))
    y = np.arange(len(bars))[::-1]
    ax_funnel.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.55)
    for yi, (lab, val, _c) in zip(y, bars):
        ax_funnel.text(val + n_pool * 0.015, yi, thousands(val), va="center",
                       fontsize=8.5, color=INK)
    ax_funnel.set_yticks(y, [b[0] for b in bars], fontsize=8.5)
    ax_funnel.set_xlim(0, n_pool * 1.25)
    ax_funnel.set_xticks([])
    ax_funnel.spines["bottom"].set_visible(False)
    cut = removed.get("C2_volume_window", 0)
    ax_funnel.set_title("what the constraints allow" + (
        f"\nC2 volume window removes {thousands(cut)}" if stage >= 1 and cut else
        "\nno hard filter active this round" if stage >= 1 else ""),
        fontsize=9, color=INK, loc="left")

    # ---- batch composition -------------------------------------------------
    if stage >= 2:
        comp = np.zeros((4, 20))
        letters = "ACDEFGHIKLMNPQRSTVWY"
        for s in rec["selected"]:
            for i, aa in enumerate(s["variant"]):
                comp[i, letters.index(aa)] += 1
        ax_batch.imshow(comp, aspect="auto", cmap="Blues", vmin=0, vmax=comp.max())
        ax_batch.set_yticks(range(4), [f"{w}{s}" for w, s in zip(WILD, SITES)], fontsize=8)
        ax_batch.set_xticks(range(20), list(letters), fontsize=7)
        ax_batch.set_title("amino acids the batch actually proposes", fontsize=9,
                           color=INK, loc="left")
    else:
        ax_batch.axis("off")
        ax_batch.text(0.0, 0.5, "the agent reads the previous audit,\n"
                                "then rewrites the rules for this round",
                      fontsize=9.5, color=MUT, va="center")
    for sp in ("left", "bottom"):
        ax_batch.spines[sp].set_visible(stage >= 2)

    # ---- best so far -------------------------------------------------------
    xs = [start_labels + (j + 1) * 96 for j in range(len(rounds))]
    ys = [rr["best_so_far"] for rr in rounds]
    upto = r_idx if stage >= 3 else r_idx - 1
    ax_curve.axhline(optimum, color=HIT, lw=0.9, ls=(0, (4, 3)))
    ax_curve.text(start_labels + 10, optimum * 1.02, f"global optimum FWAA {optimum:.3f}",
                  fontsize=8.5, color=HIT)
    if upto >= 0:
        ax_curve.plot(xs[:upto + 1], ys[:upto + 1], "-o", color=BATCH, lw=2.0, ms=5)
    if stage >= 3:
        ax_curve.annotate(f"{ys[upto]:.3f}", (xs[upto], ys[upto]),
                          textcoords="offset points", xytext=(8, -2),
                          fontsize=9.5, color=INK, fontweight="bold")
    if stage >= 2:
        jitter = np.linspace(-14, 14, len(fits))
        ax_curve.scatter(xs[r_idx] + jitter, fits, s=9, color=BATCH, alpha=0.35,
                         linewidths=0, zorder=1)
    ax_curve.set_xlim(start_labels + 40, start_labels + 96 * len(rounds) + 62)
    ax_curve.set_ylim(-0.3, optimum * 1.16)
    ax_curve.set_xlabel("measurements paid for (cumulative)")
    ax_curve.set_ylabel("fitness")
    ax_curve.set_title("best variant found so far", fontsize=9, color=INK, loc="left")

    # ---- caption line ------------------------------------------------------
    msg = {
        0: "the round opens with the audit of the previous one",
        1: "hard filters first: only legal candidates survive",
        2: "96 variants are committed (SHA-256) before any label is revealed",
        3: f"labels come back: batch best {fits.max():.3f}, mean {fits.mean():.3f}",
    }[stage]
    fig.text(0.035, 0.028, msg, fontsize=9.5, color=MUT)
    if stage >= 3 and upto >= 0 and abs(ys[upto] - optimum) < 1e-6:
        w = fig.text(0.035, 0.028, msg + "  ", fontsize=9.5, color=MUT, alpha=0)
        fig.canvas.draw()
        x1 = w.get_window_extent().x1 / fig.get_window_extent().width
        fig.text(x1, 0.028, "— global optimum reached", fontsize=9.5, color=HIT,
                 fontweight="bold")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--prior-eval", default="results/prior_evaluation.json")
    ap.add_argument("--arm", default="prior_constraint")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--out", default="figures/animation/loop.gif")
    ap.add_argument("--ms", type=int, default=900, help="frame duration")
    args = ap.parse_args()

    style()
    records = json.loads(Path(args.records).read_text())
    optimum = json.loads(Path(args.prior_eval).read_text())["global_optimum"]["fitness"]
    rounds = sorted([r for r in records if r["arm"] == args.arm and r["seed"] == args.seed],
                    key=lambda r: r["round"])
    # every arm starts from the same 192 labels; the curve starts at the first
    # round the campaign actually paid for
    start_labels = 192

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for i in range(len(rounds)):
        for stage in range(4):
            fig = frame(rounds, i, stage, start_labels, optimum)
            png = out.parent / f"_frame_{i}_{stage}.png"
            fig.savefig(png, facecolor="white")
            plt.close(fig)
            frames.append(Image.open(png).convert("RGB"))
            durations.append(args.ms if stage < 3 else int(args.ms * 2.1))
    durations[-1] = 2600
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True, disposal=2)
    for f in out.parent.glob("_frame_*.png"):
        f.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, "
          f"{frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == "__main__":
    main()
