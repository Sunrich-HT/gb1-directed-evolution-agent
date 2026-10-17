#!/usr/bin/env python
"""Figures of the predicted structures the structural decisions were made on.

Coordinates are projected onto two principal axes and drawn in ordinary 2D axes:
matplotlib's 3D axes reserve large unreachable margins, which makes a tight,
caption-safe layout impossible at paper width.

fig7_docking_check  - predicted complex against the crystal complex with the Fc
                      superposed, so the displacement of the GB1 domain is visible.
                      This is the picture behind refusing to build an interface
                      constraint on the prediction.
fig8_packing_basis  - what the monomer prediction does support: the local environment
                      of the four mutable sites, dense against sparse, and the
                      measured relation to fitness that became constraint C9.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_complexes import ca_map, kabsch, parse_cif_atoms
from build_structure_evidence import parse_pdb
from make_paper_figures import MM, save, style

SITES = (39, 40, 41, 54)
SITE_COLOUR = {39: "#1d4ed8", 40: "#b45309", 41: "#15803d", 54: "#7c3aed"}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def ca_array(atoms, chain):
    m = ca_map(atoms, chain)
    return np.array([m[i] for i in sorted(m)])


def basis_of(P):
    """Two principal axes of a point set, as a 3x2 projection matrix."""
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[:2].T


def smooth(P, k=8):
    if len(P) < 4:
        return P
    t = np.linspace(0, 1, len(P))
    tt = np.linspace(0, 1, len(P) * k)
    return np.column_stack([np.interp(tt, t, P[:, d]) for d in range(P.shape[1])])


def frame_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


def figure7(check, native, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 84 * MM))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.26, wspace=0.04)

    nat_fc = ca_array(native, "A")
    nat_gb1 = ca_array(native, "C")

    panels = [("single_fc_chain", "results/structure/complex/VDGV.cif",
               "单条 Fc 链（262 残基）"),
              ("native_fc_homodimer", "results/structure/complex_dimer/VDGV.cif",
               "天然 Fc 同源二聚体（468 残基）")]

    for ax, (setup, path, title) in zip(axes, panels):
        rec = next(r for r in check["setups"][setup] if r["file"] == Path(path).name)
        atoms = parse_cif_atoms(Path(path))
        P = ca_array(atoms, rec["fc_chain_touched"])
        n = min(len(P), len(nat_fc))
        R, cp, cq = kabsch(P[:n], nat_fc[:n])

        moved = {ch: (ca_array(atoms, ch) - cp) @ R + cq
                 for ch, cnt in rec["chains"].items() if cnt >= 100}
        G = (ca_array(atoms, rec["gb1_chain"]) - cp) @ R + cq
        B = basis_of(np.vstack(list(moved.values()) + [nat_gb1, G]))

        for Q in moved.values():
            q = smooth(Q @ B)
            ax.plot(q[:, 0], q[:, 1], color="#c3c7ce", lw=1.1, zorder=2)
        for arr, colour, lw, z in ((nat_gb1, "#15803d", 2.2, 4), (G, "#c0392b", 2.2, 5)):
            q = smooth(arr @ B)
            ax.plot(q[:, 0], q[:, 1], color=colour, lw=lw, zorder=z)

        a, b = (nat_gb1 @ B).mean(0), (G @ B).mean(0)
        ax.annotate("", xy=b, xytext=a, zorder=10,
                    arrowprops=dict(arrowstyle="<->", lw=1.1, color="#111827",
                                    shrinkA=0, shrinkB=0))
        ax.text(0.5, 0.02, f'结构域位置相差 {rec["gb1_pose_rmsd_to_crystal_A"]:.1f} Å',
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="#111827", zorder=11,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#e4e4e7"))
        ax.set_title(title, fontsize=8.5, pad=6)
        frame_off(ax)

    axes[0].text(-0.01, 1.07, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    axes[1].text(-0.01, 1.07, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.185, "灰 = 预测的 IgG-Fc（已叠合到晶体 1FCC，叠合 RMSD 0.8–1.0 Å）　　"
                         "绿 = 晶体中蛋白 G 结构域的位置　　红 = 预测放置结构域的位置",
             ha="center", fontsize=7.6)
    fig.text(0.5, 0.105, "结合位点落在对的那块 Fc 表面上，但结构域是用错误的一面贴上去的：",
             ha="center", fontsize=7.6)
    fig.text(0.5, 0.03, "GB1 一侧与晶体接触面的重合为 0/16，四个可变位点都不在预测界面上"
                        " —— 界面类约束不能建立在它之上。",
             ha="center", fontsize=7.6, color="#b91c1c")
    return save(fig, out, "fig7_docking_check")


def side_chain(atoms, res):
    return [a for a in atoms if a["resseq"] == res and a["name"] not in BACKBONE
            and a["element"] != "H"]


def figure8(pack, shortlist, fitness_of, out: Path) -> Path:
    dens = {v: float(np.mean([s["heavy_atoms_within_6A"] for s in o["sites"].values()]))
            for v, o in pack["observations"].items() if o["sites"]}
    here = [v for v in shortlist if v in dens]
    hi = max(here, key=lambda v: dens[v])
    lo = min(here, key=lambda v: dens[v])

    fig = plt.figure(figsize=(180 * MM, 84 * MM))
    gs = fig.add_gridspec(1, 3, left=0.015, right=0.985, top=0.88, bottom=0.30,
                          wspace=0.22, width_ratios=[1, 1, 1.22])

    for col, v in enumerate((hi, lo)):
        ax = fig.add_subplot(gs[0, col])
        atoms = parse_cif_atoms(Path(f"results/structure/mono/{v}.cif"))
        P = np.array([a["xyz"] for a in atoms if a["name"] == "CA"])
        B = basis_of(P)
        q = smooth(P @ B)
        ax.plot(q[:, 0], q[:, 1], color="#c3c7ce", lw=1.3, zorder=2)
        for s in SITES:
            sc = side_chain(atoms, s)
            if not sc:
                continue
            X = np.array([a["xyz"] for a in sc])
            cen = X.mean(0)
            near = np.array([a["xyz"] for a in atoms if a["element"] != "H"
                             and abs(a["resseq"] - s) > 1
                             and np.linalg.norm(a["xyz"] - cen) < 6.0])
            if len(near):
                n2 = near @ B
                ax.scatter(n2[:, 0], n2[:, 1], s=4.5, color=SITE_COLOUR[s], alpha=0.30,
                           lw=0, zorder=3)
            x2 = X @ B
            ax.scatter(x2[:, 0], x2[:, 1], s=20, color=SITE_COLOUR[s], edgecolor="white",
                       linewidth=0.4, zorder=5)
            c2 = cen @ B
            ax.text(c2[0], c2[1] + 1.4, str(s), fontsize=7, ha="center",
                    color=SITE_COLOUR[s], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        ax.set_title(f"{v}", fontsize=9.5, pad=10, fontweight="bold")
        ax.text(0.5, 1.015, f"预测堆积密度 {dens[v]:.0f}　实测适应度 {fitness_of[v]:.2f}",
                transform=ax.transAxes, ha="center", fontsize=7.6)
        ax.text(-0.01, 1.10, "ab"[col], transform=ax.transAxes, fontsize=9, fontweight="bold")
        frame_off(ax)

    ax = fig.add_subplot(gs[0, 2])
    x = np.array([dens[v] for v in here])
    y = np.array([fitness_of[v] for v in here])
    ax.scatter(x, y, s=17, color="#9aa3ad", edgecolor="white", linewidth=0.4, zorder=3)
    for v, colour in ((hi, "#c0392b"), (lo, "#1d4ed8")):
        ax.scatter([dens[v]], [fitness_of[v]], s=38, color=colour, edgecolor="white",
                   linewidth=0.7, zorder=5)
        ax.annotate(v, (dens[v], fitness_of[v]), textcoords="offset points",
                    xytext=(-6 if v == hi else 6, 6),
                    ha="right" if v == hi else "left", fontsize=7, color=colour)
    med = float(np.median(x))
    ax.axvline(med, color="#111827", lw=0.8, ls=(0, (3, 2)), zorder=2)
    r = spearmanr(x, y)
    ax.set_xlabel("预测局部堆积密度（6 Å 内重原子数）", fontsize=7.8, labelpad=2)
    ax.set_ylabel("实测适应度", fontsize=7.8)
    ax.set_title(f"短名单 {len(here)} 条：ρ = {r.statistic:+.2f}  (p = {r.pvalue:.3f})",
                 fontsize=8.5, pad=10)
    ax.text(med + 0.15, ax.get_ylim()[1], "中位数", fontsize=6.8, va="top", color="#111827")
    ax.text(-0.26, 1.10, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.105, "a、b 为预测的单体结构：灰线是主链走向，彩色实心点是四个可变位点的侧链重原子，"
                         "同色半透明小点是被计入「局部堆积密度」的邻近重原子。",
             ha="center", fontsize=7.6)
    fig.text(0.5, 0.028, f"c 中虚线为该量在短名单上的中位数：右半边 {len(here)//2} 条的实测均值 4.31，"
                         f"左半边 3.32；全局最优落在右半边。",
             ha="center", fontsize=7.6)
    return save(fig, out, "fig8_packing_basis")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/paper")
    args = ap.parse_args()

    style()
    mpl.rcParams["font.sans-serif"] = ["Songti SC", "Hiragino Sans GB", "Arial Unicode MS",
                                       "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    from gb1agent.data import load_landscape
    land = load_landscape(args.data)
    fitness_of = dict(zip(land["variant"].to_numpy(), land["fitness"].to_numpy(float)))

    check = json.loads(Path("results/structure/complex_check.json").read_text())
    native = parse_pdb(Path("data/1fcc.pdb"), chains=None)
    pack = json.loads(Path("results/structure/predicted_packing.json").read_text())
    shortlist = json.loads(Path("results/structure/round2_shortlist.json").read_text())["shortlist"]

    out = Path(args.out)
    print("\n".join(str(p) for p in (figure7(check, native, out),
                                     figure8(pack, shortlist, fitness_of, out))))


if __name__ == "__main__":
    main()
