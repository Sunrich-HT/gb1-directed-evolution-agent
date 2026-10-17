#!/usr/bin/env python
"""Measure predicted interface geometry from the folded variant-Fc complexes.

Two things are computed per variant:

1. a sanity check - is the domain docked where it is docked in the crystal?  The
   predicted Fc chain is superposed onto 1FCC chain A and the same transform applied
   to the predicted GB1 chain, whose C-alpha RMSD against 1FCC chain C says whether
   the binding MODE is native.  An interface area measured on a wrongly docked model
   means nothing, so this gate comes first.

2. the interface itself - buried surface area on complex formation (total, and the
   share contributed by each mutable site) and the closest approach of each site's
   side chain to Fc.

Writes results/structure/complex_features.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_structure_evidence import MAX_ASA, parse_pdb, sequence_of, shrake_rupley

SITES = (39, 40, 41, 54)


def parse_cif_atoms(path: Path) -> list:
    """Boltz-2 writes line-broken atom_site rows; tokenise the whole loop."""
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines)
             if l.strip() == "loop_" and lines[k + 1].strip().startswith("_atom_site."))
    cols, j = [], i + 1
    while lines[j].strip().startswith("_atom_site."):
        cols.append(lines[j].strip().split(".", 1)[1])
        j += 1
    toks = []
    while j < len(lines) and not lines[j].startswith(("#", "loop_", "_", "data_")):
        toks += lines[j].split()
        j += 1
    n = len(cols)
    rows = [dict(zip(cols, toks[k:k + n])) for k in range(0, len(toks) - n + 1, n)]
    out = []
    for r in rows:
        if r.get("group_PDB") not in (None, "ATOM"):
            continue
        out.append({
            "chain": r.get("label_asym_id") or r.get("auth_asym_id"),
            "resseq": int(r["label_seq_id"]),
            "name": r["label_atom_id"].strip('"'),
            "element": r.get("type_symbol", r["label_atom_id"][0]),
            "resname": r["label_comp_id"],
            "xyz": np.array([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])]),
            "bfactor": float(r.get("B_iso_or_equiv", 0.0)),
        })
    return out


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return R, P.mean(0), Q.mean(0)


def ca_map(atoms, chain):
    return {a["resseq"]: a["xyz"] for a in atoms if a["chain"] == chain and a["name"] == "CA"}


def residue_sasa(atoms) -> dict:
    """Per-residue absolute SASA, keyed (chain, resseq)."""
    table = shrake_rupley(atoms)
    return {k: v["sasa"] for k, v in table.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", default="results/structure/complex")
    ap.add_argument("--native", default="data/1fcc.pdb")
    ap.add_argument("--out", default="results/structure/complex_features.json")
    args = ap.parse_args()

    native = parse_pdb(Path(args.native), chains=None)
    nat_fc = ca_map(native, "A")
    nat_gb1 = ca_map(native, "C")

    rows = []
    for cif in sorted(Path(args.complexes).glob("*.cif")):
        v = cif.stem
        atoms = parse_cif_atoms(cif)
        chains = sorted({a["chain"] for a in atoms})
        if len(chains) < 2:
            print(v, "single chain, skipped", flush=True)
            continue
        gb1_chain, fc_chain = chains[0], chains[1]
        if sum(1 for a in atoms if a["chain"] == chains[0] and a["name"] == "CA") > 100:
            gb1_chain, fc_chain = chains[1], chains[0]

        # --- 1. is it docked where the crystal docks it? ---
        # the prediction numbers residues 1..n while the crystal keeps PDB numbering,
        # so chains are paired by sequence order, not by residue id
        pred_fc, pred_gb1 = ca_map(atoms, fc_chain), ca_map(atoms, gb1_chain)
        pf = [pred_fc[i] for i in sorted(pred_fc)]
        nf = [nat_fc[i] for i in sorted(nat_fc)]
        n_fc = min(len(pf), len(nf))
        P, Q = np.array(pf[:n_fc]), np.array(nf[:n_fc])
        R, cp, cq = kabsch(P, Q)
        fc_rmsd = float(np.sqrt((((P - cp) @ R - (Q - cq)) ** 2).sum(1).mean()))

        pg = [pred_gb1[i] for i in sorted(pred_gb1)]
        ng = [nat_gb1[i] for i in sorted(nat_gb1)]
        n_g = min(len(pg), len(ng))
        moved = (np.array(pg[:n_g]) - cp) @ R + cq
        gb1_rmsd = float(np.sqrt(((moved - np.array(ng[:n_g])) ** 2).sum(1).mean()))

        # --- 2. interface burial ---
        sasa_cx = residue_sasa(atoms)
        sasa_a = residue_sasa([a for a in atoms if a["chain"] == gb1_chain])
        buried = {}
        for res in pred_gb1:
            k_cx, k_a = (gb1_chain, res), (gb1_chain, res)
            if k_cx in sasa_cx and k_a in sasa_a:
                buried[res] = max(0.0, sasa_a[k_a] - sasa_cx[k_cx])
        total_buried = float(sum(buried.values()))
        site_buried = {str(s): round(buried.get(s, 0.0), 1) for s in SITES}

        fc_xyz = np.array([a["xyz"] for a in atoms if a["chain"] == fc_chain])
        min_dist = {}
        for s in SITES:
            side = [a["xyz"] for a in atoms if a["chain"] == gb1_chain and a["resseq"] == s
                    and a["name"] not in ("N", "C", "O")]
            min_dist[str(s)] = (round(float(np.min(np.linalg.norm(
                fc_xyz[None, :, :] - np.array(side)[:, None, :], axis=2))), 2)
                if side else None)

        plddt_sites = float(np.mean([a["bfactor"] for a in atoms
                                     if a["chain"] == gb1_chain and a["resseq"] in SITES]))
        rows.append({
            "variant": v, "n_fc_paired": int(n_fc), "n_gb1_paired": int(n_g),
            "fc_superposition_rmsd": round(fc_rmsd, 2),
            "gb1_rmsd_to_native_pose": round(gb1_rmsd, 2),
            "native_binding_mode": bool(gb1_rmsd < 5.0),
            "interface_buried_A2": round(total_buried, 1),
            "buried_by_site_A2": site_buried,
            "min_sidechain_distance_to_fc_A": min_dist,
            "plddt_at_sites": round(plddt_sites, 3),
        })
        print(f"{v}: GB1 pose RMSD {gb1_rmsd:5.2f} A | interface {total_buried:6.1f} A^2 | "
              f"site40 {site_buried['40']:5.1f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print("wrote", args.out, "|", len(rows), "complexes")


if __name__ == "__main__":
    main()
