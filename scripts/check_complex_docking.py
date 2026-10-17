#!/usr/bin/env python
"""Does the hosted structure prediction reproduce the known GB1-Fc binding mode?

An interface-based constraint is only meaningful if the predicted complex puts the
domain where the crystal puts it.  This checks that, in both setups that were tried:
a single Fc chain, and the native Fc homodimer.

For each predicted complex it superposes the Fc chain the domain actually touches
onto chain A of RCSB 1FCC, applies the same transform to the predicted GB1 chain and
compares with chain C (the copy bound to chain A); it then lists which GB1 residues
are at the predicted interface and how many of them are in the crystallographic
contact set.

Writes results/structure/complex_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_complexes import ca_map, kabsch, parse_cif_atoms
from build_structure_evidence import parse_pdb

SITES = (39, 40, 41, 54)


def ca_array(atoms, chain):
    m = ca_map(atoms, chain)
    return np.array([m[i] for i in sorted(m)])


def check(path: Path, native, native_contacts) -> dict:
    atoms = parse_cif_atoms(path)
    counts = {ch: sum(1 for a in atoms if a["chain"] == ch and a["name"] == "CA")
              for ch in sorted({a["chain"] for a in atoms})}
    gb1_ch = [ch for ch, n in counts.items() if n < 100][0]
    fc_chains = [ch for ch, n in counts.items() if n >= 100]

    G_all = np.array([a["xyz"] for a in atoms if a["chain"] == gb1_ch])
    touched, dmin = None, 1e9
    for ch in fc_chains:
        X = np.array([a["xyz"] for a in atoms if a["chain"] == ch])
        d = float(np.linalg.norm(X[:, None, :] - G_all[None, :, :], axis=2).min())
        if d < dmin:
            touched, dmin = ch, d

    P = ca_array(atoms, touched)
    Q = ca_array(native, "A")
    n = min(len(P), len(Q))
    R, cp, cq = kabsch(P[:n], Q[:n])
    fc_rmsd = float(np.sqrt((((P[:n] - cp) @ R - (Q[:n] - cq)) ** 2).sum(1).mean()))

    g_pred = ca_array(atoms, gb1_ch)
    g_nat = ca_array(native, "C")
    m = min(len(g_pred), len(g_nat))
    moved = (g_pred - cp) @ R + cq
    pose_rmsd = float(np.sqrt(((moved[:m] - g_nat[:m]) ** 2).sum(1).mean()))

    gb1_atoms = [a for a in atoms if a["chain"] == gb1_ch]
    fc_xyz = np.array([a["xyz"] for a in atoms if a["chain"] == touched])
    dmat = np.linalg.norm(np.array([a["xyz"] for a in gb1_atoms])[:, None, :]
                          - fc_xyz[None, :, :], axis=2).min(1)
    iface = sorted({gb1_atoms[i]["resseq"] for i in np.where(dmat < 4.5)[0]})

    return {"file": path.name, "chains": counts, "gb1_chain": gb1_ch,
            "fc_chain_touched": touched, "closest_approach_A": round(dmin, 2),
            "fc_superposition_rmsd_A": round(fc_rmsd, 2),
            "gb1_pose_rmsd_to_crystal_A": round(pose_rmsd, 2),
            "native_binding_mode": bool(pose_rmsd < 5.0),
            "predicted_gb1_interface": iface,
            "overlap_with_crystal_contacts": sorted(set(iface) & native_contacts),
            "n_crystal_contacts": len(native_contacts),
            "mutable_sites_at_interface": [p for p in SITES if p in iface]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default="results/structure/complex")
    ap.add_argument("--dimer", default="results/structure/complex_dimer")
    ap.add_argument("--native", default="data/1fcc.pdb")
    ap.add_argument("--structure-evidence", default="knowledge/structure_evidence.json")
    ap.add_argument("--out", default="results/structure/complex_check.json")
    args = ap.parse_args()

    native = parse_pdb(Path(args.native), chains=None)
    contacts = {r["b1_resseq"] for r in
                json.loads(Path(args.structure_evidence).read_text())
                ["fc_contact_residues_b1_numbering"]}

    out = {"crystal_contacts_gb1_side": sorted(contacts), "setups": {}}
    for label, folder in (("single_fc_chain", args.single), ("native_fc_homodimer", args.dimer)):
        rows = []
        for cif in sorted(Path(folder).glob("*.cif")):
            rows.append(check(cif, native, contacts))
            r = rows[-1]
            print(f'{label} {cif.stem}: pose {r["gb1_pose_rmsd_to_crystal_A"]:6.2f} A '
                  f'| contact overlap {len(r["overlap_with_crystal_contacts"])}/{len(contacts)} '
                  f'| mutable sites at interface {r["mutable_sites_at_interface"] or "none"}',
                  flush=True)
        out["setups"][label] = rows

    usable = any(r["native_binding_mode"] for rows in out["setups"].values() for r in rows)
    out["verdict"] = ("predicted complex reproduces the crystallographic binding mode"
                      if usable else
                      "predicted complex does NOT reproduce the crystallographic binding mode; "
                      "an interface-based constraint cannot be derived from it")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n" + out["verdict"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
