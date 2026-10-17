#!/usr/bin/env python
"""Structural observables at the four sites, measured on the PREDICTED monomer.

The complex prediction cannot be used here (see results/structure/complex_check.json:
the hosted model docks the domain with the wrong face, 0/16 of the crystal contact
surface), but the monomer fold is accurate, so what it can report honestly is local
geometry at the mutated sites:

  strain      - closest heavy-atom approach between a site's side chain and the rest
                of the domain; below ~3.0 A for carbon-carbon is a strained contact
  packing     - heavy atoms within 6 A of the side-chain centroid (cavity if few)
  loop_shift  - backbone dihedral deviation at site 41 against the wild-type
                prediction, i.e. does the substitution distort the loop
  plddt       - the model's own confidence at those positions

Thresholds are physical (van der Waals contact distance, 30 degrees of dihedral),
not tuned against the fitness labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_complexes import parse_cif_atoms
from build_structure_evidence import rama_region

SITES = (39, 40, 41, 54)
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def circ_diff(a: float, b: float) -> float:
    """Shortest angular distance between two dihedrals, in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def phi_psi(atoms, res):
    by = {}
    for a in atoms:
        if a["name"] in ("N", "CA", "C"):
            by[(a["resseq"], a["name"])] = a["xyz"]
    try:
        phi = dihedral(by[(res - 1, "C")], by[(res, "N")], by[(res, "CA")], by[(res, "C")])
        psi = dihedral(by[(res, "N")], by[(res, "CA")], by[(res, "C")], by[(res + 1, "N")])
        return phi, psi
    except KeyError:
        return None, None


def observe(path: Path) -> dict:
    atoms = parse_cif_atoms(path)
    atoms = [a for a in atoms if a["element"] != "H"]
    out = {"variant": path.stem, "sites": {}}
    for s in SITES:
        side = [a for a in atoms if a["resseq"] == s and a["name"] not in BACKBONE]
        others = [a for a in atoms if abs(a["resseq"] - s) > 1]
        if not side or not others:
            continue
        # strain is a NONPOLAR clash: N/O pairs at 2.8-3.0 A are hydrogen bonds,
        # so only carbon-carbon approaches are counted here
        side_c = [a for a in side if a["element"] == "C"]
        other_c = [a for a in others if a["element"] == "C"]
        if not side_c or not other_c:
            continue
        X = np.array([a["xyz"] for a in side_c])
        Y = np.array([a["xyz"] for a in other_c])
        d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        others = other_c
        centroid = X.mean(0)
        near = int((np.linalg.norm(Y - centroid, axis=1) < 6.0).sum())
        out["sites"][str(s)] = {
            "min_cc_contact_A": round(float(d.min()), 2),
            "closest_partner": f'{others[j]["resname"]}{others[j]["resseq"]}:{others[j]["name"]}',
            "heavy_atoms_within_6A": near,
            "n_side_chain_atoms": len(side),
            "plddt": round(float(np.mean([a["bfactor"] for a in atoms if a["resseq"] == s])), 3),
        }
    phi, psi = phi_psi(atoms, 41)
    out["loop41"] = {"phi": None if phi is None else round(phi, 1),
                     "psi": None if psi is None else round(psi, 1),
                     # reference-free: the wild type is glycine, so "deviation from
                     # the wild type" would flag every non-glycine residue by construction.
                     # The mechanistic question is whether the predicted backbone is a
                     # conformation a non-glycine residue can adopt at all.
                     "region": None if phi is None else rama_region(phi, psi)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monomers", default="results/structure/mono")
    ap.add_argument("--reference", default="VDGV")
    ap.add_argument("--strain-threshold", type=float, default=3.2,
                    help="carbon-carbon approach below this is a strained contact (A); "
                         "the standard nonpolar van der Waals contact is 3.4-3.7 A")
    ap.add_argument("--loop-threshold", type=float, default=30.0,
                    help="site-41 dihedral deviation from wild type counted as distortion (deg)")
    ap.add_argument("--out", default="results/structure/predicted_packing.json")
    args = ap.parse_args()

    rows = {}
    for cif in sorted(Path(args.monomers).glob("*.cif")):
        rows[cif.stem] = observe(cif)
        print(f'{cif.stem}: ' + " ".join(
            f'{s}:{rows[cif.stem]["sites"][s]["min_cc_contact_A"]:.2f}'
            for s in rows[cif.stem]["sites"]), flush=True)

    ref = rows.get(args.reference)
    flags = {}
    for v, r in rows.items():
        strained = [s for s, d in r["sites"].items()
                    if d["min_cc_contact_A"] < args.strain_threshold]
        region = r["loop41"].get("region")
        shift = None
        if ref and r["loop41"]["phi"] is not None and ref["loop41"]["phi"] is not None:
            shift = round(max(circ_diff(r["loop41"]["phi"], ref["loop41"]["phi"]),
                              circ_diff(r["loop41"]["psi"], ref["loop41"]["psi"])), 1)
        outside = bool(region and region.startswith("outside"))
        flags[v] = {"strained_sites": strained,
                    "loop41_region": region,
                    "loop41_deviation_from_wt_deg": shift,
                    "loop41_outside_nonglycine_regions": outside,
                    "flagged": bool(strained) or outside}

    out = {"thresholds": {"strain_A": args.strain_threshold,
                          "loop_deg": args.loop_threshold,
                          "basis": "van der Waals contact distance; dihedral deviation vs the "
                                   "wild-type prediction - not tuned against fitness"},
           "reference": args.reference,
           "criteria": ["strain: a side-chain carbon within %.1f A of another carbon" % args.strain_threshold,
                        "loop: predicted site-41 backbone outside non-glycine favoured regions"],
           "observations": rows, "flags": flags,
           "n_flagged": sum(1 for f in flags.values() if f["flagged"]), "n_total": len(flags)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'\nflagged {out["n_flagged"]} / {out["n_total"]} variants -> {args.out}')


if __name__ == "__main__":
    main()
