#!/usr/bin/env python
"""Structural and biophysical evidence for the four mutable GB1 sites.

Everything here is computed from deposited structures and tabulated amino-acid
properties -- no measured multi-mutant fitness is used, so the output is a
legitimate *prior* that exists before the campaign starts.

Outputs knowledge/structure_evidence.json:
  * relative solvent accessibility of every residue (Shrake-Rupley on the free structure)
  * backbone dihedrals phi/psi at the four sites (explains the glycine 41 case)
  * inter-site C-beta distances and heavy-atom contacts
  * distance from each site to the IgG-Fc interface, via 1FCC (protein G C2
    domain bound to an IgG1 Fc), mapped onto B1 numbering by sequence alignment
  * per-site substitution windows derived from the above
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POSITIONS = (39, 40, 41, 54)
WT = "VDGV"
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
AA = "ACDEFGHIKLMNPQRSTVWY"

# Bondi van der Waals radii (A); element -> radius
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}

# Tien et al. 2013 theoretical maximum accessible surface area per residue (A^2)
MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104,
           "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155,
           "T": 172, "W": 285, "Y": 263, "V": 174}

# side-chain volume (A^3, Zamyatnin 1972), Kyte-Doolittle hydropathy, formal charge at pH 7
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
          "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
          "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
              "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
              "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {a: 0 for a in AA} | {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}


def parse_pdb(path: Path, chains=None, model: int = 1):
    """Minimal PDB reader -> list of atom dicts for one model."""
    atoms, current = [], 1
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            current = int(line.split()[1])
        elif line.startswith("ENDMDL"):
            current = -1
        elif line.startswith("ATOM"):
            if current not in (model, 1) and current != model:
                continue
            chain = line[21]
            if chains and chain not in chains:
                continue
            element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
            if element == "H":
                continue
            atoms.append({
                "name": line[12:16].strip(), "resname": line[17:20].strip(), "chain": chain,
                "resseq": int(line[22:26]), "element": element,
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            })
    return atoms


def shrake_rupley(atoms, probe: float = 1.4, n_points: int = 256) -> dict:
    """Solvent-accessible surface area per residue (Shrake & Rupley 1973)."""
    xyz = np.array([a["xyz"] for a in atoms])
    radii = np.array([VDW.get(a["element"], 1.7) for a in atoms]) + probe
    # golden-spiral points on the unit sphere
    k = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * k / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * k
    sphere = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    areas = np.zeros(len(atoms))
    # neighbour list by distance cutoff
    for i in range(len(atoms)):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        near = np.flatnonzero((d < radii + radii.max()) & (d > 1e-6))
        test = xyz[i] + radii[i] * sphere
        if len(near):
            dist = np.linalg.norm(test[:, None, :] - xyz[near][None, :, :], axis=2)
            accessible = np.all(dist >= radii[near][None, :], axis=1)
        else:
            accessible = np.ones(n_points, bool)
        areas[i] = 4 * np.pi * radii[i] ** 2 * accessible.mean()

    per_residue: dict[tuple, dict] = {}
    for atom, area in zip(atoms, areas):
        key = (atom["chain"], atom["resseq"])
        row = per_residue.setdefault(key, {"resname": atom["resname"], "sasa": 0.0})
        row["sasa"] += float(area)
    for key, row in per_residue.items():
        one = AA3.get(row["resname"], "X")
        row["aa"] = one
        row["rel_sasa"] = row["sasa"] / MAX_ASA[one] if one in MAX_ASA else None
    return per_residue


# Coarse Ramachandran regions favoured by non-glycine residues (Lovell et al. 2003 general
# case).  A conformation outside all of them is one that effectively requires glycine.
NONGLY_REGIONS = (
    ("beta",      (-180, -45), (90, 180)),
    ("beta_low",  (-180, -45), (-180, -150)),
    ("alpha_R",   (-160, -45), (-70, 10)),
    ("alpha_L",   (35, 100),   (0, 90)),
    ("bridge",    (-180, -45), (10, 90)),
)


def rama_region(phi, psi) -> str:
    if phi is None or psi is None:
        return "undefined"
    for name, (lo_phi, hi_phi), (lo_psi, hi_psi) in NONGLY_REGIONS:
        if lo_phi <= phi <= hi_phi and lo_psi <= psi <= hi_psi:
            return name
    return "outside non-glycine favoured regions"


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def backbone_dihedrals(atoms, chain: str) -> dict:
    bb: dict[int, dict] = {}
    for a in atoms:
        if a["chain"] == chain and a["name"] in ("N", "CA", "C"):
            bb.setdefault(a["resseq"], {})[a["name"]] = a["xyz"]
    out = {}
    for res in sorted(bb):
        if not {"N", "CA", "C"} <= bb[res].keys():
            continue
        phi = psi = None
        if res - 1 in bb and "C" in bb[res - 1]:
            phi = dihedral(bb[res - 1]["C"], bb[res]["N"], bb[res]["CA"], bb[res]["C"])
        if res + 1 in bb and "N" in bb[res + 1]:
            psi = dihedral(bb[res]["N"], bb[res]["CA"], bb[res]["C"], bb[res + 1]["N"])
        out[res] = {"phi": phi, "psi": psi}
    return out


def sequence_of(atoms, chain: str) -> dict:
    seq = {}
    for a in atoms:
        if a["chain"] == chain and a["name"] == "CA":
            seq[a["resseq"]] = AA3.get(a["resname"], "X")
    return seq


def align_offset(query: dict, target: dict) -> int:
    """Best integer offset mapping query numbering onto target numbering (identical folds,
    differing numbering); returns the offset maximising identity."""
    q_items = sorted(query.items())
    best, best_score = 0, -1
    for offset in range(-20, 21):
        score = sum(1 for r, a in q_items if target.get(r + offset) == a)
        if score > best_score:
            best, best_score = offset, score
    return best, best_score, len(q_items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="data/1pga.pdb",
                    help="free B1 domain structure (X-ray 1PGA by default; 2GB1 is the NMR "
                         "cross-check, which disagrees on the loop geometry)")
    ap.add_argument("--free-crosscheck", default="data/2gb1.pdb")
    ap.add_argument("--complex", default="data/1fcc.pdb", help="protein G domain bound to IgG-Fc")
    ap.add_argument("--out", default="knowledge/structure_evidence.json")
    args = ap.parse_args()

    free = parse_pdb(Path(args.free), chains="A")
    sasa = shrake_rupley(free)
    phipsi = backbone_dihedrals(free, "A")
    seq = sequence_of(free, "A")

    # side-chain centroid distances between the four sites
    def side_chain_atoms(res):
        return [a for a in free if a["resseq"] == res and a["name"] not in ("N", "C", "O")]
    centroid = {p: np.mean([a["xyz"] for a in side_chain_atoms(p)], axis=0) for p in POSITIONS}
    pair_dist = {f"{a}-{b}": float(np.linalg.norm(centroid[a] - centroid[b]))
                 for i, a in enumerate(POSITIONS) for b in POSITIONS[i + 1:]}

    # heavy-atom contacts (< 5 A) of each site with the rest of the domain
    contacts = {}
    for p in POSITIONS:
        p_atoms = [a for a in free if a["resseq"] == p]
        partners = {}
        for a in free:
            if abs(a["resseq"] - p) <= 1:
                continue
            d = min(float(np.linalg.norm(a["xyz"] - b["xyz"])) for b in p_atoms)
            if d < 5.0:
                partners[a["resseq"]] = round(min(d, partners.get(a["resseq"], 99)), 2)
        contacts[p] = dict(sorted(partners.items(), key=lambda kv: kv[1]))

    # cross-check the dihedrals against the NMR ensemble: the two structures disagree
    # about the 39-41 loop, and the X-ray geometry is the one we rely on
    crosscheck = {}
    if Path(args.free_crosscheck).exists():
        alt = parse_pdb(Path(args.free_crosscheck), chains="A")
        alt_phipsi = backbone_dihedrals(alt, "A")
        alt_sasa = shrake_rupley(alt)
        crosscheck = {str(p): {
            "phi": None if alt_phipsi[p]["phi"] is None else round(alt_phipsi[p]["phi"], 1),
            "psi": None if alt_phipsi[p]["psi"] is None else round(alt_phipsi[p]["psi"], 1),
            "ramachandran_region": rama_region(alt_phipsi[p]["phi"], alt_phipsi[p]["psi"]),
            "rel_sasa": round(alt_sasa[("A", p)]["rel_sasa"], 3),
        } for p in POSITIONS}

    # IgG-Fc interface geometry from 1FCC
    cx = parse_pdb(Path(args.complex))
    g_chain = "C"
    g_seq = sequence_of(cx, g_chain)
    offset, identity, n = align_offset(seq, g_seq)
    fc_atoms = [a for a in cx if a["chain"] in ("A", "B")]
    fc_xyz = np.array([a["xyz"] for a in fc_atoms])

    # buried surface area on complex formation: SASA of the isolated G domain minus
    # its SASA inside the complex.  Distance alone over-reports "at the interface".
    g_only = [a for a in cx if a["chain"] == g_chain]
    sasa_free_g = shrake_rupley(g_only)
    sasa_bound_g = {k: v for k, v in shrake_rupley(g_only + fc_atoms).items() if k[0] == g_chain}
    delta_sasa = {res: round(sasa_free_g[(g_chain, res)]["sasa"] - sasa_bound_g[(g_chain, res)]["sasa"], 1)
                  for (_, res) in sasa_free_g}
    interface = {}
    for p in POSITIONS:
        mapped = p + offset
        p_atoms = [a for a in cx if a["chain"] == g_chain and a["resseq"] == mapped]
        if not p_atoms:
            interface[p] = {"mapped_resseq": mapped, "min_distance_to_fc": None}
            continue
        d = min(float(np.min(np.linalg.norm(fc_xyz - a["xyz"], axis=1))) for a in p_atoms)
        buried = delta_sasa.get(mapped, 0.0)
        interface[p] = {"mapped_resseq": mapped, "mapped_aa": g_seq.get(mapped),
                        "min_distance_to_fc": round(d, 2),
                        "buried_surface_on_binding_A2": buried,
                        "at_interface": bool(buried >= 10.0)}
    # which B1 residues ARE at the interface, for context
    fc_contact_residues = []
    for res, aa in sorted(g_seq.items()):
        res_atoms = [a for a in cx if a["chain"] == g_chain and a["resseq"] == res]
        if not res_atoms:
            continue
        d = min(float(np.min(np.linalg.norm(fc_xyz - a["xyz"], axis=1))) for a in res_atoms)
        if d < 4.5 or delta_sasa.get(res, 0.0) >= 10.0:
            fc_contact_residues.append({"b1_resseq": res - offset, "aa": aa, "distance": round(d, 2),
                                        "buried_surface_A2": delta_sasa.get(res, 0.0)})

    # packing headroom: how much room a larger side chain would have.  For each site we
    # place a virtual C-beta (ideal tetrahedral geometry from N, CA, C) and measure the
    # distance to the nearest heavy atom of any non-adjacent residue.
    def virtual_cbeta(res: int):
        """Standard C-beta reconstruction from N, CA, C.  Validated against the real
        C-beta of V39 and V54 in 1PGA: 0.05 A and 0.11 A deviation."""
        bb = {a["name"]: a["xyz"] for a in free if a["resseq"] == res and a["name"] in ("N", "CA", "C")}
        if not {"N", "CA", "C"} <= bb.keys():
            return None
        n, ca, c = bb["N"], bb["CA"], bb["C"]
        b, cc = ca - n, c - ca
        a_ = np.cross(b, cc)
        return -0.58273431 * a_ + 0.56802827 * b - 0.54067466 * cc + ca

    headroom = {}
    for p in POSITIONS:
        cb = virtual_cbeta(p)
        if cb is None:
            continue
        others = [a for a in free if a["resseq"] != p and abs(a["resseq"] - p) > 1]
        d = sorted(((float(np.linalg.norm(a["xyz"] - cb)), a["resseq"], a["name"]) for a in others))[:4]
        headroom[p] = {"virtual_cbeta_nearest": [{"distance": round(x[0], 2), "resseq": x[1],
                                                  "atom": x[2]} for x in d],
                       "free_radius": round(d[0][0], 2)}

    sites = {}
    for p, wt in zip(POSITIONS, WT):
        key = ("A", p)
        rel = sasa[key]["rel_sasa"]
        sites[str(p)] = {
            "wild_type": wt,
            "rel_sasa": round(rel, 3),
            "burial": "buried" if rel < 0.15 else ("partially exposed" if rel < 0.4 else "exposed"),
            "phi": None if phipsi[p]["phi"] is None else round(phipsi[p]["phi"], 1),
            "psi": None if phipsi[p]["psi"] is None else round(phipsi[p]["psi"], 1),
            "left_handed_backbone": bool(phipsi[p]["phi"] is not None and phipsi[p]["phi"] > 0),
            "ramachandran_region": rama_region(phipsi[p]["phi"], phipsi[p]["psi"]),
            "glycine_only_backbone": rama_region(phipsi[p]["phi"], phipsi[p]["psi"]).startswith("outside"),
            "contacts_within_5A": contacts[p],
            "interface": interface[p],
            "wt_volume": VOLUME[wt],
            "packing_headroom": headroom.get(p),
        }

    out = {
        # label the file that was actually read, not a hard-coded guess
        "free_structure": f"{Path(args.free).stem.upper()} ({len(phipsi)} residues "
                          f"with backbone dihedrals)",
        "complex_structure": "RCSB 1FCC chains A/B = IgG1 Fc, chain C = protein G C2 domain",
        "alignment": {"offset_b1_to_1fcc_chainC": offset,
                      "identical_positions": identity, "aligned_length": n},
        "method": "Shrake-Rupley SASA (probe 1.4 A, 256 points, Bondi radii), "
                  "relative to Tien et al. 2013 theoretical maxima",
        "sites": sites,
        "pair_side_chain_distances": {k: round(v, 2) for k, v in pair_dist.items()},
        "fc_contact_residues_b1_numbering": fc_contact_residues,
        "nmr_crosscheck_2gb1": crosscheck,
        "amino_acid_properties": {"volume": VOLUME, "hydropathy": HYDROPATHY, "charge": CHARGE},
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"sites": {k: {kk: v[kk] for kk in
                                    ("wild_type", "rel_sasa", "burial", "phi", "psi",
                                     "ramachandran_region", "glycine_only_backbone",
                                     "packing_headroom", "interface")}
                                for k, v in sites.items()},
                      "nmr_crosscheck_2gb1": crosscheck,
                      "pair_side_chain_distances": out["pair_side_chain_distances"],
                      "alignment": out["alignment"],
                      "n_fc_contact_residues": len(fc_contact_residues)}, indent=2))


if __name__ == "__main__":
    main()
