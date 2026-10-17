"""Compile published and structural evidence into a quantitative round-1 prior.

The prior scores the 4 x 20 substitution table plus one pair term, and every
component is switchable so that "the prior helped" is always one ablation away
from being tested.  Each component names the evidence it rests on.

Components (defaults in ``DEFAULT_WEIGHTS``)
--------------------------------------------
``mpnn``            ProteinMPNN inverse-folding log-odds on the GB1 backbone,
                    **zeroed at site 40**: a zero-shot ensemble called all 19
                    substitutions there disadvantageous while measurement says
                    the opposite (evidence EV053; EV049 on zero-shot scores not
                    transferring to non-natural objectives).
``packing_headroom``  Penalty for adding side-chain volume where there is no room.
                    Severity comes from the distance between the reconstructed
                    C-beta and the nearest neighbouring heavy atom in 1PGA, so it
                    is large at site 41 (2.11 A to the carbonyl of residue 55) and
                    negligible at 39/40/54.
``burial_charge``   Penalty for burying a formal charge, scaled by burial.
``interface_bonus`` Bonus for large/aromatic side chains at the one position that
                    actually buries surface against IgG-Fc in 1FCC (site 40,
                    41.9 A^2; the others bury <= 7 A^2) -- evidence EV034.
``pair_volume_compensation``  Bonus when the volume added at site 41 is removed at
                    site 54 (and vice versa): the strongest positive epistasis in
                    the published double-mutant map, G41L/V54G, works by exactly
                    this transfer (EV020), and V54A turns site-41 substitutions
                    from deleterious to beneficial (EV019).
``literature``      Per-substitution claims from the curated evidence base,
                    weighted by their stated confidence.
``burial_volume``   Symmetric volume term, OFF by default: measured against the
                    landscape it ranks worse than omitting it (see
                    results/prior_evaluation.json).

Leakage discipline
------------------
Rows whose source is the deep mutational scan that serves as the oracle are
excluded unless ``allow_dms_derived=True``; the measured single-mutant table is
never part of the default prior and is evaluated separately as an upper bound.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
POSITIONS = (39, 40, 41, 54)
WT = "VDGV"

LARGE_HYDROPHOBIC_AROMATIC = set("WYFLIM")


def _z(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        sd = row.std()
        out[i] = (row - row.mean()) / sd if sd > 1e-9 else 0.0
    return out


@dataclass
class PriorComponent:
    name: str
    weight: float
    matrix: np.ndarray | None                    # 4 x 20 site term (z-scored)
    provenance: str
    evidence_ids: list = field(default_factory=list)
    pair: tuple | None = None                    # (i, j) site indices for a pair term
    pair_matrix: np.ndarray | None = None        # 20 x 20, z-scored overall

    def to_dict(self) -> dict:
        d = {"name": self.name, "weight": self.weight, "provenance": self.provenance,
             "evidence_ids": self.evidence_ids, "kind": "pair" if self.pair else "site"}
        if self.matrix is not None:
            d["top_per_site"] = {
                str(p): sorted(((AA[j], round(float(self.matrix[i, j]), 2)) for j in range(20)),
                               key=lambda kv: -kv[1])[:5]
                for i, p in enumerate(POSITIONS)}
        if self.pair_matrix is not None:
            flat = np.argsort(-self.pair_matrix, axis=None)[:5]
            d["top_pairs"] = [
                {"site_%d" % POSITIONS[self.pair[0]]: AA[k // 20],
                 "site_%d" % POSITIONS[self.pair[1]]: AA[k % 20],
                 "score": round(float(self.pair_matrix[k // 20, k % 20]), 2)} for k in flat]
        return d


class EvidencePrior:
    DEFAULT_WEIGHTS = {
        "mpnn": 1.0,
        "packing_headroom": 1.0,
        "burial_charge": 0.6,
        "interface_bonus": 0.8,
        "pair_volume_compensation": 0.8,
        "literature": 1.0,
        # off by default: tested symmetric and over-packing-asymmetric, both rank the
        # landscape worse than omitting the term (rho 0.227 / 0.246 vs 0.258)
        "burial_volume": 0.0,
    }

    # zero-shot folding scores are not trusted at the interface position
    ZERO_SHOT_MUTED_SITES = (40,)

    def __init__(self, structure_evidence: dict, mpnn_log_odds: dict | None = None,
                 evidence_base: dict | None = None, weights: dict | None = None,
                 allow_dms_derived: bool = False,
                 volume_asymmetry: tuple[float, float] = (1.0, 0.25)):
        self.structure = structure_evidence
        self.mpnn = mpnn_log_odds
        self.evidence_base = evidence_base or {"evidence": []}
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.allow_dms_derived = allow_dms_derived
        self.volume_asymmetry = volume_asymmetry
        self.components: list[PriorComponent] = []
        self._build()

    # ---------------------------------------------------------------- helpers
    def _rows(self):
        return {int(k): v for k, v in self.structure["sites"].items()}

    def _volume(self):
        return self.structure["amino_acid_properties"]["volume"]

    def _headroom_severity(self, site: dict) -> float:
        """0 where a larger side chain fits, 1 where the C-beta already clashes."""
        hr = site.get("packing_headroom") or {}
        free = hr.get("free_radius")
        if free is None:
            return 0.0
        return float(np.clip((3.5 - free) / 1.5, 0.0, 1.0))

    # ------------------------------------------------------------- components
    def _component_mpnn(self):
        if not self.mpnn or self.weights.get("mpnn", 0) == 0:
            return
        table = np.zeros((4, 20))
        for i, row in enumerate(self.mpnn["log_odds"]):
            if POSITIONS[i] in self.ZERO_SHOT_MUTED_SITES:
                continue
            for a, v in row.items():
                table[i, AA_INDEX[a]] = v
        self.components.append(PriorComponent(
            "mpnn", self.weights["mpnn"], _z(table),
            f"ProteinMPNN {self.mpnn.get('model', 'soluble v_48_002')} log-odds on "
            f"{self.mpnn.get('structure', '1PGA/2GB1')}; muted at site "
            f"{self.ZERO_SHOT_MUTED_SITES} because zero-shot scores mis-rank that position",
            ["EV053", "EV049", "EV057"]))

    def _component_packing_headroom(self):
        if self.weights.get("packing_headroom", 0) == 0:
            return
        vol = self._volume()
        rows = self._rows()
        table = np.zeros((4, 20))
        severities = {}
        for i, p in enumerate(POSITIONS):
            sev = self._headroom_severity(rows[p])
            severities[p] = round(sev, 2)
            if sev == 0:
                continue
            wt_vol = vol[rows[p]["wild_type"]]
            for a in AA:
                added = max(0.0, vol[a] - wt_vol)
                table[i, AA_INDEX[a]] = -sev * (added / 40.0)
        self.components.append(PriorComponent(
            "packing_headroom", self.weights["packing_headroom"], _z(table),
            "penalty for added side-chain volume, scaled by the clash severity of the "
            f"reconstructed C-beta in 1PGA (per-site severity {severities})",
            ["EV031", "EV033", "EV004"]))

    def _component_burial_charge(self):
        if self.weights.get("burial_charge", 0) == 0:
            return
        charge = self.structure["amino_acid_properties"]["charge"]
        rows = self._rows()
        table = np.zeros((4, 20))
        for i, p in enumerate(POSITIONS):
            severity = max(0.0, 1.0 - rows[p]["rel_sasa"] / 0.25)
            for a in AA:
                table[i, AA_INDEX[a]] = -severity * abs(charge[a])
        self.components.append(PriorComponent(
            "burial_charge", self.weights["burial_charge"], _z(table),
            "desolvation penalty for a formal charge, scaled by relative SASA in 1PGA",
            ["EV028", "EV031"]))

    def _component_interface_bonus(self):
        if self.weights.get("interface_bonus", 0) == 0:
            return
        rows = self._rows()
        table = np.zeros((4, 20))
        buried = {}
        for i, p in enumerate(POSITIONS):
            area = rows[p]["interface"].get("buried_surface_on_binding_A2") or 0.0
            buried[p] = area
            if area < 10.0:
                continue
            scale = min(1.0, area / 40.0)
            for a in LARGE_HYDROPHOBIC_AROMATIC:
                table[i, AA_INDEX[a]] = scale
            table[i, AA_INDEX["W"]] = scale * 1.3
        self.components.append(PriorComponent(
            "interface_bonus", self.weights["interface_bonus"], _z(table),
            f"bonus for large/aromatic side chains at positions that bury surface against "
            f"IgG-Fc in 1FCC (buried area per site: {buried})",
            ["EV034", "EV037"]))

    def _component_burial_volume(self):
        if self.weights.get("burial_volume", 0) == 0:
            return
        vol = self._volume()
        rows = self._rows()
        over, under = self.volume_asymmetry
        table = np.zeros((4, 20))
        for i, p in enumerate(POSITIONS):
            severity = max(0.0, 1.0 - rows[p]["rel_sasa"] / 0.25)
            wt_vol = vol[rows[p]["wild_type"]]
            for a in AA:
                dv = vol[a] - wt_vol
                coef = over if dv > 0 else under
                table[i, AA_INDEX[a]] = -severity * coef * (dv / 40.0) ** 2
        self.components.append(PriorComponent(
            "burial_volume", self.weights["burial_volume"], _z(table),
            f"symmetric/asymmetric volume term (weights {self.volume_asymmetry}); retained "
            "only for the ablation that shows it does not help"))

    def _component_pair_volume_compensation(self):
        if self.weights.get("pair_volume_compensation", 0) == 0:
            return
        if 41 not in POSITIONS or 54 not in POSITIONS:
            return
        vol = self._volume()
        i41, i54 = POSITIONS.index(41), POSITIONS.index(54)
        table = np.zeros((20, 20))
        for a in AA:
            d41 = vol[a] - vol[WT[i41]]
            for b in AA:
                d54 = vol[b] - vol[WT[i54]]
                # reward transfers: volume added at 41 removed at 54, and vice versa
                transfer = min(abs(d41), abs(d54)) if d41 * d54 < 0 else 0.0
                imbalance = abs(d41 + d54)
                table[AA_INDEX[a], AA_INDEX[b]] = transfer / 60.0 - imbalance / 200.0
        table = (table - table.mean()) / (table.std() + 1e-9)
        self.components.append(PriorComponent(
            "pair_volume_compensation", self.weights["pair_volume_compensation"], None,
            "volume transfer between sites 41 and 54: the strongest published positive "
            "epistasis (G41L/V54G) works by moving side-chain volume from the buried core "
            "to the loop, and V54A flips site-41 substitutions from deleterious to beneficial",
            ["EV020", "EV019", "EV007"], pair=(i41, i54), pair_matrix=table))

    def _component_literature(self):
        if self.weights.get("literature", 0) == 0:
            return
        table = np.zeros((4, 20))
        used = []
        for row in self.evidence_base.get("evidence", []):
            if row.get("scope") not in ("site", "pair"):
                continue
            if row.get("dms_derived") and not self.allow_dms_derived:
                continue
            sign = {"beneficial": 1.0, "tolerant": 0.3, "deleterious": -1.0}.get(row.get("direction"))
            if sign is None:
                continue
            conf = float(row.get("confidence", 0.5))
            for sub in row.get("substitutions", []) or []:
                digits = "".join(ch for ch in sub if ch.isdigit())
                if not digits:
                    continue
                pos, aa = int(digits), sub[-1]
                if pos not in POSITIONS or aa not in AA_INDEX:
                    continue
                table[POSITIONS.index(pos), AA_INDEX[aa]] += sign * conf
                used.append(row.get("id"))
        if used:
            self.components.append(PriorComponent(
                "literature", self.weights["literature"], _z(table),
                "per-substitution claims from the curated evidence base, weighted by confidence"
                + ("" if self.allow_dms_derived else "; rows derived from the oracle DMS excluded"),
                sorted(set(used))))

    def _build(self):
        self._component_mpnn()
        self._component_packing_headroom()
        self._component_burial_charge()
        self._component_interface_bonus()
        self._component_burial_volume()
        self._component_pair_volume_compensation()
        self._component_literature()

    # ------------------------------------------------------------------ scoring
    @property
    def table(self) -> np.ndarray:
        total = np.zeros((4, 20))
        for comp in self.components:
            if comp.matrix is not None:
                total += comp.weight * comp.matrix
        return total

    @property
    def pair_terms(self) -> list:
        return [c for c in self.components if c.pair_matrix is not None]

    def score_variants(self, variants) -> np.ndarray:
        table = self.table
        idx = np.array([[AA_INDEX[v[i]] for i in range(4)] for v in variants])
        score = table[np.arange(4)[None, :], idx].sum(axis=1)
        for comp in self.pair_terms:
            i, j = comp.pair
            score = score + comp.weight * comp.pair_matrix[idx[:, i], idx[:, j]]
        return score

    def attribution(self, variant: str) -> dict:
        out = {}
        idx = [AA_INDEX[variant[i]] for i in range(4)]
        for comp in self.components:
            if comp.matrix is not None:
                out[comp.name] = round(float(sum(comp.weight * comp.matrix[i, idx[i]]
                                                 for i in range(4))), 3)
            else:
                i, j = comp.pair
                out[comp.name] = round(float(comp.weight * comp.pair_matrix[idx[i], idx[j]]), 3)
        out["total"] = round(float(sum(out.values())), 3)
        return out

    def to_dict(self) -> dict:
        return {"weights": self.weights, "allow_dms_derived": self.allow_dms_derived,
                "zero_shot_muted_sites": list(self.ZERO_SHOT_MUTED_SITES),
                "components": [c.to_dict() for c in self.components],
                "table": {str(p): {AA[j]: round(float(self.table[i, j]), 3) for j in range(20)}
                          for i, p in enumerate(POSITIONS)}}


def load(structure_path="knowledge/structure_evidence.json",
         mpnn_path="config/mpnn_log_odds.json",
         evidence_path="knowledge/evidence_base.json", **kwargs) -> EvidencePrior:
    structure = json.loads(Path(structure_path).read_text())
    mpnn = json.loads(Path(mpnn_path).read_text()) if Path(mpnn_path).exists() else None
    base = json.loads(Path(evidence_path).read_text()) if Path(evidence_path).exists() else None
    return EvidencePrior(structure, mpnn, base, **kwargs)
