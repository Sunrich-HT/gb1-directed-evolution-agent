"""An explicit, auditable constraint layer over batch selection.

Every design decision that is not "the model ranked it highly" is written here as
a named constraint with a parameter, an evidence source and a type:

* ``filter``  removes candidates from the legal pool (hard);
* ``penalty`` subtracts from a candidate's score (soft);
* ``quota``   shapes the composition of the chosen batch (diversity, exploration).

Each round the engine returns an audit record: how many candidates each filter
removed, how much each penalty moved the ranking, whether each quota bound, and
which picks differ from the unconstrained ranking.  That record -- not a
narrative -- is what makes the campaign interpretable.

Constraints are data, so an agent can propose changes to them between rounds
(``ConstraintSet.apply_updates``) and the change is logged with its reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = (39, 40, 41, 54)
WT = "VDGV"


@dataclass
class Constraint:
    id: str
    kind: str                     # "filter" | "penalty" | "quota"
    description: str
    params: dict
    evidence: str = ""            # provenance: evidence id, structural fact, or feedback round
    active: bool = True
    origin: str = "initial"       # "initial" | "agent_round_k" | "feedback"

    def to_dict(self) -> dict:
        return asdict(self)


def default_constraints(structure_evidence: dict) -> "ConstraintSet":
    """Constraints that follow from the structural evidence, before any label."""
    sites = {int(k): v for k, v in structure_evidence["sites"].items()}
    buried = [p for p in POSITIONS if sites[p]["rel_sasa"] < 0.15]
    # "no room to grow": the reconstructed C-beta is already close to a neighbouring
    # heavy atom, so any added side-chain volume strains the backbone
    tight = [p for p in POSITIONS
             if ((sites[p].get("packing_headroom") or {}).get("free_radius") or 9.9) < 3.0]
    # only positions that actually bury surface against IgG-Fc count as interface
    interface = [p for p in POSITIONS
                 if (sites[p]["interface"].get("buried_surface_on_binding_A2") or 0.0) >= 10.0]

    cs = [
        Constraint("C1_no_remeasure", "filter",
                   "a variant whose label is already known is never selected again",
                   {}, "campaign protocol"),
        Constraint("C2_volume_window", "filter",
                   "where the backbone leaves no room, added side-chain volume is capped",
                   {"sites": tight, "max_added_volume": 60.0},
                   f"reconstructed C-beta within 3.0 A of a neighbouring heavy atom at "
                   f"sites {tight} (1PGA); a cap, not a ban, because the published "
                   f"G41L/V54G pair shows the strain can be compensated elsewhere"),
        Constraint("C3_buried_charge", "penalty",
                   "burying a formal charge costs score",
                   {"sites": buried, "weight": 0.8},
                   f"buried sites {buried}; desolvation penalty for charged side chains"),
        # The "glycine is required at 41" rule that an earlier version of this code
        # enforced is NOT supported by the crystal structure: phi/psi there are
        # -137.6/-159.8 in 1PGA, an ordinary beta conformation.  The real constraint at
        # 41 is side-chain packing, which C2 covers and the prior scores.  The residue
        # that genuinely needs glycine in this loop is G38, which is not mutable here.
        Constraint("C4_charge_pair_balance", "penalty",
                   "two new buried charges in one variant are penalised together",
                   {"sites": buried, "weight": 0.5},
                   "desolvation cost is super-additive when several charges are buried"),
        Constraint("C5_batch_diversity", "quota",
                   "each site shows at least k distinct amino acids in the batch",
                   {"min_distinct_aa_per_site": 6, "max_fraction_same_aa_per_site": 0.5},
                   "exploration requirement: a batch that collapses onto one motif "
                   "cannot inform the next model"),
        Constraint("C6_exploration_quota", "quota",
                   "a fixed share of the batch comes from outside the top of the ranking",
                   {"fraction": 0.25},
                   "epistatic landscape: the ranking is trained on few labels and is "
                   "unreliable in the tail"),
        Constraint("C7_interface_contact", "penalty",
                   "candidates that remove every large side chain from the Fc contact "
                   "positions are penalised",
                   {"sites": interface, "weight": 0.0},
                   f"sites burying >= 10 A^2 against IgG-Fc in 1FCC: {interface} "
                   f"(off by default; the agent may switch it on)"),
        Constraint("C8_refuted_substitutions", "penalty",
                   "substitutions whose registered hypothesis was refuted lose score",
                   {"substitutions": {}, "weight": 1.0},
                   "set from the program's verdict on the previous round's hypotheses"),
    ]
    return ConstraintSet(cs, structure_evidence)


class ConstraintSet:
    def __init__(self, constraints: list[Constraint], structure_evidence: dict):
        self.constraints = {c.id: c for c in constraints}
        self.structure = structure_evidence
        self.history: list[dict] = []

    # ------------------------------------------------------------------ updates
    def apply_updates(self, updates: list[dict], round_: int) -> list[dict]:
        """Apply agent-proposed parameter changes.  Returns the accepted log."""
        accepted = []
        for upd in updates or []:
            cid = upd.get("id")
            c = self.constraints.get(cid)
            if c is None:
                continue
            before = json.loads(json.dumps(c.params))
            if "params" in upd and isinstance(upd["params"], dict):
                for k, v in upd["params"].items():
                    if k in c.params or k in ("substitutions", "weight", "fraction"):
                        c.params[k] = v
            if "active" in upd:
                c.active = bool(upd["active"])
            c.origin = f"agent_round_{round_}"
            entry = {"round": round_, "id": cid, "before": before,
                     "after": json.loads(json.dumps(c.params)), "active": c.active,
                     "reason": str(upd.get("reason", ""))[:300]}
            self.history.append(entry)
            accepted.append(entry)
        return accepted

    def set_refuted(self, substitutions: dict, round_: int) -> None:
        c = self.constraints["C8_refuted_substitutions"]
        c.params["substitutions"] = substitutions
        c.origin = f"feedback_round_{round_}"
        self.history.append({"round": round_, "id": c.id, "after": {"substitutions": substitutions},
                             "reason": "program verdict on registered hypotheses"})

    # ------------------------------------------------------------------- filters
    def legal_mask(self, variants: np.ndarray, known: set) -> tuple[np.ndarray, dict]:
        vol = self.structure["amino_acid_properties"]["volume"]
        mask = np.ones(len(variants), bool)
        audit = {}

        c = self.constraints["C1_no_remeasure"]
        if c.active:
            keep = np.array([v not in known for v in variants])
            audit[c.id] = int((~keep & mask).sum())
            mask &= keep

        c = self.constraints["C2_volume_window"]
        if c.active and c.params.get("sites"):
            limit = float(c.params.get("max_added_volume", c.params.get("max_abs_delta_volume", 60.0)))
            idx = [POSITIONS.index(p) for p in c.params["sites"]]
            keep = np.array([
                all(vol[v[i]] - vol[WT[i]] <= limit for i in idx) for v in variants])
            audit[c.id] = int((~keep & mask).sum())
            mask &= keep

        return mask, audit

    # ------------------------------------------------------------------ penalties
    def penalty(self, variants: np.ndarray) -> tuple[np.ndarray, dict]:
        charge = self.structure["amino_acid_properties"]["charge"]
        pen = np.zeros(len(variants))
        parts = {}

        c = self.constraints["C3_buried_charge"]
        if c.active and c.params.get("sites"):
            idx = [POSITIONS.index(p) for p in c.params["sites"]]
            w = float(c.params.get("weight", 0.0))
            term = np.array([w * sum(abs(charge[v[i]]) for i in idx) for v in variants])
            pen += term
            parts[c.id] = float(term.mean())

        c = self.constraints["C4_charge_pair_balance"]
        if c.active and c.params.get("sites"):
            idx = [POSITIONS.index(p) for p in c.params["sites"]]
            w = float(c.params.get("weight", 0.0))
            term = np.array([w * max(0, sum(1 for i in idx if abs(charge[v[i]]) >= 1) - 1)
                             for v in variants])
            pen += term
            parts[c.id] = float(term.mean())

        c = self.constraints["C7_interface_contact"]
        if c.active and float(c.params.get("weight", 0.0)) > 0 and c.params.get("sites"):
            idx = [POSITIONS.index(p) for p in c.params["sites"]]
            w = float(c.params["weight"])
            term = np.array([w * sum(1.0 for i in idx if v[i] in "AGS") for v in variants])
            pen += term
            parts[c.id] = float(term.mean())

        c = self.constraints["C8_refuted_substitutions"]
        subs = c.params.get("substitutions") or {}
        if c.active and subs:
            w = float(c.params.get("weight", 1.0))
            term = np.zeros(len(variants))
            for sub, strength in subs.items():
                try:
                    pos = int("".join(ch for ch in sub if ch.isdigit()))
                    aa = sub[-1]
                    i = POSITIONS.index(pos)
                except (ValueError, KeyError):
                    continue
                term += w * float(strength) * np.array([1.0 if v[i] == aa else 0.0 for v in variants])
            pen += term
            parts[c.id] = float(term.mean())

        return pen, parts

    # --------------------------------------------------------------------- quota
    def select(self, variants: np.ndarray, score: np.ndarray, batch_size: int,
               rng: np.random.Generator, known: set) -> tuple[np.ndarray, dict]:
        """Rank by constrained score, then enforce the composition quotas."""
        mask, filter_audit = self.legal_mask(variants, known)
        legal = np.flatnonzero(mask)
        pen, penalty_parts = self.penalty(variants)
        constrained = score - pen

        unconstrained_pick = set(np.argsort(-score)[:batch_size].tolist())

        order = legal[np.argsort(-constrained[legal])]
        picks: list[int] = []

        c_div = self.constraints["C5_batch_diversity"]
        c_exp = self.constraints["C6_exploration_quota"]
        explore_n = int(round(batch_size * float(c_exp.params.get("fraction", 0.0)))) \
            if c_exp.active else 0
        exploit_n = batch_size - explore_n

        max_per_site = (int(batch_size * float(c_div.params.get("max_fraction_same_aa_per_site", 1.0)))
                        if c_div.active else batch_size)
        counts = [{} for _ in range(4)]

        def fits(j: int) -> bool:
            v = variants[j]
            return all(counts[i].get(v[i], 0) < max_per_site for i in range(4))

        quota_blocks = 0
        for j in order:
            if len(picks) >= exploit_n:
                break
            if fits(j):
                picks.append(int(j))
                for i in range(4):
                    counts[i][variants[j][i]] = counts[i].get(variants[j][i], 0) + 1
            else:
                quota_blocks += 1

        # exploration draws from the legal pool outside the exploitation head
        head = set(order[: max(exploit_n * 3, batch_size)].tolist())
        tail = np.array([j for j in legal if j not in head and j not in picks])
        if explore_n and len(tail):
            chosen = rng.choice(tail, size=min(explore_n, len(tail)), replace=False)
            picks.extend(int(j) for j in chosen)

        # top up strictly by constrained rank if a quota left the batch short
        if len(picks) < batch_size:
            for j in order:
                if len(picks) >= batch_size:
                    break
                if int(j) not in picks:
                    picks.append(int(j))

        picked = np.array(picks[:batch_size])
        distinct = [len({variants[j][i] for j in picked}) for i in range(4)]
        audit = {
            "n_pool": int(len(variants)),
            "n_legal_after_filters": int(mask.sum()),
            "filters_removed": filter_audit,
            "penalty_mean_contribution": {k: round(v, 4) for k, v in penalty_parts.items()},
            "exploration_quota": explore_n,
            "diversity_blocks": quota_blocks,
            "distinct_aa_per_site": distinct,
            "diversity_satisfied": bool(
                not c_div.active or min(distinct) >= int(c_div.params.get("min_distinct_aa_per_site", 0))),
            "changed_vs_unconstrained": int(len(set(picked.tolist()) - unconstrained_pick)),
            "active_constraints": [cid for cid, c in self.constraints.items() if c.active],
        }
        return picked, audit

    def to_dict(self) -> dict:
        return {"constraints": [c.to_dict() for c in self.constraints.values()],
                "update_history": self.history}
