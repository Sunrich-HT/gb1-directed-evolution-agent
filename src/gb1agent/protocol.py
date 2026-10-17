"""Round protocol for the evidence-prior + constraint arm.

Differences from the plain LLM arm, and the reason for each:

round 1   The batch is chosen from ``w_prior * prior + w_model * model`` under the
          structural constraints, not from the model ranking alone.  Before any
          multi-mutant label exists the model has seen 192 random variants; the
          evidence prior is the only information in the loop that did not come
          from those labels.

round k>1 The agent reads: the realised fitness of its own picks, the verdict the
          program passed on each of its hypotheses, the per-site statistics of
          what worked, and the constraint audit.  It replies with (a) new
          hypotheses, (b) explicit constraint updates with reasons, (c) the
          weights for the next round's score.  The program applies the updates,
          re-selects under them, and records what changed.

The agent never names individual variants here: it sets the *rules*, and the
rules pick the batch.  That is what makes every pick attributable.
"""
from __future__ import annotations

import json

import numpy as np

from . import agent as agent_mod
from .constraints import POSITIONS, WT

SYSTEM_PROMPT = (
    "You are a protein engineer directing a machine-learning-guided directed-evolution "
    "campaign on the GB1 four-site landscape (positions 39, 40, 41, 54; wild type VDGV). "
    "Measurements are historical values from a published deep mutational scan: they are data, "
    "not experiments you are performing. You do not pick variants by name. You govern the "
    "campaign by setting (1) falsifiable hypotheses, (2) explicit constraints on which "
    "candidates are admissible and how the batch must be composed, and (3) the weights that "
    "balance the evidence prior against the fitted model. A program then selects the batch "
    "under your rules, reveals the labels, and judges your hypotheses. Reply with JSON only."
)


def site_statistics(variants, fitness) -> dict:
    """Per-site, per-amino-acid mean fitness among the labelled set -- the evidence the
    agent reasons over, computed by the program so it cannot be misquoted."""
    out = {}
    for i, pos in enumerate(POSITIONS):
        rows = {}
        for aa in sorted({v[i] for v in variants}):
            sel = np.array([v[i] == aa for v in variants])
            if sel.sum() >= 3:
                rows[aa] = {"n": int(sel.sum()), "mean": round(float(np.mean(fitness[sel])), 3),
                            "best": round(float(np.max(fitness[sel])), 3)}
        out[str(pos)] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["mean"])[:8])
    return out


def build_round_message(round_: int, prior_summary: dict, constraint_set, known_summary: dict,
                        site_stats: dict, feedback: dict | None, audit: dict | None) -> dict:
    return {
        "round": round_,
        "task": ("Set the rules for this round: hypotheses, constraint updates, and the "
                 "prior/model weights. The program selects the batch under them."),
        "evidence_prior": prior_summary,
        "current_constraints": [
            {"id": c.id, "kind": c.kind, "description": c.description, "params": c.params,
             "evidence": c.evidence, "active": c.active}
            for c in constraint_set.constraints.values()],
        "labelled_data_summary": known_summary,
        "per_site_statistics_of_labelled_data": site_stats,
        "previous_round_feedback": feedback,
        "previous_round_constraint_audit": audit,
        "required_output": {
            "hypotheses": "3 to 4 items of {statement, site, substitution, direction}; "
                          "each statement at most 25 words and testable by the next batch",
            "constraint_updates": "list of {id, params, active, reason}; only ids listed in "
                                  "current_constraints; reason at most 25 words; [] is allowed",
            "weights": "{prior: float in [0,2], model: float in [0,2]} for this round's score",
            "rationale": "at most 60 words on why these rules follow from the evidence so far",
        },
    }


def parse_round_response(text: str) -> tuple[list, list, dict, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        text = text[4:] if text.lower().startswith("json") else text
    obj = json.loads(text)
    hypotheses = []
    for h in obj.get("hypotheses", []) or []:
        hypotheses.append(agent_mod.Hypothesis(
            statement=str(h.get("statement", ""))[:300],
            site=agent_mod._as_site(h.get("site")),
            substitution=(str(h["substitution"]).strip()[-1]
                          if h.get("substitution") else None),
            direction=agent_mod._as_direction(h.get("direction")),
        ))
    updates = [u for u in (obj.get("constraint_updates") or []) if isinstance(u, dict)]
    w = obj.get("weights") or {}
    weights = {"prior": float(np.clip(float(w.get("prior", 1.0)), 0.0, 2.0)),
               "model": float(np.clip(float(w.get("model", 1.0)), 0.0, 2.0))}
    return hypotheses, updates, weights, str(obj.get("rationale", ""))[:500]


def refuted_substitution_map(hypotheses) -> dict:
    """Turn the program's verdicts into the parameters of constraint C8."""
    out = {}
    for h in hypotheses:
        site = h["site"] if isinstance(h, dict) else h.site
        sub = h["substitution"] if isinstance(h, dict) else h.substitution
        verdict = h["verdict"] if isinstance(h, dict) else h.verdict
        direction = h["direction"] if isinstance(h, dict) else h.direction
        if not sub or site not in POSITIONS:
            continue
        key = f"{WT[POSITIONS.index(site)]}{site}{sub}"
        if verdict == "refuted" and direction == "beneficial":
            out[key] = 1.0
        elif verdict == "supported" and direction == "beneficial":
            out[key] = -0.5          # a supported claim becomes a bonus
    return out
