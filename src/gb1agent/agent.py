"""The LLM agent: shortlist tools, hypothesis registration, batch selection.

Design notes
------------
* Everything the agent outputs is frozen (hashed) before any label is revealed.
* The agent must register falsifiable site-level hypotheses each round.  After
  the labels come back the *program*, not the model, decides whether each
  hypothesis was supported, refuted or left undetermined, so "the agent learned
  from feedback" is an auditable claim rather than a narrative one.
* The knowledge-augmented arm differs from the plain arm by exactly one thing:
  the knowledge block in the prompt.  Everything else -- tools, shortlist,
  batch size, feedback -- is identical, which is what makes the contrast a
  controlled ablation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .data import commit_digest
from .features import AA

POSITIONS = (39, 40, 41, 54)
WT = "VDGV"

SYSTEM_PROMPT = (
    "You are a protein engineer running machine-learning-guided directed evolution on the "
    "GB1 four-site landscape (positions 39, 40, 41, 54; wild type VDGV). Measurements are "
    "historical IgG-Fc binding enrichment values from a published deep mutational scan: they "
    "are data, not experiments you are performing. Your job each round is to (1) state "
    "falsifiable hypotheses about which substitutions help at which site, (2) choose exactly "
    "the requested number of variants from the provided shortlist, balancing exploitation of "
    "the model ranking against exploration that will inform the next model, and (3) explain "
    "your top choices. Reply with JSON only."
)

KNOWLEDGE_BLOCK = {
    "structural_context": [
        "GB1 is a 56-residue IgG-Fc binding domain with a four-strand beta-sheet packed "
        "against one alpha-helix; positions 39, 40, 41 sit in the beta3-beta4 turn region and "
        "54 on the last strand.",
        "Positions 39/41 pack against the hydrophobic core side; 40 is the most solvent-exposed "
        "of the four and tolerates the widest range of substitutions.",
    ],
    "biophysical_rules": [
        "Charged side chains change the local electrostatic environment in a context-dependent "
        "way; the sign of the effect is not fixed a priori.",
        "Side-chain volume and hydrophobicity affect packing; a backbone-compatible substitution "
        "is not automatically a binding improvement.",
        "Glycine and proline change backbone flexibility and are frequent sources of epistasis.",
        "A substitution that is deleterious on its own can be beneficial in combination: do not "
        "discard a substitution only because its single-mutant value is low.",
    ],
    "zero_shot_note": (
        "structure_prior is a ProteinMPNN inverse-folding log-odds score (soluble v_48_002) on "
        "the 2GB1 backbone: log p(substitution | backbone) - log p(wild type | backbone). It "
        "reflects how compatible a residue is with the fold, which is NOT the same thing as "
        "binding to IgG-Fc; on this landscape it correlates only weakly with measured fitness."
    ),
}


@dataclass
class Hypothesis:
    statement: str
    site: int
    substitution: str | None = None
    direction: str = "unknown"      # "beneficial" | "deleterious" | "unknown"
    round_: int = 0
    verdict: str = "pending"        # set by the program after labels are revealed
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "statement": self.statement, "site": self.site, "substitution": self.substitution,
            "direction": self.direction, "round": self.round_, "verdict": self.verdict,
            "evidence": self.evidence,
        }


def build_shortlist(variants, predictions, known_variants, prior_scores=None,
                    n_exploit: int = 180, n_diverse: int = 60, rng=None) -> list[dict]:
    """Model-ranked head plus a diverse tail, so the agent can actually explore."""
    order = np.argsort(-predictions)
    head = order[:n_exploit]
    rest = order[n_exploit:]
    if rng is not None and rest.size:
        tail = rng.choice(rest, size=min(n_diverse, rest.size), replace=False)
    else:
        tail = rest[:n_diverse]
    chosen = np.concatenate([head, tail])
    known_set = set(known_variants)
    rows = []
    for i in chosen:
        v = variants[i]
        row = {
            "variant": str(v),
            "mutations": "/".join(
                f"{WT[k]}{POSITIONS[k]}{v[k]}" for k in range(4) if v[k] != WT[k]
            ) or "WT",
            "predicted": round(float(predictions[i]), 3),
            "source": "model_top" if i in set(head.tolist()) else "diverse",
        }
        if prior_scores is not None:
            row["structure_prior"] = round(float(prior_scores[i]), 3)
        rows.append(row)
    return rows


def neighbour_summary(variant: str, known_variants, known_fitness, k: int = 3) -> list[dict]:
    dists = [sum(a != b for a, b in zip(variant, kv)) for kv in known_variants]
    order = np.argsort(dists)[:k]
    return [
        {"variant": str(known_variants[i]), "fitness": round(float(known_fitness[i]), 3),
         "hamming": int(dists[i])}
        for i in order
    ]


COMPACT_RETRY_NOTE = (
    "Your previous reply was cut off at the output-token limit and could not be parsed. "
    "Reply again, same task, but minimally: 3 hypotheses with statements under 15 words, "
    "the full selection array, and an empty reasons object."
)


def build_user_message(shortlist, batch_size, known_summary, feedback, knowledge: bool) -> dict:
    payload = {
        "batch_size": batch_size,
        "known_data_summary": known_summary,
        "shortlist": shortlist,
        "previous_round_feedback": feedback,
        "required_output": {
            "hypotheses": "3 to 4 items of {statement, site, substitution, direction}; "
                          "each statement at most 25 words",
            "selection": f"exactly {batch_size} variant ids taken from the shortlist, "
                         "as a flat JSON array of strings and nothing else",
            "reasons": "mapping from variant id to a reason of at most 15 words, for 5 picks",
            "output_limit": "return these three keys only; no commentary, no per-variant "
                            "annotations beyond the five reasons",
        },
    }
    if knowledge:
        payload["knowledge"] = KNOWLEDGE_BLOCK
    return payload


_DIRECTION = {
    "beneficial": "beneficial", "positive": "beneficial", "up": "beneficial",
    "increase": "beneficial", "improves": "beneficial", "higher": "beneficial",
    "deleterious": "deleterious", "negative": "deleterious", "down": "deleterious",
    "decrease": "deleterious", "harmful": "deleterious", "lower": "deleterious",
}


def _as_site(value) -> int:
    """Models return 39, "39", "V39M" or even "39/40/41/54"; accept the
    unambiguous cases and treat the rest as untestable."""
    if isinstance(value, (int, float)):
        return int(value)
    digits = re.findall(r"\d+", str(value))
    unique = {int(d) for d in digits if int(d) in POSITIONS}
    return unique.pop() if len(unique) == 1 else 0


def _as_direction(value) -> str:
    return _DIRECTION.get(str(value).strip().lower(), "unknown")


def parse_response(text: str, legal: set[str], batch_size: int) -> tuple[list[str], list[Hypothesis], dict]:
    """Strict parse; raises on anything that would silently corrupt the protocol."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    obj = json.loads(text)
    selection = [str(v).strip().upper() for v in obj.get("selection", [])]
    seen, unique = set(), []
    for v in selection:
        if v in legal and v not in seen:
            seen.add(v)
            unique.append(v)
    hypotheses = []
    for h in obj.get("hypotheses", []) or []:
        if not isinstance(h, dict):
            continue
        hypotheses.append(
            Hypothesis(
                statement=str(h.get("statement", ""))[:400],
                site=_as_site(h.get("site", 0)),
                substitution=(str(h["substitution"]) if h.get("substitution") else None),
                direction=_as_direction(h.get("direction", "unknown")),
            )
        )
    return unique[:batch_size], hypotheses, obj.get("reasons", {}) or {}


def judge_hypotheses(hypotheses, selected_variants, selected_fitness, incumbent: float):
    """Program-side verdicts: the model does not get to grade its own homework.

    A hypothesis naming a substitution is supported when variants carrying it do
    better than the rest of the same batch in the stated direction, refuted when
    they do worse, and undetermined when the batch contains too few of them.
    """
    xs = np.asarray(selected_fitness, float)
    for h in hypotheses:
        if not h.substitution or h.site not in POSITIONS:
            h.verdict = "undetermined"
            h.evidence = {"reason": "no testable substitution named"}
            continue
        site_idx = POSITIONS.index(h.site)
        target = h.substitution.strip().upper()[-1]
        mask = np.array([v[site_idx] == target for v in selected_variants])
        n_with = int(mask.sum())
        if n_with < 3 or n_with == mask.size:
            h.verdict = "undetermined"
            h.evidence = {"n_with_substitution": n_with, "reason": "insufficient contrast in batch"}
            continue
        with_mean, without_mean = float(xs[mask].mean()), float(xs[~mask].mean())
        delta = with_mean - without_mean
        expected_sign = {"beneficial": 1, "deleterious": -1}.get(h.direction, 0)
        if expected_sign == 0:
            h.verdict = "undetermined"
        elif np.sign(delta) == expected_sign and abs(delta) > 0.05:
            h.verdict = "supported"
        elif np.sign(delta) == -expected_sign and abs(delta) > 0.05:
            h.verdict = "refuted"
        else:
            h.verdict = "undetermined"
        h.evidence = {
            "n_with_substitution": n_with, "mean_with": round(with_mean, 4),
            "mean_without": round(without_mean, 4), "delta": round(delta, 4),
        }
    return hypotheses


def freeze(selection, hypotheses) -> str:
    return commit_digest(list(selection) + [h.statement for h in hypotheses])
