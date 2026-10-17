"""The directed-evolution loop: five arms, identical start, identical budget.

Arms
----
random          uniform draw from the legal pool (control)
greedy          predictor's top-k (the "predictor recommends directly" baseline)
greedy_explore  75% predictor top-k + 25% uniform exploration (no LLM) -- isolates
                the effect of exploration from the effect of the language model
llm_agent       LLM picks the batch from a model-ranked shortlist
llm_knowledge   same, plus a knowledge block and the zero-shot prior column
prior_constraint
                evidence-prior + constraint arm: round 1 is chosen from the
                literature/structure prior under structural constraints, and in
                later rounds the agent edits the constraint set and the
                prior/model weights rather than naming variants

Every arm starts from the same D0, spends the same number of labels per round,
and never sees the frozen test set.  Labels are revealed only after the batch
and the hypotheses are hashed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import agent as agent_mod
from . import protocol as protocol_mod
from .constraints import default_constraints
from .llm import TruncatedResponse
from .data import Oracle, commit_digest
from .metrics import batch_diversity, campaign_metrics, regression_metrics, top_percentile_precision, topk_recall
from .predictors import FitnessModel, ModelSpec

ARMS = ("random", "greedy", "greedy_explore", "llm_agent", "llm_knowledge",
        "prior_fixed", "prior_constraint")
LLM_ARMS = ("llm_agent", "llm_knowledge")
# arms driven by the evidence prior and the constraint engine; "prior_fixed" runs the
# same machinery with frozen rules and no language model, which isolates what the agent
# contributes on top of the evidence itself
RULE_ARMS = ("prior_fixed", "prior_constraint")


@dataclass
class RoundRecord:
    arm: str
    seed: int
    round: int
    commit_sha256: str
    selected: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    diversity: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    hypotheses: list = field(default_factory=list)
    best_so_far: float = 0.0
    n_labels_used: int = 0
    notes: dict = field(default_factory=dict)


def _feedback_block(record: RoundRecord | None) -> dict | None:
    if record is None:
        return None
    rows = sorted(record.selected, key=lambda r: -r["fitness"])[:10]
    return {
        "round": record.round,
        "selected_mean": record.metrics.get("mean"),
        "selected_best": record.metrics.get("best"),
        "incumbent_now": record.best_so_far,
        "top_results": [
            {"variant": r["variant"], "predicted": round(r["predicted"], 3),
             "measured": round(r["fitness"], 3)} for r in rows
        ],
        "hypothesis_verdicts": [
            {"statement": h["statement"], "verdict": h["verdict"], "evidence": h["evidence"]}
            for h in record.hypotheses
        ],
    }


def run_arm(arm, df, split, oracle: Oracle, spec: ModelSpec, *, seed: int, rounds: int,
            batch_size: int, backend=None, prior_scores=None, explore_frac: float = 0.25,
            shortlist_exploit: int = 180, shortlist_diverse: int = 60,
            log_dir: Path | None = None, evidence_prior=None,
            structure_evidence: dict | None = None) -> list[RoundRecord]:
    variants = df["variant"].to_numpy()
    fitness = df["fitness"].to_numpy(float)
    rng = np.random.default_rng(seed * 1000 + ARMS.index(arm))

    known = list(split.initial)
    pool = list(split.pool)
    best_so_far = float(fitness[known].max())
    records: list[RoundRecord] = []
    previous: RoundRecord | None = None

    constraint_set = None
    if arm in RULE_ARMS:
        if structure_evidence is None or evidence_prior is None:
            raise ValueError("the prior_constraint arm needs structure_evidence and evidence_prior")
        constraint_set = default_constraints(structure_evidence)
        prior_all = evidence_prior.score_variants(variants)
        prior_z = (prior_all - prior_all.mean()) / (prior_all.std() + 1e-9)
        round_weights = {"prior": 1.0, "model": 1.0}
        last_audit: dict | None = None

    for round_ in range(1, rounds + 1):
        model = FitnessModel(spec, seed).fit(variants[known], fitness[known])
        pool_arr = np.asarray(pool)
        predictions = model.predict(variants[pool_arr])

        hypotheses: list = []
        notes: dict = {}
        if arm == "random":
            picks_local = rng.choice(len(pool_arr), batch_size, replace=False)
        elif arm == "greedy":
            picks_local = np.argsort(-predictions)[:batch_size]
        elif arm == "greedy_explore":
            n_exploit = int(round(batch_size * (1 - explore_frac)))
            exploit = np.argsort(-predictions)[:n_exploit]
            remaining = np.setdiff1d(np.arange(len(pool_arr)), exploit, assume_unique=False)
            explore = rng.choice(remaining, batch_size - n_exploit, replace=False)
            picks_local = np.concatenate([exploit, explore])
        elif arm in LLM_ARMS:
            use_knowledge = arm == "llm_knowledge"
            prior_pool = prior_scores[pool_arr] if (use_knowledge and prior_scores is not None) else None
            shortlist = agent_mod.build_shortlist(
                variants[pool_arr], predictions, variants[known], prior_scores=prior_pool,
                n_exploit=shortlist_exploit, n_diverse=shortlist_diverse, rng=rng,
            )
            known_summary = {
                "n_known": len(known),
                "best_known_fitness": round(best_so_far, 3),
                "wild_type_fitness": 1.0,
                "top_known": [
                    {"variant": str(variants[i]), "fitness": round(float(fitness[i]), 3)}
                    for i in np.asarray(known)[np.argsort(-fitness[known])[:8]]
                ],
            }
            payload = agent_mod.build_user_message(
                shortlist, batch_size, known_summary, _feedback_block(previous), use_knowledge
            )
            messages = [
                {"role": "system", "content": agent_mod.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            legal = {row["variant"] for row in shortlist}
            try:
                text = backend.chat(messages, tag=f"{arm}_s{seed}_r{round_}",
                                    response_format={"type": "json_object"})
            except TruncatedResponse:
                # One declared retry with a hard brevity instruction.  Recorded in the
                # round notes and in the ledger as a separate call -- never silent.
                notes["truncation_retry"] = True
                messages = messages + [
                    {"role": "assistant", "content": "(reply truncated at the output limit)"},
                    {"role": "user", "content": agent_mod.COMPACT_RETRY_NOTE},
                ]
                text = backend.chat(messages, tag=f"{arm}_s{seed}_r{round_}_compact",
                                    response_format={"type": "json_object"})
            selection, hypotheses, reasons = agent_mod.parse_response(text, legal, batch_size)
            notes["n_selected_by_llm"] = len(selection)
            notes["reasons"] = {k: v for k, v in list(reasons.items())[:10]}
            if len(selection) < batch_size:
                # declared, never silent: top up by model rank and record the shortfall
                notes["topped_up"] = batch_size - len(selection)
                ranked = [shortlist[i]["variant"] for i in np.argsort(
                    [-row["predicted"] for row in shortlist])]
                for v in ranked:
                    if len(selection) >= batch_size:
                        break
                    if v not in selection:
                        selection.append(v)
            index_of = {str(v): i for i, v in enumerate(variants[pool_arr])}
            picks_local = np.array([index_of[v] for v in selection])
            if log_dir:
                (Path(log_dir) / f"{arm}_s{seed}_r{round_}_llm.json").write_text(
                    json.dumps({"payload": payload, "response": text}, ensure_ascii=False, indent=2))
        elif arm in RULE_ARMS:
            known_summary = {
                "n_known": len(known),
                "best_known_fitness": round(best_so_far, 3),
                "wild_type_fitness": 1.0,
                "top_known": [
                    {"variant": str(variants[i]), "fitness": round(float(fitness[i]), 3)}
                    for i in np.asarray(known)[np.argsort(-fitness[known])[:8]]
                ],
            }
            prior_summary = {
                "components": [{"name": c.name, "weight": c.weight, "provenance": c.provenance}
                               for c in evidence_prior.components],
                "top_substitutions_per_site": {
                    str(pos): [f"{aa}({score:+.2f})" for aa, score in
                               sorted(((a, evidence_prior.table[i, j])
                                       for j, a in enumerate(agent_mod.AA)), key=lambda kv: -kv[1])[:6]]
                    for i, pos in enumerate(protocol_mod.POSITIONS)},
            }
            if arm == "prior_fixed":
                # frozen rules: the evidence prior and the default constraints, no agent
                model_z = (predictions - predictions.mean()) / (predictions.std() + 1e-9)
                score = model_z + prior_z[pool_arr]
                picks_local, audit = constraint_set.select(
                    variants[pool_arr], score, batch_size, rng, set(variants[known].tolist()))
                last_audit = audit
                notes.update({"weights": {"prior": 1.0, "model": 1.0},
                              "constraint_audit": audit, "agent": "none (fixed rules)"})
                picks_global = pool_arr[np.asarray(picks_local, int)]
                picked_variants = [str(v) for v in variants[picks_global]]
                commit = commit_digest(picked_variants)
                revealed = oracle.reveal(picks_global, arm=arm, round_=round_, commit_sha256=commit)
                predicted_for_picks = predictions[np.asarray(picks_local, int)]
                record = RoundRecord(
                    arm=arm, seed=seed, round=round_, commit_sha256=commit,
                    selected=[{"variant": v, "predicted": float(pp), "fitness": float(y)}
                              for v, pp, y in zip(picked_variants, predicted_for_picks, revealed)],
                    metrics=campaign_metrics(revealed, best_so_far),
                    diversity=batch_diversity(picked_variants),
                    hypotheses=[], n_labels_used=len(known) - len(split.initial) + len(picks_global),
                    notes=notes)
                best_so_far = max(best_so_far, float(revealed.max()))
                record.best_so_far = best_so_far
                test_pred = model.predict(variants[split.test])
                test_true = oracle.truth_for_evaluation(split.test)
                record.test_metrics = {
                    **regression_metrics(test_true, test_pred),
                    "top100_recall": topk_recall(test_true, test_pred, 100),
                    "top5pct_precision": top_percentile_precision(test_true, test_pred, 5.0)}
                known = known + list(picks_global)
                pool = [i for i in pool if i not in set(picks_global.tolist())]
                records.append(record)
                previous = record
                continue

            stats = protocol_mod.site_statistics(variants[known], fitness[known])
            payload = protocol_mod.build_round_message(
                round_, prior_summary, constraint_set, known_summary, stats,
                _feedback_block(previous), last_audit)
            messages = [
                {"role": "system", "content": protocol_mod.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            try:
                text = backend.chat(messages, tag=f"{arm}_s{seed}_r{round_}",
                                    response_format={"type": "json_object"})
            except TruncatedResponse:
                notes["truncation_retry"] = True
                messages = messages + [
                    {"role": "assistant", "content": "(reply truncated at the output limit)"},
                    {"role": "user", "content": agent_mod.COMPACT_RETRY_NOTE},
                ]
                text = backend.chat(messages, tag=f"{arm}_s{seed}_r{round_}_compact",
                                    response_format={"type": "json_object"})
            hypotheses, updates, round_weights, rationale = protocol_mod.parse_round_response(text)
            accepted = constraint_set.apply_updates(updates, round_)
            model_pool = predictions
            model_z = (model_pool - model_pool.mean()) / (model_pool.std() + 1e-9)
            score = round_weights["model"] * model_z + round_weights["prior"] * prior_z[pool_arr]
            picks_local, audit = constraint_set.select(
                variants[pool_arr], score, batch_size, rng, set(variants[known].tolist()))
            last_audit = audit
            notes.update({"weights": round_weights, "rationale": rationale,
                          "constraint_updates": accepted, "constraint_audit": audit,
                          "prior_attribution_examples": {
                              str(variants[pool_arr][j]): evidence_prior.attribution(
                                  str(variants[pool_arr][j]))
                              for j in picks_local[:5]}})
            if log_dir:
                (Path(log_dir) / f"{arm}_s{seed}_r{round_}_llm.json").write_text(
                    json.dumps({"payload": payload, "response": text}, ensure_ascii=False, indent=2))
        else:
            raise ValueError(f"unknown arm {arm}")

        picks_global = pool_arr[np.asarray(picks_local, int)]
        picked_variants = [str(v) for v in variants[picks_global]]
        commit = commit_digest(picked_variants)

        revealed = oracle.reveal(picks_global, arm=arm, round_=round_, commit_sha256=commit)
        predicted_for_picks = predictions[np.asarray(picks_local, int)]

        record = RoundRecord(
            arm=arm, seed=seed, round=round_, commit_sha256=commit,
            selected=[
                {"variant": v, "predicted": float(p), "fitness": float(y)}
                for v, p, y in zip(picked_variants, predicted_for_picks, revealed)
            ],
            metrics=campaign_metrics(revealed, best_so_far),
            diversity=batch_diversity(picked_variants),
            hypotheses=[h.to_dict() for h in agent_mod.judge_hypotheses(
                hypotheses, picked_variants, revealed, best_so_far)],
            n_labels_used=len(known) - len(split.initial) + len(picks_global),
            notes=notes,
        )
        best_so_far = max(best_so_far, float(revealed.max()))
        record.best_so_far = best_so_far

        if constraint_set is not None:
            refuted = protocol_mod.refuted_substitution_map(record.hypotheses)
            if refuted:
                constraint_set.set_refuted(refuted, round_)
            record.notes["constraint_set_after_round"] = constraint_set.to_dict()

        test_pred = model.predict(variants[split.test])
        test_true = oracle.truth_for_evaluation(split.test)
        record.test_metrics = {
            **regression_metrics(test_true, test_pred),
            "top100_recall": topk_recall(test_true, test_pred, 100),
            "top5pct_precision": top_percentile_precision(test_true, test_pred, 5.0),
        }

        known = known + list(picks_global)
        pool = [i for i in pool if i not in set(picks_global.tolist())]
        records.append(record)
        previous = record

    return records


def records_to_rows(records) -> list[dict]:
    rows = []
    for r in records:
        d = asdict(r)
        d["selected"] = len(r.selected)
        rows.append(d)
    return rows
