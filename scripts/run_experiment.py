#!/usr/bin/env python
"""Run the five-arm directed-evolution comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import Oracle, load_landscape, make_split
from gb1agent.llm import DeepSeekBackend, Ledger, MockBackend
from gb1agent.loop import ARMS, LLM_ARMS, run_arm
from gb1agent.predictors import ModelSpec, select_model
from gb1agent.priors import load_log_odds, score_variants, zscore_by_substitution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/experiment")
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--backend", choices=["mock", "deepseek"], default="mock")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--prior", default="config/mpnn_log_odds.json")
    ap.add_argument("--budget-usd", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_landscape(args.data)
    variants = df["variant"].to_numpy()

    prior_scores = None
    prior_path = Path(args.prior)
    if prior_path.exists():
        table = zscore_by_substitution(load_log_odds(prior_path))
        prior_scores = score_variants(variants, table)

    ledger = Ledger(path=out / "llm_ledger.json", cap_usd=args.budget_usd)
    if args.backend == "deepseek":
        backend = DeepSeekBackend(ledger, model=args.model)
    else:
        backend = MockBackend()

    all_records = []
    for seed in args.seeds:
        split = make_split(df, seed=seed)
        spec, cv_report = select_model(variants[split.initial],
                                       df["fitness"].to_numpy(float)[split.initial], seed=seed)
        (out / f"model_selection_seed{seed}.json").write_text(
            json.dumps({"selected": spec.label(), "cv": cv_report}, indent=2))
        for arm in args.arms:
            if arm in LLM_ARMS and args.backend == "mock":
                pass  # the mock backend still exercises the full agent path
            oracle = Oracle(df, log_path=out / "oracle_queries.jsonl")
            records = run_arm(
                arm, df, split, oracle, spec, seed=seed, rounds=args.rounds,
                batch_size=args.batch_size, backend=backend, prior_scores=prior_scores,
                log_dir=out,
            )
            for r in records:
                all_records.append(
                    {k: v for k, v in r.__dict__.items() if k != "selected"} |
                    {"selected": r.selected}
                )
            print(f"[{arm} seed={seed}] best_so_far={records[-1].best_so_far:.4f} "
                  f"gain={sum(x.metrics['gain'] for x in records):.4f}", flush=True)

    (out / "records.json").write_text(json.dumps(all_records, indent=2))
    summary = {
        "arms": args.arms, "seeds": args.seeds, "rounds": args.rounds,
        "batch_size": args.batch_size, "backend": args.backend,
        "llm_calls": len(ledger.calls), "llm_spend_usd": round(ledger.spent(), 4),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
