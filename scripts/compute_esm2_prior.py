#!/usr/bin/env python
"""Compute the ESM-2 masked-marginal prior for the four mutable GB1 sites.

Because the four sites are scored independently under the masked-marginal
convention, the whole zero-shot prior collapses to a 4 x 20 table, which is
cached in config/esm2_log_odds.json so that every downstream step reproduces
without a GPU or a model download.

Run once:  python scripts/compute_esm2_prior.py --out config/esm2_log_odds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

GB1_DOMAIN = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POSITIONS = (39, 40, 41, 54)   # 1-indexed in the 56-aa domain
WT = "VDGV"
AA = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--out", default="config/esm2_log_odds.json")
    ap.add_argument("--backend", choices=["huggingface", "fair-esm"], default="huggingface")
    args = ap.parse_args()

    assert len(GB1_DOMAIN) == 56
    for pos, wt in zip(POSITIONS, WT):
        assert GB1_DOMAIN[pos - 1] == wt, (pos, GB1_DOMAIN[pos - 1], wt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = []
    if args.backend == "huggingface":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        name = f"facebook/{args.model}"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name).eval().to(device)
        tokens = tokenizer(GB1_DOMAIN, return_tensors="pt").to(device)
        with torch.no_grad():
            for pos, wt in zip(POSITIONS, WT):
                masked = tokens["input_ids"].clone()
                masked[0, pos] = tokenizer.mask_token_id          # +1 for the CLS token
                logits = model(input_ids=masked, attention_mask=tokens["attention_mask"]).logits
                log_probs = torch.log_softmax(logits[0, pos], dim=-1)
                wt_lp = log_probs[tokenizer.convert_tokens_to_ids(wt)].item()
                table.append({aa: log_probs[tokenizer.convert_tokens_to_ids(aa)].item() - wt_lp
                              for aa in AA})
    else:
        import esm

        model, alphabet = getattr(esm.pretrained, args.model)()
        model = model.eval().to(device)
        converter = alphabet.get_batch_converter()
        _, _, tokens = converter([("gb1", GB1_DOMAIN)])
        tokens = tokens.to(device)
        with torch.no_grad():
            for pos, wt in zip(POSITIONS, WT):
                masked = tokens.clone()
                masked[0, pos] = alphabet.mask_idx                # +1 for the BOS token
                logits = model(masked)["logits"]
                log_probs = torch.log_softmax(logits[0, pos], dim=-1)
                wt_lp = log_probs[alphabet.get_idx(wt)].item()
                table.append({aa: log_probs[alphabet.get_idx(aa)].item() - wt_lp for aa in AA})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "backend": args.backend,
        "sequence": GB1_DOMAIN,
        "positions": list(POSITIONS),
        "wild_type": WT,
        "convention": "masked-marginal log-odds, log p(mut) - log p(wt), per site",
        "log_odds": table,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
