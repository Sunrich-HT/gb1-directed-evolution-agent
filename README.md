# GB1 Directed-Evolution Agent

A machine-learning-guided directed-evolution agent for the GB1 four-site fitness
landscape, built as a controlled comparison: **random**, **predictor-greedy**,
**predictor-greedy + exploration**, **LLM agent**, and **knowledge-augmented LLM
agent**, all starting from the same labels and spending the same budget.

The measured landscape is used as a *virtual* evaluator. No wet-lab experiment is
performed or claimed anywhere in this repository.

---

## What question this answers

Given a few hundred measured variants, can an agent propose which variants to
measure next so that each round finds better protein than the last — and does a
language model in the loop beat simply trusting the predictor?

Three things make the comparison informative:

1. **The candidate pool contains the answer.** The pool is the whole unobserved
   landscape, including its global optimum (FWAA, fitness 8.76). A pool whose
   ceiling is below the starting incumbent would force a null result before any
   model runs.
2. **An exploration-only control.** `greedy_explore` is the predictor with a 25%
   random exploration quota and no language model, so any LLM advantage can be
   separated from the advantage of simply not being greedy.
3. **Hypotheses are graded by the program.** The agent must register falsifiable
   site-level hypotheses before labels are revealed; the code, not the model,
   decides whether each one was supported or refuted.

## Results (v2: evidence prior + explicit constraints)

Protocol: 192 starting labels, 3 rounds x 96 picks, **15 seeds**, identical start and budget for every arm.
Candidate pool = the 139,169 unobserved variants, which contains the global optimum FWAA (fitness 8.762).

| arm | final best (mean of 15 seeds) | median | hits the global optimum | batch diversity |
|---|---|---|---|---|
| random | 4.336 | 4.437 | 0/15 | 3.80 |
| predictor-greedy | 7.595 | 7.313 | 3/15 | 2.45 |
| predictor + 25% exploration | 7.129 | 7.313 | 3/15 | 2.99 |
| LLM agent (DeepSeek) | 7.617 | 7.555 | 3/15 | 2.45 |
| LLM + knowledge block | 7.595 | 7.313 | 3/15 | 2.47 |
| **evidence prior + constraints, no LLM** | 8.031 | 8.762 | 9/15 | 3.15 |
| evidence prior + constraints, LLM-steered | 8.070 | 8.762 | 9/15 | 3.27 |

**What the comparison shows**

1. *Evidence beats more model rounds.* A prior built only from the crystal structure, biophysics and
   62 curated literature rows — no multi-mutant label — ranks the landscape at Spearman
   rho = +0.295 on the >=3-mutation subset, matching the measured-single-mutant upper
   bound (+0.294) that uses oracle labels.
2. *Constraints triple the hit rate.* Evidence prior + constraints reaches the global optimum in 9/15 seeds
   against 3/15 for predictor-greedy (paired McNemar, 6 vs 0 discordant seeds, exact p = 0.031).
3. *The language model adds nothing on top.* Letting DeepSeek rewrite the constraint set each round is
   indistinguishable from freezing the rules (8.070 vs 8.031, Wilcoxon p = 1.00), and using the LLM as a
   *picker* reproduces predictor-greedy on 14 of 15 seeds.
4. *Zero-shot folding scores are not binding predictors.* ProteinMPNN alone reaches rho = +0.170;
   NVIDIA-hosted Boltz-2 confidence correlates with fitness at +0.21 and adds -0.002 Spearman as a feature.
5. *A structural assumption was falsified mid-project.* The common "glycine is required at 41" rule is not
   supported by the crystal structure (phi/psi -137.6/-159.8 in 1PGA, ordinary beta region); the real
   constraint there is side-chain packing, compensable at site 54. The filter was removed.

Paper: [`paper/main.pdf`](paper/main.pdf) (7 pages, Chinese, LaTeX/XeLaTeX, real citations).
Figures: `figures/paper/` — built to `figures/FIGURE_CONTRACT.md` and passing panel-alignment,
PDF text-collision and 5 pt glyph audits.
Evidence base: `knowledge/evidence_base.json` (62 rows, 21 sources) + `knowledge/references.bib`.
Cost: 154 DeepSeek calls across both campaigns, $0.523 total; ledgers under `results/`.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data (once)

```sh
mkdir -p data
curl -L -o data/four_mutations_full_data.csv.zip \
  https://raw.githubusercontent.com/J-SNACKKB/FLIP/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/four_mutations_full_data.csv.zip
```

The loader verifies the archive's SHA-256 (`85692d80…c383bff`) and refuses to run
on anything else.

## Small-scale reproduction (about a minute, no API key, no network)

```sh
python scripts/reproduce_small.py data/four_mutations_full_data.csv.zip
```

Runs the complete chain — split, cross-validated model selection, agent tool
loop, commit-then-reveal oracle, feedback, metrics — with a deterministic mock
agent backend. It verifies the machinery; it is not a scientific result.

## Full experiment

```sh
# baseline prediction and high-value identification
python scripts/run_baselines.py --data data/four_mutations_full_data.csv.zip

# five-arm campaign; omit --backend deepseek to stay offline
export DEEPSEEK_API_KEY=...          # never committed; see .env.example
python scripts/run_experiment.py --data data/four_mutations_full_data.csv.zip \
    --backend deepseek --seeds 301 302 303 --rounds 3 --batch-size 96

# interpretability report
python scripts/run_interpret.py --data data/four_mutations_full_data.csv.zip \
    --records results/experiment/records.json
```

### Structure-informed prior (optional, already cached)

`config/mpnn_log_odds.json` holds a 4x20 inverse-folding log-odds table computed
locally from ProteinMPNN (soluble `v_48_002`, pinned commit
`8907e667…cdef57`, weights SHA-256 `0877f840…a7c448`) on RCSB **2GB1** chain A.
Recompute with:

```sh
python vendor/proteinmpnn/protein_mpnn_run.py --pdb_path data/2gb1.pdb \
    --pdb_path_chains A --out_folder /tmp/mpnn_out --unconditional_probs_only 1 \
    --path_to_model_weights vendor/proteinmpnn/soluble_model_weights --model_name v_48_002
```

## Layout

```
src/gb1agent/
  data.py         landscape loading, frozen splits, commit-then-reveal oracle
  features.py     additive (80-dim) and pairwise/epistatic (2480-dim) encodings
  predictors.py   ridge / random forest / FCNN + cross-validated selection
  priors.py       zero-shot priors and the substitution-wise z-scoring
  agent.py        prompts, shortlist tools, hypothesis registration and grading
  llm.py          DeepSeek client with a hard budget ledger; offline mock
  nvidia.py       NVIDIA hosted inference with multi-key rotation
  loop.py         the five-arm campaign driver
  interpret.py    model-, decision- and failure-layer interpretability
  metrics.py      Spearman, top-k recall, top-5% precision, per-round gain
scripts/          runnable entry points (baselines, campaign, structure, interpretability,
                  figures, report)
config/           cached zero-shot prior tables
vendor/           third-party source, pinned and checksummed
```

## Secrets

No key, token or credential is stored in this repository. `llm.py` and
`nvidia.py` read `DEEPSEEK_API_KEY` and `NVIDIA_API_KEYS` from the environment
only, failures are reported by key *index*, and the request logs under
`results/` are git-ignored.

## External resources used

| Resource | Use | Licence |
|---|---|---|
| FLIP GB1 four-site DMS (Wu et al. 2016; Dallago et al. 2021) | fitness data and virtual evaluator | data CC BY 4.0, code AFL-3.0 |
| ProteinMPNN `v_48_002` (Dauparas et al. 2022) | structure-informed zero-shot prior | MIT |
| RCSB 2GB1 (Gronenborn et al. 1991) | backbone for the inverse-folding prior | public |
| DeepSeek chat API | the LLM agent arms | commercial API |
| NVIDIA hosted inference | optional structure services | commercial API |

Every number in the report comes from a run recorded under
`results/`.

## Licence

MIT (see `LICENSE`).
