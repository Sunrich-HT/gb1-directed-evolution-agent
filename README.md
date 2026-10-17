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

## Results actually measured in this repository

Protocol: 192 starting labels, 3 rounds x 96 picks, 3 seeds, identical start and budget for every arm;
candidate pool = the 139,169 unobserved variants, which contains the global optimum FWAA (fitness 8.762).

| arm | final best-so-far (mean of 3 seeds) | per-seed | hits the global optimum | batch diversity |
|---|---|---|---|---|
| random | **4.376** | 3.88 / 4.37 / 4.88 | 0/3 | 3.80 |
| predictor-greedy | **7.670** | 8.76 / 6.69 / 7.56 | 1/3 | 2.43 |
| predictor + 25% exploration | **7.670** | 8.76 / 6.69 / 7.56 | 1/3 | 3.04 |
| LLM agent (DeepSeek) | **7.732** | 8.76 / 6.88 / 7.56 | 1/3 | 2.44 |
| knowledge-augmented LLM agent | **7.732** | 8.76 / 6.88 / 7.56 | 1/3 | 2.44 |

* **Model guidance is what matters.** Every arm with a predictor in it ends near 7.7; random ends at 4.38.
* **The language model adds little here.** Round 1 the agent picks exactly the predictor's top-96
  (batch overlap 1.00); by round 3 overlap has fallen to 0.82, and the final best-so-far is 7.73 vs 7.67.
  The knowledge block changes almost nothing (overlap between the two LLM arms stays >= 0.99).
* **Hypotheses are graded, not narrated.** 72 registered hypotheses:
  20 supported, 15 refuted, 37 undetermined.
* **Predictors from 192 labels are weak at the top.** Spearman 0.35, top-5% precision 0.28 on a held-out
  10,000-variant test set; on the official FLIP `sampled` split (6,988 training labels) the same model
  reaches Spearman 0.919 and top-100 recall 0.76.
* **Structure confidence carries no usable signal for binding.** 146 variants spanning the whole fitness
  range were folded with NVIDIA-hosted Boltz-2 (5 rotating keys, 146/146 succeeded, 25 rate limits
  absorbed by rotation). pLDDT at the four sites correlates with fitness at Spearman
  +0.21 against +0.56
  for the sequence model; adding the structure columns moves 5-fold CV Spearman by
  -0.002 and top-5% precision by +0.000.
* **Why the optimum is hard.** Epistasis carries 83% of the
  pairwise model's weight, and two of the four components of FWAA (V39F, G41A) are deleterious on their own.
* **Cost.** 18 DeepSeek requests, 0.0758 USD, every call in
  `results/llm_deepseek/llm_ledger.json`.

Report: [`report/GB1_agent_report.pdf`](report/GB1_agent_report.pdf) (3 pages, Chinese).
Rebuild everything with `scripts/run_baselines.py`, `scripts/run_experiment.py`,
`scripts/run_structure_ablation.py`, `scripts/analyze_structure.py`, `scripts/run_interpret.py`,
`scripts/make_figures.py`, `scripts/make_report.py`.

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
