# Figure contract (nature-figure protocol)

Backend: **Python / matplotlib**, exclusive for all drawing, preview, export and visual QA.
Export: 180 mm double-column width, 600 dpi raster, PDF with embedded TrueType (`pdf.fonttype 42`),
every rendered glyph >= 5 pt. Each figure passes `audit_panel_alignment.py` (render-time plot-area
geometry, 1.5 pt tolerance) and `audit_figure_collisions.py` (PyMuPDF text/stroke collisions) before
it is used in the paper.

Every number in every panel is read from a file under `results/` or `knowledge/`; nothing is retyped.

---

## Figure 1 — What is known before the first label

**Core conclusion.** Structure and published evidence, with no measured multi-mutant label, split the
four sites into two mechanisms and produce a prior that ranks this landscape as well as the measured
single-mutant table does.

**Results-level question.** Is there enough prior information to design round 1 on something other
than a predictor trained on 192 random labels?

| panel | inferential role | source |
|---|---|---|
| a | the mechanism split: which site can act through the IgG-Fc interface, which act through packing | `knowledge/structure_evidence.json` |
| b | what the prior actually asserts, substitution by substitution | `results/v2/evidence_prior.json` |
| c | prior quality against the landscape, with the leakage upper bound as the reference line | `results/prior_evaluation.json` |
| d | the operational consequence: best fitness inside one 96-variant batch | `results/prior_evaluation.json` |

**Archetype.** Quantitative grid. **Review risk.** Panel c invites the objection that the prior was
tuned on the oracle; the caption states the one oracle-informed choice (the volume term was switched
off after measurement) and panel c shows that ablation explicitly.

---

## Figure 2 — Six arms under one budget

**Core conclusion.** Under an identical start and an identical label budget, the ordering of the six
strategies is decided by what information enters round 1, not by whether a language model is in the loop.

**Results-level question.** Does the evidence prior plus explicit constraints beat predictor-greedy,
and does a language model add anything beyond that?

| panel | inferential role | source |
|---|---|---|
| a | the campaign trace: best-so-far against labels spent, all arms, mean and spread over seeds | `results/v2/records.json` |
| b | per-seed endpoint, so the reader sees the variance behind the means | `results/v2/records.json` |
| c | cumulative gain per round: where each arm's improvement actually happens | `results/v2/records.json` |
| d | hit rate of the global optimum, the exam's own success criterion | `results/v2/records.json` |

**Archetype.** Quantitative grid. **Review risk.** Five seeds is a small n; panel b shows every seed
and the caption gives the paired difference rather than only the means.

---

## Figure 3 — Why each batch looks the way it does

**Core conclusion.** Every pick is attributable to a named evidence component and a named constraint,
and the constraint set demonstrably changes in response to refuted hypotheses.

**Results-level question.** Is the agent's reasoning auditable, or only narrated?

| panel | inferential role | source |
|---|---|---|
| a | prior attribution: which evidence component paid for the top picks | `results/v2/records.json` (`prior_attribution_examples`) |
| b | constraint audit: how many candidates each constraint removed, per round | `results/v2/records.json` (`constraint_audit`) |
| c | hypothesis verdicts per round, assigned by the program after the labels | `results/v2/records.json` |
| d | consequence of the rules: batch diversity and overlap with the unconstrained ranking | `results/v2/records.json` |

**Archetype.** Quantitative grid. **Review risk.** Panel a must not imply causality for the campaign
outcome; it reports the decomposition of the prior score, which the caption says explicitly.

---

## Figure 4 — The landscape that the rules are fighting

**Core conclusion.** Pairwise epistasis, not single-mutant quality, decides this landscape, and the
41/54 volume-transfer mechanism that the prior encodes is visible in the measured data.

**Results-level question.** Do the mechanisms the prior assumes exist in the measurements?

| panel | inferential role | source |
|---|---|---|
| a | measured single-mutant effects per site, the baseline expectation | `results/interpretability.json` |
| b | epistasis share of the fitted pairwise model | `results/interpretability.json` |
| c | the 41/54 volume-transfer test: measured fitness against the volume imbalance the prior penalises | landscape + `knowledge/structure_evidence.json` |
| d | why greedy single-mutant screening misses the optimum: its components are deleterious alone | `results/interpretability.json` |

**Archetype.** Quantitative grid. **Review risk.** Panel c is correlational across the landscape; the
caption states the n and that the prior term was defined before this panel was plotted.

---

## Extended data

**E1 — structure services.** Boltz-2 folding confidence against measured fitness and the ablation that
shows it adds nothing (`results/structure/structure_analysis.json`).
