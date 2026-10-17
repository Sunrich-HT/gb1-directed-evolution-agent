# Reproduction targets.  `make help` lists them.
PY ?= python
DATA ?= data/gb1_flip.zip

.PHONY: help install test hygiene data prior baselines campaign figures paper demo all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package and its dependencies
	$(PY) -m pip install -e .

test:  ## run the unit tests
	$(PY) tests/test_pipeline.py

hygiene:  ## fail if the tree carries a credential, absolute path or build artefact
	$(PY) scripts/check_repo_hygiene.py

data:  ## fetch the landscape and the reference structures
	$(PY) scripts/fetch_data.py

prior:  ## rebuild the structural evidence, the knowledge graph and the evidence prior
	$(PY) scripts/build_structure_evidence.py
	$(PY) scripts/build_knowledge_graph.py --data $(DATA)
	$(PY) scripts/evaluate_prior.py --data $(DATA)

baselines:  ## fit the baseline predictors and report Spearman / Pearson / MSE / top-k
	$(PY) scripts/run_baselines.py --data $(DATA) --out results/baselines.json

campaign:  ## run the seven-arm campaign (needs DEEPSEEK_API_KEY for the two LLM arms)
	$(PY) scripts/run_experiment.py --data $(DATA) --arms all --seeds 301 302 303 304 305

analysis:  ## mutation-order, uncertainty and first-batch-information checks
	$(PY) scripts/analyze_order_and_uncertainty.py --data $(DATA)
	$(PY) scripts/check_first_batch_information.py --data $(DATA)

figures:  ## rebuild every figure
	$(PY) scripts/make_paper_figures.py --data $(DATA) --records results/v2/records_final.json
	$(PY) scripts/make_structure_figures.py --data $(DATA)

paper:  ## rebuild the values, the generated sections and the PDF
	$(PY) scripts/make_paper_values.py
	$(PY) scripts/make_new_sections.py
	$(PY) scripts/make_case_appendix.py --data $(DATA)
	cd paper && xelatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
		xelatex -interaction=nonstopmode main.tex >/dev/null && xelatex -interaction=nonstopmode main.tex >/dev/null
	@echo "paper/main.pdf"

demo:  ## rebuild the call-trace demo page
	$(PY) scripts/capture_trace.py --data $(DATA)
	$(PY) scripts/capture_structure_trace.py
	$(PY) scripts/make_demo_page.py

recommend:  ## one-shot recommendation from the wild-type sequence
	$(PY) scripts/recommend.py --top-k 10

all: test hygiene prior baselines analysis figures paper demo  ## everything that does not need API keys

clean:  ## remove LaTeX build artefacts
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.toc
