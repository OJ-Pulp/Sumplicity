# Sumplicity benchmarks

Scripts that reproduce every number in the SoftwareX article (Tables 4–7 and
Figure 1). Nothing here is needed to *use* Sumplicity.

## Environment

```bash
python -m venv .venv-bench && source .venv-bench/bin/activate
pip install -e ..                       # Sumplicity itself
pip install -r requirements-lock.txt    # exact versions used for the paper
python env_info.py                      # writes results/environment.json
```

Paper runs: Apple M4 Pro (8 performance + 4 efficiency cores), 24 GB RAM,
macOS 26.6, Python 3.14.3, NumPy 2.5.5 (Apple Accelerate BLAS), NLTK 3.10.4,
Sumy 0.14.0. The full record is in `results/environment.json`.

## Data

Every dataset is downloaded from the Hugging Face Hub **at a pinned commit**
(see `DATASETS` / `BOOKS` in `common.py` and `HOTPOT` in `rag_demo.py`); only
the test/validation files are fetched.

| Dataset | Split | Docs | Selection | Summary length *k* |
|---|---|---|---|---|
| CNN/Daily Mail 3.0.0 | test (11,490) | 1,000 | uniform random, seed 42 | 3 sentences (Lead-3 convention) |
| arXiv (ccdv) | test (6,440) | 1,000 | uniform random, seed 42 | 6 (median abstract length) |
| BillSum (US Congress) | test (3,269) | 1,000 | uniform random, seed 42 | 4 (median summary length) |
| IsmaelMousa/books | train+validation | — | length sweep, 1K–128K tokens | 10 |
| HotpotQA distractor | validation (7,405) | 200 | uniform random, seed 42 | — |

Documents with fewer than *k*+1 candidate sentences are dropped before sampling.
The sampled row indices are saved in `results/<dataset>/sample_rows.csv`.

## Fairness protocol

Sumy runs on **Sumplicity's own preprocessing** through `sumy_adapter.py`.
The adapter replaces Sumy's tokenizer with one whose `to_words` returns
Sumplicity's final terms (NLTK `word_tokenize`, lower-casing, the word filter,
NLTK English stop words, Snowball stemming). Sumy is then given a no-op stemmer
and an empty stop-word list, so none of its own steps change the terms. This
also covers its KL summarizer, which never stems. `verify_preprocessing.py`
checks, for every sentence of every sampled document, that each of the seven
Sumy summarizers sees exactly the terms in Sumplicity's TF matrix; the result
(0 mismatches) is in `results/preprocessing_check.json`.

Both libraries also get the **same candidate sentences** (segmented once in
`common.split_sentences`; Sumy receives them as an `ObjectDocumentModel` and
never re-segments) and the **same k**. Selected
sentences are put back in document order before scoring. `sumy_default`
repeats the run with Sumy's own stop-word list, as in the Sumy README.

## Scripts

| Script | Output | Paper |
|---|---|---|
| `run_quality.py --dataset {cnn_dailymail,arxiv,billsum}` | `results/<ds>/rouge.csv`, `selections.jsonl` | Tables 4, 5 |
| `run_bertscore.py --dataset …` | `results/<ds>/bertscore.csv` | Tables 4, 5 |
| `stats.py` | `results/<ds>/paired_tests.csv`, `results/tables.tex` | Tables 4, 5 |
| `run_speed.py` | `results/speed/speed.jsonl` | Table 6, Fig. 1 |
| `rag_demo.py` (optional; needs `ANTHROPIC_API_KEY`) | `results/rag/` | not reported in the paper |
| `make_figures.py` | `results/figures/` | Fig. 1 |

Statistics: per-document scores; 95% percentile-bootstrap CIs (10,000
resamples); paired two-sided Wilcoxon signed-rank tests on Sumplicity − Sumy,
Holm–Bonferroni corrected over the 28 tests (7 algorithms × 4 metrics) per
dataset.

Runtime: quality runs take roughly 5 min (CNN/DM), 10 min (BillSum) and 1–2 h
(arXiv, dominated by Sumy) on the machine above; the speed sweep takes about 1 h.
