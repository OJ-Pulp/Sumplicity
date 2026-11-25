# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - unreleased
### Changed — algorithms now follow their original publications
- `Luhn_sum`: Luhn's significance factor (clusters of significant words with
  at most 4 intervening words; score = significant² / span; best cluster per
  sentence). Uses the new position matrix.
- `SumBasic`: average over distinct words; each pick must contain the current
  highest-probability word (step 3 of the original algorithm).
- `KL_Sum`: summary distribution smoothed by interpolation with the document
  distribution (`lam=0.01`); greedy selection stops when KL no longer decreases
  (`stop_early=True`).
- `LSA_Sum`: Steinberger & Ježek's sentence-vector length over dimensions with
  σ ≥ ½σ_max (previously an L1 score over all dimensions).
- `TextRank`: distinct-word overlap / (log|Si| + log|Sj|), no self-loops; the
  ranking's fixed point is computed exactly with one linear solve.
- `LexRank`: IDF-modified cosine, 0.1 threshold with self-links, damping per
  Eq. 9 of Erkan & Radev; exact stationary distribution via one linear solve.
  `continuous=True` gives continuous LexRank.
- Ties now go to the earlier sentence. Scores are rounded to 12 significant
  digits before ranking, so mathematically equal scores that differ only by
  floating-point rounding (which varies between x86 and ARM processors) are
  treated as ties; the same text gives the same summary on every machine.
- Requires Python 3.10 or later (Python 3.9 reached end of life in October 2025).

### Added
- `DegreeCentrality` (Erkan & Radev 2004, §3.1); `Graph_Reduction` is now a
  deprecated alias of it (graph reduction has no published definition).
- `preprocessor.tf(..., positions=True)` returns a sentences x positions matrix
  of term indices (word order), used by Luhn.
- `summarize(..., max_words=...)` word budget.
- `_math.idf`, `_math.idf_cosine`.

### Changed
- NLTK `stopwords` and English `punkt_tab` data now ship inside the package
  (`sumplicity/data`); no download is needed. Requires `nltk>=3.9`.
- `preprocessor` keeps only word-like tokens, set by a new `token_pattern`
  argument (default: a letter followed by letters, apostrophes or hyphens).
  Results differ from 0.1.0.

### Added
- `preprocessor.terms(chunk)` returns the lower-cased content tokens of a chunk.
- `tests/test_equations.py` checks every summarizer against a plain-loop
  transcription of its published equation.
- `benchmarks/`: reproducible quality, speed/memory and RAG benchmarks, including
  `sumy_adapter.py`, which runs Sumy on Sumplicity's preprocessing.

### Fixed
- The token filter used a substring test against `string.punctuation`, so quote
  tokens (``` `` ```, `''`), dashes (`--`), possessive `'s`, numbers and dotted
  abbreviations became terms.


## [0.1.0] - 2026-09-22
### Added
- Luhn, SumBasic, KL-Sum, LSA, Graph Reduction, TextRank and LexRank summarizers
  operating on a shared term-frequency matrix.
- `preprocessor` with NLTK tokenization, Snowball stemming, stop-word filtering,
  TF, TF-IDF and BM25 weighting, and a heuristic noise filter (`clean`).
- `BaseSummarizer` so new algorithms only implement a `rank(TF, top_k)` method.
- `ensure_nltk_data()` helper, test suite, and CI.

### Fixed
- `preprocessor.clean` tokenized individual characters when given a string.
- `TextRank` now computes the stationary distribution by power iteration
  (previously it summed columns of the transition matrix).
- `get_all_summarizer()` now returns all seven algorithms.
