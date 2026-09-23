# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]
### Changed
- NLTK `stopwords` and English `punkt_tab` data now ship inside the package
  (`sumplicity/data`); no download is needed. Requires `nltk>=3.9`.


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
