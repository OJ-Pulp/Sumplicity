# Sumplicity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24540692.svg)](https://doi.org/10.5281/zenodo.24540692)

**Sumplicity** (from *Sumy* to *Simplicity*) is a small, transparent Python library
of classical extractive summarization algorithms. Every algorithm is written as a
handful of NumPy matrix operations on one shared term-frequency (TF) matrix, so
preprocessing and scoring are fully separated and each step can be read, audited,
and modified.

Each algorithm follows its original publication. Where a paper leaves a constant
unspecified, the value is a documented constructor argument (default shown).
`tests/test_equations.py` checks every implementation against a plain-loop
transcription of the paper's definition and, where the paper gives one, against
its worked example.

| Class | Algorithm | Reference | Unspecified constants (default) |
|---|---|---|---|
| `Luhn_sum` | Best significance factor, (significant words)² / cluster span | Luhn (1958) | `min_freq=2`, `max_gap=4` |
| `SumBasic` | Average word probability, top-word constraint, probability squaring | Nenkova & Vanderwende (2005); Vanderwende et al. (2007) | — |
| `KL_Sum` | Greedy minimisation of KL(P_D ‖ P_S), stop when it no longer decreases | Haghighi & Vanderwende (2009) | smoothing `lam=0.01` |
| `LSA_Sum` | Sentence-vector length in the reduced latent space | Steinberger & Ježek (2004) | `ratio=0.5` (dimensions with σ ≥ ½σ_max) |
| `DegreeCentrality` | Number of neighbours in the thresholded IDF-cosine graph | Erkan & Radev (2004), §3.1 | `threshold=0.1` |
| `TextRank` | Weighted PageRank over distinct-word overlap | Mihalcea & Tarau (2004) | `damping=0.85` |
| `LexRank` | PageRank over the thresholded IDF-cosine graph (`continuous=True` for weighted) | Erkan & Radev (2004) | `threshold=0.1`, `jump=0.15` |

`Graph_Reduction` remains as a deprecated alias of `DegreeCentrality`.

`summarize(text, top_k=3, max_words=None)` limits the summary by number of
sentences or, as in the original evaluations, by number of words.

### Roadmap

Sumplicity aims to be a reference collection of Bag-of-Words extractive
summarizers for research baselines. This release covers the algorithms studies
commonly take from Sumy. Next comes Edmundson's method (1969), then other
widely used baselines such as centroid-based summarization and maximal marginal
relevance (MMR), and query-focused variants. Every addition follows the same
rules: implemented from its original paper, tested against that paper's
definition and examples, and written as matrix operations.

## Installation

```bash
pip install git+https://github.com/OJ-Pulp/Sumplicity.git
```

The NLTK data Sumplicity needs (`stopwords` and the English `punkt_tab` tokenizer)
ships inside the package, so no download step is required and it works offline.

## Quick start

```python
from sumplicity import LexRank

text = open("article.txt").read()
for sentence in LexRank().summarize(text, top_k=3):
    print(sentence)
```

`summarize` accepts either a raw string (split into sentences with NLTK) or a list
of pre-chunked strings (sentences, paragraphs, retrieved passages, ...), and
returns the selected chunks in ranked order. See `examples/quickstart.py` for all
seven algorithms side by side, and `examples/rag_example.py` for a minimal
retrieve-compress-generate pipeline.

## Working with the TF matrix directly

```python
from sumplicity import preprocessor, SumBasic

pp = preprocessor()
chunks = pp.chunk(text)
TF, terms = pp.tf(chunks, keys=True)     # rows = stemmed terms, columns = chunks
order = SumBasic().rank(TF, top_k=3)     # chunk indices, best first
```

`preprocessor` also provides `tfidf`, `bm25`, and `clean` (a heuristic filter that
drops noisy sentences such as tables or markup).

## Adding an algorithm

Subclass `BaseSummarizer` and implement `rank(self, TF, top_k) -> list[int]`:

```python
import numpy as np
from sumplicity import BaseSummarizer

class LengthBaseline(BaseSummarizer):
    def rank(self, TF, top_k):
        return list(np.argsort(TF.sum(axis=0))[::-1][:top_k])
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

## Citing

If you use Sumplicity in research, please cite using [CITATION.cff](CITATION.cff)

## Contributing and support

Bug reports, questions and feature requests go in the
[issue tracker](https://github.com/OJ-Pulp/Sumplicity/issues).
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes. This project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
