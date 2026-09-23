# Sumplicity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Sumplicity** (from *Sumy* to *Simplicity*) is a small, transparent Python library
of classical extractive summarization algorithms. Every algorithm is written as a
handful of NumPy matrix operations on one shared term-frequency (TF) matrix, so
preprocessing and scoring are fully separated and each step can be read, audited,
and modified.

Included algorithms:

| Class | Algorithm | Reference |
|---|---|---|
| `Luhn_sum` | Frequency-weighted sentence scoring | Luhn (1958) |
| `SumBasic` | Average word probability with redundancy squaring | Nenkova & Vanderwende (2005) |
| `KL_Sum` | Greedy KL-divergence minimisation | Haghighi & Vanderwende (2009) |
| `LSA_Sum` | Singular-value-weighted latent topics | Steinberger & Jezek (2004) |
| `Graph_Reduction` | Degree centrality on a cosine-similarity graph | — |
| `TextRank` | PageRank over term-overlap similarity | Mihalcea & Tarau (2004) |
| `LexRank` | PageRank over cosine similarity | Erkan & Radev (2004) |

## Installation

```bash
pip install git+https://github.com/YOUR-USERNAME/sumplicity.git
python -c "import sumplicity; sumplicity.ensure_nltk_data()"   # one-time NLTK data download
```

Offline machines: download the NLTK `punkt`, `punkt_tab` and `stopwords` resources
elsewhere, copy them to a directory, and call `nltk.data.path.append("<dir>")`
before importing Sumplicity.

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
seven algorithms side by side.

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

If you use Sumplicity in research, please cite this repository of the academic

## Contributing and support

Bug reports, questions and feature requests go in the
[issue tracker](https://github.com/YOUR-USERNAME/sumplicity/issues).
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes. This project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
