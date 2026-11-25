---
title: 'Sumplicity: Transparent, vectorized extractive summarization in Python'
tags:
  - Python
  - natural language processing
  - text summarization
  - extractive summarization
  - bag-of-words
  - retrieval-augmented generation
authors:
  - name: Osiris J. Terry
    orcid: 0009-0004-4453-2874
    corresponding: true
    affiliation: 1
  - name: Christine M. Schubert Kabban
    affiliation: 2
  - name: Kenneth M. Hopkinson
    affiliation: 1
affiliations:
  - name: Department of Electrical and Computer Engineering, Air Force Institute of Technology, United States
    index: 1
  - name: Department of Mathematics and Statistics, Air Force Institute of Technology, United States
    index: 2
date: 22 September 2026
bibliography: paper.bib
---

# Summary

Automatic summarization shortens a document while keeping its most important
content. *Extractive* summarizers do this by selecting whole sentences from the
original text instead of writing new ones. Classical extractive methods rely on
simple word statistics, such as how often a word occurs and which sentences share
words, which makes them fast, predictable, and free of training data. These
properties have made them useful again as lightweight components in modern
pipelines, including retrieval-augmented generation (RAG), where a short, faithful
selection of source text is passed to a large language model.

Sumplicity is a Python library that implements seven classical extractive
summarization algorithms: Luhn [@luhn1958], SumBasic [@nenkova2005], KL-Sum
[@haghighi2009], Latent Semantic Analysis [@steinberger2004], graph degree
centrality, TextRank [@mihalcea2004], and LexRank [@erkan2004]. Each algorithm is
expressed as a few vectorized NumPy [@harris2020] operations on a single shared
term-by-sentence frequency matrix produced by an NLTK-based [@bird2009]
preprocessing step. A researcher can read each algorithm in a few lines, change
one preprocessing choice and see its effect across every method, or add a new
algorithm by writing one function.

# Statement of need

Classical summarizers remain standard baselines in summarization research and
practical building blocks in larger systems. In published work they are most often
obtained from the Sumy package [@sumy], which has been used as a comparison
baseline [@lamsiyah2021; @shukla2022; @gusev2020], as a component in application
pipelines [@vimalaksha2018; @barua2019], and as the subject of library comparisons
[@sharma2021; @giarelis2023]. Sumy was written as a diploma thesis for Czech and
Slovak text; its author notes that its documentation is limited and that some
language-specific behaviour is hard-coded [@sumy]. Preprocessing, tokenization,
and scoring are interleaved inside each algorithm, which makes it difficult to
tell whether a reported difference between two summarizers comes from the
algorithms themselves or from differences in how each one tokenizes, stems, and
filters text. Gensim's TextRank summarizer, once a common alternative
[@rehurek2010], was removed in Gensim 4.0.

Sumplicity addresses this by giving researchers a reference implementation in
which every algorithm consumes exactly the same preprocessed representation. Its
target audience is (1) NLP researchers who need reproducible, inspectable
classical baselines, (2) engineers building retrieval and RAG systems who need a
fast, dependency-light sentence selector, and (3) instructors and students who
want to see how these algorithms work in terms of linear algebra.

# State of the field

Sumy [@sumy] is the most widely used package offering several classical
summarizers in one interface. Other options cover only one or two methods
(for example, standalone TextRank or LexRank packages) or are general NLP
frameworks whose summarization components were removed or are unmaintained
[@rehurek2010]. Neural extractive and abstractive summarizers achieve higher
quality on many benchmarks but require GPUs, training data, and large model
downloads, and they are not suitable as transparent baselines.

We considered contributing to Sumy instead of writing a new package. We chose a
new package because the change we needed is architectural: Sumy's algorithms each
operate on their own document objects and perform their own preprocessing, so
separating preprocessing from scoring and vectorizing every method would have
amounted to rewriting each algorithm and changing the public interface that
existing users depend on. Sumplicity keeps the familiar algorithm set but makes
the shared matrix representation the central, documented abstraction.

# Software design

The library has three parts. `preprocessor` performs sentence splitting and word
tokenization with NLTK, lowercasing, stop-word and punctuation removal, and
Snowball stemming, and returns a dense matrix whose rows are stemmed terms and
whose columns are text chunks. It also provides TF-IDF and BM25 weightings and a
heuristic filter for noisy sentences. `_math` contains the shared numerical
routines: cosine and term-overlap similarity matrices and power iteration for the
stationary distribution of a Markov chain. `summarizers` contains one class per
algorithm; every class derives from `BaseSummarizer` and implements only
`rank(TF, top_k)`, which maps the TF matrix to ranked chunk indices.

The central trade-off was to use a single representation for all methods. This
costs some algorithm-specific flexibility (each method sees the same stemming and
stop-word choices), but it guarantees that comparisons between methods isolate the
scoring logic, and it lets a user change preprocessing once and have the change
apply everywhere. We chose dense NumPy arrays over sparse matrices because
sentence-level matrices for typical documents are small, dense operations are
simpler to read, and they let each algorithm be written close to its published
equations. Iterative methods (SumBasic and KL-Sum) update a probability or summary
vector in place of recomputing statistics from text, and TextRank and LexRank
share one PageRank implementation that differs only in the similarity function.
`summarize` accepts either a raw string or a list of pre-chunked strings, so the
same code can rank sentences, paragraphs, or passages returned by a retriever.
The English NLTK tokenizer model and stop-word list are bundled with the package,
so it runs on offline and air-gapped systems without a separate download step.

# Research impact statement

Classical summarizers are used throughout the literature as baselines, pipeline
components, and objects of comparison, and in the studies we reviewed they are
most often taken from Sumy
[@lamsiyah2021; @shukla2022; @meng2021; @gusev2020; @vimalaksha2018; @barua2019;
@bhattacharya2021; @feijo2018; @sharma2021; @giarelis2023]. Results obtained this
way depend on implementation details that the reporting papers cannot inspect.
Sumplicity was built to supply these baselines in a form that can be audited and
reproduced, and it was the implementation used in a benchmark study comparing
classical extractive summarizers with Sumy [@terry2026].

In that study, both libraries were run on the same 1,000 CNN/Daily Mail articles
[@hermann2015] and scored with ROUGE-1, ROUGE-2, and ROUGE-L F1 [@lin2004].
Sumplicity scored higher than Sumy on all three metrics for all seven algorithms,
with an average relative improvement of about 19%. The gains were largest for
KL-Sum, LSA, graph reduction, and TextRank, and smallest for Luhn and LexRank.
Sumplicity's seven algorithms also scored within a narrow range of one another
(ROUGE-1 F1 between 0.221 and 0.244), while Sumy's varied more widely (0.191 to
0.234), which suggests that much of the variation between methods reported with
Sumy comes from differences in implementation.

Runtime was measured on the same news articles (about 1,000 tokens each) and on
a collection of books averaging about 64,000 tokens [@mousa_books]. Sumplicity
produced summaries in roughly 0.012 to 0.017 seconds per article for every
algorithm and was 2 to 10 times faster than Sumy on articles. On books, the
speedup grew to between 2 and 239 times depending on the algorithm, with the
largest gains for the iterative and graph-based methods (KL-Sum, graph
reduction, TextRank, and LexRank), whose Sumy implementations scale poorly with
document length. These results show that the library is ready for use both as a
reproducible research baseline and as a fast sentence selector inside retrieval
and RAG pipelines.

# AI usage disclosure

Generative AI (Claude, Anthropic) was used to help prepare the software
repository: packaging the authors' existing modules, drafting unit tests,
continuous-integration workflows, and repository documentation, and to help
draft this paper from the authors' full-length manuscript. The summarization
algorithms, their matrix formulations, the benchmark study, and all design
decisions are the authors' own. The authors reviewed, edited, and tested all
AI-assisted material and take full responsibility for its accuracy.

# Acknowledgements

We thank the author of Sumy, whose package made classical extractive
summarization widely accessible and motivated this work. This work received no
external funding. The views expressed in
this paper are those of the authors and do not reflect the official policy or
position of the United States Air Force, the Department of Defense, or the U.S.
Government.

# References
