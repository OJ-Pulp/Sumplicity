"""Shared helpers for the Sumplicity vs. Sumy benchmarks.

Fairness protocol (reviewer issue B)
------------------------------------
Every system, for every document, receives

* the *same list of candidate sentences* (segmented once, here, and handed to
  both libraries; Sumy never re-segments the text),
* the *same summary length* ``k`` (number of sentences, fixed per dataset),
* the *same word tokens*: NLTK ``word_tokenize``, lower-casing, the same
  word filter (a letter, then letters/apostrophes/hyphens) and the same
  NLTK English stop-word list. Sumy receives these tokens through
  :class:`SharedTokenizer`, so its own tokenizer and stop-word filter are
  bypassed;
* the *same stemmer*: NLTK Snowball (English). Sumy's ``Stemmer("english")``
  is NLTK's ``EnglishStemmer``; Sumy's KL summarizer, which never stems, is
  patched to stem like every other summarizer (:class:`StemmingKL`).

``verify_preprocessing.py`` checks, sentence by sentence, that every Sumy
summarizer sees exactly the stems in the corresponding Sumplicity TF column.

Selected sentences are always emitted in original document order before
scoring, so ROUGE-L is not affected by the ranking order a library returns.
A second Sumy configuration, ``sumy_default``, uses Sumy exactly as its
README shows (its own tokenizer, Stemmer and stop-word list) to show what an
out-of-the-box user gets.
"""
from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import sumplicity
from sumplicity import get_all_summarizer, preprocessor

SEED = 42
HF_CACHE = os.environ.get(
    "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
)

# Pinned dataset revisions (Hugging Face commit SHAs) used for the paper.
DATASETS = {
    "cnn_dailymail": dict(
        repo="abisee/cnn_dailymail",
        revision="96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        file="3.0.0/test-00000-of-00001.parquet",
        text="article", ref="highlights", k=3,
    ),
    "arxiv": dict(
        repo="ccdv/arxiv-summarization",
        revision="240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
        file="document/test-00000-of-00001.parquet",
        text="article", ref="abstract", k=6,
    ),
    "billsum": dict(
        repo="FiscalNote/billsum",
        revision="3d8510441c06a3d9dfb32eb0d7f80151730bcc4f",
        file="data/test-00000-of-00001.parquet",
        text="text", ref="summary", k=4,
    ),
}
BOOKS = dict(
    repo="IsmaelMousa/books",
    revision="c7b67ef6f5dba23fa6ad852d1d7f908c866eb1b5",
    files=["train.json", "validation.json"],
)

ALGORITHMS = ["Luhn", "SumBasic", "KL", "LSA", "Degree", "TextRank", "LexRank"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def hf_file(repo: str, revision: str, filename: str) -> str:
    """Download (or reuse from cache) one file of a dataset at a pinned revision."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def split_sentences(text: str, dataset: str) -> List[str]:
    """Segment a document once; the result is shared by every system."""
    if dataset == "arxiv":
        # ccdv/arxiv-summarization is lower-cased and pre-tokenised, with
        # sentence-final periods written as " . "; Punkt mis-splits it.
        sents = re.split(r"(?<= \.) ", text)
    else:
        sents = sent_tokenize(re.sub(r"\s+", " ", text))
    return [s.strip() for s in sents if len(s.split()) >= 3]


def split_reference(ref: str, dataset: str) -> List[str]:
    if dataset in ("cnn_dailymail", "arxiv"):
        return [s.strip() for s in ref.split("\n") if s.strip()]
    return sent_tokenize(re.sub(r"\s+", " ", ref))


def load_sample(dataset: str, n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Uniform random sample of ``n`` test documents (without replacement).

    Documents that yield fewer than ``k + 1`` candidate sentences are removed
    *before* sampling, so every system has a real choice to make.
    """
    cfg = DATASETS[dataset]
    df = pd.read_parquet(hf_file(cfg["repo"], cfg["revision"], cfg["file"]))
    df = df.reset_index().rename(columns={"index": "row"})
    df["sentences"] = df[cfg["text"]].map(lambda t: split_sentences(t, dataset))
    df = df[df["sentences"].map(len) > cfg["k"]]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["reference"] = df[cfg["ref"]].map(lambda r: "\n".join(split_reference(r, dataset)))
    return df[["row", "sentences", "reference"]]


def load_books_text() -> str:
    """All books concatenated (used to build inputs of controlled length)."""
    import json

    texts = []
    for f in BOOKS["files"]:
        for rec in json.load(open(hf_file(BOOKS["repo"], BOOKS["revision"], f))):
            texts.append(rec["EN"])
    return "\n\n".join(texts)


# --------------------------------------------------------------------------
# systems
# --------------------------------------------------------------------------

NLTK_STOP = sorted(set(stopwords.words("english")))


from sumy_adapter import make_sumy, run_sumy, sumy_document  # noqa: E402,F401


def run_sumplicity(summarizer, sentences: List[str], k: int) -> List[int]:
    if summarizer.needs_positions:  # Luhn also needs word order
        TF, P = summarizer.preprocessor.tf(sentences, positions=True)
        return sorted(int(i) for i in summarizer.rank(TF, k, P))
    TF = summarizer.preprocessor.tf(sentences)
    return sorted(int(i) for i in summarizer.rank(TF, k))


def sumplicity_systems() -> Dict[str, object]:
    return {name: cls() for name, cls in get_all_summarizer().items()}


def lead_k(sentences: List[str], k: int) -> List[int]:
    return list(range(min(k, len(sentences))))


def random_k(sentences: List[str], k: int, doc_id: int) -> List[int]:
    rng = random.Random(SEED * 100003 + doc_id)
    return sorted(rng.sample(range(len(sentences)), min(k, len(sentences))))
