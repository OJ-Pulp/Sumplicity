"""Extractive summarizers implemented as matrix operations on a TF matrix.

Each summarizer accepts either a raw string (split into sentences with
NLTK) or a pre-chunked list of strings, and returns the ``top_k`` chunks
it selects, in ranked order.
"""
from typing import Dict, List, Type, Union

import numpy as np

from ._math import cos_sim_mat, markov_chain, overlap_sim_mat
from .preprocess import preprocessor

TextInput = Union[str, List[str]]


class BaseSummarizer:
    """Shared plumbing: input normalisation and TF-matrix construction.

    Subclasses implement :meth:`rank`, which maps a TF matrix to an ordered
    list of chunk indices. Swapping the preprocessor (e.g. a different
    stemmer or stop-word list) changes every algorithm consistently.
    """

    def __init__(self, preprocessor_: preprocessor = None):
        self.preprocessor = preprocessor_ if preprocessor_ is not None else preprocessor()

    def _chunks(self, text: TextInput) -> List[str]:
        if isinstance(text, str):
            return self.preprocessor.chunk(text)
        if isinstance(text, list):
            return text
        raise TypeError(f"text must be str or List[str], got {type(text)}")

    def rank(self, TF: np.ndarray, top_k: int, **kwargs) -> List[int]:
        raise NotImplementedError

    def summarize(self, text: TextInput, top_k: int = 3, **kwargs) -> List[str]:
        chunks = self._chunks(text)
        TF = self.preprocessor.tf(chunks)
        return [chunks[int(idx)] for idx in self.rank(TF, top_k, **kwargs)]


def _top(scores: np.ndarray, top_k: int) -> List[int]:
    return [int(i) for i in np.argsort(scores)[::-1][:top_k]]


class Luhn_sum(BaseSummarizer):
    """Frequency-weighted sentence scoring after Luhn (1958)."""

    def rank(self, TF, top_k):
        # Weight each term by its total frequency, then sum per chunk.
        chunk_scores = (TF * TF.sum(axis=1).reshape(-1, 1)).sum(axis=0)
        return _top(chunk_scores, top_k)


class SumBasic(BaseSummarizer):
    """SumBasic (Nenkova & Vanderwende, 2005) with probability squaring."""

    def rank(self, TF, top_k):
        results = []
        B = TF.sum(axis=1) / TF.sum()
        A = (TF > 0).astype(int)
        sent_lens = np.clip(np.sum(TF, axis=0), 1, None)
        while len(results) < min(top_k, TF.shape[1]):
            chunk_scores = np.dot(B, TF) / sent_lens
            top_idx = next(i for i in np.argsort(chunk_scores)[::-1] if i not in results)
            results.append(int(top_idx))
            # Square the probabilities of terms in the chosen chunk.
            B = B * (B * A[:, top_idx] + 1 - A[:, top_idx])
        return results


class KL_Sum(BaseSummarizer):
    """Greedy KL-divergence summarization (Haghighi & Vanderwende, 2009)."""

    def rank(self, TF, top_k):
        results = []
        B = (TF.sum(axis=1) / TF.sum()).reshape(-1, 1)
        summary_vector = np.zeros((TF.shape[0], 1))
        while len(results) < min(top_k, TF.shape[1]):
            Q = (TF + summary_vector) / np.clip(np.sum(TF + summary_vector, axis=0), 1, None)
            KL = np.sum(np.log(B / np.clip(Q, 1e-100, 1.0)) * B - B + Q, axis=0)
            top_idx = next(i for i in np.argsort(KL) if i not in results)
            results.append(int(top_idx))
            summary_vector = summary_vector + TF[:, top_idx].reshape(-1, 1)
        return results


class LSA_Sum(BaseSummarizer):
    """Latent Semantic Analysis scoring (Steinberger & Jezek, 2004)."""

    def rank(self, TF, top_k):
        # Thin SVD: identical singular values / right vectors, far less memory.
        _U, s, Vt = np.linalg.svd(TF, full_matrices=False)
        chunk_scores = np.sum(np.abs(Vt) * np.abs(s.reshape(-1, 1)), axis=0)
        return _top(chunk_scores, top_k)


class Graph_Reduction(BaseSummarizer):
    """Degree centrality on a cosine-similarity sentence graph."""

    def rank(self, TF, top_k):
        chunk_scores = np.sum(cos_sim_mat(TF), axis=1)
        return _top(chunk_scores, top_k)


class _RankBase(BaseSummarizer):
    """PageRank-style stationary distribution over a similarity graph."""

    def _similarity(self, TF):
        raise NotImplementedError

    def rank(self, TF, top_k, damping: float = 0.85, max_iter: int = 100):
        sim_matrix = self._similarity(TF)
        sim_matrix = sim_matrix / np.clip(np.sum(sim_matrix, axis=1, keepdims=True), 1e-10, None)
        n = sim_matrix.shape[0]
        transition_matrix = (1 - damping) * np.ones((n, n)) / n + damping * sim_matrix
        chunk_scores = markov_chain(transition_matrix, max_iter=max_iter)
        return _top(chunk_scores, top_k)


class TextRank(_RankBase):
    """TextRank (Mihalcea & Tarau, 2004): term-overlap similarity."""

    def _similarity(self, TF):
        return overlap_sim_mat(TF)


class LexRank(_RankBase):
    """LexRank (Erkan & Radev, 2004): cosine similarity."""

    def _similarity(self, TF):
        return cos_sim_mat(TF)


def get_all_summarizer() -> Dict[str, Type[BaseSummarizer]]:
    """Registry of every summarizer, keyed by display name."""
    return {
        "Luhn": Luhn_sum,
        "SumBasic": SumBasic,
        "KL": KL_Sum,
        "LSA": LSA_Sum,
        "GraphReduction": Graph_Reduction,
        "TextRank": TextRank,
        "LexRank": LexRank,
    }
