"""Extractive summarizers implemented as matrix operations.

Every summarizer follows the same pipeline: the preprocessor builds the
term-by-sentence matrix ``TF`` (T x N), the summarizer applies the matrix
operations that define the published algorithm, and returns sentence indices
in ranked order. Each class cites the paper it implements; where a paper leaves
a constant unspecified, the chosen value is a documented constructor argument.

Each summarizer accepts either a raw string (split into sentences with NLTK)
or a pre-chunked list of strings.
"""
import warnings
from typing import Dict, List, Optional, Type, Union

import numpy as np

from ._math import idf_cosine
from .preprocess import preprocessor

TextInput = Union[str, List[str]]


def _top(scores: np.ndarray, top_k: int) -> List[int]:
    """Indices of the ``top_k`` highest scores; ties go to the earlier sentence."""
    return [int(i) for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:top_k]]


class BaseSummarizer:
    """Shared plumbing: input normalisation and matrix construction.

    Subclasses implement :meth:`rank`, which maps a TF matrix to an ordered
    list of chunk indices. Swapping the preprocessor (e.g. a different
    stemmer or stop-word list) changes every algorithm consistently.
    """

    #: True for algorithms that also need word order (the position matrix).
    needs_positions = False

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

    def summarize(self, text: TextInput, top_k: Optional[int] = 3,
                  max_words: Optional[int] = None) -> List[str]:
        """Return the selected chunks in ranked order.

        ``top_k`` limits the number of chunks; ``max_words`` (optional) limits
        the total number of whitespace-separated words, the length unit used in
        the original evaluations. Chunks are added in rank order while they fit.
        """
        chunks = self._chunks(text)
        k = len(chunks) if (top_k is None or max_words is not None) else top_k
        if self.needs_positions:
            TF, P = self.preprocessor.tf(chunks, positions=True)
            order = self.rank(TF, k, P)
        else:
            order = self.rank(self.preprocessor.tf(chunks), k)
        if max_words is not None:
            chosen, used = [], 0
            for idx in order[: (top_k or len(order))]:
                n = len(chunks[idx].split())
                if used + n > max_words:
                    break
                chosen.append(idx)
                used += n
            order = chosen
        return [chunks[int(idx)] for idx in order]


class Luhn_sum(BaseSummarizer):
    """Luhn (1958), "The automatic creation of literature abstracts".

    Significant words are content words occurring at least ``min_freq`` times.
    In each sentence, significant words separated by at most ``max_gap``
    non-significant words form a cluster; a cluster's significance factor is
    (significant words)^2 / (all words in the cluster), and a sentence scores
    the factor of its best cluster (Luhn 1958, p. 161 and Fig. 2).
    """

    needs_positions = True

    def __init__(self, preprocessor_: preprocessor = None, min_freq: int = 2, max_gap: int = 4):
        super().__init__(preprocessor_)
        self.min_freq, self.max_gap = min_freq, max_gap

    def rank(self, TF, top_k, P=None):
        if P is None:
            raise ValueError(
                "Luhn needs word order: pass the position matrix from "
                "preprocessor.tf(chunks, positions=True)."
            )
        N, L = P.shape
        significant = TF.sum(axis=1) >= self.min_freq
        sig = (P >= 0) & significant[np.maximum(P, 0)]                 # N x L
        pos = np.broadcast_to(np.arange(L), (N, L))
        far = -(L + self.max_gap + 2)
        last = np.maximum.accumulate(np.where(sig, pos, far), axis=1)  # last significant position
        prev = np.hstack([np.full((N, 1), far), last[:, :-1]])         # previous significant position
        start = sig & (pos - prev - 1 > self.max_gap)                  # a new cluster begins
        cluster = np.cumsum(start, axis=1) + np.arange(N)[:, None] * (L + 1)
        cid, p = cluster[sig], pos[sig]
        size = N * (L + 1)
        count = np.bincount(cid, minlength=size)                       # significant words per cluster
        first = np.full(size, np.inf)
        np.minimum.at(first, cid, p)
        lastp = np.full(size, -np.inf)
        np.maximum.at(lastp, cid, p)
        span = np.where(count > 0, lastp - first + 1, 1)               # all words in the cluster
        factor = np.where(count > 0, count ** 2 / span, 0.0)
        return _top(factor.reshape(N, L + 1).max(axis=1), top_k)


class SumBasic(BaseSummarizer):
    """SumBasic (Nenkova & Vanderwende, 2005), steps as in Vanderwende et al. (2007, §2).

    p(w) = n/N; a sentence's weight is the average p(w) of its distinct words;
    the chosen sentence is the best-weighted one containing the currently most
    probable word; the probabilities of the chosen sentence's words are squared.
    Ties, which the paper does not address, go to the word first seen in the
    text and to the earlier sentence.
    """

    def rank(self, TF, top_k):
        A = (TF > 0).astype(float)
        p = TF.sum(axis=1) / TF.sum()
        types = np.maximum(A.sum(axis=0), 1)
        available = np.ones(TF.shape[1], dtype=bool)
        chosen = []
        while len(chosen) < min(top_k, TF.shape[1]):
            weight = (p @ A) / types
            present = A[:, available].any(axis=1)
            candidates = np.zeros_like(available)
            if present.any():
                best_word = np.argmax(np.where(present, p, -np.inf))
                candidates = available & (A[best_word] > 0)
            if not candidates.any():          # only empty sentences remain
                candidates = available
            idx = np.flatnonzero(candidates)
            k = int(idx[np.argmax(weight[idx])])
            chosen.append(k)
            available[k] = False
            p = np.where(A[:, k] > 0, p ** 2, p)
        return chosen


class KL_Sum(BaseSummarizer):
    """KLSum (Haghighi & Vanderwende, 2009, §3.2 and §4).

    Greedily adds the sentence that minimises KL(P_D || P_S), and stops when no
    sentence lowers it. P_S is smoothed so it has a small amount of mass on all
    document words (their footnote 10): P_S = (1 - lam) * P_summary + lam * P_D.
    """

    def __init__(self, preprocessor_: preprocessor = None, lam: float = 0.01,
                 stop_early: bool = True):
        super().__init__(preprocessor_)
        self.lam, self.stop_early = lam, stop_early

    def _kl(self, counts: np.ndarray, p: np.ndarray) -> np.ndarray:
        total = counts.sum(axis=0, keepdims=True)
        p_s = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0)
        q = (1 - self.lam) * p_s + self.lam * p[:, None]
        return (p[:, None] * np.log(p[:, None] / q)).sum(axis=0)

    def rank(self, TF, top_k):
        p = TF.sum(axis=1) / TF.sum()
        s = np.zeros(TF.shape[0])
        current = np.inf
        available = np.ones(TF.shape[1], dtype=bool)
        chosen = []
        while len(chosen) < min(top_k, TF.shape[1]):
            idx = np.flatnonzero(available)
            kl = self._kl(TF[:, idx] + s[:, None], p)
            b = int(np.argmin(kl))
            if self.stop_early and kl[b] >= current:
                break
            k = int(idx[b])
            chosen.append(k)
            available[k] = False
            s += TF[:, k]
            current = kl[b]
        return chosen


class LSA_Sum(BaseSummarizer):
    """LSA summarization (Steinberger & Jezek, 2004, §4).

    A sentence's score is the length of its vector in the reduced latent space,
    sqrt(sum_i v_ki^2 sigma_i^2), over the dimensions whose singular value is at
    least ``ratio`` times the largest.
    """

    def __init__(self, preprocessor_: preprocessor = None, ratio: float = 0.5):
        super().__init__(preprocessor_)
        self.ratio = ratio

    def rank(self, TF, top_k):
        _U, sigma, Vt = np.linalg.svd(TF, full_matrices=False)
        n = int((sigma >= self.ratio * sigma[0]).sum())
        scores = np.sqrt(((Vt[:n] ** 2) * (sigma[:n, None] ** 2)).sum(axis=0))
        return _top(scores, top_k)


class DegreeCentrality(BaseSummarizer):
    """Degree centrality (Erkan & Radev, 2004, §3.1).

    A sentence's score is its number of neighbours (itself included) in the
    graph linking sentences whose IDF-modified cosine exceeds ``threshold``.
    """

    def __init__(self, preprocessor_: preprocessor = None, threshold: float = 0.1):
        super().__init__(preprocessor_)
        self.threshold = threshold

    def rank(self, TF, top_k):
        return _top((idf_cosine(TF) > self.threshold).sum(axis=1), top_k)


class Graph_Reduction(DegreeCentrality):
    """Deprecated alias of :class:`DegreeCentrality` (graph reduction has no
    published definition; degree centrality is its closest published method)."""

    def __init__(self, *args, **kwargs):
        warnings.warn("Graph_Reduction is deprecated; use DegreeCentrality.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class TextRank(BaseSummarizer):
    """TextRank sentence extraction (Mihalcea & Tarau, 2004, §2.2 and §4.1).

    Similarity = (distinct words shared) / (log|S_i| + log|S_j|), no self-loops;
    scores are the fixed point of WS = (1 - d) + d W^T WS, where W is the
    row-normalised similarity matrix. The fixed point is computed exactly:
    WS = (1 - d) (I - d W^T)^-1 1.
    """

    def __init__(self, preprocessor_: preprocessor = None, damping: float = 0.85):
        super().__init__(preprocessor_)
        self.damping = damping

    def rank(self, TF, top_k):
        A = (TF > 0).astype(float)
        overlap = A.T @ A
        length = TF.sum(axis=0)
        log_len = np.log(np.where(length > 0, length, 1))
        denom = log_len[:, None] + log_len[None, :]
        S = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0)
        np.fill_diagonal(S, 0.0)
        out = S.sum(axis=1, keepdims=True)
        W = np.divide(S, out, out=np.zeros_like(S), where=out > 0)
        N = TF.shape[1]
        d = self.damping
        scores = np.linalg.solve(np.eye(N) - d * W.T, np.full(N, 1 - d))
        return _top(scores, top_k)


class LexRank(BaseSummarizer):
    """LexRank (Erkan & Radev, 2004, Eqs. 2, 6, 9 and 10).

    Sentences are linked when their IDF-modified cosine exceeds ``threshold``
    (self-links included); the row-normalised graph is combined with a uniform
    random jump of probability ``jump``. Scores are the stationary distribution,
    computed exactly: p = (jump/N) (I - (1 - jump) B^T)^-1 1. With
    ``continuous=True`` the cosine values weight the edges (Eq. 10).
    """

    def __init__(self, preprocessor_: preprocessor = None, threshold: float = 0.1,
                 jump: float = 0.15, continuous: bool = False):
        super().__init__(preprocessor_)
        self.threshold, self.jump, self.continuous = threshold, jump, continuous

    def rank(self, TF, top_k):
        S = idf_cosine(TF)
        B = S if self.continuous else (S > self.threshold).astype(float)
        B = B / B.sum(axis=1, keepdims=True)
        N = B.shape[0]
        p = np.linalg.solve(np.eye(N) - (1 - self.jump) * B.T, np.full(N, self.jump / N))
        return _top(p, top_k)


def get_all_summarizer() -> Dict[str, Type[BaseSummarizer]]:
    """Registry of every summarizer, keyed by display name."""
    return {
        "Luhn": Luhn_sum,
        "SumBasic": SumBasic,
        "KL": KL_Sum,
        "LSA": LSA_Sum,
        "Degree": DegreeCentrality,
        "TextRank": TextRank,
        "LexRank": LexRank,
    }
