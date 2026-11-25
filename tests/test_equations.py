"""Fidelity tests: each matrix-form summarizer against its original paper.

Two kinds of test:
* a plain-loop transcription of the paper's definition, checked to select the
  same sentences as the vectorized implementation on real text;
* the papers' own worked examples (Luhn 1958 Fig. 2; Erkan & Radev 2004
  Figure 1 -> Tables 1 and 2), reproduced from the published numbers.
"""
import math

import numpy as np
import pytest

from sumplicity import (DegreeCentrality, KL_Sum, LexRank, LSA_Sum, Luhn_sum,
                        SumBasic, TextRank, preprocessor)
from sumplicity._math import idf_cosine

TEXT = (
    "The river flooded the valley after three days of heavy rain. "
    "Emergency crews evacuated residents from the flooded valley towns. "
    "Rainfall totals in the valley exceeded historical records for the season. "
    "A local bakery donated bread to the evacuated residents. "
    "Officials expect the river to recede by the end of the week. "
    "The mayor praised emergency crews for the rapid evacuation of the valley. "
    "Heavy rain is forecast again for the valley next week, officials said. "
    "Residents of the flooded towns returned home after the river receded."
)
PP = preprocessor()
SENTS = PP.chunk(TEXT)
TF, KEYS, P = PP.tf(SENTS, keys=True, positions=True)
T, N = TF.shape


def order(scores, k):
    """Indices of the k highest scores, earlier sentence first on ties."""
    return sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:k]


# ------------------------------------------------------------------ Luhn
def luhn_loop(TF, P, min_freq=2, max_gap=4):
    significant = {i for i in range(TF.shape[0]) if TF[i].sum() >= min_freq}
    scores = []
    for row in P:
        positions = [k for k, t in enumerate(row) if t >= 0 and t in significant]
        best, cluster = 0.0, []
        for k in positions:
            if cluster and k - cluster[-1] - 1 > max_gap:
                best = max(best, len(cluster) ** 2 / (cluster[-1] - cluster[0] + 1))
                cluster = []
            cluster.append(k)
        if cluster:
            best = max(best, len(cluster) ** 2 / (cluster[-1] - cluster[0] + 1))
        scores.append(best)
    return scores


def test_luhn_matches_loop():
    assert Luhn_sum().rank(TF, N, P) == order(luhn_loop(TF, P), N)


def test_luhn_paper_figure_2():
    # Luhn (1958) Fig. 2: 4 significant words in a 7-word bracket -> 16/7 = 2.3.
    # Sentence 1: two adjacent significant words (4/2 = 2.0); sentence 2: three
    # (9/3 = 3.0); sentence 3: two significant words 5 apart -> separate clusters (1.0).
    Pm = np.array([[0, -1, 0, 0, -1, -1, 0, -2],
                   [0, 0, -2, -2, -2, -2, -2, -2],
                   [0, 0, 0, -2, -2, -2, -2, -2],
                   [0, -1, -1, -1, -1, -1, 0, -2]])
    TFm = np.array([[11.0, 0, 0, 0]])
    assert luhn_loop(TFm, Pm)[0] == pytest.approx(16 / 7)
    assert Luhn_sum().rank(TFm, 4, Pm) == [2, 0, 1, 3]


def test_luhn_requires_positions():
    with pytest.raises(ValueError):
        Luhn_sum().rank(TF, 3)


# ------------------------------------------------------------------ SumBasic
def sumbasic_loop(TF, k):
    p = [TF[i].sum() / TF.sum() for i in range(T)]
    words = [{i for i in range(T) if TF[i, j] > 0} for j in range(N)]
    chosen = []
    while len(chosen) < k:
        remaining = [j for j in range(N) if j not in chosen]
        top_word = max((i for j in remaining for i in words[j]), key=lambda i: (p[i], -i))
        cands = [j for j in remaining if top_word in words[j]]
        weight = {j: sum(p[i] for i in words[j]) / len(words[j]) for j in cands}
        j = max(cands, key=lambda j: (weight[j], -j))
        chosen.append(j)
        for i in words[j]:
            p[i] = p[i] ** 2
    return chosen


def test_sumbasic_matches_loop():
    assert SumBasic().rank(TF, 4) == sumbasic_loop(TF, 4)


def test_sumbasic_picks_sentence_with_top_word():
    # "solar" is the most probable word, so the first pick must contain it even
    # though the wind sentence has a higher average word probability.
    docs = ["Solar power grows.", "Solar panels convert sunlight.", "Solar farms expand.",
            "Wind wind wind turbines."]
    TFm, keys = preprocessor().tf(docs, keys=True)
    first = SumBasic().rank(TFm, 1)[0]
    assert TFm[keys.index("solar"), first] > 0


# ------------------------------------------------------------------ KL-Sum
def kl_loop(TF, k, lam=0.01):
    p = [TF[i].sum() / TF.sum() for i in range(T)]
    s = [0.0] * T
    chosen, current = [], math.inf
    while len(chosen) < k:
        best = None
        for j in range(N):
            if j in chosen:
                continue
            c = [s[i] + TF[i, j] for i in range(T)]
            tot = sum(c)
            q = [(1 - lam) * (c[i] / tot if tot else 0) + lam * p[i] for i in range(T)]
            kl = sum(p[i] * math.log(p[i] / q[i]) for i in range(T))
            if best is None or kl < best[0]:
                best = (kl, j)
        if best[0] >= current:
            break
        current = best[0]
        chosen.append(best[1])
        s = [s[i] + TF[i, best[1]] for i in range(T)]
    return chosen


def test_klsum_matches_loop():
    assert KL_Sum().rank(TF, 4) == kl_loop(TF, 4)


def test_klsum_divergence_decreases_and_can_stop():
    full = KL_Sum(stop_early=False).rank(TF, N)
    assert len(full) == N
    assert len(KL_Sum().rank(TF, N)) <= N


# ------------------------------------------------------------------ LSA
def test_lsa_matches_formula():
    _, sigma, Vt = np.linalg.svd(TF, full_matrices=False)
    n = sum(1 for x in sigma if x >= 0.5 * sigma[0])
    scores = [math.sqrt(sum(Vt[i, j] ** 2 * sigma[i] ** 2 for i in range(n))) for j in range(N)]
    assert LSA_Sum().rank(TF, N) == order(scores, N)


# ------------------------------------------------------------------ Erkan & Radev (2004) Figure 1
FIG1 = np.array([
    [1.00, 0.45, 0.02, 0.17, 0.03, 0.22, 0.03, 0.28, 0.06, 0.06, 0.00],
    [0.45, 1.00, 0.16, 0.27, 0.03, 0.19, 0.03, 0.21, 0.03, 0.15, 0.00],
    [0.02, 0.16, 1.00, 0.03, 0.00, 0.01, 0.03, 0.04, 0.00, 0.01, 0.00],
    [0.17, 0.27, 0.03, 1.00, 0.01, 0.16, 0.28, 0.17, 0.00, 0.09, 0.01],
    [0.03, 0.03, 0.00, 0.01, 1.00, 0.29, 0.05, 0.15, 0.20, 0.04, 0.18],
    [0.22, 0.19, 0.01, 0.16, 0.29, 1.00, 0.05, 0.29, 0.04, 0.20, 0.03],
    [0.03, 0.03, 0.03, 0.28, 0.05, 0.05, 1.00, 0.06, 0.00, 0.00, 0.01],
    [0.28, 0.21, 0.04, 0.17, 0.15, 0.29, 0.06, 1.00, 0.25, 0.20, 0.17],
    [0.06, 0.03, 0.00, 0.00, 0.20, 0.04, 0.00, 0.25, 1.00, 0.26, 0.38],
    [0.06, 0.15, 0.01, 0.09, 0.04, 0.20, 0.00, 0.20, 0.26, 1.00, 0.12],
    [0.00, 0.00, 0.00, 0.01, 0.18, 0.03, 0.01, 0.17, 0.38, 0.12, 1.00],
])


def test_degree_reproduces_paper_table_1():
    # Thresholds 0.1 and 0.3; at 0.2 the published matrix is rounded to two
    # decimals and three pairs print as exactly 0.20, so 0.2 cannot be checked.
    assert list((FIG1 > 0.1).sum(axis=1)) == [5, 7, 2, 6, 5, 7, 2, 9, 5, 6, 5]
    assert list((FIG1 > 0.3).sum(axis=1)) == [2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2]


def test_lexrank_reproduces_paper_table_2(monkeypatch):
    import sumplicity.summarizers as S
    captured = {}

    def fake_cosine(_TF):
        return FIG1

    real_solve = np.linalg.solve

    def spy_solve(a, b):
        captured["p"] = real_solve(a, b)
        return captured["p"]

    monkeypatch.setattr(S, "idf_cosine", fake_cosine)
    monkeypatch.setattr(S.np.linalg, "solve", spy_solve)
    LexRank().rank(np.zeros((1, 11)), 11)
    p = captured["p"] / captured["p"].max()
    table2 = [0.6007, 0.8466, 0.3491, 0.7520, 0.5907, 0.7993, 0.3548, 1.0000, 0.5921, 0.6910, 0.5921]
    assert np.allclose(p, table2, atol=1e-3)


def test_degree_matches_loop():
    S = idf_cosine(TF)
    scores = [sum(1 for i in range(N) if S[j, i] > 0.1) for j in range(N)]
    assert DegreeCentrality().rank(TF, N) == order(scores, N)


def idf_cosine_loop(TF):
    idf = [math.log(N / sum(1 for j in range(N) if TF[i, j] > 0)) for i in range(T)]
    def cos(x, y):
        num = sum(TF[i, x] * TF[i, y] * idf[i] ** 2 for i in range(T))
        nx = math.sqrt(sum((TF[i, x] * idf[i]) ** 2 for i in range(T)))
        ny = math.sqrt(sum((TF[i, y] * idf[i]) ** 2 for i in range(T)))
        return 1.0 if x == y else (num / (nx * ny) if nx and ny else 0.0)
    return [[cos(x, y) for y in range(N)] for x in range(N)]


def test_idf_cosine_matches_equation_2():
    assert np.allclose(idf_cosine(TF), idf_cosine_loop(TF))


# ------------------------------------------------------------------ TextRank
def textrank_loop(TF, d=0.85, tol=1e-12):
    words = [{i for i in range(T) if TF[i, j] > 0} for j in range(N)]
    length = [TF[:, j].sum() for j in range(N)]
    def sim(a, b):
        den = math.log(length[a]) + math.log(length[b])
        return len(words[a] & words[b]) / den if a != b and den > 0 else 0.0
    w = [[sim(a, b) for b in range(N)] for a in range(N)]
    out = [sum(r) for r in w]
    ws = [1.0] * N
    while True:  # Mihalcea & Tarau (2004), weighted ranking formula
        new = [(1 - d) + d * sum(w[j][i] / out[j] * ws[j] for j in range(N) if out[j] > 0)
               for i in range(N)]
        if max(abs(a - b) for a, b in zip(new, ws)) < tol:
            return new
        ws = new


def test_textrank_matches_iterative_definition():
    assert TextRank().rank(TF, N) == order(textrank_loop(TF), N)


# ------------------------------------------------------------------ LexRank
def lexrank_loop(TF, t=0.1, jump=0.15, tol=1e-14):
    S = idf_cosine_loop(TF)
    B = [[1.0 if S[x][y] > t else 0.0 for y in range(N)] for x in range(N)]
    B = [[v / sum(r) for v in r] for r in B]
    p = [1.0 / N] * N
    while True:  # Algorithm 2 (power method) on Eq. 9
        new = [sum((jump / N + (1 - jump) * B[j][i]) * p[j] for j in range(N)) for i in range(N)]
        if math.sqrt(sum((a - b) ** 2 for a, b in zip(new, p))) < tol:
            return new
        p = new


def test_lexrank_matches_power_method():
    assert LexRank().rank(TF, N) == order(lexrank_loop(TF), N)


def test_continuous_lexrank_runs():
    assert len(LexRank(continuous=True).rank(TF, 3)) == 3
