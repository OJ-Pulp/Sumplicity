import numpy as np
import pytest

import sumplicity
from sumplicity import get_all_summarizer, preprocessor
from sumplicity._math import cos_sim_mat, markov_chain, overlap_sim_mat

TEXT = (
    "The river flooded the valley after three days of heavy rain. "
    "Emergency crews evacuated residents from the flooded valley towns. "
    "Rainfall totals in the valley exceeded historical records. "
    "A local bakery donated bread to the evacuated residents. "
    "Officials expect the river to recede by the end of the week. "
    "The mayor praised emergency crews for the rapid evacuation."
)
SENTENCES = preprocessor().chunk(TEXT)
ALL = list(get_all_summarizer().items())


# ---------- preprocessing ----------

def test_tf_shape_and_counts():
    pp = preprocessor()
    TF, keys = pp.tf(["Cats chase cats.", "Dogs sleep."], keys=True)
    assert TF.shape == (len(keys), 2)
    assert TF[keys.index("cat"), 0] == 2
    assert "the" not in keys and "." not in keys


def test_tf_rejects_empty_content():
    with pytest.raises(ValueError):
        preprocessor().tf(["the a an", "."])


def test_clean_accepts_string_and_list():
    pp = preprocessor()
    noisy = TEXT + " @@@ ### $$$ %%% ^^^ &&&."
    from_str = pp.clean(noisy)
    from_list = pp.clean(pp.chunk(noisy))
    assert from_str == from_list
    assert all(isinstance(s, str) for s in from_str)
    assert not any("@@@" in s for s in from_str)


def test_idf_and_bm25_shapes():
    pp = preprocessor()
    TF = pp.tf(SENTENCES)
    assert pp.idf(TF).shape == (TF.shape[0], 1)
    assert pp.bm25(TF).shape == TF.shape
    assert pp.tfidf(TF).shape == TF.shape


# ---------- math utilities ----------

def test_cosine_similarity_properties():
    TF = preprocessor().tf(SENTENCES)
    S = cos_sim_mat(TF)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)


def test_overlap_similarity_symmetric():
    S = overlap_sim_mat(preprocessor().tf(SENTENCES))
    assert np.allclose(S, S.T)


def test_markov_chain_stationary_distribution():
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    v = markov_chain(P, max_iter=1000, tol=1e-12)
    assert np.allclose(v, [5 / 6, 1 / 6], atol=1e-6)
    assert np.isclose(v.sum(), 1.0)


# ---------- summarizers ----------

def make(cls):
    # KL-Sum stops early when no sentence lowers the divergence (paper §4);
    # the generic length tests use it without that stopping rule.
    return cls(stop_early=False) if cls is sumplicity.KL_Sum else cls()


@pytest.mark.parametrize("name,cls", ALL)
def test_returns_top_k_unique_sentences(name, cls):
    out = make(cls).summarize(SENTENCES, top_k=3)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert all(s in SENTENCES for s in out)


@pytest.mark.parametrize("name,cls", ALL)
def test_string_and_list_inputs_agree(name, cls):
    assert cls().summarize(TEXT, top_k=2) == cls().summarize(SENTENCES, top_k=2)


@pytest.mark.parametrize("name,cls", ALL)
def test_top_k_larger_than_document(name, cls):
    out = make(cls).summarize(SENTENCES, top_k=100)
    assert len(out) == len(SENTENCES)


@pytest.mark.parametrize("name,cls", ALL)
def test_max_words_budget(name, cls):
    out = make(cls).summarize(SENTENCES, top_k=None, max_words=25)
    assert sum(len(s.split()) for s in out) <= 25
    assert out == make(cls).summarize(SENTENCES, top_k=len(SENTENCES))[: len(out)]


@pytest.mark.parametrize("name,cls", ALL)
def test_deterministic(name, cls):
    assert cls().summarize(SENTENCES, 3) == cls().summarize(SENTENCES, 3)


def test_off_topic_sentence_not_selected_by_centrality_methods():
    for cls in (sumplicity.LexRank, sumplicity.TextRank, sumplicity.DegreeCentrality):
        top = cls().summarize(SENTENCES, top_k=2)
        assert not any("bakery" in s for s in top)


def test_sumbasic_redundancy_penalty():
    # After the first solar sentence is chosen, its words' probabilities are
    # squared, so the most probable remaining word is "wind" and the second
    # pick is a wind sentence rather than the near-duplicate solar sentence.
    docs = [
        "Solar panels convert sunlight into electricity.",
        "Solar panels convert sunlight into electricity efficiently.",
        "Wind turbines generate power from moving air.",
        "Wind farms are growing.",
    ]
    out = sumplicity.SumBasic().summarize(docs, top_k=2)
    assert out[0] == docs[0]
    assert "Wind" in out[1]


def test_lsa_matches_full_svd():
    TF = preprocessor().tf(SENTENCES)
    _, s, Vt = np.linalg.svd(TF)               # full SVD gives the same leading factors
    n = int((s >= 0.5 * s[0]).sum())
    full = np.sqrt(((Vt[:n] ** 2) * (s[:n, None] ** 2)).sum(axis=0))
    expected = [int(i) for i in np.argsort(-full, kind="stable")[:3]]
    assert sumplicity.LSA_Sum().rank(TF, 3) == expected


def test_registry_complete():
    assert set(get_all_summarizer()) == {
        "Luhn", "SumBasic", "KL", "LSA", "Degree", "TextRank", "LexRank"
    }


def test_graph_reduction_is_deprecated_alias():
    with pytest.warns(DeprecationWarning):
        g = sumplicity.Graph_Reduction()
    assert isinstance(g, sumplicity.DegreeCentrality)


def test_position_matrix():
    TF, keys, P = preprocessor().tf(["The cats chase the cats.", "Dogs sleep."], keys=True, positions=True)
    cat = keys.index("cat")
    # the(stop) cats chase the(stop) cats ; punctuation is not a position
    assert P[0].tolist() == [-1, cat, keys.index("chase"), -1, cat]
    assert P[1].tolist() == [keys.index("dog"), keys.index("sleep"), -2, -2, -2]


def test_token_filter_drops_punctuation_numbers_and_quotes():
    pp = preprocessor()
    terms = pp.terms("``The U.S.C. says'' -- in 2014 -- that state-level rules don't apply.")
    assert terms == ["says", "state-level", "rules", "n't", "apply"]


def test_token_pattern_is_configurable():
    pp = preprocessor(token_pattern=r"^\w+$")
    assert "2014" in pp.terms("Rules changed in 2014.")
