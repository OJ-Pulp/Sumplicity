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

@pytest.mark.parametrize("name,cls", ALL)
def test_returns_top_k_unique_sentences(name, cls):
    out = cls().summarize(SENTENCES, top_k=3)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert all(s in SENTENCES for s in out)


@pytest.mark.parametrize("name,cls", ALL)
def test_string_and_list_inputs_agree(name, cls):
    assert cls().summarize(TEXT, top_k=2) == cls().summarize(SENTENCES, top_k=2)


@pytest.mark.parametrize("name,cls", ALL)
def test_top_k_larger_than_document(name, cls):
    out = cls().summarize(SENTENCES, top_k=100)
    assert len(out) == len(SENTENCES)


@pytest.mark.parametrize("name,cls", ALL)
def test_deterministic(name, cls):
    assert cls().summarize(SENTENCES, 3) == cls().summarize(SENTENCES, 3)


def test_off_topic_sentence_not_selected_by_centrality_methods():
    for cls in (sumplicity.LexRank, sumplicity.TextRank, sumplicity.Graph_Reduction):
        top = cls().summarize(SENTENCES, top_k=2)
        assert not any("bakery" in s for s in top)


def test_sumbasic_redundancy_penalty():
    docs = [
        "Solar panels convert sunlight into electricity.",
        "Solar panels convert sunlight into electricity efficiently.",
        "Wind turbines generate power from moving air.",
    ]
    out = sumplicity.SumBasic().summarize(docs, top_k=2)
    assert any("Wind" in s for s in out)


def test_lsa_matches_full_svd():
    TF = preprocessor().tf(SENTENCES)
    _, s, Vt = np.linalg.svd(TF)
    full = np.sum(np.abs(Vt[: len(s)]) * s.reshape(-1, 1), axis=0)
    expected = [int(i) for i in np.argsort(full)[::-1][:3]]
    assert sumplicity.LSA_Sum().rank(TF, 3) == expected


def test_registry_complete():
    assert set(get_all_summarizer()) == {
        "Luhn", "SumBasic", "KL", "LSA", "GraphReduction", "TextRank", "LexRank"
    }
