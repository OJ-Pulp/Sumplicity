"""Run Sumy on Sumplicity's preprocessing.

Sumy summarizers read a sentence's words from ``sentence.words``, which Sumy
obtains by calling ``tokenizer.to_words(text)``. Each summarizer then
lower-cases the words, drops its stop words and applies its stemmer (except
KL, which never stems). This adapter plugs Sumplicity's preprocessor into that
first step, so ``to_words`` returns the *final* Sumplicity terms, already
filtered and stemmed, and turns Sumy's own steps into no-ops:

* stemmer   -> ``null_stemmer`` (only lower-cases; stems are already lower case)
* stop words -> empty set (stop words were already removed; filtering again
  could wrongly drop a stem that happens to spell a stop word, e.g. "wills" ->
  "will")

The result: every Sumy summarizer sees, for every sentence, exactly the terms
in the corresponding column of Sumplicity's TF matrix. ``verify_preprocessing.py``
checks this sentence by sentence.
"""
from __future__ import annotations

from typing import List

from sumplicity import preprocessor


class SumplicityTokenizer:
    """Drop-in replacement for ``sumy.nlp.tokenizers.Tokenizer``."""

    def __init__(self, pp: preprocessor | None = None):
        self.pp = pp or preprocessor()

    @property
    def language(self) -> str:
        return "english"

    def to_words(self, sentence: str) -> tuple:
        return tuple(self.pp.stem(t) for t in self.pp.terms(sentence))

    def to_sentences(self, paragraph: str) -> tuple:  # not used: sentences are pre-split
        return tuple(self.pp.chunk(paragraph))


def sumy_document(sentences: List[str], tokenizer) -> object:
    """Wrap pre-segmented sentences as a Sumy document (no re-segmentation)."""
    from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence

    return ObjectDocumentModel([Paragraph([Sentence(s, tokenizer) for s in sentences])])


def make_sumy(name: str, config: str = "matched"):
    """Return ``(summarizer, tokenizer)``.

    config="matched":    Sumy's own pipeline with the same components as
                         Sumplicity (tokenizer, word filter, stemmer, NLTK stop
                         words). Used for the paper's comparison.
    config="sumplicity": Sumy algorithms fed Sumplicity's final terms
                         (diagnostic only; bypasses Sumy's preprocessing).
    config="default":    Sumy exactly as its README shows (own tokenizer,
                         Snowball stemmer, Sumy's English stop-word list).
    """
    from sumplicity import preprocessor as _pp
    NLTK_STOP = sorted(_pp().stop_words)
    from sumy.nlp.stemmers import Stemmer, null_stemmer
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.reduction import ReductionSummarizer
    from sumy.summarizers.sum_basic import SumBasicSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words

    cls = {
        "Luhn": LuhnSummarizer, "SumBasic": SumBasicSummarizer, "KL": KLSummarizer,
        "LSA": LsaSummarizer, "Degree": ReductionSummarizer,  # Sumy has no degree centrality; its graph reduction is the counterpart
        "TextRank": TextRankSummarizer, "LexRank": LexRankSummarizer,
    }[name]
    if config == "matched":
        # Sumy's own pipeline, built from the same components Sumplicity uses:
        # NLTK word_tokenize + the same word filter (Sumy's Tokenizer), the NLTK
        # Snowball English stemmer, and the NLTK English stop-word list.
        s = cls(Stemmer("english"))
        s.stop_words = frozenset(NLTK_STOP)
        return s, Tokenizer("english")
    if config == "sumplicity":
        s = cls(null_stemmer)
        s.stop_words = frozenset()
        return s, SumplicityTokenizer()
    if config == "default":
        s = cls(Stemmer("english"))
        s.stop_words = get_stop_words("english")
        return s, Tokenizer("english")
    raise ValueError(config)


def run_sumy(system, sentences: List[str], k: int) -> List[int]:
    summarizer, tokenizer = system
    doc = sumy_document(sentences, tokenizer)
    chosen = summarizer(doc, k)
    index = {id(s): i for i, s in enumerate(doc.sentences)}
    return sorted(index[id(s)] for s in chosen)
