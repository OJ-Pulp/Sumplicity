"""Text preprocessing: sentence splitting, tokenization, stemming, and
construction of the term-by-sentence frequency (TF) matrix.

Every summarizer in Sumplicity consumes the TF matrix produced here, so
preprocessing is fully separated from the scoring mathematics.
"""
import string
from typing import List, Union

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}


def ensure_nltk_data(quiet: bool = True) -> None:
    """Download the NLTK resources Sumplicity needs if they are missing.

    To use a local/offline copy instead, place the resources in a directory
    and add it with ``nltk.data.path.append("<dir>")`` before importing.
    """
    import nltk

    for package, resource in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=quiet)


class preprocessor:
    """Tokenizes, stems and filters text, and builds term-frequency matrices.

    Rows of the TF matrix are (stemmed) terms; columns are text chunks
    (usually sentences).
    """

    def __init__(self, language: str = "english"):
        self.stemmer = SnowballStemmer(language)
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError as err:
            raise LookupError(
                "NLTK stopwords are not installed. Run "
                "`python -c \"import sumplicity; sumplicity.ensure_nltk_data()\"` "
                "or append a local nltk_data directory to nltk.data.path."
            ) from err

    def _keep(self, token: str) -> bool:
        return (
            token not in self.stop_words
            and token not in string.punctuation
            and token.strip() != ""
        )

    def query_terms(self, text: str, stem: bool = True) -> List[str]:
        """Extracts the query terms from the text."""
        chunks = self.tokenize(self.chunk(text))
        terms = []
        for chunk in chunks:
            for token in chunk:
                token = token.lower()
                if self._keep(token):
                    terms.append(self.stem(token) if stem else token)
        return terms

    def tf(self, text: List[str], include_terms: bool = False, keys: bool = False):
        """Builds the (terms x chunks) term-frequency matrix.

        Returns the TF matrix, optionally followed by the stemmed keys and/or
        the first surface form observed for each term.
        """
        tokens = self.tokenize(text)
        num_chunks = len(tokens)
        term_dict = {}
        original_tokens = []

        for i, chunk in enumerate(tokens):
            for token in chunk:
                token = token.lower()
                if self._keep(token):
                    base_token = self.stem(token)
                    if base_token not in term_dict:
                        term_dict[base_token] = np.zeros(num_chunks)
                        original_tokens.append(token)
                    term_dict[base_token][i] += 1

        if not term_dict:
            raise ValueError(
                "No content terms remained after preprocessing; the input is "
                "empty or contains only stop words and punctuation."
            )

        tf_matrix = np.vstack(list(term_dict.values()))
        if include_terms and keys:
            return tf_matrix, list(term_dict.keys()), original_tokens
        if include_terms:
            return tf_matrix, original_tokens
        if keys:
            return tf_matrix, list(term_dict.keys())
        return tf_matrix

    def idf(self, tf: np.ndarray) -> np.ndarray:
        """Computes the IDF score for each term in the document."""
        num_sent = tf.shape[1]
        return np.log(num_sent / (np.sum(tf > 0, axis=1) + 1)).reshape(-1, 1)

    def tfidf(self, tf: np.ndarray) -> np.ndarray:
        """Computes the TF-IDF score for each term in the document."""
        return tf * self.idf(tf)

    def bm25(self, tf: np.ndarray, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Computes the BM25 score for each term in the document."""
        idf = self.idf(tf)
        doc_len = np.sum(tf, axis=0, keepdims=True)
        avgdl = np.mean(doc_len)
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl)) * idf

    def tokenize(self, text: List[str]) -> List[List[str]]:
        return [word_tokenize(text_chunk) for text_chunk in text]

    def stem(self, token: str) -> str:
        return self.stemmer.stem(token)

    def chunk(self, text: str) -> List[str]:
        """Splits the text into chunks based on sentence boundaries."""
        return sent_tokenize(text)

    def _word_score(self, word: str) -> float:
        """Fraction of ASCII letters in a word (0 for very long tokens)."""
        word_len = len(word)
        if 0 < word_len < 15:
            return len([c for c in word if c in string.ascii_letters]) / word_len
        return 0.0

    def _sentence_score(self, sentence: List[str]) -> float:
        """Mean word score of a tokenized sentence (0 if too short/long)."""
        sentence_len = len(sentence)
        if 4 < sentence_len < 60:
            return float(np.sum([self._word_score(w) for w in sentence]) / sentence_len)
        return 0.0

    def clean(self, text: Union[str, List[str]], threshold: float = 0.75) -> List[str]:
        """Drops sentences that look like noise (tables, URLs, markup, etc.).

        Returns the surviving sentences as strings.
        """
        if isinstance(text, str):
            chunks = self.chunk(text)
        elif isinstance(text, list):
            chunks = text
        else:
            raise ValueError(
                f"Input must be a string or a list of strings. Got {type(text)}."
            )

        scores = np.array([self._sentence_score(s) for s in self.tokenize(chunks)])
        return [chunk for chunk, score in zip(chunks, scores) if score > threshold]
