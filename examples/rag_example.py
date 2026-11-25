"""A minimal retrieve -> compress -> generate pipeline with Sumplicity.

1. Retrieve: score passages against the question with BM25, computed from
   Sumplicity's preprocessor (no other retrieval library needed).
2. Compress: summarize the retrieved passages with LexRank to a word budget,
   so the language model receives a shorter context.
3. Generate: ask a language model (Claude, via the `anthropic` package) to
   answer from the compressed context. Without ANTHROPIC_API_KEY set, the
   script prints the prompt it would send instead.

Run:  python examples/rag_example.py
"""
import os

import numpy as np

from sumplicity import LexRank, preprocessor

PASSAGES = [
    "The Harlow River flooded the valley after three days of heavy rain. "
    "Water levels peaked at 6.2 metres on Tuesday, the highest since records began in 1921.",
    "Emergency crews evacuated about 1,800 residents from Millbrook and Easton. "
    "Most were housed overnight in the Millbrook community centre.",
    "The regional council approved a new flood barrier for Easton last spring. "
    "Construction had not started when the river rose.",
    "A local bakery donated bread to evacuated families. "
    "Volunteers also collected blankets and children's clothing.",
    "Forecasters expect the river to fall below flood level by Friday. "
    "Further rain is possible early next week.",
    "The town's football club postponed its weekend fixtures. "
    "The club's ground lies on the river's floodplain.",
]
QUESTION = "How many residents were evacuated, and where did they stay?"


def retrieve(question: str, passages: list, k: int = 3) -> list:
    """Top-k passages by BM25 score of the question's terms."""
    pp = preprocessor()
    TF, keys = pp.tf(passages, keys=True)        # terms x passages
    query = set(pp.query_terms(question))
    rows = [i for i, term in enumerate(keys) if term in query]
    scores = pp.bm25(TF)[rows].sum(axis=0)
    return [passages[i] for i in np.argsort(-scores, kind="stable")[:k]]


def compress(passages: list, max_words: int = 40) -> str:
    """Keep the most central sentences of the retrieved passages, in text order."""
    pp = preprocessor()
    sentences = [s for p in passages for s in pp.chunk(p)]
    kept = LexRank().summarize(sentences, top_k=None, max_words=max_words)
    return " ".join(s for s in sentences if s in kept)


def generate(context: str, question: str) -> str:
    prompt = f"Answer using only this context.\n\nContext: {context}\n\nQuestion: {question}"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "[ANTHROPIC_API_KEY not set; prompt that would be sent]\n" + prompt
    import anthropic

    reply = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5", max_tokens=2000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if reply.stop_reason == "refusal":
        return "[the model declined to answer]"
    return "".join(b.text for b in reply.content if b.type == "text")


if __name__ == "__main__":
    retrieved = retrieve(QUESTION, PASSAGES)
    context = compress(retrieved)
    print("Retrieved:", *retrieved, sep="\n  ")
    print("\nCompressed context:", context)
    print("\nAnswer:", generate(context, QUESTION))
