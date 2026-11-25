"""Retrieve -> compress -> generate demonstration (reviewer issue I).

A minimal retrieval-augmented generation (RAG) pipeline on HotpotQA
(distractor setting, validation split). Each question comes with ten Wikipedia
paragraphs, two of which contain the supporting facts. Four context-building
strategies are compared; the same LLM answers every question:

  full        all ten paragraphs (no retrieval, no compression)
  bm25        top-4 paragraphs by BM25 (Sumplicity's preprocessor.bm25)
  bm25+lex    top-4 paragraphs, compressed to 40% of their sentences with
              generic (query-agnostic) LexRank
  bm25+qlex   top-4 paragraphs, compressed to 40% with query-biased LexRank
              (Otterbacher et al., 2005) written as a 15-line Sumplicity
              subclass: the random-jump vector is proportional to each
              sentence's BM25 similarity to the question.

Reported per strategy: exact match (EM) and token F1 against the gold answer
(official HotpotQA normalisation), the fraction of gold supporting-fact
sentences kept in the context, LLM input tokens, and compression time.

Requires ANTHROPIC_API_KEY. Usage:
  python benchmarks/rag_demo.py --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
import pandas as pd

import common as C
from sumplicity import BaseSummarizer, LexRank, preprocessor
from sumplicity._math import cos_sim_mat, markov_chain

MODEL = "claude-opus-5-5"
TOP_PARAGRAPHS = 4
KEEP = 0.4
OUT = os.path.join(os.path.dirname(__file__), "results", "rag")
HOTPOT = dict(repo="hotpotqa/hotpot_qa", revision="1908d6afbbead072334abe2965f91bd2709910ab",
              file="distractor/validation-00000-of-00001.parquet")
SYSTEM = (
    "Answer the question using only the provided context. Reply with the answer "
    "alone: a short span of a few words copied from the context, or yes/no for "
    "yes/no questions. Do not explain. If the context does not contain the "
    "answer, reply exactly: unknown"
)
PP = preprocessor()


# ---------------------------------------------------------------- Sumplicity
class QueryLexRank(BaseSummarizer):
    """Query-biased LexRank: PageRank with a question-dependent random jump."""

    def __init__(self, query: str, damping: float = 0.85):
        super().__init__()
        self.query, self.damping = query, damping

    def rank(self, TF, top_k, terms=None):
        q = np.array([t in set(PP.query_terms(self.query)) for t in terms], dtype=float)
        bias = q @ PP.bm25(TF) + 1e-3                    # relevance of each sentence
        bias = bias / bias.sum()
        S = cos_sim_mat(TF)
        S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-10, None)
        Z = (1 - self.damping) * np.tile(bias, (len(bias), 1)) + self.damping * S
        scores = markov_chain(Z, v=bias)
        return [int(i) for i in np.argsort(scores)[::-1][:top_k]]


def bm25_paragraphs(question: str, paragraphs: list, top: int) -> list:
    TF, keys = PP.tf([" ".join(p) for p in paragraphs], keys=True)
    q = set(PP.query_terms(question))
    w = PP.bm25(TF)[[i for i, k in enumerate(keys) if k in q]].sum(axis=0)
    return sorted(np.argsort(-w, kind="stable")[:top].tolist())


def build_context(row, strategy: str):
    titles, paras = row["titles"], row["paragraphs"]
    t0 = time.perf_counter()
    if strategy == "full":
        chosen = [(i, j) for i in range(len(paras)) for j in range(len(paras[i]))]
    else:
        keep_p = bm25_paragraphs(row["question"], paras, TOP_PARAGRAPHS)
        chosen = [(i, j) for i in keep_p for j in range(len(paras[i]))]
        if strategy != "bm25":
            sents = [paras[i][j] for i, j in chosen]
            k = max(1, round(KEEP * len(sents)))
            try:
                TF, terms = PP.tf(sents, keys=True)
                if strategy == "bm25+lex":
                    idx = LexRank().rank(TF, k)
                else:
                    idx = QueryLexRank(row["question"]).rank(TF, k, terms=terms)
                chosen = [chosen[i] for i in sorted(idx)]
            except ValueError:  # no content terms
                chosen = chosen[:k]
    sec = time.perf_counter() - t0
    blocks = {}
    for i, j in chosen:
        blocks.setdefault(i, []).append(paras[i][j].strip())
    text = "\n\n".join(f"Title: {titles[i]}\n{' '.join(s)}" for i, s in blocks.items())
    gold = {(t, s) for t, s in row["support"]}
    kept = {(titles[i], j) for i, j in chosen}
    recall = len(gold & kept) / len(gold) if gold else 1.0
    return text, sec, recall


# ---------------------------------------------------------------- scoring
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(pt), common / len(gt)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- LLM
CLIENT = None


def ask(context: str, question: str) -> dict:
    try:
        r = CLIENT.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
    except anthropic.BadRequestError as e:
        return {"answer": "", "error": f"bad_request: {e.message}", "in_tok": 0, "out_tok": 0}
    except anthropic.APIStatusError as e:
        return {"answer": "", "error": f"status_{e.status_code}", "in_tok": 0, "out_tok": 0}
    except anthropic.APIConnectionError:
        return {"answer": "", "error": "connection", "in_tok": 0, "out_tok": 0}
    if r.stop_reason == "refusal":
        return {"answer": "", "error": "refusal", "in_tok": r.usage.input_tokens,
                "out_tok": r.usage.output_tokens}
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    return {"answer": text, "error": None, "model": r.model,
            "in_tok": r.usage.input_tokens, "out_tok": r.usage.output_tokens}


def load(n: int) -> pd.DataFrame:
    df = pd.read_parquet(C.hf_file(HOTPOT["repo"], HOTPOT["revision"], HOTPOT["file"]))
    df = df.sample(n=n, random_state=C.SEED).reset_index(drop=True)
    df["titles"] = df["context"].map(lambda c: list(c["title"]))
    df["paragraphs"] = df["context"].map(lambda c: [list(s) for s in c["sentences"]])
    df["support"] = df["supporting_facts"].map(lambda s: list(zip(s["title"], s["sent_id"])))
    return df


def main():
    global CLIENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="reuse successful calls in results/rag/rag_results.jsonl; re-send only failed ones")
    a = ap.parse_args()
    CLIENT = anthropic.Anthropic(max_retries=6)
    os.makedirs(OUT, exist_ok=True)

    df = load(a.n)
    strategies = ["full", "bm25", "bm25+lex", "bm25+qlex"]
    jobs = []
    for qi, row in df.iterrows():
        for s in strategies:
            ctx, sec, rec = build_context(row, s)
            jobs.append({"q": qi, "id": row["id"], "strategy": s, "question": row["question"],
                         "gold": row["answer"], "context": ctx, "compress_sec": sec,
                         "support_recall": rec})

    def run(job):
        return {**job, **ask(job["context"], job["question"])}

    done = {}
    path = os.path.join(OUT, "rag_results.jsonl")
    if a.resume and os.path.exists(path):  # keep successful calls, re-send failed ones
        for line in open(path):
            r = json.loads(line)
            if not r.get("error"):
                done[(r["q"], r["strategy"])] = r
        print(f"resume: {len(done)} successful calls kept, {len(jobs) - len(done)} to send")
    todo = [j for j in jobs if (j["q"], j["strategy"]) not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        fresh = {(r["q"], r["strategy"]): r for r in ex.map(run, todo)}
    res = []
    for j in jobs:
        key = (j["q"], j["strategy"])
        res.append(fresh[key] if key in fresh else {**j, **done[key]})
    for r in res:
        r["em"] = float(normalize(r["answer"]) == normalize(r["gold"]))
        r["f1"] = f1(r["answer"], r["gold"])
    with open(os.path.join(OUT, "rag_results.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps({k: v for k, v in r.items() if k != "context"}) + "\n")

    out = pd.DataFrame(res)
    summ = out.groupby("strategy", sort=False).agg(
        EM=("em", "mean"), F1=("f1", "mean"), support_recall=("support_recall", "mean"),
        input_tokens=("in_tok", "mean"), compress_ms=("compress_sec", lambda x: 1000 * x.mean()),
        errors=("error", lambda x: x.notna().sum()))
    rng = np.random.default_rng(C.SEED)
    for s in strategies:  # 95% bootstrap CIs over questions
        v = out[out.strategy == s]["f1"].to_numpy()
        means = v[rng.integers(0, len(v), (10_000, len(v)))].mean(axis=1)
        summ.loc[s, "F1_lo"], summ.loc[s, "F1_hi"] = np.percentile(means, [2.5, 97.5])
    summ.to_csv(os.path.join(OUT, "rag_summary.csv"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
