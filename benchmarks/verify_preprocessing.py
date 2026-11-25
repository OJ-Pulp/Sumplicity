"""Verify that Sumy and Sumplicity see identical preprocessed text.

For every sentence of every sampled benchmark document, compares the multiset
of terms each Sumy summarizer works with (read through that summarizer's own
internal methods) against the corresponding column of Sumplicity's TF matrix.
Writes results/preprocessing_check.json and exits non-zero on any mismatch.

Usage: python benchmarks/verify_preprocessing.py [--config sumplicity|default]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import common as C

ALGOS = None
CONFIG = "sumplicity"


def _init(config):
    global ALGOS, CONFIG
    CONFIG = config
    ALGOS = {name: C.make_sumy(name, config) for name in C.ALGORITHMS}


def sumy_views(name, summarizer, doc):
    """Per-sentence term multisets as seen inside each Sumy summarizer."""
    sents = doc.sentences
    if name in ("TextRank", "LexRank", "Degree"):
        return [Counter(summarizer._to_words_set(s)) for s in sents]
    if name in ("SumBasic", "KL"):
        return [Counter(summarizer._get_content_words_in_sentence(s)) for s in sents]
    if name == "Luhn":  # rate_sentence stems every word of the sentence
        return [Counter(summarizer.stem_word(w) for w in s.words) for s in sents]
    if name == "LSA":
        d = summarizer._create_dictionary(doc)
        inv = {i: w for w, i in d.items()}
        M = summarizer._create_matrix(doc, d)
        return [Counter({inv[r]: int(M[r, c]) for r in range(M.shape[0]) if M[r, c]})
                for c in range(M.shape[1])]
    raise ValueError(name)


def check(args):
    ds, doc_id, sentences = args
    pp = C.preprocessor()
    TF, keys = pp.tf(sentences, keys=True)
    ref = [Counter({keys[r]: int(TF[r, c]) for r in range(TF.shape[0]) if TF[r, c]})
           for c in range(TF.shape[1])]
    out = {"sentences": len(sentences), "mismatch": {}}
    for name, (summarizer, tok) in ALGOS.items():
        doc = C.sumy_document(sentences, tok)
        view = sumy_views(name, summarizer, doc)
        bad = [i for i, (a, b) in enumerate(zip(ref, view)) if a != b]
        if bad:
            i = bad[0]
            out["mismatch"][name] = {"n": len(bad), "example": sentences[i],
                                     "only_sumplicity": dict(ref[i] - view[i]),
                                     "only_sumy": dict(view[i] - ref[i])}
    return ds, doc_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sumplicity")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    jobs = []
    for ds in C.DATASETS:
        df = C.load_sample(ds, a.n)
        jobs += [(ds, i, s) for i, s in enumerate(df["sentences"])]
    with Pool(os.cpu_count() - 2, initializer=_init, initargs=(a.config,)) as p:
        res = p.map(check, jobs, chunksize=8)

    report = {}
    for ds in C.DATASETS:
        r = [o for d, _, o in res if d == ds]
        report[ds] = {"documents": len(r), "sentences": sum(o["sentences"] for o in r),
                      "mismatched_sentences": {n: sum(o["mismatch"].get(n, {}).get("n", 0) for o in r)
                                               for n in C.ALGORITHMS}}
        ex = next((o["mismatch"] for o in r if o["mismatch"]), None)
        if ex:
            report[ds]["first_example"] = ex
    report["config"] = a.config
    path = os.path.join(os.path.dirname(__file__), "results",
                        "preprocessing_check.json" if a.config == "sumplicity" else f"preprocessing_check_{a.config}.json")
    json.dump(report, open(path, "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str)[:4000])
    total = sum(sum(v["mismatched_sentences"].values()) for k, v in report.items() if k != "config")
    print(f"\nTOTAL mismatched sentence-views: {total}")
    sys.exit(1 if total and a.config == "sumplicity" else 0)


if __name__ == "__main__":
    main()
