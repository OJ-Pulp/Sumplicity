"""Summary-quality benchmark: Sumplicity vs. Sumy (+ Lead-k / Random-k).

Writes, per dataset,
  results/<dataset>/selections.jsonl  selected sentence indices per system/doc
  results/<dataset>/rouge.csv         per-document ROUGE-1/2/Lsum P/R/F1

Usage:
  python benchmarks/run_quality.py --dataset cnn_dailymail --n 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import pandas as pd
from rouge_score import rouge_scorer

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
_SYSTEMS = None


def _init():
    global _SYSTEMS
    _SYSTEMS = {}
    for name, obj in C.sumplicity_systems().items():
        _SYSTEMS[f"sumplicity:{name}"] = ("sumplicity", obj)
        _SYSTEMS[f"sumy:{name}"] = ("sumy", C.make_sumy(name, "matched"))
        _SYSTEMS[f"sumy_default:{name}"] = ("sumy", C.make_sumy(name, "default"))


def _select(args):
    doc_id, sentences, k = args
    out = {"doc": doc_id, "sel": {}, "sec": {}}
    for sysname, (lib, obj) in _SYSTEMS.items():
        t0 = time.perf_counter()
        idx = (C.run_sumplicity if lib == "sumplicity" else C.run_sumy)(obj, sentences, k)
        out["sec"][sysname] = time.perf_counter() - t0
        out["sel"][sysname] = idx
    out["sel"]["baseline:Lead"] = C.lead_k(sentences, k)
    out["sel"]["baseline:Random"] = C.random_k(sentences, k, doc_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()

    k = C.DATASETS[a.dataset]["k"]
    df = C.load_sample(a.dataset, a.n)
    odir = os.path.join(OUT, a.dataset)
    os.makedirs(odir, exist_ok=True)
    df[["row"]].to_csv(os.path.join(odir, "sample_rows.csv"), index_label="doc")

    jobs = [(i, s, k) for i, s in enumerate(df["sentences"])]
    with Pool(a.workers, initializer=_init) as pool:
        results = sorted(pool.imap_unordered(_select, jobs, chunksize=4), key=lambda r: r["doc"])
    with open(os.path.join(odir, "selections.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    for r in results:
        sents, ref = df["sentences"][r["doc"]], df["reference"][r["doc"]]
        for sysname, idx in r["sel"].items():
            cand = "\n".join(sents[i] for i in idx)
            sc = scorer.score(ref, cand)
            row = {"doc": r["doc"], "system": sysname, "summary": cand}
            for m, v in sc.items():
                row[f"{m}_p"], row[f"{m}_r"], row[f"{m}_f"] = v.precision, v.recall, v.fmeasure
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(odir, "rouge.csv"), index=False)
    print(pd.DataFrame(rows).groupby("system")[["rouge1_f", "rouge2_f", "rougeLsum_f"]].mean().round(4))


if __name__ == "__main__":
    main()
