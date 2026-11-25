"""Recompute only Sumplicity's side of the quality benchmark.

Used after a Sumplicity change that leaves Sumy untouched (e.g. tie-breaking):
re-selects sentences for every sumplicity:* system, rewrites those rows in
selections.jsonl and rouge.csv, and re-scores BERTScore only for summaries that
changed. Afterwards run run_budget.py, run_bertscore with SUMPLICITY_SUFFIX
(incrementally, via this script's --budget mode) and stats.py.

Usage:  python benchmarks/refresh_sumplicity.py [--budget]
"""
import argparse
import json
import os
from multiprocessing import Pool

import pandas as pd
from rouge_score import rouge_scorer

import common as C

RES = os.path.join(os.path.dirname(__file__), "results")


def _select(args):
    ds, doc, sents = args
    k = C.DATASETS[ds]["k"]
    return doc, {f"sumplicity:{n}": C.run_sumplicity(o, sents, k)
                 for n, o in C.sumplicity_systems().items()}


def bertscore_missing(df_rows: pd.DataFrame, old_bs: pd.DataFrame, refs) -> pd.DataFrame:
    """BERTScore for (doc, system) rows whose summary is new; reuse the rest."""
    import torch
    from bert_score import BERTScorer

    keep = df_rows.merge(old_bs, on=["doc", "system"], how="left")
    todo = keep[keep["bertscore_f"].isna()]
    if len(todo):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        scorer = BERTScorer(model_type="roberta-base", lang="en", rescale_with_baseline=True, device=device)
        P, R, F = scorer.score(todo["summary"].tolist(), todo["doc"].map(refs).tolist(), batch_size=64)
        keep.loc[todo.index, ["bertscore_p", "bertscore_r", "bertscore_f"]] = list(
            zip(P.numpy(), R.numpy(), F.numpy()))
    return keep[["doc", "system", "bertscore_p", "bertscore_r", "bertscore_f"]], len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", action="store_true",
                    help="only refresh BERTScore for rouge_budget.csv (after run_budget.py)")
    a = ap.parse_args()
    for ds in C.DATASETS:
        d = os.path.join(RES, ds)
        df = C.load_sample(ds, 1000)
        refs = df["reference"]
        if a.budget:
            rb = pd.read_csv(os.path.join(d, "rouge_budget.csv"), keep_default_na=False)
            old = pd.read_csv(os.path.join(d, "bertscore_budget.csv"))
            old = old.merge(pd.read_csv(os.path.join(d, "rouge_budget.csv.prev"), keep_default_na=False)[["doc", "system", "summary"]],
                            on=["doc", "system"])
            cur = rb[["doc", "system", "summary"]]
            same = cur.merge(old, on=["doc", "system", "summary"], how="left")
            bs, n = bertscore_missing(cur, same.drop(columns=["summary"]).dropna(), refs)
            bs.to_csv(os.path.join(d, "bertscore_budget.csv"), index=False)
            print(ds, "budget BERTScore re-scored:", n)
            continue
        sel = [json.loads(l) for l in open(os.path.join(d, "selections.jsonl"))]
        jobs = [(ds, r["doc"], df["sentences"][r["doc"]]) for r in sel]
        with Pool(max(1, os.cpu_count() - 2)) as p:
            new = dict(p.map(_select, jobs, chunksize=8))
        changed = 0
        for r in sel:
            for key, idx in new[r["doc"]].items():
                changed += r["sel"][key] != idx
                r["sel"][key] = idx
        with open(os.path.join(d, "selections.jsonl"), "w") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        rouge = pd.read_csv(os.path.join(d, "rouge.csv"), keep_default_na=False)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
        rows = []
        for r in sel:
            sents = df["sentences"][r["doc"]]
            for key, idx in new[r["doc"]].items():
                cand = "\n".join(sents[i] for i in idx)
                row = {"doc": r["doc"], "system": key, "summary": cand}
                for m, v in scorer.score(refs[r["doc"]], cand).items():
                    row[f"{m}_p"], row[f"{m}_r"], row[f"{m}_f"] = v.precision, v.recall, v.fmeasure
                rows.append(row)
        rouge = pd.concat([rouge[~rouge.system.str.startswith("sumplicity:")], pd.DataFrame(rows)])
        rouge = rouge.sort_values(["doc", "system"]).reset_index(drop=True)
        rouge.to_csv(os.path.join(d, "rouge.csv"), index=False)
        old_bs = pd.read_csv(os.path.join(d, "bertscore.csv"))
        prev = pd.read_csv(os.path.join(d, "rouge.csv.prev"), keep_default_na=False)[["doc", "system", "summary"]]
        old_bs = old_bs.merge(prev, on=["doc", "system"])
        cur = rouge[["doc", "system", "summary"]]
        same = cur.merge(old_bs, on=["doc", "system", "summary"], how="left").drop(columns=["summary"]).dropna()
        bs, n = bertscore_missing(cur, same, refs)
        bs.to_csv(os.path.join(d, "bertscore.csv"), index=False)
        print(ds, "selections changed:", changed, "BERTScore re-scored:", n)


if __name__ == "__main__":
    main()
