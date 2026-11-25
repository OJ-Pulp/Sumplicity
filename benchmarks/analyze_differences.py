"""Evidence for *why* Sumplicity and Sumy scores differ, per algorithm.

For every dataset and algorithm, compares the two libraries' selections:
  * agreement: share of documents with identical selections, and mean Jaccard
    overlap of the selected sentence sets;
  * summary length in words and mean position of the selected sentences
    (0 = first sentence of the document, 1 = last);
  * ROUGE-1 precision and recall (F1 differences are decomposed into these).

Writes results/differences.csv and prints a table per dataset.
"""
import json
import os

import numpy as np
import pandas as pd

import common as C

RES = os.path.join(os.path.dirname(__file__), "results")


def analyse(ds: str) -> pd.DataFrame:
    df = C.load_sample(ds, 1000)
    sel = [json.loads(l) for l in open(os.path.join(RES, ds, "selections.jsonl"))]
    rouge = pd.read_csv(os.path.join(RES, ds, "rouge.csv"), keep_default_na=False)
    pr = rouge.groupby("system")[["rouge1_p", "rouge1_r", "rouge1_f"]].mean()
    rows = []
    for algo in C.ALGORITHMS:
        a_key, b_key = f"sumplicity:{algo}", f"sumy:{algo}"
        same, jac, len_a, len_b, pos_a, pos_b = [], [], [], [], [], []
        for r in sel:
            sents = df["sentences"][r["doc"]]
            a, b = set(r["sel"][a_key]), set(r["sel"][b_key])
            same.append(a == b)
            jac.append(len(a & b) / len(a | b) if a | b else 1.0)
            len_a.append(sum(len(sents[i].split()) for i in a))
            len_b.append(sum(len(sents[i].split()) for i in b))
            n = max(len(sents) - 1, 1)
            pos_a.append(np.mean([i / n for i in a]) if a else np.nan)
            pos_b.append(np.mean([i / n for i in b]) if b else np.nan)
        rows.append({
            "dataset": ds, "algo": algo,
            "identical_selection": np.mean(same), "jaccard": np.mean(jac),
            "words_sumplicity": np.mean(len_a), "words_sumy": np.mean(len_b),
            "position_sumplicity": np.nanmean(pos_a), "position_sumy": np.nanmean(pos_b),
            "r1_p_sumplicity": pr.loc[a_key, "rouge1_p"], "r1_p_sumy": pr.loc[b_key, "rouge1_p"],
            "r1_r_sumplicity": pr.loc[a_key, "rouge1_r"], "r1_r_sumy": pr.loc[b_key, "rouge1_r"],
            "r1_f_sumplicity": pr.loc[a_key, "rouge1_f"], "r1_f_sumy": pr.loc[b_key, "rouge1_f"],
        })
    ref_words = df["reference"].map(lambda x: len(x.split())).mean()
    out = pd.DataFrame(rows)
    out["reference_words"] = ref_words
    return out


def main():
    frames = [analyse(ds) for ds in C.DATASETS
              if os.path.exists(os.path.join(RES, ds, "selections.jsonl"))]
    out = pd.concat(frames)
    out.to_csv(os.path.join(RES, "differences.csv"), index=False)
    for ds, g in out.groupby("dataset", sort=False):
        print(f"\n=== {ds} (reference summary: {g.reference_words.iloc[0]:.0f} words)")
        print(g.drop(columns=["dataset", "reference_words"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
