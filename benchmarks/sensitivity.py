"""Sensitivity of the baseline error to the constants the papers leave open.

Re-runs only Sumplicity, with alternative values of each open constant, and
recomputes Sumy's ROUGE-1 baseline error against each variant using the Sumy
results already stored in results/<dataset>/rouge.csv. No algorithm changes:
only constructor arguments differ.

  KL-Sum   smoothing lam      0.001, 0.1     (paper: "a small amount"; default 0.01)
  Luhn     min_freq           3              (paper: "stipulated low frequency"; default 2)
  LexRank  jump probability   0.10, 0.20     (paper: "typically [0.1, 0.2]"; default 0.15)

Writes results/sensitivity.csv.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon

import common as C
from sumplicity import KL_Sum, LexRank, Luhn_sum

RES = os.path.join(os.path.dirname(__file__), "results")
VARIANTS = {
    ("KL", "lam=0.01 (default)"): lambda: KL_Sum(lam=0.01),
    ("KL", "lam=0.001"): lambda: KL_Sum(lam=0.001),
    ("KL", "lam=0.1"): lambda: KL_Sum(lam=0.1),
    ("Luhn", "min_freq=2 (default)"): lambda: Luhn_sum(min_freq=2),
    ("Luhn", "min_freq=3"): lambda: Luhn_sum(min_freq=3),
    ("LexRank", "jump=0.15 (default)"): lambda: LexRank(jump=0.15),
    ("LexRank", "jump=0.10"): lambda: LexRank(jump=0.10),
    ("LexRank", "jump=0.20"): lambda: LexRank(jump=0.20),
}
_DS = None


def _work(args):
    doc, sentences, reference, k = args
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    out = []
    for (algo, label), make in VARIANTS.items():
        idx = C.run_sumplicity(make(), sentences, k)
        cand = "\n".join(sentences[i] for i in idx)
        out.append({"doc": doc, "algo": algo, "variant": label,
                    "rouge1_f": sc.score(reference, cand)["rouge1"].fmeasure})
    return out


def main():
    rows = []
    for ds in C.DATASETS:
        k = C.DATASETS[ds]["k"]
        df = C.load_sample(ds, 1000)
        jobs = [(i, s, r, k) for i, (s, r) in enumerate(zip(df["sentences"], df["reference"]))]
        with Pool(max(1, os.cpu_count() - 2)) as p:
            res = [r for chunk in p.imap(_work, jobs, chunksize=8) for r in chunk]
        sp = pd.DataFrame(res)
        sumy = pd.read_csv(os.path.join(RES, ds, "rouge.csv"), keep_default_na=False)
        for (algo, label), g in sp.groupby(["algo", "variant"]):
            sy = sumy[sumy.system == f"sumy:{algo}"].set_index("doc")["rouge1_f"]
            a = g.set_index("doc")["rouge1_f"].reindex(sy.index)
            rows.append({"dataset": ds, "algo": algo, "variant": label,
                         "sumplicity_r1": a.mean(), "sumy_r1": sy.mean(),
                         "baseline_error_pct": 100 * (sy.mean() - a.mean()) / a.mean(),
                         "wilcoxon_p": wilcoxon(sy, a, zero_method="zsplit").pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "sensitivity.csv"), index=False)
    print(out.pivot_table(index=["algo", "variant"], columns="dataset",
                          values="baseline_error_pct").round(1).to_string())


if __name__ == "__main__":
    main()
