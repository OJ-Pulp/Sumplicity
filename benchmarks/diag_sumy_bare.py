"""Diagnostic: Sumy with its constructor defaults (no stemmer, no stop words).

Many published uses of Sumy instantiate summarizers without a stemmer or a
stop-word list. This script re-scores the CNN/DM sample with that "bare"
configuration so the effect of configuration alone can be quantified.
Output: results/cnn_dailymail/rouge_sumy_bare.csv
"""
import os
from multiprocessing import Pool

import pandas as pd
from rouge_score import rouge_scorer

import common as C

ALGOS = None
TOK = None


def _init():
    global ALGOS, TOK
    from sumy.nlp.tokenizers import Tokenizer
    TOK = Tokenizer("english")
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.reduction import ReductionSummarizer
    from sumy.summarizers.sum_basic import SumBasicSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    ALGOS = {"Luhn": LuhnSummarizer(), "SumBasic": SumBasicSummarizer(), "KL": KLSummarizer(),
             "LSA": LsaSummarizer(), "Degree": ReductionSummarizer(),
             "TextRank": TextRankSummarizer(), "LexRank": LexRankSummarizer()}


def _run(args):
    i, sents, k = args
    return i, {n: C.run_sumy((s, TOK), sents, k) for n, s in ALGOS.items()}


def main(ds="cnn_dailymail"):
    k = C.DATASETS[ds]["k"]
    df = C.load_sample(ds, 1000)
    with Pool(os.cpu_count() - 2, initializer=_init) as p:
        res = dict(p.map(_run, [(i, s, k) for i, s in enumerate(df["sentences"])], chunksize=8))
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    for i, sel in res.items():
        for n, idx in sel.items():
            cand = "\n".join(df["sentences"][i][j] for j in idx)
            s = sc.score(df["reference"][i], cand)
            rows.append({"doc": i, "system": f"sumy_bare:{n}", "summary": cand,
                         **{f"{m}_f": v.fmeasure for m, v in s.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.os.path.dirname(__file__) or ".", "results", ds, "rouge_sumy_bare.csv"), index=False)
    print(out.groupby("system")[["rouge1_f", "rouge2_f", "rougeLsum_f"]].mean().round(4))


if __name__ == "__main__":
    main()
