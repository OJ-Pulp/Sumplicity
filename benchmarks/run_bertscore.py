"""BERTScore F1 for every system summary (reviewer issue C, second metric).

Reads results/<dataset>/rouge.csv (written by run_quality.py) and writes
results/<dataset>/bertscore.csv with one row per (doc, system).

Model: roberta-base (layer 10, the bert-score default for this model), with
baseline rescaling so scores are spread over a readable range. Inputs longer
than the model's 512-token window are truncated by bert-score.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from bert_score import BERTScorer

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
S = os.environ.get("SUMPLICITY_SUFFIX", "")  # "_budget": word-budget files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    odir = os.path.join(OUT, a.dataset)
    df = pd.read_csv(os.path.join(odir, f"rouge{S}.csv"), keep_default_na=False)
    refs = C.load_sample(a.dataset, int(df["doc"].max()) + 1)["reference"]
    df["reference"] = df["doc"].map(refs)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = BERTScorer(model_type="roberta-base", lang="en", rescale_with_baseline=True, device=device)

    # Score each unique (candidate, reference) pair once.
    pairs = df[["summary", "reference"]].drop_duplicates().reset_index(drop=True)
    P, R, F = scorer.score(pairs["summary"].tolist(), pairs["reference"].tolist(), batch_size=a.batch)
    pairs["bertscore_p"], pairs["bertscore_r"], pairs["bertscore_f"] = P.numpy(), R.numpy(), F.numpy()
    out = df.merge(pairs, on=["summary", "reference"], how="left")
    out[["doc", "system", "bertscore_p", "bertscore_r", "bertscore_f"]].to_csv(
        os.path.join(odir, f"bertscore{S}.csv"), index=False)
    print(out.groupby("system")["bertscore_f"].mean().round(4))


if __name__ == "__main__":
    main()
