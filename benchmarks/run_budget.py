"""Word-budget evaluation: every summary truncated to the same length.

The original papers limit summaries by words (TextRank 100, SumBasic and
KL-Sum 250, LexRank 665 bytes), whereas the main benchmark fixes the number of
sentences, so systems that prefer long sentences produce longer summaries.
Here every system's summary (its selected sentences, in document order) is cut
to the same word budget, as in the DUC evaluations, and re-scored. This
separates which content a system selects from how long its sentences are.

Budget per dataset = median reference-summary length in words (from the
sample). Summaries shorter than the budget are left as they are.

Writes results/<dataset>/rouge_budget.csv (same columns as rouge.csv).
Run run_bertscore.py --suffix _budget and stats.py --suffix _budget afterwards.
"""
import argparse
import json
import os

import pandas as pd
from rouge_score import rouge_scorer

import common as C

RES = os.path.join(os.path.dirname(__file__), "results")


def truncate(sentences, budget):
    """Keep whole sentences while they fit, then cut the next one to the budget."""
    out, used = [], 0
    for s in sentences:
        words = s.split()
        if used + len(words) <= budget:
            out.append(s)
            used += len(words)
        else:
            if budget - used > 0:
                out.append(" ".join(words[: budget - used]))
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="score only the first N documents")
    a = ap.parse_args()

    df = C.load_sample(a.dataset, 1000)
    budget = int(df["reference"].map(lambda x: len(x.split())).median())
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    for line in open(os.path.join(RES, a.dataset, "selections.jsonl")):
        r = json.loads(line)
        if a.limit is not None and r["doc"] >= a.limit:
            continue
        sents, ref = df["sentences"][r["doc"]], df["reference"][r["doc"]]
        for sysname, idx in r["sel"].items():
            cand = "\n".join(truncate([sents[i] for i in sorted(idx)], budget))
            row = {"doc": r["doc"], "system": sysname, "summary": cand,
                   "words": len(cand.split()), "budget": budget}
            for m, v in scorer.score(ref, cand).items():
                row[f"{m}_p"], row[f"{m}_r"], row[f"{m}_f"] = v.precision, v.recall, v.fmeasure
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, a.dataset, "rouge_budget.csv"), index=False)
    print(f"{a.dataset}: budget {budget} words")
    print(out.groupby("system")[["words", "rouge1_f", "rouge2_f", "rougeLsum_f"]].mean().round(4))


if __name__ == "__main__":
    main()
