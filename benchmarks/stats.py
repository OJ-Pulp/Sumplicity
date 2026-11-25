"""Confidence intervals and significance tests (reviewer issue C).

For every dataset, metric and algorithm:
  * mean score of each system with a 95% percentile-bootstrap CI over
    documents (10,000 resamples);
  * the paired difference Sumplicity - Sumy (matched configuration) with a
    95% paired-bootstrap CI;
  * a two-sided Wilcoxon signed-rank test on the per-document differences,
    Holm-Bonferroni corrected over all algorithm x metric tests within a
    dataset;
  * a TOST equivalence test (alpha = .05): the two libraries are declared
    equivalent when the 90% paired-bootstrap CI of the difference lies
    entirely inside (-MARGIN, +MARGIN), and Sumplicity is declared
    non-inferior when the CI's lower bound exceeds -MARGIN. MARGIN = 0.01
    (one F1 point) was fixed before the matched-preprocessing results were
    seen.

Writes results/<dataset>/summary.csv, results/<dataset>/paired_tests.csv and
results/tables.tex (LaTeX tables used in the paper).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
# "_budget" selects the word-budget evaluation files (run_budget.py)
S = os.environ.get("SUMPLICITY_SUFFIX", "")
METRICS = {"rouge1_f": "R-1", "rouge2_f": "R-2", "rougeLsum_f": "R-L", "bertscore_f": "BS"}
B = 10_000
# Equivalence margin, fixed before the matched-preprocessing results were
# seen: 1 point (0.01) of F1 on every metric. Override with SUMPLICITY_MARGIN.
MARGIN = float(os.environ.get("SUMPLICITY_MARGIN", "0.01"))
DISPLAY = {"Luhn": "Luhn", "SumBasic": "SumBasic", "KL": "KL-Sum", "LSA": "LSA",
           "Degree": "Degree centrality", "TextRank": "TextRank", "LexRank": "LexRank"}
NICE = {"cnn_dailymail": "CNN/Daily Mail", "arxiv": "arXiv", "billsum": "BillSum"}


def bootstrap_means(x: np.ndarray, rng) -> np.ndarray:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)


def bootstrap_ci(x: np.ndarray, rng) -> tuple:
    means = bootstrap_means(x, rng)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(dataset: str) -> pd.DataFrame:
    d = os.path.join(OUT, dataset)
    df = pd.read_csv(os.path.join(d, f"rouge{S}.csv"), keep_default_na=False).drop(columns=["summary"])
    bs = os.path.join(d, f"bertscore{S}.csv")
    if os.path.exists(bs):
        df = df.merge(pd.read_csv(bs), on=["doc", "system"])
    return df


def analyse(dataset: str):
    rng = np.random.default_rng(C.SEED)
    df = load(dataset)
    metrics = [m for m in METRICS if m in df.columns]
    wide = {m: df.pivot(index="doc", columns="system", values=m) for m in metrics}

    summ = []
    for m in metrics:
        for sysname in wide[m].columns:
            x = wide[m][sysname].to_numpy()
            lo, hi = bootstrap_ci(x, rng)
            summ.append({"system": sysname, "metric": m, "mean": x.mean(), "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(summ)

    tests = []
    for other in ["sumy", "sumy_default"]:
        for algo in C.ALGORITHMS:
            for m in metrics:
                a = wide[m][f"sumplicity:{algo}"].to_numpy()
                b = wide[m][f"{other}:{algo}"].to_numpy()
                d = a - b
                bm = bootstrap_means(d, rng)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                lo90, hi90 = np.percentile(bm, [5, 95])
                p = wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
                tests.append({"vs": other, "algo": algo, "metric": m, "sumplicity": a.mean(),
                              "other": b.mean(), "diff": d.mean(), "diff_lo": lo, "diff_hi": hi,
                              "diff_lo90": lo90, "diff_hi90": hi90,
                              # TOST equivalence (alpha=.05): 90% CI inside (-margin, +margin)
                              "equivalent": bool(-MARGIN < lo90 and hi90 < MARGIN),
                              # one-sided non-inferiority (alpha=.05): Sumplicity not worse by >= margin
                              "noninferior": bool(lo90 > -MARGIN),
                              "rel_pct": 100 * d.mean() / b.mean(), "p": p,
                              "win": (d > 0).mean(), "tie": (d == 0).mean()})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    for other in tests["vs"].unique():
        mask = tests["vs"] == other
        tests.loc[mask, "p_holm"] = holm(tests.loc[mask, "p"].to_numpy())
    d = os.path.join(OUT, dataset)
    summ.to_csv(os.path.join(d, f"summary{S}.csv"), index=False)
    tests.to_csv(os.path.join(d, f"paired_tests{S}.csv"), index=False)
    return summ, tests


def star(p):
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def latex_table(dataset: str, summ: pd.DataFrame, tests: pd.DataFrame) -> str:
    metrics = [m for m in METRICS if m in summ["metric"].unique()]
    mean = summ.set_index(["system", "metric"])["mean"]
    t = tests[tests.vs == "sumy"].set_index(["algo", "metric"])
    cols = " ".join(["r"] * len(metrics))
    head = " & ".join(METRICS[m] for m in metrics)
    lines = [
        r"\begin{tabular}{l " + cols + " " + cols + "}", r"\toprule",
        rf" & \multicolumn{{{len(metrics)}}}{{c}}{{Sumplicity}} & \multicolumn{{{len(metrics)}}}{{c}}{{Sumy (matched)}} \\",
        rf"\cmidrule(lr){{2-{1+len(metrics)}}}\cmidrule(lr){{{2+len(metrics)}-{1+2*len(metrics)}}}",
        rf"Algorithm & {head} & {head} \\", r"\midrule",
    ]
    for algo in C.ALGORITHMS:
        cells_a, cells_b = [], []
        for m in metrics:
            a, b = mean[(f"sumplicity:{algo}", m)], mean[(f"sumy:{algo}", m)]
            s = star(t.loc[(algo, m), "p_holm"])
            fa = f"{a:.3f}".lstrip("0") if a >= 0 else f"{a:.3f}"
            fb = f"{b:.3f}".lstrip("0") if b >= 0 else f"{b:.3f}"
            if a > b and s:
                fa = r"\mathbf{" + fa + "}" + s
            elif b > a and s:
                fb = r"\mathbf{" + fb + "}" + s
            cells_a.append(f"${fa}$"); cells_b.append(f"${fb}$")
        lines.append(f"{DISPLAY[algo]} & " + " & ".join(cells_a + cells_b) + r" \\")
    lines.append(r"\midrule")
    for base, label in [("baseline:Lead", f"Lead-{C.DATASETS[dataset]['k']}"), ("baseline:Random", "Random")]:
        cells = [f"{mean[(base, m)]:.3f}".lstrip("0") for m in metrics]
        lines.append(f"{label} & " + " & ".join(cells) + " & " + " & ".join(["--"] * len(metrics)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    tex = []
    for ds in C.DATASETS:
        if not os.path.exists(os.path.join(OUT, ds, f"rouge{S}.csv")):
            continue
        summ, tests = analyse(ds)
        tex.append(f"% ---- {NICE[ds]} (k = {C.DATASETS[ds]['k']}) ----\n" + latex_table(ds, summ, tests))
        t = tests[tests.vs == "sumy"]
        print(f"\n=== {ds}")
        print(t[["algo", "metric", "sumplicity", "other", "diff", "diff_lo", "diff_hi", "rel_pct", "p_holm", "win"]]
              .round(4).to_string(index=False))
        sig = t["p_holm"] < .05
        print(f"--- vs sumy (same preprocessing): Sumplicity significantly higher {((t['diff'] > 0) & sig).sum()}/{len(t)}, "
              f"significantly lower {((t['diff'] < 0) & sig).sum()}/{len(t)}, no significant difference {(~sig).sum()}/{len(t)}; "
              f"equivalent within ±{MARGIN} {t['equivalent'].sum()}/{len(t)}; non-inferior {t['noninferior'].sum()}/{len(t)}")
        td = tests[tests.vs == "sumy_default"]
        print(f"--- vs sumy_default: Sumplicity better & significant in "
              f"{((td['diff'] > 0) & (td['p_holm'] < .05)).sum()}/{len(td)}, worse & significant in "
              f"{((td['diff'] < 0) & (td['p_holm'] < .05)).sum()}/{len(td)}")
    open(os.path.join(OUT, f"tables{S}.tex"), "w").write("\n\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
