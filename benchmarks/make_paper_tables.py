"""LaTeX for the paper's results tables, generated from the result files.

Table 4: quality (ROUGE-1 and BERTScore F1) per dataset and algorithm.
Table 5: baseline error of Sumy relative to the faithful implementation.
Table 6: algorithm-only speed and memory.
Writes results/paper_tables.tex.
"""
import os

import numpy as np
import pandas as pd

import common as C

RES = os.path.join(os.path.dirname(__file__), "results")
NAME = {"Luhn": "Luhn", "SumBasic": "SumBasic", "KL": "KL-Sum", "LSA": "LSA",
        "Degree": "Degree", "TextRank": "TextRank", "LexRank": "LexRank"}
DS = ["cnn_dailymail", "arxiv", "billsum"]


def fmt(x):
    return f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"$-${abs(x):.3f}".replace("0.", ".", 1)


def quality_table():
    lines = [r"\begin{tabular}{l" + "rr" * 6 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{CNN/Daily Mail} & \multicolumn{4}{c}{arXiv} & \multicolumn{4}{c}{BillSum} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
             r" & \multicolumn{2}{c}{R-1} & \multicolumn{2}{c}{BERTScore} & \multicolumn{2}{c}{R-1} & \multicolumn{2}{c}{BERTScore} & \multicolumn{2}{c}{R-1} & \multicolumn{2}{c}{BERTScore} \\",
             r"Algorithm" + " & Spl. & Sumy" * 6 + r" \\", r"\midrule"]
    tests = {ds: pd.read_csv(os.path.join(RES, ds, "paired_tests.csv")).query("vs == 'sumy'") for ds in DS}
    for algo in C.ALGORITHMS:
        cells = []
        for ds in DS:
            for m in ["rouge1_f", "bertscore_f"]:
                r = tests[ds].query("algo == @algo and metric == @m").iloc[0]
                a, b = fmt(r.sumplicity), fmt(r.other)
                if r.p_holm < 0.05:
                    if r.sumplicity > r.other:
                        a = r"\textbf{" + a + "}"
                    else:
                        b = r"\textbf{" + b + "}"
                if r.equivalent:
                    b += r"$^{\approx}$"
                cells += [a, b]
        lines.append(f"{NAME[algo]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def baseline_error_table():
    def err(path_suffix, vs):
        out = {}
        for ds in DS:
            t = pd.read_csv(os.path.join(RES, ds, f"paired_tests{path_suffix}.csv"))
            t = t[(t.vs == vs) & (t.metric == "rouge1_f")].set_index("algo")
            out[ds] = 100 * (t.other - t.sumplicity) / t.sumplicity
            out[ds + "_sig"] = t.p_holm < 0.05
        return out
    fixed, budget = err("", "sumy"), err("_budget", "sumy")
    bare = {}
    for ds in DS:
        b = pd.read_csv(os.path.join(RES, ds, "rouge_sumy_bare.csv"), keep_default_na=False).groupby("system").rouge1_f.mean()
        f = pd.read_csv(os.path.join(RES, ds, "rouge.csv"), keep_default_na=False).groupby("system").rouge1_f.mean()
        bare[ds] = {a: 100 * (b[f"sumy_bare:{a}"] - f[f"sumplicity:{a}"]) / f[f"sumplicity:{a}"] for a in C.ALGORITHMS}

    def cell(v, sig=True):
        s = f"{v:+.1f}".replace("-", "$-$")
        return s if sig else r"\textit{" + s + "}"

    lines = [r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Same components, $k$ sentences} & \multicolumn{3}{c}{Same components, word budget} & \multicolumn{3}{c}{Sumy constructor defaults} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Algorithm & CNN & arXiv & Bill & CNN & arXiv & Bill & CNN & arXiv & Bill \\", r"\midrule"]
    for algo in C.ALGORITHMS:
        c = [cell(fixed[ds][algo], fixed[ds + "_sig"][algo]) for ds in DS]
        c += [cell(budget[ds][algo], budget[ds + "_sig"][algo]) for ds in DS]
        c += [cell(bare[ds][algo]) for ds in DS]
        lines.append(f"{NAME[algo]} & " + " & ".join(c) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def speed_table():
    t = pd.read_json(os.path.join(RES, "speed", "speed.jsonl"), lines=True)
    m, t = t[t.get("mode") == "memory"], t[(t.get("mode") == "time") & (t.threads == "1-thread")]
    real = pd.read_csv(os.path.join(RES, "speed", "realdocs_summary.csv"))
    lines = [r"\begin{tabular}{l rrr rr rr rr r}", r"\toprule",
             r" & \multicolumn{3}{c}{Scoring time, 64K tokens} & \multicolumn{2}{c}{End to end $\times$} & \multicolumn{2}{c}{Real documents} & \multicolumn{2}{c}{Peak MiB, 64K} & 128K \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"Algorithm & Spl. (s) & Sumy (s) & $\times$ & 1K & 64K & median $\times$ & range $\times$ & Spl. & Sumy & Spl. (s) \\", r"\midrule"]
    for algo in C.ALGORITHMS:
        q = lambda lib, L, col, d=t: d[(d.lib == lib) & (d.algo == algo) & (d.length == L)][col]
        sp, sy = q("sumplicity", 64000, "rank_seconds").median(), q("sumy", 64000, "rank_seconds").median()
        r = real[real.algo == algo]
        msp = q("sumplicity", 64000, "rank_peak_bytes", m).max() / 2**20
        msy = q("sumy", 64000, "rank_peak_bytes", m).max() / 2**20
        s128 = q("sumplicity", 128000, "rank_seconds").median()
        e2e = {L: q("sumy", L, "seconds").median() / q("sumplicity", L, "seconds").median() for L in (1000, 64000)}
        lines.append(f"{NAME[algo]} & {sp:.2f} & {sy:.1f} & {sy/sp:.0f} & {e2e[1000]:.1f} & {e2e[64000]:.0f} & "
                     f"{np.median(r.median_speedup):.0f} & "
                     f"{r.median_speedup.min():.0f}--{r.median_speedup.max():.0f} & {msp:.0f} & {msy:.0f} & {s128:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    tex = "\n\n".join(["% Table 4", quality_table(), "% Table 5", baseline_error_table(),
                       "% Table 6", speed_table()])
    open(os.path.join(RES, "paper_tables.tex"), "w").write(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
