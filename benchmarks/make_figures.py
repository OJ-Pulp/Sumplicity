"""Figure 2 (algorithm-only runtime and memory vs. input length) and the speed tables.

Times are the scoring stage only (both libraries share the same preprocessing
components; the cost of Sumplicity's preprocessing is drawn as a grey
reference line). Memory is the peak allocated by the scoring stage
(tracemalloc), excluding the preprocessed input.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
NAMES = {"Luhn": "Luhn", "SumBasic": "SumBasic", "KL": "KL-Sum", "LSA": "LSA",
         "Degree": "Degree", "TextRank": "TextRank", "LexRank": "LexRank"}
COLORS = dict(zip(C.ALGORITHMS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                                 "#66a61e", "#e6ab02", "#a6761d"]))


def load():
    rows = [json.loads(l) for l in open(os.path.join(RES, "speed", "speed.jsonl"))]
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    t = df[(df.get("mode") == "time")]
    m = df[(df.get("mode") == "memory")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    sp_t = t[(t.lib == "sumplicity") & (t.threads == "1-thread")]
    sy_t = t[(t.lib == "sumy") & (t.threads == "1-thread")]

    # (a) speed-up of the scoring stage vs. input length
    ax = axes[0]
    for algo in C.ALGORITHMS:
        a_ = sp_t[sp_t.algo == algo].groupby("length")["rank_seconds"].median()
        b_ = sy_t[sy_t.algo == algo].groupby("length")["rank_seconds"].median()
        r = (b_ / a_).dropna()
        ax.plot(r.index, r.values, "o-", ms=3, color=COLORS[algo], label=NAMES[algo])
    ax.axhline(1, color="k", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Input length (tokens)",
           ylabel="Speed-up (Sumy time / Sumplicity time)", title="(a) Scoring speed-up")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8, ncol=2)

    # (b) scoring time at 64K tokens, both libraries side by side
    ax = axes[1]
    L = 64000
    x = np.arange(len(C.ALGORITHMS))
    a_ = sp_t[sp_t.length == L].groupby("algo")["rank_seconds"].median().reindex(C.ALGORITHMS)
    b_ = sy_t[sy_t.length == L].groupby("algo")["rank_seconds"].median().reindex(C.ALGORITHMS)
    pre = sp_t[sp_t.length == L]["pre_seconds"].median()
    ax.bar(x - 0.2, a_.values, 0.4, color="#1b6ca8", label="Sumplicity")
    ax.bar(x + 0.2, b_.values, 0.4, color="#d1495b", label="Sumy")
    ax.axhline(pre, color="grey", ls=":", lw=2, label="shared preprocessing")
    for i, (u, v) in enumerate(zip(a_.values, b_.values)):
        ax.text(i, max(u, v) * 1.4, f"{v / u:.0f}x", ha="center", fontsize=8)
    ax.set(yscale="log", xticks=x, ylabel="Median scoring time (s)",
           title="(b) Scoring time at 64K tokens")
    ax.set_xticklabels([NAMES[a] for a in C.ALGORITHMS], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=.3)

    # (c) peak memory of the scoring stage
    ax = axes[2]
    for lib, ls in [("sumplicity", "-"), ("sumy", "--")]:
        sub = m[m.lib == lib]
        for algo in C.ALGORITHMS:
            g = sub[sub.algo == algo].groupby("length")["rank_peak_bytes"].max() / 2**20
            ax.plot(g.index, g.values, ls, marker="o", ms=3, color=COLORS[algo])
    ax.plot([], [], "k-", label="Sumplicity"); ax.plot([], [], "k--", label="Sumy")
    ax.set(xscale="log", yscale="log", xlabel="Input length (tokens)",
           ylabel="Peak memory of scoring stage (MiB)", title="(c) Peak memory")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"scaling.{ext}"), dpi=200)

    # Table: medians at selected lengths, speed-up, memory
    rows = []
    for algo in C.ALGORITHMS:
        r = {"algo": algo}
        for L in [1000, 8000, 64000, 128000]:
            sp = t[(t.lib == "sumplicity") & (t.threads == "1-thread") & (t.algo == algo) & (t.length == L)]["rank_seconds"].median()
            sy = t[(t.lib == "sumy") & (t.algo == algo) & (t.length == L)]["rank_seconds"].median()
            r[f"sp_{L}"], r[f"sy_{L}"] = sp, sy
            r[f"x_{L}"] = sy / sp if sy == sy else np.nan
            r[f"mem_sp_{L}"] = m[(m.lib == "sumplicity") & (m.algo == algo) & (m.length == L)]["rank_peak_bytes"].max() / 2**20
            r[f"mem_sy_{L}"] = m[(m.lib == "sumy") & (m.algo == algo) & (m.length == L)]["rank_peak_bytes"].max() / 2**20
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "speed", "speed_table.csv"), index=False)
    # log-log scaling exponents (time ~ length^b) between 8K and 64K
    exps = []
    for (lib, thr), sub in t.groupby(["lib", "threads"]):
        for algo, g in sub.groupby("algo"):
            g = g[(g.length >= 8000) & (g.length <= 64000)].groupby("length")["rank_seconds"].median()
            if len(g) >= 2:
                b = np.polyfit(np.log(g.index), np.log(g.values), 1)[0]
                exps.append({"lib": lib, "threads": thr, "algo": algo, "exponent": b})
    pd.DataFrame(exps).to_csv(os.path.join(RES, "speed", "scaling_exponents.csv"), index=False)
    seg = df[df.kind == "segmentation"].groupby("length")["seconds"].median()
    seg.to_csv(os.path.join(RES, "speed", "segmentation.csv"))
    pre = t[(t.lib == "sumplicity") & (t.threads == "1-thread")].groupby("length")[["pre_seconds"]].median()
    pre.to_csv(os.path.join(RES, "speed", "sumplicity_preprocessing.csv"))
    # paired per-document speed-ups on real benchmark documents
    from scipy.stats import wilcoxon
    rd = pd.read_json(os.path.join(RES, "speed", "realdocs.jsonl"), lines=True)
    w = rd.pivot_table(index=["dataset", "doc", "algo"], columns="lib", values="rank_seconds").dropna()
    real = []
    for (ds, algo), g in w.groupby(level=["dataset", "algo"]):
        real.append({"dataset": ds, "algo": algo, "n": len(g),
                     "median_speedup": float((g["sumy"] / g["sumplicity"]).median()),
                     "sumplicity_median_ms": 1000 * g["sumplicity"].median(),
                     "sumy_median_ms": 1000 * g["sumy"].median(),
                     "wilcoxon_p": float(wilcoxon(g["sumy"], g["sumplicity"]).pvalue)})
    pd.DataFrame(real).to_csv(os.path.join(RES, "speed", "realdocs_summary.csv"), index=False)
    print(pd.DataFrame(real).round(4).to_string())
    print(tab.round(4).to_string())
    print(pd.DataFrame(exps).pivot_table(index="algo", columns=["lib", "threads"], values="exponent").round(2))
    print("segmentation", seg.round(3).to_dict())
    print("sumplicity TF build", pre.round(3).to_dict())


if __name__ == "__main__":
    main()
