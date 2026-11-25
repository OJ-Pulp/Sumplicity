"""Runtime and memory scaling benchmark (reviewer issues G and H).

Inputs of controlled length are cut from the concatenated IsmaelMousa/books
corpus (whitespace tokens). Every measurement runs in a *fresh subprocess*
after a warm-up call, so import costs, NLTK model loading and allocator state
from earlier measurements do not leak into later ones.

Timed region (both libraries): from the list of pre-segmented sentences to the
list of selected sentence indices. For Sumplicity this includes tokenisation,
stop-word removal, stemming and TF-matrix construction; for Sumy it includes
building the document model and its tokenisation/stemming. Sentence
segmentation (NLTK Punkt, identical for both) is timed separately.

Memory: a separate subprocess per measurement records the tracemalloc peak
(Python + NumPy allocations made inside the timed region). tracemalloc slows
pure-Python code, so it is never active during timing runs.

Usage:
  python benchmarks/run_speed.py               # full sweep
  python benchmarks/run_speed.py --quick       # small smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "speed")
LENGTHS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
SUMY_MAX = 64000          # Sumy at 128K tokens takes many minutes per algorithm
K = 10                    # sentences selected


INPUTS = os.path.join(OUT, "inputs")


def input_text(length: int, rep: int, words=None) -> str:
    """Input of ``length`` whitespace tokens at a deterministic offset (cached on disk)."""
    path = os.path.join(INPUTS, f"{length}_{rep}.txt")
    if not os.path.exists(path):
        import common as C

        words = words or C.load_books_text().split()
        start = (rep * 997_331) % (len(words) - length)  # deterministic offsets
        os.makedirs(INPUTS, exist_ok=True)
        open(path, "w").write(" ".join(words[start:start + length]))
    return open(path).read()


def build_input(length: int, rep: int) -> list:
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(input_text(length, rep))


def worker(spec: dict) -> dict:
    """Runs inside a subprocess: one (library, algorithm, length, rep) cell."""
    import time
    import tracemalloc

    import common as C

    sentences = build_input(spec["length"], spec["rep"])
    pre, rank = stages(spec["lib"], spec["algo"])
    rank(pre(sentences[:50]))  # warm-up

    if spec["mode"] == "memory":
        tracemalloc.start()
        x = pre(sentences)
        _, pre_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        rank(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {**spec, "n_sentences": len(sentences), "pre_peak_bytes": pre_peak,
                "rank_peak_bytes": peak - base, "peak_bytes": max(pre_peak, peak)}

    rss0 = _peak_rss()
    t0, c0 = time.perf_counter(), time.process_time()
    x = pre(sentences)
    t1, c1 = time.perf_counter(), time.process_time()
    rss1 = _peak_rss()
    rank(x)
    t2, c2 = time.perf_counter(), time.process_time()
    rss2 = _peak_rss()
    return {**spec, "n_sentences": len(sentences),
            "seconds": t2 - t0, "pre_seconds": t1 - t0, "rank_seconds": t2 - t1,
            # CPU time summed over all threads of the process (system load)
            "cpu_seconds": c2 - c0, "pre_cpu_seconds": c1 - c0, "rank_cpu_seconds": c2 - c1,
            # process-level peak resident memory (whole interpreter) and its growth
            "rss_peak_bytes": rss2, "rss_baseline_bytes": rss0,
            "pre_rss_growth_bytes": rss1 - rss0, "rank_rss_growth_bytes": rss2 - rss1}


FOOTPRINT = {
    "python": "pass",
    "sumplicity": ("from sumplicity import LexRank\n"
                   "LexRank().summarize(open(INPUT).read(), top_k=10)"),
    "sumy": ("from sumy.parsers.plaintext import PlaintextParser\n"
             "from sumy.nlp.tokenizers import Tokenizer\n"
             "from sumy.nlp.stemmers import Stemmer\n"
             "from sumy.summarizers.lex_rank import LexRankSummarizer\n"
             "doc = PlaintextParser.from_string(open(INPUT).read(), Tokenizer('english')).document\n"
             "LexRankSummarizer(Stemmer('english'))(doc, 10)"),
}


def import_footprint(length: int = 4000) -> list:
    """Peak RSS of a fresh interpreter that only imports one library and
    summarizes one document (no benchmark tooling loaded)."""
    path = os.path.join(INPUTS, f"{length}_0.txt")
    # Sumy needs NLTK's punkt/stopword data, which Sumplicity bundles; point
    # Sumy's process at the same files without importing Sumplicity.
    from sumplicity.preprocess import NLTK_DATA_DIR
    env = {**os.environ, "NLTK_DATA": NLTK_DATA_DIR}
    out = []
    for lib, code in FOOTPRINT.items():
        script = (f"INPUT = {path!r}\nimport resource, sys, time\nt = time.perf_counter()\n{code}\n"
                  "r = resource.getrusage(resource.RUSAGE_SELF)\n"
                  "print(r.ru_maxrss if sys.platform == 'darwin' else r.ru_maxrss * 1024, "
                  "time.perf_counter() - t, r.ru_utime + r.ru_stime)")
        vals = []
        for _ in range(3):
            p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               check=True, env=env, cwd=INPUTS)
            vals.append([float(v) for v in p.stdout.split()])
        rss, wall, cpu = (sorted(c)[1] for c in zip(*vals))  # median of 3
        out.append({"kind": "footprint", "lib": lib, "length": length,
                    "rss_peak_bytes": rss, "seconds_incl_import": wall, "cpu_seconds": cpu})
    return out


def _peak_rss() -> int:
    """Peak resident set size of this process so far, in bytes."""
    import resource
    import sys as _sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if _sys.platform == "darwin" else r * 1024  # Linux reports KiB


def stages(lib: str, algo: str, k: int = K):
    """Split a library's run into (preprocess, rank) callables.

    Both libraries preprocess with Sumplicity's preprocessor (Sumy through
    sumy_adapter), so ``pre`` is the same work for both; ``rank`` is the
    algorithm alone: Sumplicity's ``rank(TF, k)`` vs. Sumy's
    ``summarizer(document, k)`` on a document whose words are already
    tokenized, filtered and stemmed.
    """
    import common as C

    if lib == "sumplicity":
        obj = C.sumplicity_systems()[algo]
        if obj.needs_positions:  # Luhn: TF plus the position matrix
            return (lambda sents: obj.preprocessor.tf(sents, positions=True),
                    lambda x: obj.rank(x[0], k, x[1]))
        return obj.preprocessor.tf, lambda TF: obj.rank(TF, k)
    summarizer, tok = C.make_sumy(algo, "matched")

    def pre(sentences):
        doc = C.sumy_document(sentences, tok)
        for s in doc.sentences:
            s.words  # cached_property: tokenize + filter + stem now, not inside rank
        return doc

    return pre, lambda doc: summarizer(doc, k)


def real_documents(n: int = 200) -> None:
    """Algorithm-only timing on real benchmark documents (single-threaded).

    For each dataset, the first ``n`` documents of the quality sample (already
    a uniform random draw) are summarized by every algorithm in both libraries,
    in alternating order, after one warm-up document.
    """
    import time

    import numpy as np

    import common as C

    out = open(os.path.join(OUT, "realdocs.jsonl"), "w")
    for ds in C.DATASETS:
        k = C.DATASETS[ds]["k"]
        df = C.load_sample(ds, 1000).head(n + 1)
        for algo in C.ALGORITHMS:
            st = {lib: stages(lib, algo, k) for lib in ("sumplicity", "sumy")}
            for lib, (pre, rank) in st.items():  # warm-up on doc 0
                rank(pre(df["sentences"][0]))
            for i in range(1, len(df)):
                sents = df["sentences"][i]
                order = ("sumplicity", "sumy") if i % 2 else ("sumy", "sumplicity")
                for lib in order:
                    pre, rank = st[lib]
                    rec = {"dataset": ds, "doc": i, "algo": algo, "lib": lib, "n_sentences": len(sents)}
                    for attempt in (1, 2):  # LAPACK SVD occasionally fails to converge; retry once
                        x = pre(sents)
                        try:
                            t0, c0 = time.perf_counter(), time.process_time()
                            rank(x)
                            t1, c1 = time.perf_counter(), time.process_time()
                            rec.update(rank_seconds=t1 - t0, rank_cpu_seconds=c1 - c0,
                                       attempts=attempt)
                            break
                        except np.linalg.LinAlgError as e:
                            rec.update(error=str(e), attempts=attempt)
                    out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"realdocs {ds} {algo} done", flush=True)
    out.close()


def segmentation_time(length: int, rep: int) -> float:
    import time

    from nltk.tokenize import sent_tokenize

    text = input_text(length, rep)
    sent_tokenize(text[:2000])
    t0 = time.perf_counter()
    sent_tokenize(text)
    return time.perf_counter() - t0


def spawn(spec: dict, env: dict) -> dict:
    """Run one cell in a fresh process; retry once (e.g. LAPACK non-convergence)."""
    for attempt in (1, 2):
        p = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(spec)],
            capture_output=True, text=True, env=env, cwd=HERE,
        )
        if p.returncode == 0:
            return {**json.loads(p.stdout.strip().splitlines()[-1]), "attempts": attempt}
    return {**spec, "error": p.stderr.strip().splitlines()[-1], "attempts": attempt,
            "seconds": float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--realdocs", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip-realdocs", action="store_true")
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(json.loads(a.worker))))
        return
    if a.realdocs is not None:
        real_documents(a.realdocs)
        return

    import common as C

    os.makedirs(OUT, exist_ok=True)
    lengths = [1000, 4000] if a.quick else LENGTHS
    words = C.load_books_text().split()  # cut every input once, up front
    for L in lengths:
        for rep in range(5):
            input_text(L, rep, words)
    single = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(os.path.join(OUT, "footprint.json"), "w") as f:
        json.dump(import_footprint(), f, indent=2)
    if not (a.quick or a.skip_realdocs):
        subprocess.run([sys.executable, __file__, "--realdocs", "200"], env=single, cwd=HERE, check=True)
    configs = [("sumplicity", single, "1-thread"), ("sumplicity", dict(os.environ), "default"),
               ("sumy", single, "1-thread")]
    out = open(os.path.join(OUT, "quick.jsonl" if a.quick else "speed.jsonl"), "w")
    for L in lengths:
        reps = 5 if L <= 16000 else 3
        if a.quick:
            reps = 1
        for rep in range(reps):
            rec = {"kind": "segmentation", "length": L, "rep": rep,
                   "seconds": segmentation_time(L, rep)}
            out.write(json.dumps(rec) + "\n"); out.flush()
        for algo in C.ALGORITHMS:
            for lib, env, threads in configs:
                if lib == "sumy" and L > SUMY_MAX:
                    continue
                for rep in range(reps):
                    r = spawn({"lib": lib, "algo": algo, "length": L, "rep": rep,
                               "mode": "time", "threads": threads}, env)
                    out.write(json.dumps(r) + "\n"); out.flush()
                    print(f"{L:>6} {lib:<10} {threads:<8} {algo:<14} rep{rep} {r['seconds']:.4f}s", flush=True)
                if threads != "default":  # memory once per lib/algo/length
                    r = spawn({"lib": lib, "algo": algo, "length": L, "rep": 0,
                               "mode": "memory", "threads": threads}, env)
                    out.write(json.dumps(r) + "\n"); out.flush()
    out.close()


if __name__ == "__main__":
    main()
