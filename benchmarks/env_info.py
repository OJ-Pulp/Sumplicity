"""Record the hardware/software environment of a benchmark run (issue G)."""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata


def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


def main():
    import numpy as np
    import psutil

    pkgs = ["sumplicity", "numpy", "nltk", "sumy", "rouge-score", "bert-score",
            "torch", "transformers", "scipy", "anthropic", "datasets", "huggingface-hub"]
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "perf_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in pkgs},
        "numpy_blas": _blas(np),
    }
    out = os.path.join(os.path.dirname(__file__), "results", "environment.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(info, open(out, "w"), indent=2)
    print(json.dumps(info, indent=2))


def _version(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


def _blas(np):
    try:
        cfg = np.show_config(mode="dicts")
        return cfg["Build Dependencies"]["blas"]
    except Exception:
        return None


if __name__ == "__main__":
    main()
