"""Benchmark NeuroGraph 0.2 through its own command-line tools.

For each dataset: neurograph_train on <name>_train.csv (default 300 s
allowance, i.e. the wizard's "Balanced"), then neurograph_predict on the
untouched <name>_test.csv, then the same scorer used for the scikit-learn
reference numbers (bench.py score). Nothing is dropped by hand: identifier
and date columns are left to the product's automatic handling.

  python run_new.py [name ...]        (default: all)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Folder with neurograph_train.exe and neurograph_predict.exe.
BIN = os.environ.get("NEUROGRAPH_BIN", os.path.join(HERE, "..", "..", "build", "Release"))
TRAIN = os.path.join(BIN, "neurograph_train.exe")
PREDICT = os.path.join(BIN, "neurograph_predict.exe")
BENCH = os.environ.get("NEUROGRAPH_BENCH_DIR", os.path.abspath("bench-data"))
OUT = os.path.join(BENCH, "new")

SETS = {
    "iris": "species", "wine": "cultivar", "bcancer": "diagnosis",
    "digitsS": "digit", "churn": "churned", "churnC": "churned",
    "mid": "label", "midR": "label", "big": "label",
    "diabetes": "progression", "friedman": "y", "friedmanR": "y",
}


def run_one(name, target, budget="300"):
    bundle = os.path.join(OUT, f"{name}_bundle")
    preds = os.path.join(OUT, f"{name}_preds.csv")
    t0 = time.time()
    tr = subprocess.run([TRAIN, "--data", os.path.join(BENCH, f"{name}_train.csv"),
                         "--target", target, "--out", bundle, "--time", budget,
                         "--json", "--quiet"],
                        capture_output=True, text=True, encoding="utf-8")
    train_s = time.time() - t0
    if tr.returncode != 0:
        return {"name": name, "error": (tr.stdout + tr.stderr)[-2000:], "exit": tr.returncode}
    report = json.loads(tr.stdout)

    t1 = time.time()
    pr = subprocess.run([PREDICT, "--model", bundle,
                         "--in", os.path.join(BENCH, f"{name}_test.csv"), "--out", preds],
                        capture_output=True, text=True, encoding="utf-8")
    predict_s = time.time() - t1
    if pr.returncode != 0:
        return {"name": name, "error": pr.stderr[-2000:], "exit": pr.returncode}

    sc = subprocess.run([sys.executable, os.path.join(HERE, "bench.py"),
                         "score", name, preds], capture_output=True, text=True, cwd=BENCH)
    scored = json.loads(sc.stdout.strip().splitlines()[-1])[name]

    size = sum(os.path.getsize(os.path.join(bundle, f)) for f in os.listdir(bundle))
    return {
        "name": name,
        "train_seconds": round(train_s, 2),
        "predict_seconds": round(predict_s, 2),
        "chosen": report["model"]["description"],
        "leaderboard": [(c["name"], c["check_score"], round(c["seconds"], 2), c["stopped_by_time"])
                        for c in report["leaderboard"]],
        "internal": {"evaluated_on": report["evaluated_on"], "n": report["n_evaluated"],
                     **report["metrics"]},
        "verdict": report["verdict"],
        "curve": report["learning_curve"]["message"],
        "top_columns": [c["column"] for c in report["importance"][:5]],
        "test": scored,
        "bundle_kb": round(size / 1024),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(SETS)
    path = os.path.join(OUT, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for n in names:
        r = run_one(n, SETS[n])
        results[n] = r
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
        key = "error" if "error" in r else "test"
        print(n, json.dumps(r.get(key))[:300], r.get("train_seconds"), flush=True)


if __name__ == "__main__":
    main()
