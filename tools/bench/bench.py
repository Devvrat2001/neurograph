"""Reference models + scoring for the NeuroGraph benchmark.

Data folder: $NEUROGRAPH_BENCH_DIR (default ./bench-data).

  python bench.py baselines            -> fit sklearn models, write sk_results.json
  python bench.py score <name> <preds> -> score a NeuroGraph predictions CSV
"""
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
D = os.environ.get("NEUROGRAPH_BENCH_DIR", os.path.abspath("bench-data"))

# name -> (target column, task, columns a sensible analyst would drop as identifiers)
SETS = {
    "iris":     ("species", "cls", []),
    "wine":     ("cultivar", "cls", []),
    "bcancer":  ("diagnosis", "cls", []),
    "digitsS":  ("digit", "cls", []),
    "churn":    ("churned", "cls", ["customer_id", "signup_date"]),
    "mid":      ("label", "cls", []),
    "big":      ("label", "cls", []),
    "diabetes": ("progression", "reg", []),
    "friedman": ("y", "reg", []),
    # variants kept for comparison with version 0.1, which hashed numeric
    # columns with more than 1,024 distinct values as if they were text
    "friedmanR": ("y", "reg", []),
    "midR":      ("label", "cls", []),
    "churnC":    ("churned", "cls", []),
    # churn with NOTHING dropped: ids + dates one-hot encoded, as a naive tool would
    "churnNaive": ("churned", "cls", []),
}


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [row for row in r]
    return hdr, rows


def is_num(v):
    if v == "":
        return True
    try:
        float(v)
        return True
    except ValueError:
        return False


def frame(name):
    target, task, drop = SETS[name]
    hdr, tr = read(os.path.join(D, f"{name}_train.csv"))
    _, te = read(os.path.join(D, f"{name}_test.csv"))
    ti = hdr.index(target)
    feat = [i for i, h in enumerate(hdr) if i != ti and h not in drop]
    num = [i for i in feat if all(is_num(r[i]) for r in tr)]
    cat = [i for i in feat if i not in num]

    def build(rows):
        Xn = np.array([[float(r[i]) if r[i] != "" else np.nan for i in num] for r in rows],
                      dtype=float).reshape(len(rows), len(num))
        Xc = np.array([[r[i] if r[i] != "" else "__missing__" for i in cat] for r in rows],
                      dtype=object).reshape(len(rows), len(cat))
        y = [r[ti] for r in rows]
        return Xn, Xc, y

    Xn_tr, Xc_tr, y_tr = build(tr)
    Xn_te, Xc_te, y_te = build(te)
    if task == "reg":
        y_tr = np.array(y_tr, dtype=float)
        y_te = np.array(y_te, dtype=float)
    else:
        y_tr = np.array(y_tr, dtype=object)
        y_te = np.array(y_te, dtype=object)
    X_tr = np.concatenate([Xn_tr.astype(object), Xc_tr], axis=1)
    X_te = np.concatenate([Xn_te.astype(object), Xc_te], axis=1)
    n_num, n_cat = len(num), len(cat)
    return task, X_tr, y_tr, X_te, y_te, n_num, n_cat


def prep(n_num, n_cat):
    parts = []
    if n_num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), list(range(n_num))))
    if n_cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                      list(range(n_num, n_num + n_cat))))
    return ColumnTransformer(parts)


def cls_metrics(y_true, pred, proba, classes):
    out = {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
           "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4)}
    try:
        if len(classes) == 2:
            pos = sorted(classes)[-1]
            col = list(classes).index(pos)
            out["roc_auc"] = round(float(roc_auc_score(
                (np.asarray(y_true) == pos).astype(int), proba[:, col])), 4)
        else:
            out["roc_auc"] = round(float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=list(classes))), 4)
    except Exception as e:  # noqa: BLE001
        out["roc_auc"] = None
        out["roc_auc_err"] = str(e)[:80]
    return out


def reg_metrics(y_true, pred):
    return {"r2": round(float(r2_score(y_true, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, pred)), 4)}


def baselines(names=None, out_name="sk_results.json"):
    res = {}
    for name in (names or [n for n in SETS if n not in ("friedmanR", "midR", "churnC", "churnNaive")]):
        task, X_tr, y_tr, X_te, y_te, n_num, n_cat = frame(name)
        if task == "cls":
            # integer-encode labels (sorted order) — sklearn's MLP early
            # stopping cannot score string labels.
            le = LabelEncoder().fit(y_tr)
            y_tr = le.transform(y_tr)
            y_te = le.transform(y_te)
        n_feat_raw = n_num + n_cat
        hidden = int(min(64, max(8, 2 * n_feat_raw)))
        if task == "cls":
            models = {
                "logistic_regression": LogisticRegression(max_iter=2000),
                "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
                "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=0),
                "sklearn_mlp_same_width": MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=2000,
                                                         early_stopping=len(y_tr) >= 2000, random_state=0),
            }
        else:
            models = {
                "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
                "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0),
                "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=0),
                "sklearn_mlp_same_width": MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=5000,
                                                        early_stopping=len(y_tr) >= 2000, random_state=0),
            }
        res[name] = {"task": task, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                     "n_features": n_feat_raw, "models": {}}
        if task == "cls":
            vals, counts = np.unique(y_te, return_counts=True)
            res[name]["majority_class_accuracy"] = round(float(counts.max() / counts.sum()), 4)
        for mname, model in models.items():
            pipe = Pipeline([("prep", prep(n_num, n_cat)), ("m", model)])
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            if task == "cls":
                m = cls_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te),
                                pipe.named_steps["m"].classes_)
            else:
                m = reg_metrics(y_te, pipe.predict(X_te))
            m["fit_seconds"] = round(fit_s, 2)
            res[name]["models"][mname] = m
            print(f"{name:9s} {mname:24s} {m}", flush=True)
    with open(os.path.join(D, out_name), "w") as f:
        json.dump(res, f, indent=2)


def score(name, preds_path):
    target, task, _ = SETS[name]
    hdr, te = read(os.path.join(D, f"{name}_test.csv"))
    ti = hdr.index(target)
    ph, pr = read(preds_path)
    if len(pr) != len(te):
        print(f"ROW MISMATCH: test={len(te)} preds={len(pr)}")
    n = min(len(pr), len(te))
    if task == "cls":
        y_true = np.array([r[ti] for r in te[:n]], dtype=object)
        pc = ph.index("predicted_class")
        pred = np.array([r[pc] for r in pr[:n]], dtype=object)
        classes = sorted(set(y_true))
        prob_cols = []
        for c in classes:
            cand = [i for i, h in enumerate(ph) if h in (c, f"prob_{c}", f"p_{c}", f"proba_{c}")]
            prob_cols.append(cand[0] if cand else None)
        if all(c is not None for c in prob_cols):
            proba = np.array([[float(r[i]) for i in prob_cols] for r in pr[:n]])
            m = cls_metrics(y_true, pred, proba, np.array(classes, dtype=object))
        else:
            m = {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
                 "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
                 "roc_auc": None, "note": f"prob columns not matched; header={ph[:8]}"}
        vals, counts = np.unique(pred, return_counts=True)
        m["predicted_class_counts"] = {str(v): int(c) for v, c in zip(vals, counts)}
    else:
        y_true = np.array([float(r[ti]) for r in te[:n]])
        pv = ph.index("predicted_value")
        pred = np.array([float(r[pv]) for r in pr[:n]])
        m = reg_metrics(y_true, pred)
    print(json.dumps({name: m}))


if __name__ == "__main__":
    if sys.argv[1] == "baselines":
        if len(sys.argv) > 2:
            baselines(sys.argv[2:], "sk_results_variants.json")
        else:
            baselines()
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
