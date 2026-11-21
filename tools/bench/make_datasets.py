"""Generate the benchmark CSVs into $NEUROGRAPH_BENCH_DIR (default ./bench-data).

Every dataset is split ONCE here into <name>_train.csv / <name>_test.csv.
NeuroGraph only ever sees the train file (it makes its own internal
train/val split); sklearn reference models are fitted on the same train
file. Both are scored on the identical untouched test file.

No downloads: sklearn bundled datasets + synthetic generators only.
"""
import csv
import os

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

OUT = os.environ.get("NEUROGRAPH_BENCH_DIR", os.path.abspath("bench-data"))
os.makedirs(OUT, exist_ok=True)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return repr(round(float(v), 6))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return v


def emit(name, header, X_rows, y, test_size, stratify=True, seed=0):
    idx = np.arange(len(y))
    strat = y if stratify else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    for part, sel in (("train", tr), ("test", te)):
        rows = [[fmt(v) for v in X_rows[i]] + [fmt(y[i])] for i in sel]
        write_csv(os.path.join(OUT, f"{name}_{part}.csv"), header, rows)
    print(f"{name:10s} train={len(tr):6d} test={len(te):6d} cols={len(header)}")


def clean(names):
    return [n.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for n in names]


# --- A. iris -------------------------------------------------------------
d = datasets.load_iris()
y = np.array([d.target_names[t] for t in d.target], dtype=object)
emit("iris", clean(d.feature_names) + ["species"], d.data.tolist(), y, 0.30)

# --- B. wine -------------------------------------------------------------
d = datasets.load_wine()
y = np.array([f"cultivar_{t}" for t in d.target], dtype=object)
emit("wine", clean(d.feature_names) + ["cultivar"], d.data.tolist(), y, 0.30)

# --- C. breast cancer ----------------------------------------------------
d = datasets.load_breast_cancer()
y = np.array([d.target_names[t] for t in d.target], dtype=object)
emit("bcancer", clean(d.feature_names) + ["diagnosis"], d.data.tolist(), y, 0.30)

# --- D. digits (integer-coded labels: tests task auto-detection) ----------
d = datasets.load_digits()
hdr = [f"px{i}" for i in range(64)]
emit("digits", hdr + ["digit"], d.data.astype(int).tolist(), d.target.astype(int), 0.30)
# same data, string labels (unambiguous classification)
y = np.array([f"d{t}" for t in d.target], dtype=object)
emit("digitsS", hdr + ["digit"], d.data.astype(int).tolist(), y, 0.30)

# --- E. diabetes (regression) --------------------------------------------
d = datasets.load_diabetes()
emit("diabetes", clean(d.feature_names) + ["progression"], d.data.tolist(),
     d.target.astype(float), 0.30, stratify=False)

# --- F. friedman1 (nonlinear regression) ---------------------------------
X, yv = datasets.make_friedman1(n_samples=5000, n_features=10, noise=1.0, random_state=1)
emit("friedman", [f"x{i}" for i in range(10)] + ["y"], X.tolist(), yv, 0.20, stratify=False)

# --- G. churn-like messy business table ----------------------------------
rng = np.random.default_rng(7)
n = 6000
plan = rng.choice(["Basic", "Pro", "Enterprise"], size=n, p=[0.55, 0.33, 0.12])
region = rng.choice(["North", "South", "East", "West"], size=n)
pay = rng.choice(["Card", "UPI", "NetBanking", "Invoice"], size=n, p=[0.4, 0.3, 0.15, 0.15])
auto_renew = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
tenure = rng.integers(1, 73, size=n)
base_fee = np.where(plan == "Basic", 499, np.where(plan == "Pro", 1499, 4999)).astype(float)
fee = np.round(base_fee * rng.uniform(0.8, 1.3, size=n), 2)
tickets = rng.poisson(1.6, size=n).astype(float)
usage = np.round(rng.gamma(2.0, 12.0, size=n), 1)
last_login = rng.integers(0, 91, size=n)
seats = np.where(plan == "Enterprise", rng.integers(20, 300, size=n), rng.integers(1, 25, size=n))
year = rng.integers(2021, 2026, size=n)
month = rng.integers(1, 13, size=n)
day = rng.integers(1, 29, size=n)
signup = [f"{y_}-{m_:02d}-{d_:02d}" for y_, m_, d_ in zip(year, month, day)]

logit = (
    -0.55
    + 1.6 * (tickets >= 4)
    + 1.3 * (last_login > 30)
    - 0.035 * tenure
    + 0.9 * ((plan == "Basic") & (fee > 520))
    - 1.1 * (auto_renew == "Yes")
    + 0.7 * (pay == "Invoice")
    - 0.35 * np.log1p(usage)
    + 1.7 * ((tenure < 6) & (tickets >= 2))
    + 0.9 * ((usage < 8) & (last_login > 14))
    + rng.normal(0, 0.35, size=n)
)
p = 1 / (1 + np.exp(-logit))
churn = np.where(rng.uniform(size=n) < p, "Yes", "No").astype(object)

# inject missingness the way real exports have it
tickets_m = tickets.copy()
tickets_m[rng.uniform(size=n) < 0.06] = np.nan
usage_m = usage.copy()
usage_m[rng.uniform(size=n) < 0.08] = np.nan
region_m = region.astype(object)
region_m[rng.uniform(size=n) < 0.05] = ""

cust_id = [f"CUST-{100000 + i}" for i in range(n)]
hdr = ["customer_id", "signup_date", "plan", "region", "payment_method", "auto_renew",
       "seats", "monthly_fee", "tenure_months", "support_tickets", "usage_hours",
       "last_login_days", "churned"]
rows = [[cust_id[i], signup[i], plan[i], region_m[i], pay[i], auto_renew[i],
         int(seats[i]), float(fee[i]), int(tenure[i]), tickets_m[i], usage_m[i],
         int(last_login[i])] for i in range(n)]
emit("churn", hdr, rows, churn, 0.20)
print("   churn positive rate:", round(float((churn == "Yes").mean()), 3))

# --- H. larger table for wall-clock scaling ------------------------------
X, yv = datasets.make_classification(
    n_samples=60000, n_features=40, n_informative=15, n_redundant=5,
    n_clusters_per_class=3, class_sep=0.9, flip_y=0.02, random_state=3)
y = np.array(["pos" if t == 1 else "neg" for t in yv], dtype=object)
emit("big", [f"f{i}" for i in range(40)] + ["label"], np.round(X, 5).tolist(), y, 1 / 6)

# --- I. mid-size table (same generator, 10k rows) ------------------------
X, yv = datasets.make_classification(
    n_samples=12000, n_features=20, n_informative=10, n_redundant=4,
    n_clusters_per_class=3, class_sep=0.9, flip_y=0.02, random_state=5)
y = np.array(["pos" if t == 1 else "neg" for t in yv], dtype=object)
emit("mid", [f"f{i}" for i in range(20)] + ["label"], np.round(X, 5).tolist(), y, 1 / 6)

print("written to", OUT)
