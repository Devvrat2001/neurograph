#!/usr/bin/env python3
"""
tools/gen_sklearn_golden.py — Regenerate the sklearn parity fixture.

Produces tests/fixtures/eval/sklearn_golden.json: a small deterministic
classification problem plus the metrics sklearn computes on it. The
companion C++ test (`tests/unit/test_sklearn_parity.cpp`) loads the
JSON, runs `evaluate_classification` on the same input, and asserts
the values match within 1e-6.

This script is dev-only. The generated JSON IS committed — the fixture
lives in version control so the C++ test doesn't depend on Python at
build time. Re-run this script only when:
  - sklearn changes a metric definition (rare; check release notes).
  - You want a different fixture shape (more classes, more samples,
    edge cases).

Run:  python tools/gen_sklearn_golden.py

Requirements: scikit-learn >= 1.0, numpy. No other deps.

Determinism: seeded with np.random.seed(2026). Same seed + same sklearn
version ⇒ byte-identical JSON. Bump SKLEARN_VERSION_TESTED in the
output if you knowingly upgrade sklearn and re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 2026
N_SAMPLES = 60
N_CLASSES = 3
CLASS_LABELS = ["red", "green", "blue"]


def make_dataset(seed: int = SEED):
    """Generate (y_true, probs) such that every class has nonzero support
    and the predictions are non-degenerate (no row that's all-zero, no
    class that's never the argmax). Matters because:
      - ROC AUC is undefined when a class has zero positives.
      - Per-class precision/recall is 0/0 when a class is never predicted.
    The synthesised distribution puts ~70% mass on a randomly chosen
    "true" class with the rest spread (Dirichlet-like) across the others,
    matching the harness in tests/perf/test_perf_classification_metrics.cpp.
    """
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, N_CLASSES, size=N_SAMPLES)
    probs = np.zeros((N_SAMPLES, N_CLASSES), dtype=np.float64)

    winner_mass = 0.7
    remaining_mass = 1.0 - winner_mass

    for i in range(N_SAMPLES):
        truth = int(y_true[i])
        # Random non-uniform spread over the non-truth classes.
        noise = rng.uniform(0.0, 1.0, size=N_CLASSES)
        noise[truth] = 0.0
        total = noise.sum()
        if total > 0:
            noise *= remaining_mass / total
        noise[truth] = winner_mass
        probs[i] = noise

    # Deliberately mis-label ~15% of samples so the metrics aren't all
    # trivially 1.0 — gives ROC AUC, F1, etc. a real non-degenerate
    # value to validate against.
    flip_mask = rng.uniform(size=N_SAMPLES) < 0.15
    flip_idx = np.where(flip_mask)[0]
    for i in flip_idx:
        # Swap winner mass to a different class to force a "wrong" argmax.
        truth = int(y_true[i])
        wrong = (truth + 1 + rng.randint(N_CLASSES - 1)) % N_CLASSES
        probs[i, truth], probs[i, wrong] = probs[i, wrong], probs[i, truth]

    return y_true, probs


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Wrap sklearn calls. Each metric is computed independently rather
    than via classification_report so we get raw floats with full
    precision (no string formatting / rounding)."""
    y_pred = probs.argmax(axis=1)

    per_class = []
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for c in range(N_CLASSES):
        per_class.append({
            "label":     CLASS_LABELS[c],
            "precision": float(p_per[c]),
            "recall":    float(r_per[c]),
            "f1":        float(f_per[c]),
            "support":   int(np.sum(y_true == c)),
        })

    return {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "macro_f1":      float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)),
        "weighted_f1":   float(f1_score(y_true, y_pred, average="weighted",
                                         zero_division=0)),
        # multi_class='ovr' + average='macro' = arithmetic mean of one-vs-rest
        # ROC AUCs, which is what evaluate_classification's roc_auc_macro
        # reports.
        "roc_auc_macro": float(roc_auc_score(y_true, probs,
                                              multi_class="ovr",
                                              average="macro")),
        "per_class":     per_class,
        # confusion_matrix returns int ndarray; cast for JSON.
        "confusion":     confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "tests" / "fixtures" / "eval" / "sklearn_golden.json"

    y_true, probs = make_dataset()
    expected = compute_metrics(y_true, probs)

    payload = {
        "description": (
            "Deterministic classification fixture for sklearn-parity. "
            "Input + sklearn-computed reference metrics. "
            "Regenerate via tools/gen_sklearn_golden.py."
        ),
        "seed":                  SEED,
        "sklearn_version_tested": sklearn.__version__,
        "n_samples":              int(N_SAMPLES),
        "n_classes":              int(N_CLASSES),
        "class_labels":           CLASS_LABELS,
        "y_true":                 [int(x) for x in y_true],
        "probs":                  probs.tolist(),
        "expected":               expected,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys for stable diffs across regenerations.
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {out_path.relative_to(repo_root)}")
    print(f"  sklearn={sklearn.__version__} numpy={np.__version__}")
    print(f"  shape: {N_SAMPLES} samples × {N_CLASSES} classes")
    print(f"  accuracy={expected['accuracy']:.6f} "
          f"macro_f1={expected['macro_f1']:.6f} "
          f"roc_auc_macro={expected['roc_auc_macro']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
