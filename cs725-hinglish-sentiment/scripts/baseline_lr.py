"""First baseline: TF-IDF + Logistic Regression on raw SentiMix text.

Everyone runs this once with the default settings. Everyone must get the same
macro-F1 - that proves the data, the splits and the scoring are consistent.

    python scripts/baseline_lr.py --owner person1
    python scripts/baseline_lr.py --owner person1 --features char
    python scripts/baseline_lr.py --owner person1 --tune     # pick C on validation, then test once
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.data import load_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--owner", required=True, choices=["person1", "person2", "person3", "person4"])
ap.add_argument("--features", default="word", choices=["word", "char", "both"])
ap.add_argument("--tune", action="store_true", help="choose C on validation from 0.1, 0.3, 1, 3, 10")
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--results-dir", default="results")
a = ap.parse_args()

train, val, test = (load_split(s, a.data_dir) for s in ("train", "val", "test"))
X_tr, X_va, X_te, vec = build_features(train.text, val.text, test.text, kind=a.features)
print(f"train {X_tr.shape[0]} tweets, {X_tr.shape[1]} features ({a.features})")


def fit(C):
    clf = LogisticRegression(C=C, max_iter=2000)
    t0 = time.time()
    clf.fit(X_tr, train.label)
    return clf, time.time() - t0


common = dict(model="logreg", owner=a.owner, preprocessing="raw", features=a.features, results_dir=a.results_dir)
if a.tune:
    best = None
    for C in (0.1, 0.3, 1, 3, 10):
        clf, _ = fit(C)
        r = evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val",
                     notes=f"C={C}", save_plot=False, **common)
        if best is None or r["macro_f1"] > best[1]:
            best = (C, r["macro_f1"])
    C = best[0]
    print(f"best C on validation: {C} (macro-F1 {best[1]:.4f})")
else:
    C = 1.0

clf, train_time = fit(C)
t0 = time.time()
pred = clf.predict(X_te)
infer = (time.time() - t0) / len(test) * 1000
evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val", notes=f"C={C}", save_plot=False, **common)
evaluate(test.label, pred, ids=test.id, dataset="sentimix_test", notes=f"C={C}",
         train_time_s=round(train_time, 2), infer_time_s=round(infer, 4), params=X_tr.shape[1] * 3, **common)
