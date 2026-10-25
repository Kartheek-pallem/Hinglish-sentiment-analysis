"""Linear SVM on SentiMix, using the same data, features and scoring as the baseline.

A linear SVM draws the boundary between the classes so that the closest tweets of
each class are as far from it as possible. Like logistic regression it learns one
weight per word (or word piece) for each class; it is usually a little stronger on
TF-IDF text features.

The setting C controls how strict the model is:
  small C -> simpler model, tolerates some training mistakes
  large C -> tries hard to get every training tweet right (may memorise)

Examples (run from the project folder):
    python scripts/svm.py --owner person3                       # default C = 1, word features, raw text
    python scripts/svm.py --owner person3 --features char       # character n-grams
    python scripts/svm.py --owner person3 --tune                # pick C on validation, then test once
    python scripts/svm.py --owner person3 --tune --preprocessing cleaned
    python scripts/svm.py --owner person3 --tune --preprocessing normalized --dict-size 100

--preprocessing cleaned / normalized need src/clean.py (clean(text)) and
src/normalize.py (normalize(text, size)); until those exist, use raw.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import LinearSVC  # noqa: E402

from src.data import load_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402

C_VALUES = (0.01, 0.03, 0.1, 0.3, 1, 3)

ap = argparse.ArgumentParser(description="Linear SVM baseline")
ap.add_argument("--owner", required=True, choices=["person1", "person2", "person3", "person4"])
ap.add_argument("--features", default="word", choices=["word", "char", "both"])
ap.add_argument("--preprocessing", default="raw", choices=["raw", "cleaned", "normalized"])
ap.add_argument("--dict-size", default="all", help="normalization dictionary size: 0, 50, 100 or all (only with --preprocessing normalized)")
ap.add_argument("--tune", action="store_true", help=f"choose C on validation from {C_VALUES}")
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--results-dir", default="results")
a = ap.parse_args()


def prepare(texts):
    """Apply the chosen preprocessing to a column of tweets."""
    if a.preprocessing == "raw":
        return list(texts)
    try:
        from src.clean import clean
    except ImportError:
        sys.exit("src/clean.py with clean(text) is not in the project yet - run with --preprocessing raw for now.")
    out = [clean(t) for t in texts]
    if a.preprocessing == "normalized":
        try:
            from src.normalize import normalize
        except ImportError:
            sys.exit("src/normalize.py with normalize(text, size) is not in the project yet.")
        size = None if a.dict_size == "all" else int(a.dict_size)
        out = [normalize(t, size) for t in out]
    return out


train, val, test = (load_split(s, a.data_dir) for s in ("train", "val", "test"))
X_tr, X_va, X_te, vec = build_features(prepare(train.text), prepare(val.text), prepare(test.text), kind=a.features)
print(f"train {X_tr.shape[0]} tweets, {X_tr.shape[1]} features ({a.features}, {a.preprocessing})")


def fit(C):
    clf = LinearSVC(C=C, dual="auto", max_iter=5000)
    t0 = time.time()
    clf.fit(X_tr, train.label)
    return clf, time.time() - t0


common = dict(model="svm", owner=a.owner, preprocessing=a.preprocessing, features=a.features,
              dict_size=a.dict_size if a.preprocessing == "normalized" else "", results_dir=a.results_dir)

if a.tune:
    best = None
    for C in C_VALUES:
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
infer = (time.time() - t0) / len(test) * 1000          # seconds per 1,000 tweets
evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val", notes=f"C={C} final", save_plot=False, **common)
evaluate(test.label, pred, ids=test.id, dataset="sentimix_test", notes=f"C={C}",
         train_time_s=round(train_time, 2), infer_time_s=round(infer, 4), params=clf.coef_.size, **common)
