"""Naive Bayes on SentiMix, using the same data, features and scoring as the baseline.

Naive Bayes counts how often each word appears in positive, negative and neutral
training tweets. For a new tweet it multiplies those word-by-word chances to find
the most likely class. "Naive" because it treats every word as independent of the
others. It is very fast and a classic baseline for text.

Two versions (choose with --variant):
  multinomial  the standard text version (default)
  complement   learns from "everything except this class"; often better when
               classes are a little unbalanced

The setting alpha is smoothing: it adds a small count to every word so a word
never seen with a class does not give that class a chance of exactly zero.
  small alpha -> trusts the counts more
  large alpha -> smooths more

Examples (run from the project folder):
    python scripts/naive_bayes.py --owner person3                       # default alpha = 1, word features, raw text
    python scripts/naive_bayes.py --owner person3 --features char       # character n-grams
    python scripts/naive_bayes.py --owner person3 --tune                # pick alpha on validation, then test once
    python scripts/naive_bayes.py --owner person3 --tune --variant complement
    python scripts/naive_bayes.py --owner person3 --tune --preprocessing normalized --dict-size 100

--preprocessing cleaned / normalized need src/clean.py (clean(text)) and
src/normalize.py (normalize(text, size)); until those exist, use raw.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.naive_bayes import ComplementNB, MultinomialNB  # noqa: E402

from src.data import load_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402

ALPHA_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0)

ap = argparse.ArgumentParser(description="Naive Bayes baseline")
ap.add_argument("--owner", required=True, choices=["person1", "person2", "person3", "person4"])
ap.add_argument("--variant", default="multinomial", choices=["multinomial", "complement"])
ap.add_argument("--features", default="word", choices=["word", "char", "both"])
ap.add_argument("--preprocessing", default="raw", choices=["raw", "cleaned", "normalized"])
ap.add_argument("--dict-size", default="all", help="normalization dictionary size: 0, 50, 100 or all (only with --preprocessing normalized)")
ap.add_argument("--tune", action="store_true", help=f"choose alpha on validation from {ALPHA_VALUES}")
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

NB = MultinomialNB if a.variant == "multinomial" else ComplementNB
MODEL_NAME = "naive_bayes" if a.variant == "multinomial" else "complement_nb"


def fit(alpha):
    clf = NB(alpha=alpha)
    t0 = time.time()
    clf.fit(X_tr, train.label)
    return clf, time.time() - t0


common = dict(model=MODEL_NAME, owner=a.owner, preprocessing=a.preprocessing, features=a.features,
              dict_size=a.dict_size if a.preprocessing == "normalized" else "", results_dir=a.results_dir)

if a.tune:
    best = None
    for alpha in ALPHA_VALUES:
        clf, _ = fit(alpha)
        r = evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val",
                     notes=f"alpha={alpha}", save_plot=False, **common)
        if best is None or r["macro_f1"] > best[1]:
            best = (alpha, r["macro_f1"])
    alpha = best[0]
    print(f"best alpha on validation: {alpha} (macro-F1 {best[1]:.4f})")
else:
    alpha = 1.0

clf, train_time = fit(alpha)
t0 = time.time()
pred = clf.predict(X_te)
infer = (time.time() - t0) / len(test) * 1000          # seconds per 1,000 tweets
evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val", notes=f"alpha={alpha} final", save_plot=False, **common)
evaluate(test.label, pred, ids=test.id, dataset="sentimix_test", notes=f"alpha={alpha}",
         train_time_s=round(train_time, 2), infer_time_s=round(infer, 4), params=clf.feature_log_prob_.size, **common)
