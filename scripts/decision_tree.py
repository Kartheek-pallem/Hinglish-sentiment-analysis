"""Decision Tree on SentiMix, using the same data, features and scoring as the baseline.

A decision tree is a flowchart of yes/no questions learned from the training
tweets, for example:
    does the tweet contain "bekaar"?  yes -> does it contain "nahi"?  ...
    no  -> does it contain "achha"?   ...
Each path ends in a leaf that gives an answer (positive, negative or neutral).
It is easy to explain - you can read the rules - but a single tree tends to
memorise the training tweets, so limiting its size matters.

Trees are slow on huge vocabularies, so this script keeps only the most frequent
n-grams (--vocab, default 20,000).

Settings tried with --tune (chosen on validation, never on test):
  max_depth         how many questions in a row: 20, 50, 100 or no limit
  min_samples_leaf  smallest group a leaf may hold: 1 (detailed) or 5 (smoother)

Examples (run from the project folder):
    python scripts/decision_tree.py --owner person4                     # default: no depth limit, leaf 1
    python scripts/decision_tree.py --owner person4 --tune              # pick settings on validation
    python scripts/decision_tree.py --owner person4 --tune --features char
    python scripts/decision_tree.py --owner person4 --tune --preprocessing normalized --dict-size 100

--preprocessing cleaned / normalized need src/clean.py (clean(text)) and
src/normalize.py (normalize(text, size)); until those exist, use raw.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.data import load_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402

GRID = [(depth, leaf) for depth in (20, 50, 100, None) for leaf in (1, 5)]   # (max_depth, min_samples_leaf)
SEED = 42

ap = argparse.ArgumentParser(description="Decision Tree baseline")
ap.add_argument("--owner", required=True, choices=["person1", "person2", "person3", "person4"])
ap.add_argument("--features", default="word", choices=["word", "char"])
ap.add_argument("--preprocessing", default="raw", choices=["raw", "cleaned", "normalized"])
ap.add_argument("--dict-size", default="all", help="normalization dictionary size: 0, 50, 100 or all (only with --preprocessing normalized)")
ap.add_argument("--vocab", type=int, default=20000, help="keep only the N most frequent n-grams (trees are slow on huge vocabularies)")
ap.add_argument("--tune", action="store_true", help="choose (max_depth, min_samples_leaf) on validation")
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
X_tr, X_va, X_te, vec = build_features(prepare(train.text), prepare(val.text), prepare(test.text),
                                       kind=a.features, max_features=a.vocab)
print(f"train {X_tr.shape[0]} tweets, {X_tr.shape[1]} features ({a.features}, {a.preprocessing})")


def fit(max_depth, min_samples_leaf):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED)
    t0 = time.time()
    clf.fit(X_tr, train.label)
    return clf, time.time() - t0


common = dict(model="decision_tree", owner=a.owner, preprocessing=a.preprocessing, features=a.features,
              dict_size=a.dict_size if a.preprocessing == "normalized" else "", seed=SEED, results_dir=a.results_dir)

if a.tune:
    best = None
    for depth, leaf in GRID:
        clf, secs = fit(depth, leaf)
        r = evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val",
                     notes=f"depth={depth} leaf={leaf} vocab={a.vocab}", save_plot=False, **common)
        if best is None or r["macro_f1"] > best[1]:
            best = ((depth, leaf), r["macro_f1"])
    depth, leaf = best[0]
    print(f"best on validation: depth={depth}, leaf={leaf} (macro-F1 {best[1]:.4f})")
else:
    depth, leaf = None, 1

clf, train_time = fit(depth, leaf)
t0 = time.time()
pred = clf.predict(X_te)
infer = (time.time() - t0) / len(test) * 1000          # seconds per 1,000 tweets
note = f"depth={depth} leaf={leaf} vocab={a.vocab}"
print(f"tree size: {clf.tree_.node_count} nodes, {clf.get_depth()} levels deep")
evaluate(val.label, clf.predict(X_va), ids=val.id, dataset="sentimix_val", notes=note + " final", save_plot=False, **common)
evaluate(test.label, pred, ids=test.id, dataset="sentimix_test", notes=note,
         train_time_s=round(train_time, 2), infer_time_s=round(infer, 4), params=clf.tree_.node_count, **common)