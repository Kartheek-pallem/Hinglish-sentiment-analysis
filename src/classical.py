"""Classical baselines: TF-IDF (word or char n-grams) + LR / Linear SVM / Naive Bayes / Tree.

Default (--quick, ~2 min on Colab CPU): the configurations that matter for the final
comparison — each model on cleaned + normalized text with char features, plus the
word-feature SVM. Saves test predictions and the best SVM for the web app.

--full: the 24-run grid from the mid-review (4 models x 3 texts x 2 features) plus
the dictionary-size ablation. Takes ~10-15 min because of the decision tree.
"""

import argparse
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import LABELS, MODELS_DIR, RESULTS_DIR, Timer, load_splits, save_cost, save_predictions, set_seed
from preprocess import add_text_versions

FEATURES = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+"),
    "char": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000),
}

MODELS = {
    "logreg": (lambda v: LogisticRegression(C=v, max_iter=2000), [0.3, 1, 3, 10]),
    "svm": (lambda v: LinearSVC(C=v), [0.03, 0.1, 0.3, 1]),
    "nb": (lambda v: MultinomialNB(alpha=v), [0.03, 0.1, 0.3, 1]),
    "tree": (lambda v: DecisionTreeClassifier(max_depth=v, min_samples_leaf=5, random_state=42), [50, None]),
}


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def run(model, feature, text_col, splits, save=False):
    """Tune the one hyper-parameter on validation, then score the chosen model on test."""
    build, grid = MODELS[model]
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    with Timer() as vec_timer:                      # vectorize once, reuse for every grid value
        vec = TfidfVectorizer(**FEATURES[feature])
        X_tr = vec.fit_transform(tr[text_col])
    X_va = vec.transform(va[text_col])

    best = None
    for value in grid:
        with Timer() as t:
            clf = build(value).fit(X_tr, tr.label)
        f1 = macro_f1(va.label, clf.predict(X_va))
        if best is None or f1 > best[0]:
            best = (f1, value, Pipeline([("tfidf", vec), ("clf", clf)]), vec_timer.seconds + t.seconds)

    val_f1, value, pipe, train_s = best
    with Timer() as t:
        pred = pipe.predict(te[text_col])
    test_f1 = macro_f1(te.label, pred)
    name = f"{model}_{feature}_{text_col}"
    print(f"{name:28s} best={value!s:6s} val={val_f1:.4f} test={test_f1:.4f}")

    if save:
        save_predictions(name, te.id, te.label, pred)
        size_mb = len(pickle.dumps(pipe)) / 1e6
        save_cost(name, family="classical", params=len(pipe[0].vocabulary_) * 3,   # one weight per feature per class
                  train_seconds=round(train_s, 2), infer_ms_per_1k=round(t.seconds / len(te) * 1e6, 2),
                  size_mb=round(size_mb, 2), device="cpu")
    return {"model": model, "feature": feature, "text": text_col, "setting": value,
            "val_f1": round(val_f1, 4), "test_f1": round(test_f1, 4)}, pipe


def dictionary_ablation(splits):
    rows = []
    for size in [0, 50, 100, 200, None]:
        s = {k: v.copy() for k, v in splits.items()}
        s, d = add_text_versions(s, max_pairs=size)
        r, _ = run("svm", "char", "normalized", s)
        rows.append({"dict_pairs": len(d), "test_f1": r["test_f1"], "val_f1": r["val_f1"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dictionary_ablation.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run all 24 configurations + dictionary ablation")
    args = ap.parse_args()
    set_seed()

    splits, dictionary = add_text_versions(load_splits())
    print(f"normalization dictionary: {len(dictionary)} pairs")
    pd.Series(dictionary).to_csv(RESULTS_DIR / "normalization_dictionary.csv", header=["canonical"])

    if args.full:
        plan = [(m, f, t) for m in MODELS for t in ("raw", "cleaned", "normalized") for f in ("word", "char")]
    else:
        plan = [(m, "char", t) for m in ("svm", "logreg", "nb") for t in ("cleaned", "normalized")]
        plan += [("svm", "word", "cleaned"), ("tree", "word", "cleaned")]   # word features keep the tree fast

    rows, best_svm = [], None
    for model, feature, text_col in plan:
        row, pipe = run(model, feature, text_col, splits, save=True)
        rows.append(row)
        if model == "svm" and (best_svm is None or row["val_f1"] > best_svm[0]["val_f1"]):
            best_svm = (row, pipe)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "classical_runs.csv", index=False)
    print("\n", table.pivot_table(index="model", columns=["text", "feature"], values="test_f1"))

    # the web app uses the best SVM (chosen on validation) because it is fast on CPU
    row, pipe = best_svm
    with open(MODELS_DIR / "svm_best.pkl", "wb") as f:
        pickle.dump({"pipeline": pipe, "text_version": row["text"], "dictionary": dictionary}, f)
    print(f"\nsaved saved_models/svm_best.pkl ({row['feature']} features, {row['text']} text)")

    if args.full:
        dictionary_ablation(splits)


if __name__ == "__main__":
    main()
