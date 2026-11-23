"""Compare every model that wrote predictions to results/preds/.

Produces (all in results/):
  main_results.csv     macro-F1, accuracy, per-class F1 for every model
  slices.csv           macro-F1 on hard tweets: negation, long, heavily mixed, emoji, sarcasm tag
  bootstrap.csv        paired bootstrap of each model vs the reference SVM (is the gap real?)
  cost.csv             parameters, training time, CPU latency, size  (Q3: is a transformer worth it?)
  errors_*.csv         example tweets for error analysis
  figures/*.png        confusion matrices, slice chart, F1-vs-latency chart
  SUMMARY.md           all tables in one file, ready to copy into slides

    python src/evaluate.py
    python src/evaluate.py --reference svm_char_cleaned --cmi 30
"""

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR, LABELS, PREDS_DIR, RESULTS_DIR, load_splits
from preprocess import clean, has_negation

EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")
SARCASM_RE = re.compile(r"sarcas|irony|ironic|\bwah\s+re\b|\bwaah\b", re.I)


# ---------------------------------------------------------------- metrics

def macro_f1(gold, pred):
    """Fast macro-F1 on integer arrays (used 1000s of times by the bootstrap)."""
    f1s = []
    for c in range(3):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def per_class_f1(gold, pred):
    out = {}
    for c, name in enumerate(LABELS):
        tp = np.sum((pred == c) & (gold == c))
        denom = np.sum(pred == c) + np.sum(gold == c)
        out[f"f1_{name}"] = round(0.0 if denom == 0 else 2 * tp / denom, 4)
    return out


def paired_bootstrap(gold, pred_a, pred_b, n=1000, seed=0):
    """How often does model A beat model B when we resample the test set?"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gold), size=(n, len(gold)))
    diffs = np.array([macro_f1(gold[i], pred_a[i]) - macro_f1(gold[i], pred_b[i]) for i in idx])
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), float(np.mean(diffs <= 0))


# ---------------------------------------------------------------- slices

def build_slices(test, cmi_threshold):
    cleaned = test.text.map(clean)
    n_words = cleaned.str.split().map(len)          # real words only, not "@", "#" or URL pieces
    slices = {
        "All tweets": np.ones(len(test), bool),
        "With negation": cleaned.map(has_negation).values,
        "Long (> 20 words)": (n_words > 20).values,
        f"Heavily mixed (CMI >= {cmi_threshold:g})": (test.cmi >= cmi_threshold).values,
        "With emoji": cleaned.map(lambda t: bool(EMOJI_RE.search(t))).values,
    }
    sarcasm = test.text.map(lambda t: bool(SARCASM_RE.search(t))).values
    if sarcasm.sum() >= 30:                          # too few tweets -> score means nothing
        slices["Sarcasm marker"] = sarcasm
    return slices


# ---------------------------------------------------------------- figures

def plot_confusions(preds, gold, names):
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.zeros((3, 3), int)
        np.add.at(cm, (gold, preds[name]), 1)
        cm_norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), LABELS, fontsize=8)
        ax.set_yticks(range(3), LABELS, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_slices(slice_table, names):
    t = slice_table.loc[names]
    ax = t.T.plot(kind="bar", figsize=(10, 4.5), width=0.8)
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(max(0, t.values.min() - 0.1), min(1, t.values.max() + 0.05))
    ax.set_xticklabels([re.sub(r" \(.*?\)", "", c) for c in t.columns], rotation=0, fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "slices.png", dpi=150)
    plt.close()


def plot_cost(cost):
    # one point per model type (best variant), otherwise the 24 classical runs crowd the chart
    cost = cost.assign(kind=[n.split("_")[0] for n in cost.index])
    cost = cost.sort_values("macro_f1", ascending=False).groupby("kind", sort=False).head(1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"classical": "tab:blue", "deep": "tab:orange", "transformer": "tab:red"}
    for name, r in cost.iterrows():
        ax.scatter(r.infer_ms_per_1k, r.macro_f1, s=60, color=colors.get(r.family, "gray"))
        ax.annotate(name, (r.infer_ms_per_1k, r.macro_f1), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("CPU time per 1,000 tweets (ms, log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Accuracy vs cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_vs_cost.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None, help="model to compare against (default: svm_char_cleaned)")
    ap.add_argument("--cmi", type=float, default=40, help="CMI threshold for 'heavily mixed'")
    args = ap.parse_args()

    test = load_splits()["test"]
    test["id"] = test["id"].astype(str)
    gold = test.label.map(LABELS.index).values

    preds = {}
    for f in sorted(PREDS_DIR.glob("*.csv")):
        p = pd.read_csv(f, dtype={"id": str}).set_index("id").loc[test.id]
        preds[f.stem] = p.pred.map(LABELS.index).values
    if not preds:
        raise SystemExit("No predictions in results/preds/. Train some models first.")

    # main table
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "macro_f1": round(macro_f1(gold, p), 4),
                     "accuracy": round(float(np.mean(gold == p)), 4), **per_class_f1(gold, p)})
    main_table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).set_index("model")
    main_table.to_csv(RESULTS_DIR / "main_results.csv")

    classical = [m for m in main_table.index if m.split("_")[0] in ("svm", "logreg", "nb", "tree")
                 and "hybrid" not in m]
    # default reference = the mid-review best (SVM, cleaned text, char features)
    ref = args.reference or ("svm_char_cleaned" if "svm_char_cleaned" in preds else
                             (classical[0] if classical else main_table.index[0]))

    # slices
    slices = build_slices(test, args.cmi)
    slice_rows = {}
    for name, p in preds.items():
        slice_rows[name] = {s: round(macro_f1(gold[m], p[m]), 4) if m.sum() else np.nan for s, m in slices.items()}
    slice_table = pd.DataFrame(slice_rows).T.loc[main_table.index]
    slice_table.columns = [f"{s} [{slices[s].sum()}]" for s in slices]
    slice_table.to_csv(RESULTS_DIR / "slices.csv")

    # bootstrap vs reference
    boot = []
    for name, p in preds.items():
        if name == ref:
            continue
        mean, lo, hi, p_val = paired_bootstrap(gold, p, preds[ref])
        boot.append({"model": name, "vs": ref, "delta_f1": round(mean, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "p_not_better": round(p_val, 3),
                     "significant": "yes" if (lo > 0 or hi < 0) else "no"})
    boot_table = pd.DataFrame(boot).sort_values("delta_f1", ascending=False)
    boot_table.to_csv(RESULTS_DIR / "bootstrap.csv", index=False)

    # cost (Q3)
    cost_path = RESULTS_DIR / "costs.json"
    cost = pd.DataFrame(json.loads(cost_path.read_text())).T if cost_path.exists() else pd.DataFrame()
    if len(cost):
        cost = cost[cost.index.isin(main_table.index)].copy()
        cost["macro_f1"] = main_table.loc[cost.index, "macro_f1"]
        cost = cost.sort_values("macro_f1", ascending=False)
        cost.to_csv(RESULTS_DIR / "cost.csv")
        plot_cost(cost)

    # errors: what does the best model fix, and what does nobody get right?
    best = main_table.index[0]
    view = test[["id", "text", "label", "cmi"]].copy()
    view["ref_pred"] = [LABELS[i] for i in preds[ref]]
    view["best_pred"] = [LABELS[i] for i in preds[best]]
    view["negation"] = slices["With negation"]
    fixed = view[(preds[ref] != gold) & (preds[best] == gold)]
    all_wrong = view[np.all([p != gold for p in preds.values()], axis=0)]
    fixed.to_csv(RESULTS_DIR / f"errors_fixed_by_{best}.csv", index=False)
    all_wrong.to_csv(RESULTS_DIR / "errors_all_models_wrong.csv", index=False)

    # how many mistakes involve the neutral class?
    neutral_share = {}
    for name, p in preds.items():
        wrong = p != gold
        neutral_share[name] = round(float(np.mean((gold[wrong] == 1) | (p[wrong] == 1))), 3) if wrong.any() else 0

    # figures
    top = [ref] + [m for m in main_table.index if m != ref][:3]
    plot_confusions(preds, gold, top)
    families = [ref] + [m for m in main_table.index if m.startswith(("bilstm", "cnn", "lstm", "charcnn"))][:2]
    families += [m for m in main_table.index if m in cost.index and cost.loc[m, "family"] == "transformer"][:2] \
        if len(cost) else []
    plot_slices(slice_table, list(dict.fromkeys(families)) or top)

    # summary
    lines = ["# Results summary\n", f"Reference model for comparisons: **{ref}**  |  best model: **{best}**\n",
             "## Test scores\n", main_table.to_markdown(), "\n## Hard tweets (macro-F1, [number of tweets])\n",
             slice_table.to_markdown(), "\n## Paired bootstrap vs reference (1000 resamples)\n",
             boot_table.to_markdown(index=False),
             "\n## Share of mistakes that involve the neutral class\n",
             pd.Series(neutral_share, name="neutral_share").to_frame().to_markdown()]
    if len(cost):
        lines += ["\n## Cost (Q3)\n", cost[["family", "params", "train_seconds", "infer_ms_per_1k", "size_mb",
                                           "device", "macro_f1"]].to_markdown()]
    vs = RESULTS_DIR / "vader_study.csv"
    if vs.exists():
        lines += ["\n## Base-paper check (VADER labels)\n", pd.read_csv(vs).to_markdown(index=False)]
    lines += [f"\n## Error files\n- errors_fixed_by_{best}.csv: {len(fixed)} tweets the reference got wrong "
              f"and {best} got right\n- errors_all_models_wrong.csv: {len(all_wrong)} tweets no model got right\n"]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))

    pd.set_option("display.width", 200)
    print(main_table, "\n")
    print(slice_table, "\n")
    print(boot_table.to_string(index=False))
    print("\nWrote results/SUMMARY.md and figures in results/figures/")


if __name__ == "__main__":
    main()
