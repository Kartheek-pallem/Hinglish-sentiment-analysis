"""The one scoring function everyone uses, so every number is comparable.

    from src.evaluate import evaluate
    evaluate(y_true, y_pred, ids=test.id, model="logreg", owner="person1",
             dataset="sentimix_test", preprocessing="raw", features="word")

What it does:
  * prints accuracy, macro-F1 and per-class F1;
  * appends one row to results/results_<owner>.csv (columns below, always in this order);
    each person has their own file, so git never has merge conflicts;
    load_results() combines all four;
  * saves the predictions to results/preds/<run_id>.csv (id, true, pred), which
    slice analysis and significance tests need;
  * saves a confusion-matrix image to results/confusion/<run_id>.png.
"""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

LABELS = ["negative", "neutral", "positive"]
COLUMNS = ["run_id", "date", "owner", "model", "dataset", "preprocessing", "features", "dict_size", "seed",
           "train_fraction", "n", "accuracy", "precision_macro", "recall_macro", "macro_f1",
           "f1_negative", "f1_neutral", "f1_positive", "train_time_s", "infer_time_s", "params", "notes"]


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(s)).strip("-").lower()


def evaluate(y_true, y_pred, *, model: str, owner: str, dataset: str, preprocessing: str = "raw",
             features: str = "", ids=None, dict_size="", seed="", train_fraction=1.0,
             train_time_s="", infer_time_s="", params="", notes="", results_dir: str = "results",
             save_plot: bool = True, verbose: bool = True) -> dict:
    y_true = [str(y).strip().lower() for y in y_true]
    y_pred = [str(y).strip().lower() for y in y_pred]
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} items but y_pred has {len(y_pred)}")
    unknown = (set(y_true) | set(y_pred)) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels {sorted(unknown)}; use {LABELS}")
    if owner not in {"person1", "person2", "person3", "person4"}:
        raise ValueError("owner must be person1, person2, person3 or person4")

    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "owner": owner, "model": model, "dataset": dataset,
        "preprocessing": preprocessing, "features": features, "dict_size": dict_size, "seed": seed,
        "train_fraction": train_fraction, "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "f1_negative": round(per_class[0], 4), "f1_neutral": round(per_class[1], 4), "f1_positive": round(per_class[2], 4),
        "train_time_s": train_time_s, "infer_time_s": infer_time_s, "params": params, "notes": notes,
    }
    parts = [owner, model, dataset, preprocessing, features, f"d{dict_size}" if dict_size != "" else "",
             f"s{seed}" if seed != "" else "", f"f{train_fraction}" if train_fraction != 1.0 else ""]
    row["run_id"] = "_".join(_slug(p) for p in parts if p != "")

    os.makedirs(os.path.join(results_dir, "preds"), exist_ok=True)
    res_path = os.path.join(results_dir, f"results_{owner}.csv")
    new = not os.path.exists(res_path)
    with open(res_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)

    pred_path = os.path.join(results_dir, "preds", row["run_id"] + ".csv")
    ids = list(ids) if ids is not None else list(range(len(y_true)))
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "true", "pred"])
        w.writerows(zip(ids, y_true, y_pred))

    if save_plot:
        _plot_confusion(y_true, y_pred, row["run_id"], os.path.join(results_dir, "confusion"))
    if verbose:
        print(f"{row['run_id']}: macro-F1 {row['macro_f1']:.4f} | accuracy {row['accuracy']:.4f} | "
              f"F1 neg/neu/pos {row['f1_negative']:.3f}/{row['f1_neutral']:.3f}/{row['f1_positive']:.3f}")
    return row


def load_results(results_dir: str = "results"):
    """All logged runs from every person, as one DataFrame."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(results_dir, "results_person*.csv")))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _plot_confusion(y_true, y_pred, run_id, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(run_id, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, run_id + ".png"))
    plt.close(fig)
