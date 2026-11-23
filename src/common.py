"""Shared paths, labels and small helpers used by every script."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
PREDS_DIR = RESULTS_DIR / "preds"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "saved_models"

LABELS = ["negative", "neutral", "positive"]   # same order as the paper: 0, 1, 2
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

for d in (RAW_DIR, PREDS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_splits():
    """Read the processed table written by data.py."""
    path = DATA_DIR / "sentimix.csv"
    if not path.exists():
        raise SystemExit("data/sentimix.csv not found. Run:  python src/data.py")
    df = pd.read_csv(path, keep_default_na=False)
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "test")}


def save_predictions(name, ids, gold, pred, probs=None):
    """Every model writes the same CSV, so evaluate.py can compare them all."""
    out = pd.DataFrame({"id": ids, "gold": gold, "pred": pred})
    if probs is not None:
        for i, lab in enumerate(LABELS):
            out[f"p_{lab}"] = np.round(probs[:, i], 5)
    out.to_csv(PREDS_DIR / f"{name}.csv", index=False)


def save_cost(name, **info):
    """Store training/inference cost so analysis.py can build the cost table (Q3)."""
    path = RESULTS_DIR / "costs.json"
    costs = json.loads(path.read_text()) if path.exists() else {}
    costs[name] = info
    path.write_text(json.dumps(costs, indent=2))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
