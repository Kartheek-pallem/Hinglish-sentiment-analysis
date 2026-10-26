"""Report classifier performance on each Person 1 difficulty slice.

Run from the project root after building slices.csv and model predictions:
    python -m src.slice_report

The default reports only test prediction files, never validation predictions.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]
BOOLEAN_SLICES = ["code_mixed", "has_negation", "has_spelling_variation"]


def _score_group(frame: pd.DataFrame) -> dict:
    """Calculate metrics for a subset of tweets."""
    return {
        "n": len(frame),
        "accuracy": round(accuracy_score(frame["true"], frame["pred"]), 4),
        "macro_f1": round(
            f1_score(
                frame["true"], frame["pred"], labels=LABELS,
                average="macro", zero_division=0,
            ),
            4,
        ),
    }


def slice_report(
    predictions_dir: str = "results/preds",
    slices_path: str = "data/processed/slices.csv",
    output_path: str = "results/slice_report.csv",
) -> pd.DataFrame:
    """Compute accuracy and Macro-F1 for each test run and each slice.

    Prediction CSVs must have columns id,true,pred (written by evaluate.py).
    A slice score is calculated only on the tweets belonging to that slice.
    """
    slices = pd.read_csv(slices_path, dtype={"id": str}, keep_default_na=False)
    if slices.empty:
        raise ValueError(f"No slice rows found in {slices_path}")

    paths = sorted(glob.glob(os.path.join(predictions_dir, "*sentimix-test*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No test prediction files matching *sentimix-test*.csv in {predictions_dir}"
        )

    rows = []
    for path in paths:
        pred = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        required = {"id", "true", "pred"}
        if not required.issubset(pred.columns):
            print(f"Skipping {path}: expected columns {sorted(required)}")
            continue

        joined = pred.merge(
            slices.drop(columns=["text", "label"], errors="ignore"),
            on="id", how="inner", validate="many_to_one"
        )
        if joined.empty:
            print(f"Skipping {path}: no prediction IDs matched slices.csv")
            continue

        run_id = os.path.basename(path)[:-4]
        for col in BOOLEAN_SLICES:
            # CSV readers may represent booleans as True/False or 1/0.
            in_slice = joined[col].astype(str).str.lower().isin(["true", "1"])
            for group_name, mask in (("in_slice", in_slice), ("outside_slice", ~in_slice)):
                subset = joined.loc[mask]
                if subset.empty:
                    continue
                rows.append({"run_id": run_id, "slice": col, "group": group_name,
                             **_score_group(subset)})

        if "code_mix_level" in joined.columns:
            for level in sorted(joined["code_mix_level"].unique()):
                subset = joined.loc[joined["code_mix_level"] == level]
                if subset.empty:
                    continue
                rows.append({"run_id": run_id, "slice": "code_mix_level", "group": str(level),
                             **_score_group(subset)})

    report = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Saved {len(report)} slice-score rows to {output_path}")
    if not report.empty:
        print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/preds")
    parser.add_argument("--slices", default="data/processed/slices.csv")
    parser.add_argument("--output", default="results/slice_report.csv")
    args = parser.parse_args()
    slice_report(args.predictions_dir, args.slices, args.output)
