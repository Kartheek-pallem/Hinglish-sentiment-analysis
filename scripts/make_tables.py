"""Turn everyone's logged runs into the first Q1 and Q2 results.  Owners: Q1 Person 2, Q2 Persons 1 and 4.

Run from the project folder after the runs are done (and after `python3 -m src.slices`):
    python3 scripts/make_tables.py

It reads results/results_person*.csv and results/preds/*.csv and writes:

  results/tables/q1_preprocessing.csv / .md   test macro-F1 for raw, cleaned, normalized (every model)
  results/tables/q1_significance.csv / .md    is each change real?  paired bootstrap, 95% interval
  results/tables/q1_dict_size.csv / .md       macro-F1 as the dictionary grows (where those runs exist)
  results/tables/q2_slices.csv / .md          macro-F1 on each slice, best run of each model
  results/figures/q1_preprocessing.png
  results/figures/q1_dict_size.png
  results/figures/q2_slices_heatmap.png

Paired bootstrap, in plain words: draw 3,000 test tweets with replacement, score
both runs on the same draw, note the difference, repeat 1,000 times. If 95% of the
differences are on the same side of zero, the change is unlikely to be luck.

Which run represents a model in Q2: the one with the best VALIDATION macro-F1, so
the test set is never used to make a choice.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import LABELS, load_results  # noqa: E402

# slice columns and the order to show their levels in (other columns found in the slices file are added too)
LEVELS = {"cmi_level": ["low", "medium", "high"], "negation": ["no", "yes"], "spelling_var": ["no", "yes"],
          "length": ["short", "medium", "long"], "emoji": ["no", "yes"]}

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="results")
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--boot", type=int, default=1000, help="bootstrap repeats")
a = ap.parse_args()

TABLES = os.path.join(a.results_dir, "tables")
FIGS = os.path.join(a.results_dir, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
STAGES = ["raw", "cleaned", "normalized"]
NICE = {"logreg": "Logistic regression", "svm": "Linear SVM", "naive_bayes": "Naive Bayes",
        "complement_nb": "Complement NB", "decision_tree": "Decision tree"}

# colours: one blue ramp, light -> dark, for raw -> cleaned -> normalized (an ordered sequence)
STAGE_COLOURS = {"raw": "#86b6ef", "cleaned": "#2a78d6", "normalized": "#104281"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def save(df, name, floatfmt=".4f"):
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # tabulate not installed
        md = df.to_string(index=False)
    with open(os.path.join(TABLES, name + ".md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\n== {name} ==\n{md}")


def stage_name(row):
    if row.preprocessing != "normalized":
        return row.preprocessing
    return "normalized" if str(row.dict_size) in ("all", "nan", "") else f"normalized_d{row.dict_size}"


def macro_f1(y, p):
    """Fast macro-F1 for label arrays coded 0, 1, 2."""
    f1 = []
    for k in range(3):
        tp = np.sum((y == k) & (p == k), axis=-1)
        fp = np.sum((y != k) & (p == k), axis=-1)
        fn = np.sum((y == k) & (p != k), axis=-1)
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0))
    return np.mean(f1, axis=0)


def load_pred(run_id):
    path = os.path.join(a.results_dir, "preds", run_id + ".csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"id": str})
    return df.set_index("id")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- the runs
res = load_results(a.results_dir)
if res.empty:
    sys.exit("no results yet - run some models first")
res["dict_size"] = res["dict_size"].astype(str).replace("nan", "")
res["stage"] = res.apply(stage_name, axis=1)
res["features"] = res["features"].fillna("")
# a run logged twice (re-run) -> keep the latest; for val keep the final row of the run
test = res[res.dataset == "sentimix_test"].drop_duplicates("run_id", keep="last")
val = res[res.dataset == "sentimix_val"].drop_duplicates("run_id", keep="last")
# the matching validation run has the same run_id with "val" in place of "test"
val_by_run = val.set_index(val.run_id.str.replace("sentimix-val", "sentimix-test", regex=False)).macro_f1
test = test.assign(val_f1=test.run_id.map(val_by_run))
test["model_name"] = test.model.map(NICE).fillna(test.model)

# ----------------------------------------------------------------------------- Q1: preprocessing
q1 = (test[test.stage.isin(STAGES)]
      .pivot_table(index=["model_name", "features"], columns="stage", values="macro_f1", aggfunc="last")
      .reindex(columns=STAGES).reset_index())
if "cleaned" in q1 and "normalized" in q1:
    q1["change_raw_to_normalized"] = q1.normalized - q1.raw
q1 = q1.rename(columns={"model_name": "model"})
save(q1, "q1_preprocessing")

# paired bootstrap on the saved test predictions
code = {lab: i for i, lab in enumerate(LABELS)}
rng = np.random.default_rng(0)
sig_rows = []
for (model, feats), g in test[test.stage.isin(STAGES)].groupby(["model_name", "features"]):
    runs = g.drop_duplicates("stage", keep="last").set_index("stage").run_id
    preds = {s: load_pred(runs[s]) for s in runs.index}
    for before, after in [("raw", "cleaned"), ("cleaned", "normalized"), ("raw", "normalized")]:
        if preds.get(before) is None or preds.get(after) is None:
            continue
        p0, p1 = preds[before], preds[after].reindex(preds[before].index)
        y = p0["true"].map(code).to_numpy()
        a0, a1 = p0["pred"].map(code).to_numpy(), p1["pred"].map(code).to_numpy()
        idx = rng.integers(0, len(y), size=(a.boot, len(y)))
        diffs = macro_f1(y[idx], a1[idx]) - macro_f1(y[idx], a0[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig_rows.append({"model": model, "features": feats, "comparison": f"{before} -> {after}",
                         "change": macro_f1(y, a1) - macro_f1(y, a0), "low_95": lo, "high_95": hi,
                         "real_change": "yes" if lo > 0 or hi < 0 else "no (could be luck)"})
if sig_rows:
    save(pd.DataFrame(sig_rows), "q1_significance")

# dictionary size curve: cleaned = size 0, then each normalized size
curve = test[test.preprocessing.isin(["cleaned", "normalized"])].copy()
if len(curve):
    n_all = None
    lex = os.path.join("data", "lexicon", "norm_dict.csv")
    if os.path.exists(lex):
        lx = pd.read_csv(lex)
        n_all = int((lx["keep"].astype(str).str.strip() != "0").sum())
    curve["size"] = np.where(curve.preprocessing == "cleaned", "0", curve.dict_size)
    curve["size_n"] = pd.to_numeric(curve["size"].replace("all", n_all), errors="coerce")
    has_curve = curve.groupby(["model_name", "features"]).size_n.nunique()
    keep = has_curve[has_curve >= 3].index
    curve = curve.set_index(["model_name", "features"]).loc[keep].reset_index() if len(keep) else curve.iloc[0:0]
if len(curve):
    curve = curve.sort_values(["model_name", "features", "size_n"]).drop_duplicates(
        ["model_name", "features", "size_n"], keep="last")
    save(curve[["model_name", "features", "size", "size_n", "macro_f1", "val_f1"]]
         .rename(columns={"model_name": "model", "size_n": "rows_used"}), "q1_dict_size")

# ----------------------------------------------------------------------------- Q2: slices
slices_path = os.path.join(a.data_dir, "slices_test.csv")
if not os.path.exists(slices_path):
    print("\n(no slices yet - run `python3 -m src.slices`, then this script again, for Q2)")
    q2 = None
else:
    slices = pd.read_csv(slices_path, dtype={"id": str}).set_index("id")
    # use every column that looks like a slice: a few distinct values, not the label or a raw number
    found = {}
    for col in slices.columns:
        if col in ("label", "split") or slices[col].nunique() > 10:
            continue
        values = [str(v) for v in slices[col].dropna().unique()]
        order = [v for v in LEVELS.get(col, []) if v in values]
        found[col] = order + sorted(v for v in values if v not in order)
    slices = slices.astype({c: str for c in found})
    LEVELS = found
    best = test.dropna(subset=["val_f1"]).sort_values("val_f1").drop_duplicates("model", keep="last")
    best = best.sort_values("macro_f1", ascending=False)
    rows = []
    for col, levels in LEVELS.items():
        for lv in levels:
            ids = slices.index[slices[col] == lv]
            row = {"slice": col, "level": lv, "n_tweets": len(ids)}
            for r in best.itertuples():
                p = load_pred(r.run_id)
                if p is None:
                    continue
                p = p.reindex(ids).dropna()
                row[r.model_name] = macro_f1(p["true"].map(code).to_numpy(), p["pred"].map(code).to_numpy())
            rows.append(row)
    q2 = pd.DataFrame(rows)
    overall = {"slice": "ALL", "level": "all test tweets", "n_tweets": len(slices)}
    overall.update({r.model_name: r.macro_f1 for r in best.itertuples()})
    q2 = pd.concat([pd.DataFrame([overall]), q2], ignore_index=True)
    save(q2, "q2_slices")
    with open(os.path.join(TABLES, "q2_runs_used.txt"), "w") as f:
        f.write("\n".join(f"{r.model_name}: {r.run_id} (val {r.val_f1:.4f}, test {r.macro_f1:.4f})"
                          for r in best.itertuples()) + "\n")

# ----------------------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

# Q1 grouped bars
q1p = q1.dropna(subset=["raw"]).dropna(subset=[c for c in ("cleaned", "normalized") if c in q1], how="all").copy()
if len(q1p):
    q1p = q1p.sort_values(["features", "raw"], ascending=[False, False])
    labels = [f"{m}\n({f})" for m, f in zip(q1p.model, q1p.features)]
    x = np.arange(len(q1p))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(q1p)), 3.8), dpi=160)
    for i, s in enumerate(STAGES):
        if s in q1p:
            ax.bar(x + (i - 1) * (w + 0.02), q1p[s], w, color=STAGE_COLOURS[s], label=s, zorder=3)
    lo = np.nanmin(q1p[[s for s in STAGES if s in q1p]].to_numpy())
    ax.set_ylim(max(0, lo - 0.05), min(1, np.nanmax(q1p[[s for s in STAGES if s in q1p]].to_numpy()) + 0.03))
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("test macro-F1", color=INK)
    ax.set_title("Q1 · Does cleaning and normalizing the text help?", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q1_preprocessing.png"))
    plt.close(fig)

# Q1 dictionary-size curve
if len(curve):
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=160)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, ((m, f), g) in enumerate(curve.groupby(["model_name", "features"])):
        ax.plot(g.size_n, g.macro_f1, marker="o", markersize=5, linewidth=2, color=palette[i % 4], label=f"{m} ({f})")
    mid = curve.macro_f1.mean()
    ax.set_ylim(mid - 0.03, mid + 0.03)  # a fixed +-3 point window, so tiny wiggles are not blown up
    ax.set_xlabel("dictionary rows used (0 = cleaned text only)", color=INK)
    ax.set_ylabel("test macro-F1", color=INK)
    ax.set_title("Q1 · How big must the dictionary be?", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q1_dict_size.png"))
    plt.close(fig)

# Q2 heatmap: slice F1 minus the model's overall F1 (blue = better, red = worse)
if q2 is not None and len(q2.columns) > 3:
    models = [c for c in q2.columns if c not in ("slice", "level", "n_tweets")]
    body = q2[q2.slice != "ALL"]
    gap = body[models].to_numpy() - q2.loc[q2.slice == "ALL", models].to_numpy()
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    lim = max(0.05, np.nanmax(np.abs(gap)))
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * len(models), 0.36 * len(body) + 1.4), dpi=160)
    ax.imshow(gap, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            if not np.isnan(gap[i, j]):
                ax.text(j, i, f"{gap[i, j]:+.3f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(models)), models, fontsize=8)
    ax.set_yticks(range(len(body)), [f"{s} = {lv}  (n={n})" for s, lv, n in zip(body.slice, body.level, body.n_tweets)],
                  fontsize=8)
    ax.set_title("Q2 · Slice macro-F1 minus overall macro-F1\n(red = the model does worse on these tweets)",
                 loc="left", fontsize=10)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q2_slices_heatmap.png"))
    plt.close(fig)

print(f"\nTables in {TABLES}/, figures in {FIGS}/")