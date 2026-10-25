# CS 725 · Hinglish sentiment analysis

Three-class sentiment (positive, negative, neutral) on code-mixed Hinglish tweets from
**SentiMix Hinglish** (SemEval-2020 Task 9): 14,000 train / 3,000 validation / 3,000 test.

Research questions:
* **Q1** How much does spelling normalization help, and for which models?
* **Q2** Which kinds of Hinglish text make models fail (code-mixing, spelling variation, negation, sarcasm)?
* **Q3** When is a transformer actually needed?

## Start here (day 1)

Open `notebooks/01_first_baseline.ipynb` in Colab or Kaggle and run it top to bottom.
It downloads the data, runs the tests and trains TF-IDF + Logistic Regression on raw text.
**Everyone posts their test macro-F1; all four numbers must match.**

Or locally:

```bash
pip install -r requirements.txt
python -m src.data                                   # download + build CSVs (prints checks)
python -m unittest discover -s tests                 # 7 tests, all should pass
python scripts/baseline_lr.py --owner person1        # first baseline
```

## What is here

| File | What it does | Owner from now on |
|---|---|---|
| `src/data.py` | Downloads SentiMix from Hugging Face, parses the CoNLL format, attaches test labels, checks counts, writes CSVs | Person 2 |
| `src/features.py` | TF-IDF word n-grams (1–2), character n-grams (2–5), or both; fitted on train only | Person 3 |
| `src/evaluate.py` | `evaluate()` scores a run, logs it, saves predictions and a confusion matrix; `load_results()` combines everyone's runs | Person 3 |
| `scripts/baseline_lr.py` | The day-1 baseline | Everyone |
| `notebooks/01_first_baseline.ipynb` | Colab/Kaggle notebook for the steps above | Person 4 |
| `tests/` | Unit tests, with small made-up files that copy the real format's quirks | Everyone |

Still to build (see the team plan): `clean.py` and `normalize.py` (Person 2), `slices.py` and
`slice_report.py` (Person 1), `compare.py` paired bootstrap and `learning_curve.py` (Person 3),
`dl_template.py` (Person 4).

## Data files

`python -m src.data` writes to `data/processed/`:

| File | Columns |
|---|---|
| `sentimix_train.csv`, `sentimix_val.csv`, `sentimix_test.csv` | `id, text, label, source, split` |
| `word_tags.csv` | `id, split, position, token, tag` — tag is `Eng`, `Hin`, `O` (no language) or `EMT` (emoticon) |

`text` is the tokens joined with spaces, unchanged. Cleaning and normalization happen on the fly,
so the raw text is never overwritten. Load a split with `load_split("train")`; IDs are strings.

Quirks the loader handles: Windows line endings, an empty token, a token with no tag, and a last
record with no blank line after it. The test file has no labels; they come from
`test_labels_hinglish.txt` and are matched by ID (the loader reports any mismatch).

## Logging results

Every score goes through `evaluate()`:

```python
from src.evaluate import evaluate
evaluate(test.label, preds, ids=test.id, model="svm", owner="person3",
         dataset="sentimix_test", preprocessing="normalized", features="char", dict_size=100)
```

* Each person's runs go to `results/results_<owner>.csv`, so git never has merge conflicts.
* Predictions go to `results/preds/<run_id>.csv` (`id, true, pred`); slice analysis and significance tests read these.
* `infer_time_s` is seconds per 1,000 tweets; `params` is the number of trained weights.
* Commit your own results file and predictions after each batch of runs.

## Rules

* Use the official splits. Never re-split the data.
* Tune on `sentimix_val` only. Score `sentimix_test` once per final setting.
* Main metric: macro-F1. Before comparing with the SemEval-2020 best score (75.0), check which F1 the task paper used.
* Random seed 42 for anything random; deep models use seeds 42, 43, 44.
* Never commit `.env`, raw downloads or model checkpoints.
