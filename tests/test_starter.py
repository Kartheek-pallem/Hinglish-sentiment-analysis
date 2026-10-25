"""Run with:  python -m unittest discover -s tests -v

The fixtures are a few made-up tweets that copy the real file format,
including its quirks (Windows line endings, an empty token, a missing tag,
and a last record without a blank line).
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIX = os.path.join(ROOT, "tests", "fixtures")

from src import data  # noqa: E402
from src.evaluate import COLUMNS, evaluate, load_results  # noqa: E402
from src.features import build_features  # noqa: E402


class TestParser(unittest.TestCase):
    def test_parses_quirks(self):
        tw = data.parse_conll(os.path.join(FIX, "mini_train.txt"))
        self.assertEqual([t["id"] for t in tw], ["101", "102", "103"])
        self.assertEqual([t["label"] for t in tw], ["positive", "neutral", "negative"])
        self.assertEqual(" ".join(tw[0]["tokens"]), "All the best jeet ke aana")  # \r\n handled
        self.assertEqual(tw[1]["tokens"][0], "RT")  # empty token dropped
        self.assertEqual(tw[2]["tags"], ["Hin", "EMT", "Hin", "O"])  # missing tag -> O, last record kept

    def test_unlabelled_test_file(self):
        tw = data.parse_conll(os.path.join(FIX, "mini_test.txt"))
        self.assertEqual([(t["id"], t["label"]) for t in tw], [("201", None), ("202", None)])


class TestBuildCsvs(unittest.TestCase):
    def test_build_and_merge_labels(self):
        raw, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(FIX, "mini_train.txt"), os.path.join(raw, data.FILES["train"]))
            shutil.copy(os.path.join(FIX, "mini_train.txt"), os.path.join(raw, data.FILES["val"]))
            shutil.copy(os.path.join(FIX, "mini_test.txt"), os.path.join(raw, data.FILES["test"]))
            shutil.copy(os.path.join(FIX, "mini_test_labels.txt"), os.path.join(raw, data.FILES["test_labels"]))
            data.build_csvs(raw, out)
            test = data.load_split("test", out)
            self.assertEqual(test.label.tolist(), ["positive", "negative"])
            self.assertEqual(list(test.columns), ["id", "text", "label", "source", "split"])
            tags = pd.read_csv(os.path.join(out, "word_tags.csv"), dtype={"id": str})
            self.assertEqual(set(tags.split), {"train", "val", "test"})
        finally:
            shutil.rmtree(raw), shutil.rmtree(out)


class TestEvaluate(unittest.TestCase):
    def test_row_preds_and_plot(self):
        d = tempfile.mkdtemp()
        try:
            y = ["positive", "negative", "neutral", "positive"]
            p = ["positive", "neutral", "neutral", "positive"]
            r = evaluate(y, p, ids=["a", "b", "c", "d"], model="logreg", owner="person3", dataset="sentimix_test",
                         preprocessing="raw", features="word", results_dir=d, verbose=False)
            self.assertAlmostEqual(r["accuracy"], 0.75)
            res = pd.read_csv(os.path.join(d, "results_person3.csv"))
            self.assertEqual(len(load_results(d)), 1)
            self.assertEqual(list(res.columns), COLUMNS)
            preds = pd.read_csv(os.path.join(d, "preds", r["run_id"] + ".csv"))
            self.assertEqual(preds.id.tolist(), ["a", "b", "c", "d"])
            self.assertTrue(os.path.exists(os.path.join(d, "confusion", r["run_id"] + ".png")))
        finally:
            shutil.rmtree(d)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            evaluate(["positive"], ["happy"], model="m", owner="person1", dataset="x", save_plot=False, verbose=False)
        with self.assertRaises(ValueError):
            evaluate(["positive"], ["positive"], model="m", owner="sai", dataset="x", save_plot=False, verbose=False)


class TestFeatures(unittest.TestCase):
    def test_fit_on_train_only(self):
        tr = ["bahut achha hai yaar", "bahut bekaar hai", "achha laga", "bekaar service"]
        te = ["totally newword here"]
        for kind in ("word", "char", "both"):
            X_tr, X_te, vec = build_features(tr, te, kind=kind, **({} if kind == "both" else {"min_df": 1}))
            self.assertEqual(X_tr.shape[0], 4)
            self.assertEqual(X_te.shape[1], X_tr.shape[1])


class TestEndToEnd(unittest.TestCase):
    """Runs the baseline script on a small synthetic dataset in the processed format."""
    def test_baseline_script(self):
        d, res = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            rng = random.Random(0)
            words = {"positive": ["achha", "mast", "badhiya", "love", "great"],
                     "negative": ["bekaar", "worst", "ganda", "hate", "bad"],
                     "neutral": ["kab", "kya", "when", "where", "news"]}
            for split, n in (("train", 300), ("val", 60), ("test", 60)):
                rows = []
                for i in range(n):
                    lab = rng.choice(list(words))
                    toks = rng.choices(words[lab], k=3) + rng.choices(["hai", "the", "yaar", "is"], k=2)
                    rows.append({"id": f"{split}{i}", "text": " ".join(toks), "label": lab, "source": "sentimix", "split": split})
                pd.DataFrame(rows).to_csv(os.path.join(d, f"sentimix_{split}.csv"), index=False)
            out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "baseline_lr.py"), "--owner", "person1",
                                  "--tune", "--data-dir", d, "--results-dir", res], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            r = load_results(res)
            self.assertGreater(r[r.dataset == "sentimix_test"].macro_f1.iloc[-1], 0.9)
        finally:
            shutil.rmtree(d), shutil.rmtree(res)


if __name__ == "__main__":
    unittest.main()
