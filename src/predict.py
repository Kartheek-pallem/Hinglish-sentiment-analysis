"""Load a saved model and label new Hinglish text (used by the web app too).

    python src/predict.py --text "movie bahut bakwas thi yaar"
    python src/predict.py --model hingbert --input youtube_comments.csv --column comment --output labelled.csv

Models: svm (always available after classical.py), bilstm, hingbert (after their training scripts).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, MODELS_DIR
from preprocess import clean, normalize


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Predictor:
    def __init__(self, model="svm"):
        self.kind = model
        if model == "svm":
            saved = pickle.load(open(MODELS_DIR / "svm_best.pkl", "rb"))
            self.pipe, self.version, self.dictionary = saved["pipeline"], saved["text_version"], saved["dictionary"]
        elif model == "bilstm":
            import torch
            from deep_models import build_model
            saved = pickle.load(open(MODELS_DIR / "bilstm_vocab.pkl", "rb"))
            self.vocab, self.version = saved["vocab"], saved["text_version"]
            self.net = build_model("bilstm", len(self.vocab))
            self.net.load_state_dict(torch.load(MODELS_DIR / "bilstm.pt", map_location="cpu"))
            self.net.eval()
            self.dictionary = saved.get("dictionary", {})
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = MODELS_DIR / model
            self.tok = AutoTokenizer.from_pretrained(path)
            self.net = AutoModelForSequenceClassification.from_pretrained(path).eval()
            self.version = (path / "text_version.txt").read_text().strip()
            self.dictionary = {}

    def prepare(self, texts):
        cleaned = [clean(str(t)) for t in texts]
        if self.version == "normalized":
            return [normalize(t, self.dictionary) for t in cleaned]
        return cleaned if self.version == "cleaned" else [str(t) for t in texts]

    def predict(self, texts, batch_size=64):
        """Returns (labels, probabilities[n, 3])."""
        texts = self.prepare(texts)
        if self.kind == "svm":
            probs = softmax(self.pipe.decision_function(texts))   # SVM scores -> rough confidence
        elif self.kind == "bilstm":
            from deep_models import predict as run
            ids, lens = self.vocab.encode(texts, 60)
            probs = run(self.net, ids, lens, "cpu")
        else:
            import torch
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=64,
                                   padding=True, return_tensors="pt")
                    out.append(torch.softmax(self.net(**enc).logits, dim=1).numpy())
            probs = np.concatenate(out) if out else np.zeros((0, 3))
        return [LABELS[i] for i in probs.argmax(1)], probs


def available_models():
    names = []
    if (MODELS_DIR / "svm_best.pkl").exists():
        names.append("svm")
    if (MODELS_DIR / "bilstm.pt").exists():
        names.append("bilstm")
    names += sorted(p.name for p in MODELS_DIR.iterdir() if (p / "config.json").exists())
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm")
    ap.add_argument("--text")
    ap.add_argument("--input", help="CSV file with a text column")
    ap.add_argument("--column", default="text")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    p = Predictor(args.model)
    if args.text:
        label, prob = p.predict([args.text])
        print(label[0], dict(zip(LABELS, np.round(prob[0], 3))))
        return
    df = pd.read_csv(args.input)
    df["sentiment"], probs = p.predict(df[args.column].astype(str).tolist())
    df["confidence"] = probs.max(1).round(3)
    df.to_csv(args.output, index=False)
    print(df.sentiment.value_counts())
    if "label" in df:   # hand-labelled file -> out-of-domain score
        from sklearn.metrics import classification_report
        print(classification_report(df.label.str.lower(), df.sentiment, digits=3))
    print("saved", args.output)


if __name__ == "__main__":
    main()
