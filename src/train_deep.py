"""Train the deep models from scratch and save their test predictions.

    python src/train_deep.py                         # lstm bilstm cnn charcnn on cleaned text
    python src/train_deep.py --models bilstm --text normalized
    python src/train_deep.py --seeds 42 43 44        # repeat with 3 seeds (mean/std printed)

Time on a Colab T4: about 2-3 minutes for all four models.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import LABEL2ID, LABELS, MODELS_DIR, RESULTS_DIR, Timer, load_splits, save_cost, save_predictions, set_seed
from deep_models import Vocab, build_model, count_params, predict, train
from preprocess import add_text_versions

MAX_LEN = {"word": 60, "char": 280}


def encode_all(splits, text_col, char_level):
    vocab = Vocab(splits["train"][text_col], char_level=char_level)
    max_len = MAX_LEN["char" if char_level else "word"]
    data = {}
    for name, df in splits.items():
        ids, lens = vocab.encode(df[text_col].tolist(), max_len)
        y = torch.tensor(df.label.map(LABEL2ID).values)
        data[name] = (ids, lens, y)
    return vocab, data


def cpu_latency(model, data):
    """ms per 1,000 tweets on CPU, so every model is timed on the same hardware."""
    model = model.to("cpu")
    ids, lens, _ = data
    with Timer() as t:
        predict(model, ids, lens, "cpu")
    return round(t.seconds / len(ids) * 1e6, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "bilstm", "cnn", "charcnn"])
    ap.add_argument("--text", default="cleaned", choices=["raw", "cleaned", "normalized"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    splits, dictionary = add_text_versions(load_splits())
    test = splits["test"]
    summary = []

    for name in args.models:
        char_level = name == "charcnn"
        vocab, data = encode_all(splits, args.text, char_level)
        scores = []
        for seed in args.seeds:
            set_seed(seed)
            print(f"\n{name} (seed {seed}, vocab {len(vocab)})")
            model = build_model(name, len(vocab))
            with Timer() as t:
                model, val_f1 = train(model, data["train"], data["validation"], device, epochs=args.epochs)
            probs = predict(model, data["test"][0], data["test"][1], device)
            pred = [LABELS[i] for i in probs.argmax(1)]
            test_f1 = f1_score(test.label, pred, average="macro")
            scores.append(test_f1)
            print(f"  -> val={val_f1:.4f} test={test_f1:.4f} ({t.seconds:.0f}s)")

            if seed == args.seeds[0]:   # keep predictions/costs of the first seed
                run_name = f"{name}_{args.text}"
                save_predictions(run_name, test.id, test.label, pred, probs)
                save_cost(run_name, family="deep", params=count_params(model),
                          train_seconds=round(t.seconds, 1), infer_ms_per_1k=cpu_latency(model, data["test"]),
                          size_mb=round(count_params(model) * 4 / 1e6, 2), device=device)
                if name == "bilstm":
                    torch.save(model.state_dict(), MODELS_DIR / "bilstm.pt")
                    with open(MODELS_DIR / "bilstm_vocab.pkl", "wb") as f:
                        pickle.dump({"vocab": vocab, "text_version": args.text, "dictionary": dictionary}, f)

        summary.append({"model": name, "text": args.text, "test_f1_mean": round(np.mean(scores), 4),
                        "test_f1_std": round(np.std(scores), 4), "seeds": len(scores)})

    table = pd.DataFrame(summary)
    path = RESULTS_DIR / "deep_runs.csv"
    if path.exists():   # keep rows from earlier calls (e.g. another text version)
        old = pd.read_csv(path)
        old = old[~old.set_index(["model", "text"]).index.isin(table.set_index(["model", "text"]).index)]
        table = pd.concat([old, table], ignore_index=True)
    table.to_csv(path, index=False)
    print("\n", table)


if __name__ == "__main__":
    main()
