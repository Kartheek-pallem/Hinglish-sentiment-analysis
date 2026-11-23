"""Download SentiMix (SemEval-2020 Task 9, Hinglish) and turn it into one tidy CSV.

The Hugging Face repo RTT1/SentiMix holds the original CoNLL files:
    meta<TAB>30258<TAB>positive
    modi<TAB>Eng
    badhai<TAB>Hin
    (blank line between tweets)

Output: data/sentimix.csv with columns
    id, text, label, split, source, langs, cmi, n_hi, n_en
`langs` keeps the per-word language tags so we can find "heavily mixed" tweets later.

Usage:
    python src/data.py                 # download from Hugging Face
    python src/data.py --local         # files already copied into data/raw/
"""

import argparse

import pandas as pd

from common import DATA_DIR, RAW_DIR, LABELS

REPO = "RTT1/SentiMix"
FILES = {
    "train": "train_14k_split_conll.txt",
    "validation": "dev_3k_split_conll.txt",
    "test": "Hindi_test_unalbelled_conll_updated.txt",
}
TEST_LABELS = "test_labels_hinglish.txt"


def download():
    from huggingface_hub import hf_hub_download
    for name in list(FILES.values()) + [TEST_LABELS]:
        hf_hub_download(REPO, name, repo_type="dataset", local_dir=RAW_DIR)
        print("downloaded", name)


def parse_conll(path):
    """Yield one dict per tweet: id, label (may be None), tokens, langs."""
    tweet = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if parts[0] == "meta":
                if tweet:
                    yield tweet
                label = parts[2].strip().lower() if len(parts) > 2 else None
                tweet = {"id": parts[1].strip(), "label": label, "tokens": [], "langs": []}
            elif tweet is not None:
                tweet["tokens"].append(parts[0])
                tweet["langs"].append(parts[1].strip() if len(parts) > 1 else "O")
    if tweet:
        yield tweet


def code_mixing_index(langs):
    """CMI (Das & Gambäck, 2014): 0 = one language only, up to 50 = perfectly mixed."""
    n_hi = sum(l.lower().startswith("hin") for l in langs)
    n_en = sum(l.lower().startswith("eng") for l in langs)
    n_lang = n_hi + n_en
    cmi = 0.0 if n_lang == 0 else 100.0 * (1 - max(n_hi, n_en) / n_lang)
    return cmi, n_hi, n_en


def build_table():
    rows = []
    for split, fname in FILES.items():
        for t in parse_conll(RAW_DIR / fname):
            cmi, n_hi, n_en = code_mixing_index(t["langs"])
            rows.append({
                "id": t["id"], "text": " ".join(t["tokens"]), "label": t["label"],
                "split": split, "source": "sentimix", "langs": " ".join(t["langs"]),
                "cmi": round(cmi, 2), "n_hi": n_hi, "n_en": n_en,
            })
    df = pd.DataFrame(rows)

    # test labels come in a separate file: "Uid,Sentiment"
    gold = pd.read_csv(RAW_DIR / TEST_LABELS, dtype=str)
    gold.columns = [c.strip().lower() for c in gold.columns]
    gold_map = dict(zip(gold["uid"].str.strip(), gold["sentiment"].str.strip().str.lower()))
    is_test = df.split == "test"
    df.loc[is_test, "label"] = df.loc[is_test, "id"].map(gold_map)

    before = len(df)
    df = df[df.label.isin(LABELS)].reset_index(drop=True)
    if before != len(df):
        print(f"dropped {before - len(df)} tweets with missing labels")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="skip download, use files in data/raw/")
    args = ap.parse_args()
    if not args.local:
        download()

    df = build_table()
    df.to_csv(DATA_DIR / "sentimix.csv", index=False)

    print("\nTweets per split and label:")
    print(pd.crosstab(df.split, df.label, margins=True))
    tags = " ".join(df.langs).split()
    total = len(tags)
    for tag in ("Hin", "Eng", "O"):
        print(f"{tag}: {100 * tags.count(tag) / total:.1f}% of words")
    print("\nSaved", DATA_DIR / "sentimix.csv")


if __name__ == "__main__":
    main()
