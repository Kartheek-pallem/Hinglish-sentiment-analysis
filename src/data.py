"""SentiMix Hinglish loader (SemEval-2020 Task 9).

Downloads the four Hinglish files from the Hugging Face dataset RTT1/SentiMix,
parses the CoNLL-style format and writes clean CSVs in the team's shared format.

File format (checked against the real files):
    meta<TAB>20134<TAB>neutral      <- header: id and label (the test file has no label)
    RT<TAB>Eng                      <- one token per line with a language tag:
    @<TAB>O                            Eng, Hin, O (no language) or EMT (emoticon)
                                    <- blank line ends the tweet
Test labels live in a separate CSV: Uid,Sentiment.

Known quirks the parser tolerates: Windows line endings, an empty token
("<TAB>O"), a token with no tag, and a last record with no trailing blank line.

Usage:
    python -m src.data            # download (if needed) and build the CSVs
    python -m src.data --no-download --raw-dir data/raw   # files already in data/raw
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

HF_REPO = "RTT1/SentiMix"
FILES = {
    "train": "train_14k_split_conll.txt",
    "val": "dev_3k_split_conll.txt",
    "test": "Hindi_test_unalbelled_conll_updated.txt",
    "test_labels": "test_labels_hinglish.txt",
}
EXPECTED = {"train": 14000, "val": 3000, "test": 3000}
LABELS = ("negative", "neutral", "positive")
TAGS = ("Eng", "Hin", "O", "EMT")


def download_sentimix(raw_dir: str = "data/raw") -> dict:
    """Download the four Hinglish files (skips files already present)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for key, fname in FILES.items():
        target = os.path.join(raw_dir, fname)
        if not os.path.exists(target):
            hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=fname, local_dir=raw_dir)
        paths[key] = target
    return paths


def parse_conll(path: str) -> list[dict]:
    """Parse one SentiMix CoNLL file into a list of tweets.

    Each tweet is a dict: id (str), label (str or None), tokens (list), tags (list).
    """
    tweets, cur = [], None
    problems = Counter()

    def close():
        nonlocal cur
        if cur is not None:
            if cur["tokens"]:
                tweets.append(cur)
            else:
                problems["empty tweet dropped"] += 1
        cur = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                close()
                continue
            parts = line.split("\t")
            if parts[0].strip() == "meta":
                close()
                fields = [p.strip() for p in parts if p.strip()]
                if len(fields) < 2:  # e.g. "meta 123" with a space instead of a tab
                    fields = line.split()
                tid = fields[1] if len(fields) > 1 else None
                label = fields[2].lower() if len(fields) > 2 else None
                cur = {"id": tid, "label": label, "tokens": [], "tags": []}
                continue
            if cur is None:
                problems["token outside a tweet skipped"] += 1
                continue
            token = parts[0].strip()
            tag = parts[-1].strip() if len(parts) > 1 else ""
            if not token:
                problems["empty token skipped"] += 1
                continue
            if tag not in TAGS:
                problems["missing or unknown tag set to O"] += 1
                tag = "O"
            cur["tokens"].append(token)
            cur["tags"].append(tag)
    close()
    for k, v in problems.items():
        print(f"  [{os.path.basename(path)}] {k}: {v}")
    return tweets


def to_frames(tweets: list[dict], split: str):
    """Return (texts_df, tags_df) in the shared format."""
    rows = [{"id": t["id"], "text": " ".join(t["tokens"]), "label": t["label"],
             "source": "sentimix", "split": split} for t in tweets]
    tag_rows = [{"id": t["id"], "position": i, "token": tok, "tag": tag}
                for t in tweets for i, (tok, tag) in enumerate(zip(t["tokens"], t["tags"]))]
    return pd.DataFrame(rows), pd.DataFrame(tag_rows)


def build_csvs(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> dict:
    """Parse all splits, attach test labels, run checks and write the CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    frames, tag_frames = {}, []
    for split in ("train", "val", "test"):
        tweets = parse_conll(os.path.join(raw_dir, FILES[split]))
        df, tags = to_frames(tweets, split)
        tags.insert(1, "split", split)
        frames[split], tag_frames = df, tag_frames + [tags]

    # attach test labels by ID
    lab = pd.read_csv(os.path.join(raw_dir, FILES["test_labels"]))
    lab.columns = [c.strip().lower() for c in lab.columns]
    lab = lab.rename(columns={"uid": "id", "sentiment": "label_true"})
    lab["id"] = lab["id"].astype(str).str.strip()
    lab["label_true"] = lab["label_true"].astype(str).str.strip().str.lower()
    test = frames["test"].drop(columns=["label"]).merge(lab[["id", "label_true"]], on="id", how="left")
    test = test.rename(columns={"label_true": "label"})
    missing = int(test["label"].isna().sum())
    extra = len(set(lab["id"]) - set(test["id"]))
    print(f"  test: {missing} tweets without a label, {extra} labels without a tweet")
    frames["test"] = test[["id", "text", "label", "source", "split"]]

    # checks
    ok = True
    for split, df in frames.items():
        n, exp = len(df), EXPECTED[split]
        dups = int(df["id"].duplicated().sum())
        bad = sorted(set(df["label"].dropna()) - set(LABELS))
        flag = "OK" if n == exp else f"EXPECTED {exp}"
        print(f"  {split:5s}: {n:6d} tweets ({flag}); duplicate ids: {dups}; unknown labels: {bad or 'none'}")
        print(f"         class balance: {df['label'].value_counts(normalize=True).round(3).to_dict()}")
        ok &= (n == exp and dups == 0 and not bad)
    if missing:
        ok = False
        print("  WARNING: some test tweets have no label - check the labels file before reporting test scores.")

    paths = {}
    for split, df in frames.items():
        p = os.path.join(out_dir, f"sentimix_{split}.csv")
        df.to_csv(p, index=False)
        paths[split] = p
    tags_all = pd.concat(tag_frames, ignore_index=True)
    paths["word_tags"] = os.path.join(out_dir, "word_tags.csv")
    tags_all.to_csv(paths["word_tags"], index=False)
    share = tags_all["tag"].value_counts(normalize=True).round(3).to_dict()
    print(f"  word tags: {len(tags_all)} tokens; share by tag: {share}")
    print("  ALL CHECKS PASSED" if ok else "  SOME CHECKS NEED A LOOK (see above)")
    return paths


def load_split(split: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Load one processed split: 'train', 'val' or 'test'. IDs are kept as strings."""
    return pd.read_csv(os.path.join(data_dir, f"sentimix_{split}.csv"), dtype={"id": str}, keep_default_na=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--no-download", action="store_true", help="use files already in --raw-dir")
    a = ap.parse_args()
    if not a.no_download:
        print("Downloading SentiMix Hinglish from Hugging Face ...")
        download_sentimix(a.raw_dir)
    print("Parsing and checking ...")
    build_csvs(a.raw_dir, a.out_dir)
