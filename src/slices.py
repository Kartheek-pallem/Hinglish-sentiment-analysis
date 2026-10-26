"""Slices for research question Q2: which kinds of tweets make the models fail?  Owners: Persons 1 and 4.

A "slice" is a group of test tweets that share one property. Comparing a model's
macro-F1 on each slice with its overall macro-F1 shows where it breaks.

For every val and test tweet this file computes:
  cmi           Code-Mixing Index (Das & Gambäck, 2014), 0-50 for two languages:
                    CMI = 100 * (1 - (words of the main language) / (words tagged Hindi or English))
                0 = one language only, 50 = exactly half Hindi and half English.
                User names and links are left out (the tagger calls them "English").
  cmi_level     low (CMI <= 20), medium (20-35), high (> 35)
  negation      yes/no - contains nahi, nhi, mat, not, no, never, don't, ...
                ("na" is left out on purpose: it is often not a negation, as in "karo na")
  spelling_var  yes/no - contains a word that normalize() would rewrite (a known spelling variant)
  length        short (<= 10 words after clean()), medium (11-20), long (> 20)
  emoji         yes/no

    python3 -m src.slices            # writes data/processed/slices_val.csv and slices_test.csv

Then `python3 scripts/make_tables.py` turns saved predictions into the Q2 table.
Sarcasm is not here yet: it needs the hand-tagged sample (after the mid review).
"""
from __future__ import annotations

import os
import re

import pandas as pd

from src.clean import clean

NEGATION = {
    # Hindi (Romanized)
    "nahi", "nahin", "nhi", "nai", "nhin", "nahee", "nh", "mat", "mt", "bina", "naa",
    # English
    "not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "cannot",
    "dont", "don't", "didnt", "didn't", "doesnt", "doesn't", "isnt", "isn't", "wasnt", "wasn't",
    "cant", "can't", "wont", "won't", "shouldnt", "shouldn't", "wouldnt", "wouldn't", "aint", "n't",
}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
LEVELS = {"cmi_level": ["low", "medium", "high"], "negation": ["no", "yes"], "spelling_var": ["no", "yes"],
          "length": ["short", "medium", "long"], "emoji": ["no", "yes"]}


_LINK_PIECES = {"https", "http", ":", "//", "/", "t", ".", "co", "tco"}


def _real_word_tags(tokens: list[str], tags: list[str]) -> list[str]:
    """Language tags of real words only: drops '@ user _ name' and 'https // t . co / xyz' pieces."""
    keep, i, n = [], 0, len(tokens)
    while i < n:
        low = tokens[i].lower()
        if low in ("@", "@_"):                      # "@ Sana _ arshaa": skip the name and its "_ pieces"
            i += 2
            while i + 1 < n and tokens[i] == "_":
                i += 2
            continue
        if low.startswith("http"):                  # "https // t . co / AbC12": skip to the code
            while i < n and tokens[i].lower() in _LINK_PIECES or (i < n and tokens[i].lower().startswith("http")):
                i += 1
            i += 1                                  # the code itself
            continue
        if set(low) != {"_"}:
            keep.append(tags[i])
        i += 1
    return keep


def cmi(tags: list[str]) -> float:
    eng, hin = tags.count("Eng"), tags.count("Hin")
    return 0.0 if eng + hin == 0 else 100 * (1 - max(eng, hin) / (eng + hin))


def build_slices(split: str, data_dir: str = "data/processed") -> pd.DataFrame:
    from src.normalize import load_dict

    tweets = pd.read_csv(os.path.join(data_dir, f"sentimix_{split}.csv"), dtype={"id": str})
    tags = pd.read_csv(os.path.join(data_dir, "word_tags.csv"), dtype={"id": str, "token": str}, keep_default_na=False)
    tags = tags[tags.split == split].sort_values(["id", "position"])
    variants = load_dict(None)

    per_tweet = {i: _real_word_tags(g.token.tolist(), g.tag.tolist()) for i, g in tags.groupby("id")}
    rows = []
    for t in tweets.itertuples():
        words = clean(t.text).split()
        c = cmi(per_tweet.get(t.id, []))
        rows.append({
            "id": t.id, "label": t.label, "cmi": round(c, 1),
            "cmi_level": "low" if c <= 20 else ("medium" if c <= 35 else "high"),
            "negation": "yes" if any(w in NEGATION for w in words) else "no",
            "spelling_var": "yes" if any(w in variants for w in words) else "no",
            "n_words": len(words),
            "length": "short" if len(words) <= 10 else ("medium" if len(words) <= 20 else "long"),
            "emoji": "yes" if _EMOJI.search(clean(t.text)) else "no",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for split in ("val", "test"):
        df = build_slices(split)
        out = os.path.join("data", "processed", f"slices_{split}.csv")
        df.to_csv(out, index=False)
        print(f"\n{split}: wrote {out} ({len(df)} tweets), mean CMI {df.cmi.mean():.1f}")
        for col, levels in LEVELS.items():
            counts = df[col].value_counts()
            print(f"  {col:<13}" + "  ".join(f"{lv}: {counts.get(lv, 0)}" for lv in levels))
