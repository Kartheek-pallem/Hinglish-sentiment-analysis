"""normalize(text, size): map Hinglish spelling variants to one form.  Owner: Person 2 (preprocessing).

Romanized Hindi has no fixed spelling, so the same word appears in many forms:
    nahi / nahin / nai / nhi      bahut / bohot / bht      achha / accha / acha
A model sees each form as a different word and must learn each one separately.
normalize() rewrites every known variant to one chosen form, using a dictionary
stored in data/lexicon/norm_dict.csv (one row per variant -> canonical mapping).

Research question Q1 asks whether this helps, and how big the dictionary must be,
so the dictionary is sorted by how often each variant appears in the training
tweets, and `size` says how many rows to use:
    normalize(text, 0)     -> no change (same as cleaned text)
    normalize(text, 50)    -> only the 50 most frequent variants
    normalize(text, None)  -> every row the team has kept

    >>> normalize("bht accha laga yrr nhi pata tha", None)
    'bahut achha laga yaar nahi pata tha'

Where the dictionary comes from (`python3 -m src.normalize --build`):
  1. a short hand-written list of well-known variant groups (SEED below);
  2. automatic candidates from the TRAINING tweets only, using words tagged Hindi:
     * same "sound skeleton": doubled letters merged, w->v, z->j, q->k, ph->f
       (accha / achha / acha all become "acha", so they are grouped);
     * vowels dropped, the way people text: "bht" = "bahut" minus vowels,
       "mjhe" = "mujhe" minus vowels.
     In each group the most frequent spelling is the canonical one.

Person 2 must review it: open data/lexicon/norm_dict.csv, read at least the top 200
rows and set keep = 0 for any wrong mapping (two different words merged, such as
haar "defeat" -> har "every"). normalize() skips those rows, and a rebuild keeps
them at 0. A first few are already in REJECT below. Then rerun the experiments.

    python3 -m src.normalize --build     # (re)build the dictionary from the training data
    python3 -m src.normalize             # show the top rows and how many test words change
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict

DICT_PATH = os.path.join("data", "lexicon", "norm_dict.csv")
FIELDS = ["rank", "variant", "canonical", "variant_count", "canonical_count", "source", "keep"]

# canonical form first, then its variants (hand-written; extend freely)
SEED = [
    ["nahi", "nahin", "nai", "nhi", "nhin", "nahee", "nahii", "nahiin"],
    ["bahut", "bohot", "bahot", "bhut", "bht", "bohat", "bhot", "bahoot", "bhout"],
    ["achha", "accha", "acha", "acchha", "achchha", "achaa", "acchaa", "achhaa"],
    ["yaar", "yar", "yrr", "yr", "yaaar"],
    ["kyun", "kyu", "kyon", "kyoon", "kiun"],
    ["pyaar", "pyar", "pyaarr"],
    ["kuch", "kuchh", "kch", "kuchch"],
    ["mujhe", "mjhe", "mujhey", "muje", "mujhay"],
    ["sab", "sb"], ["kar", "kr"], ["par", "pr"], ["hum", "hm"], ["tum", "tm"],
    ["abhi", "abi", "abhee"],
    ["theek", "thik", "thk", "theekh"],
    ["matlab", "mtlb", "matlb"],
    ["bhai", "bhaii"],
]

# automatic pairs that are really two different words (found when reviewing the list).
# Add a pair here, or set keep = 0 in the CSV: both survive a rebuild.
REJECT = {
    ("baad", "bad"),     # baad = after, bad = English "bad"
    ("haar", "har"),     # haar = defeat, har = every
    ("maar", "mar"),     # maar = hit, mar = die
    ("saab", "sab"),     # saab = sir, sab = all
    ("maan", "mann"),    # maan = respect, mann = mind
    ("anna", "aana"),    # anna = elder brother, aana = to come
    ("bhee", "bhe"),     # bhee = bhi (also), not "bhe"
}

VOWELS = set("aeiou")
_cache: dict = {}


# ----------------------------------------------------------------------------- using it
def load_dict(size=None, path: str | None = None) -> dict:
    """The first `size` kept rows of the dictionary as {variant: canonical}."""
    path = path or DICT_PATH
    key = (size, path)
    if key not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python3 -m src.normalize --build` first")
        with open(path, encoding="utf-8") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["rank"]))
        rows = [r for r in rows if str(r["keep"]).strip() != "0"]
        if size is not None:
            rows = rows[: int(size)]
        _cache[key] = {r["variant"]: r["canonical"] for r in rows}
    return _cache[key]


def normalize(text, size=None) -> str:
    """Replace each known variant (whole words only) with its canonical spelling."""
    if size is not None and int(size) == 0:
        return str(text)
    mapping = load_dict(size)
    return " ".join(mapping.get(tok.lower(), tok) for tok in str(text).split(" "))


# ----------------------------------------------------------------------------- building it
def skeleton(word: str) -> str:
    """A rough 'sound' key: spellings that sound the same get the same key."""
    w = word.lower().replace("ph", "f").replace("w", "v").replace("z", "j").replace("q", "k")
    return re.sub(r"(.)\1+", r"\1", w)


def _vowels_dropped(short: str, long: str) -> bool:
    """True if `short` is `long` with only vowels removed (and the first letters agree)."""
    if short[0] != long[0] or len(short) >= len(long):
        return False
    i = 0
    for ch in long:
        if i < len(short) and ch == short[i]:
            i += 1
        elif ch not in VOWELS:
            return False
    return i == len(short)


def build(data_dir: str = "data/processed", out_path: str = DICT_PATH, min_count: int = 3,
          min_hindi: float = 0.6) -> list[dict]:
    import pandas as pd

    from src.clean import clean

    # 1) how often each word appears in the cleaned TRAINING tweets (never val/test)
    train = pd.read_csv(os.path.join(data_dir, "sentimix_train.csv"), dtype={"id": str})
    count = Counter(tok for t in train.text for tok in clean(t).split() if tok.isalpha() and tok.isascii())

    # 2) share of times each word is tagged Hindi (keeps English words like good/god apart)
    tags = pd.read_csv(os.path.join(data_dir, "word_tags.csv"), dtype={"token": str}, keep_default_na=False)
    tags = tags[tags.split == "train"]
    tok = tags.token.str.lower().str.replace(r"([a-z])\1{2,}", r"\1\1", regex=True)
    hindi_share = (tags.tag == "Hin").groupby(tok).mean().to_dict()

    def hindi(w):
        return hindi_share.get(w, 0.0) >= min_hindi

    best: dict[str, tuple] = {}  # variant -> (canonical, source)
    protected = {group[0] for group in SEED}  # hand-chosen canonical forms never become variants

    # 3) hand-written groups (always included, whatever the counts)
    for canonical, *variants in SEED:
        for v in variants:
            best[v] = (canonical, "seed")

    words = [w for w, c in count.items() if c >= min_count and hindi(w) and len(w) >= 2]

    # 4) same sound skeleton -> map to the most frequent spelling in the group
    groups = defaultdict(list)
    for w in words:
        groups[skeleton(w)].append(w)
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=lambda w: count[w])
        for v in members:
            if v != canonical and v not in best and v not in protected and len(v) >= 3:
                best[v] = (canonical, "auto: same sound")

    # 5) vowels dropped ("bht" -> "bahut"); the full form must be at least twice as common
    by_first = defaultdict(list)
    for w in words:
        by_first[w[0]].append(w)
    for v in words:
        if v in best or v in protected:
            continue
        consonant_only = not (set(v) & VOWELS)
        if not consonant_only and (len(v) < 3 or set(v[:-1]) & VOWELS):
            continue  # allow only "bht"/"kr" style words, or a vowel at the very end ("nhi", "mjhe")
        options = [c for c in by_first[v[0]] if count[c] >= 2 * count[v] and _vowels_dropped(v, c)]
        if options:
            best[v] = (max(options, key=lambda c: count[c]), "auto: vowels dropped")

    # 6) never map a word onto something that is itself mapped elsewhere
    for v, (c, s) in list(best.items()):
        seen = {v}
        while c in best and c not in seen:
            seen.add(c)
            c = best[c][0]
        best[v] = (c, s)

    # 7) keep earlier review decisions: pairs in REJECT, or rows someone set to keep = 0
    rejected = set(REJECT)
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            rejected |= {(r["variant"], r["canonical"]) for r in csv.DictReader(f) if str(r["keep"]).strip() == "0"}

    rows = [dict(variant=v, canonical=c, variant_count=count.get(v, 0), canonical_count=count.get(c, 0),
                 source=s, keep=0 if (v, c) in rejected else 1) for v, (c, s) in best.items() if v != c]
    rows.sort(key=lambda r: (-r["variant_count"], r["variant"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    _cache.clear()
    print(f"wrote {len(rows)} mappings to {out_path}")
    return rows


def report(data_dir: str = "data/processed", top: int = 30) -> None:
    import pandas as pd

    from src.clean import clean

    with open(DICT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n{'rank':>4}  {'variant':<12} -> {'canonical':<12} {'times seen':>10}  source")
    for r in rows[:top]:
        print(f"{r['rank']:>4}  {r['variant']:<12} -> {r['canonical']:<12} {r['variant_count']:>10}  {r['source']}"
              f"{'   (keep=0, skipped)' if r['keep'] == '0' else ''}")
    test = pd.read_csv(os.path.join(data_dir, "sentimix_test.csv"), dtype={"id": str})
    cleaned = [clean(t) for t in test.text]
    total = sum(len(t.split()) for t in cleaned)
    print(f"\ntest set: {total} words after clean()")
    for size in (50, 100, 200, None):
        changed = sum(a != b for t in cleaned for a, b in zip(t.split(" "), normalize(t, size).split(" ")))
        tweets = sum(t != normalize(t, size) for t in cleaned)
        print(f"  dict size {str(size or 'all'):>4}: {changed:5d} words changed ({changed / total:.1%}), "
              f"in {tweets} of {len(cleaned)} tweets")


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build()
    report()
