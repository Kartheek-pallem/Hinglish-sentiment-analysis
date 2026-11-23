"""Three text versions used in every experiment: raw, cleaned, normalized.

    raw:        @ Ravi bht acchaaa yrr https // t.co/X
    cleaned:    bht acchaa yrr
    normalized: bahut achha yaar

The spelling dictionary is learned from training tweets only (plus a small
hand-written seed list), so no test information leaks into it.
"""

import re
from collections import Counter, defaultdict

try:
    import ftfy
except ImportError:          # cleaning still works, just without emoji repair
    ftfy = None

# ---------------------------------------------------------------- cleaning

URL_RE = re.compile(r"https?\s*:?\s*/\s*/\s*\S+|\bt\.co\s*/\s*\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\s*_*\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # 3+ same letters -> 2
KEEP_RE = re.compile(r"[^a-z0-9!?\s\U0001F000-\U0001FAFF☀-➿]")
SPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(r"([\U0001F000-\U0001FAFF☀-➿])")


def clean(text):
    if ftfy is not None:
        text = ftfy.fix_text(text)              # repairs broken emoji like "ðŸ˜‚"
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ").lower()
    text = text.replace("’", "'").replace("n't", " not")
    text = REPEAT_RE.sub(r"\1\1", text)
    text = KEEP_RE.sub(" ", text)
    text = EMOJI_RE.sub(r" \1 ", text)          # each emoji becomes its own token
    return SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------- normalization

SEED_DICT = {
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "nahee": "nahi", "nehi": "nahi",
    "bht": "bahut", "bhut": "bahut", "bohot": "bahut", "bahot": "bahut", "bhot": "bahut",
    "yrr": "yaar", "yr": "yaar", "yar": "yaar",
    "acha": "achha", "accha": "achha", "acchaa": "achha", "acchha": "achha", "achchha": "achha",
    "kya": "kya", "kyaa": "kya", "kia": "kya",
    "h": "hai", "hain": "hai", "he": "hai",
    "kr": "kar", "krna": "karna", "krte": "karte", "kro": "karo",
    "plz": "please", "pls": "please", "plzz": "please",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks",
    "u": "you", "ur": "your", "r": "are",
    "sbse": "sabse", "sb": "sab", "sbko": "sabko", "jb": "jab", "tb": "tab",
    "mjhe": "mujhe", "muje": "mujhe", "tumhe": "tumhe", "aap": "aap", "ap": "aap",
    "gud": "good", "gr8": "great", "luv": "love", "dnt": "dont", "cnt": "cant",
}

# short, very common words that look alike but mean different things
PROTECTED = {
    "hai", "ho", "hi", "he", "ha", "me", "main", "mai", "mein", "na", "ne", "no", "not",
    "to", "too", "the", "ka", "ki", "ke", "ko", "se", "so", "si", "sa", "par", "per",
    "tum", "tu", "ye", "yeh", "wo", "woh", "hum", "ham", "is", "us", "was", "as",
    "a", "i", "an", "in", "on", "it", "at", "of", "or", "be", "by",
}

VOWELS = set("aeiouy")


def skeleton(word):
    """First letter + consonants, repeats collapsed: 'bahut' and 'bht' both -> 'bht'."""
    word = re.sub(r"(.)\1+", r"\1", word)
    return word[0] + "".join(c for c in word[1:] if c not in VOWELS)


def is_subsequence(short, long):
    it = iter(long)
    return all(c in it for c in short)


def learn_dictionary(train_texts, max_pairs=None, min_count=3):
    """Group training words by skeleton; map rare spellings onto the most common one.

    Returns {variant: canonical}. `max_pairs` keeps only the most frequent
    variants (used for the dictionary-size ablation: 50 / 100 / 200 / all).
    """
    counts = Counter(w for t in train_texts for w in t.split() if w.isalpha())
    groups = defaultdict(list)
    for w, c in counts.items():
        if c >= min_count and len(w) >= 2 and w not in PROTECTED:
            groups[skeleton(w)].append(w)

    pairs = {}
    for words in groups.values():
        if len(words) < 2:
            continue
        canon = max(words, key=counts.get)
        for w in words:
            if w == canon or len(w) < 3:
                continue
            # only spelling variants: rarer, and one is the other with vowels dropped/added
            rare_enough = counts[w] <= 0.5 * counts[canon]
            if rare_enough and (is_subsequence(w, canon) or is_subsequence(canon, w)):
                pairs[w] = canon

    learned = sorted(pairs, key=counts.get, reverse=True)
    if max_pairs is not None:
        learned = learned[:max_pairs]
    dictionary = {w: pairs[w] for w in learned}
    if max_pairs != 0:
        dictionary.update(SEED_DICT)        # hand-written pairs win over learned ones
    return dictionary


def normalize(text, dictionary):
    return " ".join(dictionary.get(w, w) for w in text.split())


# ---------------------------------------------------------------- stopwords (paper's step)

ENGLISH_STOP = """a an the is am are was were be been being i me my we our you your he him his
she her it its they them their this that these those of in on at to for from by with about as
into over and or but if so than too very can will just do does did have has had not no nor
don dont didnt doesnt isnt wasnt cant wont""".split()
HINGLISH_STOP = """hai hain ho tha thi the hu hoon ka ki ke ko se me mein main mai par pe ye yeh
wo woh is us aur bhi to toh tha hi na nahi nhi mat kya koi kuch sab ek jo ki""".split()
NEGATIONS = {"not", "no", "nor", "never", "dont", "didnt", "doesnt", "isnt", "wasnt", "cant",
             "wont", "nahi", "nhi", "nahin", "nai", "mat", "na", "naa", "nope", "nothing",
             "nobody", "none", "bina", "mt"}

# the paper removes English + Hinglish stopwords; most standard lists include negations
PAPER_STOPWORDS = set(ENGLISH_STOP) | set(HINGLISH_STOP)
SAFE_STOPWORDS = PAPER_STOPWORDS - NEGATIONS


def remove_stopwords(text, stopwords):
    return " ".join(w for w in text.split() if w not in stopwords)


def has_negation(text):
    return any(w in NEGATIONS for w in text.split())


def add_text_versions(splits, max_pairs=None):
    """Adds `cleaned` and `normalized` columns to every split (dictionary from train only)."""
    for df in splits.values():
        df["raw"] = df["text"].astype(str)
        df["cleaned"] = df["raw"].map(clean)
    dictionary = learn_dictionary(splits["train"]["cleaned"], max_pairs=max_pairs)
    for df in splits.values():
        df["normalized"] = df["cleaned"].map(lambda t: normalize(t, dictionary))
    return splits, dictionary


if __name__ == "__main__":
    demo = "@ Ravi bht acchaaa yrr https // t.co/X 😂😂"
    c = clean(demo)
    print("raw:       ", demo)
    print("cleaned:   ", c)
    print("normalized:", normalize(c, SEED_DICT))
