"""clean(text): light, safe clean-up of one tweet.  Owner: Person 2 (preprocessing).

"Cleaned" text is one of the three preprocessing levels in research question Q1
(raw -> cleaned -> normalized). Cleaning only removes noise; it never changes a
real word (changing spellings is normalize()'s job).

What clean() does, in order:
  1. Repairs broken characters in the test file.  The SentiMix test file was saved
     with the wrong encoding, so "…" appears as "â€¦" and 😂 as "ðŸ˜‚" in about 70%
     of test tweets, but never in train or validation.  A model trained on "…" and
     "😂" cannot recognise them in the test set, so we repair them.
  2. Removes links    "https // t . co / AbC123"  -> ""
  3. Removes user names "@ narendramodi", "@_ Mansoor" -> ""
  4. Keeps hashtag words "# IndiaWins" -> "IndiaWins"
  5. Removes the retweet marker "RT" at the start
  6. Lower-cases everything
  7. Shortens stretched letters "yaaaaar" -> "yaar", "sooooo" -> "soo"
     and repeated punctuation "!!!!!" -> "!!"
  8. Tidies spaces

Emojis are kept: they carry a lot of sentiment.

    >>> clean("@ Ravi _ 22 Bahut achhaaaa yaar !!!!! 😂 https // t . co / X57U9")
    'bahut achhaa yaar !! 😂'

Run `python3 -m src.clean` to see before/after examples from the real data.
"""
from __future__ import annotations

import re

_LINK = re.compile(r"https?\s*:?\s*/\s*/\s*(?:t\s*\.?\s*co\s*/\s*\S+|\S+)|\bhttps?\s*$|\bwww\s*\.\s*\S+", re.I)
# "t . co / xyz", "t co / xyz", "tco / xyz" left over when the link was cut
_LINK_TAIL = re.compile(r"\bt\s*\.?\s*co\s*/\s*\S+", re.I)
# "@ user", "@user", "@_ user", and split names like "@ Sana _ arshaa" or "@ PM _ Narendermodi"
# (a single "_" joins the pieces of one name; "__" or a space without "_" ends it)
_MENTION = re.compile(r"@\s*_*\s*\w+(?:\s*(?<!_)_(?!_)\s*(?!@)\w+)*")
_LONE_UNDERSCORE = re.compile(r"(?<!\S)_+(?!\S)")  # the "_" pieces of split user names
_HASHTAG = re.compile(r"#\s*(\w)")
_RT = re.compile(r"^\s*rt\b\s*:?", re.I)
_STRETCH = re.compile(r"([a-z])\1{2,}")           # 3+ same letters -> 2
_PUNCT_RUN = re.compile(r"([!?.,])\1{2,}")       # 3+ same punctuation -> 2
_SPACES = re.compile(r"\s+")
_MOJIBAKE_HINT = re.compile(r"[ÂÃâð][\x80-\xbfŒ-™]")


def _to_bytes(token: str) -> bytes | None:
    """Turn wrongly-decoded text back into its original bytes (Windows-1252, then Latin-1)."""
    out = bytearray()
    for ch in token:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))
            else:
                return None
    return bytes(out)


def fix_encoding(text: str) -> str:
    """Repair 'â€¦' -> '…' and 'ðŸ˜‚' -> '😂'. Words that are already correct are left alone."""
    if not _MOJIBAKE_HINT.search(text):
        return text
    fixed = []
    for token in text.split(" "):
        if _MOJIBAKE_HINT.search(token):
            raw = _to_bytes(token)
            if raw is not None:
                try:
                    token = raw.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        fixed.append(token)
    return " ".join(fixed)


def clean(text) -> str:
    text = fix_encoding(str(text)).replace("\u00c2", " ")   # a stray "Â" is left over from a broken space
    text = _LINK.sub(" ", text)
    text = _LINK_TAIL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _LONE_UNDERSCORE.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _RT.sub(" ", text)
    text = text.lower()
    text = _STRETCH.sub(r"\1\1", text)
    text = _PUNCT_RUN.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()


if __name__ == "__main__":
    import sys

    import pandas as pd

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    for split in ("train", "test"):
        df = pd.read_csv(f"{data_dir}/sentimix_{split}.csv", dtype={"id": str})
        print(f"\n=== {split}: 5 examples ===")
        for t in df.text.sample(5, random_state=7):
            print("BEFORE:", t)
            print("AFTER: ", clean(t), "\n")
        broken = df.text.str.contains(_MOJIBAKE_HINT).mean()
        still = pd.Series([clean(t) for t in df.text]).str.contains(_MOJIBAKE_HINT).mean()
        print(f"{split}: tweets with broken characters {broken:.1%} before clean(), {still:.1%} after")
