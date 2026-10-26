"""Create difficulty slices for the SentiMix test tweets.

A slice is a named subset of examples, such as tweets containing negation.
The code creates one row per test tweet and writes data/processed/slices.csv.

Run from the project root:
    python -m src.slices
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

# Starter heuristic list, not a perfect linguistic resource.
# Coordinate this list with Person 2's normalization dictionary.
SPELLING_VARIANTS = {
    "acha", "accha", "achchha",
    "bohot", "bht",
    "nahin", "nahi", "nhi", "nai",
    "muje", "tumhe", "kyu",
}

# Negation words in English and common Romanized Hindi/Hinglish forms.
# "na" is intentionally not included because it can be used in several ways.
NEGATION_RE = re.compile(
    r"\b(?:nahi|nahin|nhi|nai|mat|no|not|never|without|n't)\b",
    flags=re.IGNORECASE,
)


def build_slices(
    data_dir: str = "data/processed",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build test-set slice flags and code-mixing statistics.

    Inputs:
      sentimix_test.csv: id, text, label, source, split
      word_tags.csv: id, split, position, token, tag

    Output: one row per test tweet, including booleans for the slices.
    """
    test_path = os.path.join(data_dir, "sentimix_test.csv")
    tags_path = os.path.join(data_dir, "word_tags.csv")
    if output_path is None:
        output_path = os.path.join(data_dir, "slices.csv")

    test = pd.read_csv(test_path, dtype={"id": str}, keep_default_na=False)
    tags = pd.read_csv(tags_path, dtype={"id": str}, keep_default_na=False)

    # Keep only word-level annotations belonging to test tweets.
    test_tags = tags[tags["split"] == "test"].copy()

    # Count Hindi and English tokens per tweet. Punctuation/emoticons (O/EMT)
    # are excluded from the code-mixing calculation.
    lang_counts = (
        test_tags[test_tags["tag"].isin(["Eng", "Hin"])]
        .pivot_table(index="id", columns="tag", values="token",
                     aggfunc="count", fill_value=0)
        .reindex(columns=["Eng", "Hin"], fill_value=0)
    )
    lang_counts.columns = ["english_tokens", "hindi_tokens"]
    lang_counts["language_tokens"] = (
        lang_counts["english_tokens"] + lang_counts["hindi_tokens"]
    )

    # Index ranges from 0 to 1: 0 means one language dominates completely;
    # 1 means the English and Hindi token counts are equal.
    denominator = lang_counts["language_tokens"].replace(0, pd.NA)
    lang_counts["code_mix_index"] = (
        2 * lang_counts[["english_tokens", "hindi_tokens"]].min(axis=1)
        / denominator
    ).fillna(0.0)

    lang_counts["code_mixed"] = (
        (lang_counts["english_tokens"] > 0)
        & (lang_counts["hindi_tokens"] > 0)
    )

    def mix_level(row):
        if row["language_tokens"] == 0 or not row["code_mixed"]:
            return "none_or_one_language"
        if row["code_mix_index"] < 0.34:
            return "low"
        if row["code_mix_index"] < 0.67:
            return "medium"
        return "high"

    lang_counts["code_mix_level"] = lang_counts.apply(mix_level, axis=1)

    # Use the text column for simple, transparent lexical heuristics.
    result = test[["id", "text", "label"]].copy()
    result["text"] = result["text"].fillna("").astype(str)
    result["has_negation"] = result["text"].map(
        lambda text: bool(NEGATION_RE.search(text))
    )

    def has_spelling_variant(text: str) -> bool:
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        return any(token in SPELLING_VARIANTS for token in tokens)

    result["has_spelling_variation"] = result["text"].map(has_spelling_variant)

    # Every test ID should receive tag-derived language counts. Fill missing
    # tag counts safely, while keeping the tweet in the output.
    result = result.merge(lang_counts.reset_index(), on="id", how="left")
    for col in ["english_tokens", "hindi_tokens", "language_tokens"]:
        result[col] = result[col].fillna(0).astype(int)
    result["code_mix_index"] = result["code_mix_index"].fillna(0.0)
    result["code_mixed"] = result["code_mixed"].fillna(False).astype(bool)
    result["code_mix_level"] = result["code_mix_level"].fillna(
        "none_or_one_language"
    )

    # Preserve stable row order and avoid duplicating the full tweet text in
    # later reports if the team wants to share slices.csv.
    result = result.sort_values("id").reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved {len(result)} test-tweet slice rows to {output_path}")
    print(f"Tweets with code-mixing: {int(result['code_mixed'].sum())}")
    print(f"Tweets with negation: {int(result['has_negation'].sum())}")
    print(
        "Tweets with spelling-variation heuristic: "
        f"{int(result['has_spelling_variation'].sum())}"
    )
    print("Code-mixing levels:")
    print(result["code_mix_level"].value_counts().to_string())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    build_slices(args.data_dir, args.output)
