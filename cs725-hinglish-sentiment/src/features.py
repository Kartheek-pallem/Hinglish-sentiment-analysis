"""Classical features: TF-IDF word n-grams, character n-grams, or both.

Always fitted on the training texts only, then applied to the other splits.

    from src.features import build_features
    X_train, X_val, X_test, vec = build_features(train.text, val.text, test.text, kind="word")
"""
from __future__ import annotations

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


class _Both:
    """Word and character TF-IDF side by side."""
    def __init__(self, word_kw=None, char_kw=None):
        self.word = TfidfVectorizer(lowercase=True, **(word_kw or WORD))
        self.char = TfidfVectorizer(lowercase=True, **(char_kw or CHAR))

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def make_vectorizer(kind: str = "word", **overrides):
    if kind == "word":
        return TfidfVectorizer(lowercase=True, **{**WORD, **overrides})
    if kind == "char":
        return TfidfVectorizer(lowercase=True, **{**CHAR, **overrides})
    if kind == "both":
        return _Both()
    raise ValueError("kind must be 'word', 'char' or 'both'")


def build_features(train_texts, *other_texts, kind: str = "word", **overrides):
    """Fit on train_texts; return (X_train, X_other1, X_other2, ..., vectorizer)."""
    vec = make_vectorizer(kind, **overrides)
    X_train = vec.fit_transform(list(train_texts))
    others = [vec.transform(list(t)) for t in other_texts]
    return (X_train, *others, vec)
