"""Re-test the base paper's (Sinha et al., 2025) pipeline on real gold labels.

The paper labels WhatsApp messages with VADER, then trains an LSTM on those labels
and reports 97.89% accuracy. Four quick experiments (~2-3 minutes total):

 1. VADER alone vs human labels   - how good are the labels the paper trains on?
 2. Train on VADER labels         - score against VADER labels (paper-style) and
                                    against human labels (reality). Done with the
                                    paper's LSTM and with an SVM.
 3. Stopword removal              - the paper removes English + Hinglish stopwords;
                                    standard lists also delete "not"/"nahi".
 4. Hybrid features               - SVM char n-grams + VADER scores as extra features.

Writes results/vader_study.csv and prints a short table.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from common import LABEL2ID, LABELS, RESULTS_DIR, load_splits, save_predictions, set_seed
from deep_models import Vocab, build_model, predict, train
from preprocess import PAPER_STOPWORDS, SAFE_STOPWORDS, add_text_versions, has_negation, remove_stopwords

vader = SentimentIntensityAnalyzer()


def vader_scores(texts):
    return np.array([[s["neg"], s["neu"], s["pos"], s["compound"]]
                     for s in map(vader.polarity_scores, texts)])


def vader_label(compound):
    """Equation (1) in the paper."""
    return np.where(compound > 0.05, "positive", np.where(compound < -0.05, "negative", "neutral"))


def scores(name, gold, pred, **extra):
    return {"experiment": name, "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
            "accuracy": round(accuracy_score(gold, pred), 4), **extra}


def svm_char(train_text, train_y, extra_train=None, extra_test=None, test_text=None):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr, Xte = vec.fit_transform(train_text), vec.transform(test_text)
    if extra_train is not None:
        Xtr, Xte = sp.hstack([Xtr, extra_train]).tocsr(), sp.hstack([Xte, extra_test]).tocsr()
    clf = LinearSVC(C=0.1).fit(Xtr, train_y)
    return clf.predict(Xte)


def paper_lstm(train_text, train_y, val_text, val_y, test_text, device):
    vocab = Vocab(train_text)
    enc = lambda texts: vocab.encode(list(texts), 60)
    to_y = lambda labels: torch.tensor([LABEL2ID[l] for l in labels])
    model = build_model("lstm", len(vocab))
    model, _ = train(model, (*enc(train_text), to_y(train_y)), (*enc(val_text), to_y(val_y)),
                     device, epochs=6, verbose=False)
    ids, lens = enc(test_text)
    return [LABELS[i] for i in predict(model, ids, lens, device).argmax(1)]


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, _ = add_text_versions(load_splits())
    tr, te = splits["train"], splits["test"]
    rows = []

    # 1. VADER zero-shot
    tr_v, te_v = vader_scores(tr.cleaned), vader_scores(te.cleaned)
    te_vader_label = vader_label(te_v[:, 3])
    save_predictions("vader", te.id, te.label, te_vader_label)
    no_hit = float(np.mean(te_v[:, 3] == 0))
    rows.append(scores("1. VADER alone (vs human labels)", te.label, te_vader_label,
                       note=f"{100 * no_hit:.0f}% of test tweets get compound=0 (no lexicon word found)"))
    print(f"VADER finds no sentiment word in {100 * no_hit:.0f}% of test tweets")

    # 2. the paper's set-up: VADER labels, 80/20 split, LSTM
    tr_vader_label = vader_label(tr_v[:, 3])
    a_text, b_text, a_y, b_y = train_test_split(tr.cleaned.tolist(), tr_vader_label.tolist(),
                                                test_size=0.2, random_state=42)
    pred_b = paper_lstm(a_text, a_y, b_text, b_y, b_text, device)
    rows.append(scores("2a. Paper LSTM on VADER labels, scored vs VADER labels (paper-style)", b_y, pred_b))
    pred_te = paper_lstm(a_text, a_y, b_text, b_y, te.cleaned, device)
    rows.append(scores("2b. Same LSTM, scored vs human test labels", te.label, pred_te))

    pred_b = svm_char(a_text, a_y, test_text=b_text)
    rows.append(scores("2c. SVM on VADER labels, scored vs VADER labels", b_y, pred_b))
    pred_te = svm_char(a_text, a_y, test_text=te.cleaned)
    rows.append(scores("2d. Same SVM, scored vs human test labels", te.label, pred_te))

    pred_te = paper_lstm(tr.cleaned, tr.label, splits["validation"].cleaned, splits["validation"].label,
                         te.cleaned, device)
    rows.append(scores("2e. Paper LSTM trained on human labels", te.label, pred_te))

    # 3. stopword removal (paper's step) - overall and on tweets with negation
    neg = te.cleaned.map(has_negation).values
    for tag, stop in [("none", set()), ("paper list (drops not/nahi)", PAPER_STOPWORDS),
                      ("list without negations", SAFE_STOPWORDS)]:
        f = lambda s: s.map(lambda t: remove_stopwords(t, stop))
        pred = svm_char(f(tr.cleaned), tr.label, test_text=f(te.cleaned))
        rows.append(scores(f"3. SVM, stopwords removed: {tag}", te.label, pred,
                           negation_f1=round(f1_score(te.label[neg], pred[neg], average="macro"), 4)))

    # 4. hybrid: char n-grams + VADER scores
    pred = svm_char(tr.cleaned, tr.label, sp.csr_matrix(tr_v), sp.csr_matrix(te_v), te.cleaned)
    save_predictions("svm_vader_hybrid", te.id, te.label, pred)
    rows.append(scores("4. Hybrid: SVM char + VADER scores", te.label, pred))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "vader_study.csv", index=False)
    pd.set_option("display.max_colwidth", 80, "display.width", 200)
    print(table.drop(columns=[c for c in ("note",) if c in table]).to_string(index=False))


if __name__ == "__main__":
    main()
