"""Small PyTorch models trained from scratch: the paper's LSTM, BiLSTM, CNN and a char-CNN.

All of them train in a few seconds per epoch on a Colab T4 (and under a minute
per epoch on CPU), so the whole set finishes in a few minutes.
"""

import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = 0, 1


# ---------------------------------------------------------------- text -> ids

class Vocab:
    def __init__(self, texts, char_level=False, min_freq=2, max_size=30000):
        self.char_level = char_level
        counts = Counter(tok for t in texts for tok in self.split(t))
        common = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + common
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def split(self, text):
        return list(text) if self.char_level else text.split()

    def encode(self, texts, max_len):
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        lengths = np.ones(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = [self.stoi.get(w, UNK) for w in self.split(t)][:max_len]
            if toks:
                ids[i, :len(toks)] = toks
                lengths[i] = len(toks)
        return torch.from_numpy(ids), torch.from_numpy(lengths)

    def __len__(self):
        return len(self.itos)


# ---------------------------------------------------------------- models

class PaperLSTM(nn.Module):
    """Exactly the paper's design: Embedding -> LSTM(100) -> Dense(ReLU) -> softmax(3)."""

    def __init__(self, vocab_size, emb=128, hidden=100, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])


class BiLSTM(nn.Module):
    """Reads the tweet both ways; max + mean pooling over all words, not just the last one."""

    def __init__(self, vocab_size, emb=128, hidden=128, n_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != PAD).unsqueeze(-1)
        max_pool = out.masked_fill(~mask, -1e4).max(1).values
        mean_pool = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([max_pool, mean_pool], dim=1)))


class TextCNN(nn.Module):
    """Kim (2014): filters of width 3/4/5 look for short phrases like 'bilkul bakwas'."""

    def __init__(self, vocab_size, emb=128, n_filters=100, widths=(3, 4, 5), n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, n_filters, w, padding=w // 2) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.out(self.drop(torch.cat(pooled, dim=1)))


def build_model(name, vocab_size):
    if name == "lstm":
        return PaperLSTM(vocab_size)
    if name == "bilstm":
        return BiLSTM(vocab_size)
    if name in ("cnn", "charcnn"):
        if name == "charcnn":
            return TextCNN(vocab_size, emb=48, n_filters=128, widths=(3, 5, 7))
        return TextCNN(vocab_size)
    raise ValueError(name)


# ---------------------------------------------------------------- training

def predict(model, ids, lengths, device, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            logits = model(ids[i:i + batch_size].to(device), lengths[i:i + batch_size].to(device))
            probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs).numpy()


def train(model, train_data, val_data, device, epochs=8, lr=2e-3, batch_size=64, patience=2, verbose=True):
    """Adam + cross-entropy (as in the paper), early stopping on validation macro-F1."""
    x, lens, y = train_data
    xv, lv, yv = val_data
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad_epochs = -1, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx].to(device), lens[idx].to(device)), y[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        val_pred = predict(model, xv, lv, device).argmax(1)
        f1 = f1_score(yv.numpy(), val_pred, average="macro")
        if verbose:
            print(f"  epoch {epoch}: loss={total / len(x):.4f} val_macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state, bad_epochs = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1


def count_params(model):
    return sum(p.numel() for p in model.parameters())
