"""Fine-tune a pre-trained transformer (default: HingBERT) on SentiMix.

    python src/train_transformer.py                                   # HingBERT, 3 epochs
    python src/train_transformer.py --model l3cube-pune/hing-roberta --name hingroberta
    python src/train_transformer.py --model bert-base-multilingual-cased --name mbert
    python src/train_transformer.py --max_train 2000 --epochs 1       # quick check on CPU

Needs a GPU for normal use: about 3-4 minutes on a free Colab T4
(max length 64, batch 32, mixed precision).
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import LABEL2ID, LABELS, MODELS_DIR, Timer, load_splits, save_cost, save_predictions, set_seed
from preprocess import add_text_versions


def batches(enc, labels, batch_size, shuffle):
    order = np.random.permutation(len(labels)) if shuffle else np.arange(len(labels))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = {k: v[idx] for k, v in enc.items()}
        yield batch, labels[idx]


def predict(model, enc, device, batch_size=128):
    model.eval()
    n = len(enc["input_ids"])
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for i in range(0, n, batch_size):
            batch = {k: v[i:i + batch_size].to(device) for k, v in enc.items()}
            out.append(torch.softmax(model(**batch).logits.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="l3cube-pune/hing-bert")
    ap.add_argument("--name", default="hingbert")
    ap.add_argument("--text", default="cleaned", choices=["raw", "cleaned", "normalized"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_train", type=int, default=None, help="use only N training tweets (quick test)")
    ap.add_argument("--no_save", action="store_true", help="do not save the fine-tuned model")
    args = ap.parse_args()

    set_seed(42)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():      # Apple Silicon Mac
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)
    if device == "cpu":
        print("WARNING: no GPU. In Colab: Runtime > Change runtime type > T4 GPU.")

    splits, _ = add_text_versions(load_splits())
    if args.max_train:
        splits["train"] = splits["train"].sample(args.max_train, random_state=42)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)

    def encode(df):
        enc = tok(df[args.text].tolist(), truncation=True, max_length=args.max_len,
                  padding="max_length", return_tensors="pt")
        return dict(enc), torch.tensor(df.label.map(LABEL2ID).values)

    (tr_x, tr_y), (va_x, va_y), (te_x, _) = (encode(splits[s]) for s in ("train", "validation", "test"))

    steps = args.epochs * ((len(tr_y) + args.batch_size - 1) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_f1, best_state = -1, None
    with Timer() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for step, (batch, y) in enumerate(batches(tr_x, tr_y, args.batch_size, shuffle=True)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    loss = model(**batch, labels=y.to(device)).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if step % 100 == 0:
                    print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")
            val_pred = predict(model, va_x, device).argmax(1)
            f1 = f1_score(va_y.numpy(), val_pred, average="macro")
            print(f"epoch {epoch}: val macro-F1 = {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs = predict(model, te_x, device)
    test = splits["test"]
    pred = [LABELS[i] for i in probs.argmax(1)]
    test_f1 = f1_score(test.label, pred, average="macro")
    print(f"\n{args.name}: val={best_f1:.4f} test={test_f1:.4f} train time {train_timer.seconds:.0f}s")
    save_predictions(args.name, test.id, test.label, pred, probs)

    # CPU latency on 300 test tweets (same hardware as the other models)
    model_cpu = model.float().to("cpu")
    sample = {k: v[:300] for k, v in te_x.items()}
    with Timer() as t:
        predict(model_cpu, sample, "cpu", batch_size=64)
    n_params = sum(p.numel() for p in model_cpu.parameters())
    save_cost(args.name, family="transformer", params=n_params, train_seconds=round(train_timer.seconds, 1),
              infer_ms_per_1k=round(t.seconds / 300 * 1e6, 2), size_mb=round(n_params * 4 / 1e6, 1), device=device)

    if not args.no_save:
        out = MODELS_DIR / args.name
        model_cpu.save_pretrained(out)
        tok.save_pretrained(out)
        (out / "text_version.txt").write_text(args.text)
        print("saved", out)


if __name__ == "__main__":
    main()
